"""Frozen pretrained encoders -> token/frame level feature sequences (extracted once, cached).

* audio: WavLM-base+  waveform [S]       -> mean of hidden layers [lo, hi) -> [T, 768], then average-pooled in time
* text : RoBERTa-base token ids [L]      -> mean of the last K hidden layers -> [L, 768]

Sequences are stored flat (``*_flat``) with per-utterance lengths to keep the cache compact.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from .config import Config
from .data import Meta, iter_waveforms, read_meta

F16 = npt.NDArray[np.float16]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Features:
    meta: Meta
    audio_flat: F16  # [sum(Ta), D]
    audio_len: I64  # [N]
    text_flat: F16  # [sum(Tt), D]
    text_len: I64  # [N]
    tokens: list[list[str]]  # RoBERTa tokens per utterance (for plots)
    examples: dict[int, npt.NDArray[np.float32]]  # class index -> a raw waveform (for plots)

    @property
    def dim(self) -> int:
        return int(self.audio_flat.shape[1])


def _cache_key(cfg: Config) -> str:
    spec = [
        cfg.dataset_repo,
        cfg.audio_model,
        cfg.text_model,
        cfg.audio_layers,
        cfg.text_last_layers,
        cfg.audio_pool,
        cfg.max_audio_seconds,
        cfg.max_text_tokens,
        cfg.limit,
    ]
    return hashlib.sha1(json.dumps(spec).encode()).hexdigest()[:10]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def _extract_audio(cfg: Config, files: list[Path], meta: Meta) -> tuple[list[F16], dict[int, npt.NDArray[np.float32]]]:
    from transformers import AutoFeatureExtractor, AutoModel

    dev = _device()
    fe = AutoFeatureExtractor.from_pretrained(cfg.audio_model)  # type: ignore[no-untyped-call]
    model = AutoModel.from_pretrained(cfg.audio_model).to(dev).eval()
    lo, hi = cfg.audio_layers
    max_samples = int(cfg.max_audio_seconds * 16_000)
    seqs: list[F16] = []
    examples: dict[int, npt.NDArray[np.float32]] = {}
    t0 = time.time()
    for i, wav in enumerate(iter_waveforms(files, meta.row)):
        c = int(meta.label[i])
        if c not in examples and 2.5 <= len(wav) / 16_000 <= 5.0:
            examples[c] = wav.copy()
        wav = wav[:max_samples]
        x = fe(wav, sampling_rate=16_000, return_tensors="pt").input_values.to(dev)  # [1, S]
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            hs = model(x, output_hidden_states=True).hidden_states  # (L+1) x [1, T, D]
        h = torch.stack(hs[lo:hi]).float().mean(dim=0)[0]  # [T, D]
        t = (h.shape[0] // cfg.audio_pool) * cfg.audio_pool
        h = h[:t].reshape(-1, cfg.audio_pool, h.shape[-1]).mean(dim=1)  # [T/pool, D]
        seqs.append(h.cpu().numpy().astype(np.float16))
        if (i + 1) % 500 == 0 or i + 1 == len(meta):
            print(f"  audio {i + 1}/{len(meta)}  ({time.time() - t0:.0f}s)", flush=True)
    del model
    return seqs, examples


@torch.inference_mode()
def _extract_text(cfg: Config, meta: Meta) -> tuple[list[F16], list[list[str]]]:
    from transformers import AutoModel, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(cfg.text_model)
    model = AutoModel.from_pretrained(cfg.text_model).to(dev).eval()
    seqs: list[F16] = []
    tokens: list[list[str]] = []
    bs = 32
    for s in range(0, len(meta), bs):
        batch = meta.text[s : s + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=cfg.max_text_tokens, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        h = torch.stack(hs[-cfg.text_last_layers :]).mean(dim=0)  # [B, L, D]
        for b in range(h.shape[0]):
            n = int(enc["attention_mask"][b].sum())
            seqs.append(h[b, :n].cpu().numpy().astype(np.float16))
            tokens.append(list(tok.convert_ids_to_tokens(enc["input_ids"][b, :n].tolist())))
    del model
    print(f"  text {len(meta)}/{len(meta)}", flush=True)
    return seqs, tokens


def build_features(cfg: Config, files: list[Path]) -> Features:
    """Extract (or load from cache) the frozen features for the selected utterances."""
    cache = cfg.cache_dir / f"features_{_cache_key(cfg)}.npz"
    meta = read_meta(files, cfg.limit)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        tokens = json.loads(str(z["tokens"]))
        examples = {int(k): z[f"example_{k}"] for k in range(cfg.n_classes) if f"example_{k}" in z}
        print(f"features loaded from {cache}", flush=True)
        return Features(meta, z["audio_flat"], z["audio_len"], z["text_flat"], z["text_len"], tokens, examples)

    print(f"extracting features for {len(meta)} utterances on {_device()}", flush=True)
    audio, examples = _extract_audio(cfg, files, meta)
    text, tokens = _extract_text(cfg, meta)
    feats = Features(
        meta=meta,
        audio_flat=np.concatenate(audio),
        audio_len=np.array([len(a) for a in audio], dtype=np.int64),
        text_flat=np.concatenate(text),
        text_len=np.array([len(t) for t in text], dtype=np.int64),
        tokens=tokens,
        examples=examples,
    )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "audio_flat": feats.audio_flat,
        "audio_len": feats.audio_len,
        "text_flat": feats.text_flat,
        "text_len": feats.text_len,
        "tokens": np.array(json.dumps(tokens)),
    }
    arrays.update({f"example_{k}": v for k, v in examples.items()})
    np.savez(cache, **arrays)
    return feats
