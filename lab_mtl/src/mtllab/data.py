"""IEMOCAP (Hugging Face mirror): download, metadata of all tasks, streaming audio decoding."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import soundfile as sf

from .config import RAW_TO_EMOTION, Config

SAMPLE_RATE = 16_000
# Ses02M_impro02_F003: session 02, dialogue "Ses02M_impro02", utterance spoken by a female (F) speaker.
# The letter after the session number marks the dialogue; the speaker's gender is the letter of the suffix.
_UTT_RE = re.compile(r"(?P<dialogue>Ses(?P<session>\d\d)[MF]_.+)_(?P<gender>[MF])\d{3}$")


@dataclass(frozen=True)
class Meta:
    """Labels and metadata of the selected (4-emotion) utterances; every array has length ``len(meta)``."""

    row: npt.NDArray[np.int64]  # global row index in the parquet files
    utt_id: list[str]
    session: npt.NDArray[np.int64]  # 1..5, the speaker-disjoint split unit
    dialogue: npt.NDArray[np.int64]  # 0..D-1, utterances of one dialogue are strongly correlated (bootstrap unit)
    gender: npt.NDArray[np.int64]  # 0 = female, 1 = male (speaker of the utterance)
    emotion: npt.NDArray[np.int64]  # index into EMOTION_CLASSES
    vad: npt.NDArray[np.float64]  # [N, 3] valence, arousal (activation), dominance; ratings on a 1..5 scale
    text: list[str]
    seconds: npt.NDArray[np.float64]  # utterance duration from the corpus timings

    def __len__(self) -> int:
        return len(self.utt_id)


def download(cfg: Config) -> list[Path]:
    """Fetch the parquet shards (≈1.4 GB, cached in ``cfg.cache_dir/hf``)."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        cache_dir=str(cfg.cache_dir / "hf"),
    )
    files = sorted(Path(root, "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {cfg.dataset_repo}")
    return files


def clean_text(text: str) -> str:
    """Collapse whitespace; IEMOCAP transcripts keep tags such as [LAUGHTER], which are informative."""
    return re.sub(r"\s+", " ", text).strip()


def read_meta(files: list[Path], limit: int | None = None) -> Meta:
    """Read the light columns only (no audio) and keep the four target emotions.

    With ``limit`` an evenly spaced subset is taken so that all sessions stay represented.
    """
    cols = ["emotion", "titre", "to_translate", "start_time", "end_time", "valence", "activation", "dominance"]
    rows: list[int] = []
    ids: list[str] = []
    dialogues: list[str] = []
    sessions: list[int] = []
    genders: list[int] = []
    emotions: list[int] = []
    vad: list[tuple[float, float, float]] = []
    texts: list[str] = []
    secs: list[float] = []
    offset = 0
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        for i in range(len(t["emotion"])):
            utt = t["titre"][i]
            if t["emotion"][i] not in RAW_TO_EMOTION or not (m := _UTT_RE.match(utt)):
                continue
            rows.append(offset + i)
            ids.append(utt)
            dialogues.append(m["dialogue"])
            sessions.append(int(m["session"]))
            genders.append(0 if m["gender"] == "F" else 1)
            emotions.append(RAW_TO_EMOTION[t["emotion"][i]])
            vad.append((float(t["valence"][i]), float(t["activation"][i]), float(t["dominance"][i])))
            texts.append(clean_text(t["to_translate"][i]))
            secs.append(float(t["end_time"][i] - t["start_time"][i]))
        offset += len(t["emotion"])
    keep = np.arange(len(rows))
    if limit is not None and limit < len(rows):
        keep = np.unique(np.linspace(0, len(rows) - 1, limit).astype(int))
    dialogue_index = {d: k for k, d in enumerate(sorted(set(dialogues)))}
    return Meta(
        row=np.array(rows, dtype=np.int64)[keep],
        utt_id=[ids[i] for i in keep],
        session=np.array(sessions, dtype=np.int64)[keep],
        dialogue=np.array([dialogue_index[dialogues[i]] for i in keep], dtype=np.int64),
        gender=np.array(genders, dtype=np.int64)[keep],
        emotion=np.array(emotions, dtype=np.int64)[keep],
        vad=np.array(vad, dtype=np.float64)[keep],
        text=[texts[i] for i in keep],
        seconds=np.array(secs)[keep],
    )


def iter_waveforms(files: list[Path], rows: npt.NDArray[np.int64]) -> Iterator[npt.NDArray[np.float32]]:
    """Yield the mono 16 kHz waveform of every requested global row, in the order of ``rows``.

    ``rows`` must be sorted ascending; shards are streamed batch by batch so memory stays small.
    """
    wanted = {int(r): k for k, r in enumerate(rows)}
    pending: dict[int, npt.NDArray[np.float32]] = {}
    next_k = 0
    offset = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=["audio"]):
            audio = batch.column("audio").to_pylist()
            for i, item in enumerate(audio):
                k = wanted.get(offset + i)
                if k is None:
                    continue
                wav, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32", always_2d=False)
                if sr != SAMPLE_RATE:
                    raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                pending[k] = wav
            offset += len(audio)
            while next_k in pending:
                yield pending.pop(next_k)
                next_k += 1
    if next_k != len(rows):
        raise RuntimeError(f"decoded {next_k} of {len(rows)} requested utterances")
