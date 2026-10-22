"""The experiment: every model x every held-out session x every seed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from .config import MODEL_NAMES, SEARCH_SPACE, Config, sample_trials
from .features import Features
from .tensors import Dataset, Split, session_splits
from .train import History, predict, scores, trace_shapes, train_model

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
PROBE_MODELS = ("concat", "cross", "bicross")  # models that get the modality-shuffle probe


@dataclass
class AttentionExample:
    index: int  # row in the corpus
    attention: F64  # [Tt, Ta] head-averaged weights of the last cross layer


@dataclass
class TrialRecord:
    """One hyperparameter trial of one model on one outer fold (scored on the validation session only)."""

    fold: int
    kind: str
    trial: int  # 0 = the default configuration
    params: dict[str, float]
    val_ua: float
    chosen: bool


@dataclass
class Results:
    y: I64  # [N] labels
    session: I64  # [N]
    covered: npt.NDArray[np.bool_]  # [N] utterances that were in some test fold
    probs: dict[str, F64]  # kind -> [S, N, C] out-of-fold probabilities
    probs_shuffled: dict[tuple[str, str], F64]  # (kind, "audio"|"text") -> [S, N, C]
    fold_scores: dict[str, list[list[dict[str, float]]]]  # kind -> [seed][fold] metrics on the held-out session
    histories: dict[str, History]  # first fold, first seed
    attention: AttentionExample | None
    shapes: list[tuple[str, str]]
    n_params: dict[str, int]
    seconds: float
    seeds: tuple[int, ...]
    # Only with hyperparameter search: ``probs`` then come from the tuned configs, ``probs_default`` from the defaults.
    probs_default: dict[str, F64] | None = None
    tuning: list[TrialRecord] = field(default_factory=list)


def _tune(
    kind: str, cfg: Config, ds: Dataset, sp: Split, device: torch.device, d_in: int
) -> tuple[Config, list[TrialRecord]]:
    """Pick the best of ``cfg.tune_trials`` configs by UA on ``sp.val``; the test session is never touched."""
    trials = sample_trials(cfg, kind, cfg.tune_trials)
    val_ua: list[float] = []
    for t_cfg in trials:
        _, hist = train_model(kind, t_cfg, ds, sp, cfg.tune_seed, device, d_in)
        val_ua.append(hist.val_ua[hist.best_epoch])
    best = int(np.argmax(val_ua))  # ties go to the earliest trial, i.e. the default
    records = [
        TrialRecord(sp.fold, kind, i, {k: float(getattr(t, k)) for k in SEARCH_SPACE}, u, i == best)
        for i, (t, u) in enumerate(zip(trials, val_ua))
    ]
    print(f"    tune {kind:8s} default val UA {val_ua[0]:.3f} -> best {val_ua[best]:.3f} (trial {best}/{len(trials)})",
          flush=True)
    return trials[best], records


def _pick_attention_example(
    test_idx: I64, feats: Features, p_cross: F64, p_text: F64, y: I64
) -> int:
    """A short, clearly non-neutral utterance that fusion gets right and the text-only model gets wrong."""
    secs, lens = feats.meta.seconds, feats.text_len
    good = [
        int(i)
        for k, i in enumerate(test_idx)
        if y[i] != 0
        and 2.0 <= secs[i] <= 5.0
        and 6 <= lens[i] <= 16
        and p_cross[k].argmax() == y[i]
        and p_text[k].argmax() != y[i]
    ]
    if good:
        return good[0]
    fallback = [int(i) for k, i in enumerate(test_idx) if p_cross[k].argmax() == y[i] and 6 <= lens[i] <= 16]
    return fallback[0] if fallback else int(test_idx[0])


def run_experiment(cfg: Config, feats: Features, device: torch.device) -> Results:
    t0 = time.time()
    ds = Dataset(feats).to(device)
    n, c, n_seeds = len(ds), cfg.n_classes, len(cfg.seeds)
    y = feats.meta.label
    splits = session_splits(feats.meta.session, cfg.n_folds)
    covered = np.zeros(n, dtype=bool)
    probs = {k: np.zeros((n_seeds, n, c)) for k in MODEL_NAMES}
    shuffled = {(k, m): np.zeros((n_seeds, n, c)) for k in PROBE_MODELS for m in ("audio", "text")}
    fold_scores: dict[str, list[list[dict[str, float]]]] = {k: [[] for _ in cfg.seeds] for k in MODEL_NAMES}
    histories: dict[str, History] = {}
    attention: AttentionExample | None = None
    shapes: list[tuple[str, str]] = []
    n_params: dict[str, int] = {}

    tuning: list[TrialRecord] = []
    tuned = cfg.tune_trials > 1
    probs_default = {k: np.zeros((n_seeds, n, c)) for k in MODEL_NAMES} if tuned else None

    for sp in splits:
        covered[sp.test] = True
        print(f"[fold {sp.fold + 1}/{len(splits)}] test session {int(feats.meta.session[sp.test[0]])}: "
              f"train {len(sp.train)} / val {len(sp.val)} / test {len(sp.test)}", flush=True)
        chosen = {k: cfg for k in MODEL_NAMES}  # per-model config used for the final (test) runs of this fold
        if tuned:
            for kind in MODEL_NAMES:
                chosen[kind], records = _tune(kind, cfg, ds, sp, device, feats.dim)
                tuning += records
        for si, seed in enumerate(cfg.seeds):
            trained = {}
            for kind in MODEL_NAMES:
                model, hist = train_model(kind, chosen[kind], ds, sp, seed, device, feats.dim)
                p = predict(model, ds, sp.test, device)
                probs[kind][si, sp.test] = p
                if probs_default is not None:  # what the same seed gives with the untuned defaults
                    if chosen[kind] is cfg:
                        probs_default[kind][si, sp.test] = p
                    else:
                        default_model, _ = train_model(kind, cfg, ds, sp, seed, device, feats.dim)
                        probs_default[kind][si, sp.test] = predict(default_model, ds, sp.test, device)
                fold_scores[kind][si].append(scores(y[sp.test], p.argmax(1), c))
                trained[kind] = model
                if sp.fold == 0 and si == 0:
                    histories[kind] = hist
                    n_params[kind] = sum(q.numel() for q in model.parameters())
                if kind in PROBE_MODELS:
                    for m in ("audio", "text"):
                        shuffled[(kind, m)][si, sp.test] = predict(model, ds, sp.test, device, shuffle=m, seed=seed)
                s = fold_scores[kind][si][-1]
                print(f"    seed {seed} {kind:8s} UA {s['ua']:.3f}  WA {s['wa']:.3f}  F1 {s['f1']:.3f}  "
                      f"(best epoch {hist.best_epoch + 1}/{len(hist.val_ua)})", flush=True)
            if sp.fold == 0 and si == 0:
                idx = _pick_attention_example(sp.test, feats, probs["cross"][0, sp.test], probs["text"][0, sp.test], y)
                cross = trained["cross"]
                b = ds.batch(np.array([idx])).to(device)
                shapes = trace_shapes(cross, ds.batch(sp.test[:8]).to(device))
                with torch.inference_mode():
                    cross.eval()(b)
                assert cross.last_attention is not None
                w = cross.last_attention[0].cpu().numpy().astype(np.float64)  # [Tt, Ta]
                attention = AttentionExample(idx, w[: feats.text_len[idx], : feats.audio_len[idx]])
    return Results(
        y=y,
        session=feats.meta.session,
        covered=covered,
        probs=probs,
        probs_shuffled=shuffled,
        fold_scores=fold_scores,
        histories=histories,
        attention=attention,
        shapes=shapes,
        n_params=n_params,
        seconds=time.time() - t0,
        seeds=cfg.seeds,
        probs_default=probs_default,
        tuning=tuning,
    )
