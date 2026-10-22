"""Aggregation of the out-of-fold predictions: tables, paired bootstrap, example selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .config import CLASSES, MODEL_NAMES, Config
from .experiment import Results
from .features import Features
from .train import confusion, scores

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float

    def fmt(self, pct: bool = True) -> str:
        k = 100.0 if pct else 1.0
        return f"{self.mean * k:.1f} ± {self.std * k:.1f}"


def _pooled(res: Results, probs: F64) -> list[dict[str, float]]:
    """Per seed: metrics over all held-out utterances (every session is a test session once)."""
    m = res.covered
    return [scores(res.y[m], p[m].argmax(1), len(CLASSES)) for p in probs]


def summary(res: Results, probs: dict[str, F64] | None = None) -> dict[str, dict[str, MeanStd]]:
    """kind -> metric -> mean ± std over seeds of the pooled leave-session-out score."""
    out: dict[str, dict[str, MeanStd]] = {}
    for kind in MODEL_NAMES:
        per_seed = _pooled(res, (probs or res.probs)[kind])
        out[kind] = {k: MeanStd(float(np.mean([s[k] for s in per_seed])), float(np.std([s[k] for s in per_seed])))
                     for k in ("wa", "ua", "f1")}
    return out


def shuffle_summary(res: Results) -> dict[tuple[str, str], MeanStd]:
    """UA of fusion models when one modality is replaced by another utterance's (probe of reliance)."""
    return {
        key: MeanStd(
            float(np.mean([s["ua"] for s in _pooled(res, p)])), float(np.std([s["ua"] for s in _pooled(res, p)]))
        )
        for key, p in res.probs_shuffled.items()
    }


def fold_table(res: Results) -> dict[str, list[MeanStd]]:
    """kind -> UA on every held-out session (mean ± std over seeds)."""
    out: dict[str, list[MeanStd]] = {}
    for kind in MODEL_NAMES:
        n_folds = len(res.fold_scores[kind][0])
        out[kind] = [
            MeanStd(
                float(np.mean([res.fold_scores[kind][s][f]["ua"] for s in range(len(res.seeds))])),
                float(np.std([res.fold_scores[kind][s][f]["ua"] for s in range(len(res.seeds))])),
            )
            for f in range(n_folds)
        ]
    return out


def per_class_recall(res: Results, kind: str) -> F64:
    """[C] recall per class, averaged over seeds."""
    m = res.covered
    recs = []
    for p in res.probs[kind]:
        cm = confusion(res.y[m], p[m].argmax(1), len(CLASSES)).astype(np.float64)
        recs.append(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))
    return np.mean(recs, axis=0)


def mean_confusion(res: Results, kind: str) -> F64:
    """Row-normalised confusion matrix of the seed-averaged probabilities, [C, C]."""
    m = res.covered
    cm = confusion(res.y[m], res.probs[kind].mean(axis=0)[m].argmax(1), len(CLASSES)).astype(np.float64)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)


@dataclass(frozen=True)
class BootDiff:
    a: str
    b: str
    metric: str
    diff: float  # mean(a) - mean(b) on the observed data
    lo: float
    hi: float
    p_not_better: float  # share of resamples where a is not better than b


def bootstrap_diff(res: Results, a: str, b: str, cfg: Config, metric: str = "ua") -> BootDiff:
    """Paired bootstrap over utterances; the score is averaged over seeds inside each resample."""
    m = np.flatnonzero(res.covered)
    y = res.y[m]
    pa = np.stack([p[m].argmax(1) for p in res.probs[a]])  # [S, n]
    pb = np.stack([p[m].argmax(1) for p in res.probs[b]])
    rng = np.random.default_rng(12345)

    def score(pred: I64, idx: I64) -> float:
        return float(np.mean([scores(y[idx], row[idx], len(CLASSES))[metric] for row in pred]))

    full = np.arange(len(m))
    obs = score(pa, full) - score(pb, full)
    diffs = np.empty(cfg.n_bootstrap)
    for i in range(cfg.n_bootstrap):
        idx = rng.integers(0, len(m), len(m))
        diffs[i] = score(pa, idx) - score(pb, idx)
    return BootDiff(a, b, metric, obs, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)),
                    float((diffs <= 0).mean()))


def complementarity(res: Results) -> dict[str, float]:
    """Seed-averaged-probability agreement: where does one unimodal model fix the other's error?"""
    m = res.covered
    y = res.y[m]
    pa = res.probs["audio"].mean(axis=0)[m].argmax(1)
    pt = res.probs["text"].mean(axis=0)[m].argmax(1)
    pf = res.probs["cross"].mean(axis=0)[m].argmax(1)
    ok_a, ok_t, ok_f = pa == y, pt == y, pf == y
    return {
        "both_right": float((ok_a & ok_t).mean()),
        "only_audio": float((ok_a & ~ok_t).mean()),
        "only_text": float((~ok_a & ok_t).mean()),
        "both_wrong": float((~ok_a & ~ok_t).mean()),
        "oracle": float((ok_a | ok_t).mean()),
        "cross_fixes_text": float((ok_f & ~ok_t).sum() / max((~ok_t).sum(), 1)),
        "cross_breaks_text": float((~ok_f & ok_t).sum() / max(ok_t.sum(), 1)),
    }


def example_rows(res: Results, feats: Features, limit: int = 8) -> list[dict[str, str]]:
    """Utterances where cross-attention fusion is right while text-only is wrong (and audio-only is right)."""
    m = res.covered
    pa = res.probs["audio"].mean(axis=0).argmax(1)
    pt = res.probs["text"].mean(axis=0).argmax(1)
    pf = res.probs["cross"].mean(axis=0).argmax(1)
    rows: list[dict[str, str]] = []
    for i in np.flatnonzero(m & (pf == res.y) & (pt != res.y) & (pa == res.y)):
        text = feats.meta.text[i]
        if 15 <= len(text) <= 90:
            rows.append({"text": text, "true": CLASSES[res.y[i]], "text_pred": CLASSES[pt[i]],
                         "audio_pred": CLASSES[pa[i]], "fusion_pred": CLASSES[pf[i]]})
        if len(rows) >= limit:
            break
    return rows


def default_summary(res: Results) -> dict[str, dict[str, MeanStd]] | None:
    """Same table as ``summary`` for the untuned default configs (only when a hyperparameter search ran)."""
    return None if res.probs_default is None else summary(res, res.probs_default)


def tuning_rows(res: Results) -> list[dict[str, str]]:
    """One row per (model, fold): validation UA of the default vs. the chosen config and the chosen values."""
    rows: list[dict[str, str]] = []
    for kind in MODEL_NAMES:
        for fold in sorted({t.fold for t in res.tuning if t.kind == kind}):
            trials = [t for t in res.tuning if t.kind == kind and t.fold == fold]
            default = next(t for t in trials if t.trial == 0)
            best = next(t for t in trials if t.chosen)
            p = best.params
            rows.append(
                {
                    "kind": kind,
                    "fold": str(fold + 1),
                    "val_default": f"{default.val_ua * 100:.1f}",
                    "val_best": f"{best.val_ua * 100:.1f}",
                    "lr": f"{p['lr']:g}",
                    "d_model": f"{int(p['d_model'])}",
                    "dropout": f"{p['dropout']:g}",
                    "weight_decay": f"{p['weight_decay']:g}",
                    "layers": f"{int(p['n_cross_layers'])}" if kind in ("cross", "bicross") else "—",
                    "is_default": "да" if best.trial == 0 else "нет",
                }
            )
    return rows
