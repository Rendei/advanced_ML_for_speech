"""Aggregation of out-of-fold predictions: metric tables, cluster bootstrap, gradient-conflict and weight summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .config import BASELINE, REGIME_BY_NAME, TASKS, VAD_TASKS, Config, is_regression, primary_metric
from .experiment import Results
from .tasks import ccc, class_scores, task_scores

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float

    def fmt(self, digits: int = 1, pct: bool = True) -> str:
        k = 100.0 if pct else 1.0
        return f"{self.mean * k:.{digits}f} ± {self.std * k:.{digits}f}"


def truth(res: Results, task: str) -> F64 | I64:
    """Ground truth of a task for all utterances (class indices or values on the original 1..5 scale)."""
    if task == "emotion":
        return res.emotion
    if task == "gender":
        return res.gender
    return res.vad[:, VAD_TASKS.index(task)]


def regimes_with(res: Results, task: str) -> list[str]:
    return [r for r in res.regimes if task in REGIME_BY_NAME[r].tasks]


def per_seed(res: Results, regime: str, task: str) -> list[dict[str, float]]:
    """Metrics of one (regime, task) per seed, over all held-out utterances (every session is a test session once)."""
    m = res.covered
    y = truth(res, task)[m]
    return [task_scores(task, y, p[m]) for p in res.preds[(regime, task)]]


def metric(res: Results, regime: str, task: str, name: str | None = None) -> MeanStd:
    """Mean ± std over seeds of one metric (the primary one unless ``name`` is given)."""
    key = name or primary_metric(task)
    values = [s[key] for s in per_seed(res, regime, task)]
    return MeanStd(float(np.mean(values)), float(np.std(values)))


def fold_metric(res: Results, regime: str, task: str) -> list[MeanStd]:
    """Primary metric on every held-out session (mean ± std over seeds)."""
    key = primary_metric(task)
    scores = res.fold_scores[(regime, task)]
    return [
        MeanStd(
            float(np.mean([scores[s][f][key] for s in range(len(scores))])),
            float(np.std([scores[s][f][key] for s in range(len(scores))])),
        )
        for f in range(len(scores[0]))
    ]


@dataclass(frozen=True)
class BootDiff:
    a: str
    b: str
    task: str
    diff: float  # metric(a) - metric(b) on the observed data
    lo: float
    hi: float
    p_not_better: float  # share of resamples in which a is not better than b


def bootstrap_diff(res: Results, a: str, b: str, task: str, cfg: Config) -> BootDiff:
    """Paired *cluster* bootstrap: whole dialogues are resampled, because utterances of one dialogue are correlated.

    The score is averaged over seeds inside every resample; resamples of the two regimes share the same dialogues.
    """
    m = np.flatnonzero(res.covered)
    y = truth(res, task)[m]
    groups = [np.flatnonzero(res.dialogue[m] == g) for g in np.unique(res.dialogue[m])]
    regression = is_regression(task)
    key = primary_metric(task)

    def prepare(regime: str) -> list[F64 | I64]:
        arrs = res.preds[(regime, task)]
        return [p[m] if regression else p[m].argmax(axis=1) for p in arrs]

    pa, pb = prepare(a), prepare(b)
    n_classes = 4 if task == "emotion" else 2

    def score(rows: list[F64 | I64], idx: I64) -> float:
        if regression:
            return float(
                np.mean([ccc(np.asarray(y[idx], dtype=np.float64), np.asarray(r[idx], dtype=np.float64)) for r in rows])
            )
        return float(
            np.mean(
                [
                    class_scores(np.asarray(y[idx], dtype=np.int64), np.asarray(r[idx], dtype=np.int64), n_classes)[key]
                    for r in rows
                ]
            )
        )

    rng = np.random.default_rng(12345)
    full = np.arange(len(m))
    obs = score(pa, full) - score(pb, full)
    diffs = np.empty(cfg.n_bootstrap)
    for i in range(cfg.n_bootstrap):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        diffs[i] = score(pa, idx) - score(pb, idx)
    return BootDiff(
        a, b, task, obs, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), float((diffs <= 0).mean())
    )


def emotion_gains(res: Results, cfg: Config) -> list[BootDiff]:
    """Every emotion-trained regime against the single-task emotion baseline."""
    return [bootstrap_diff(res, r, BASELINE, "emotion", cfg) for r in regimes_with(res, "emotion") if r != BASELINE]


def aux_gains(
    res: Results, cfg: Config, regimes: tuple[str, ...] = ("mtl_equal", "mtl_uncertainty", "mtl_pcgrad")
) -> list[BootDiff]:
    """For every auxiliary task: all-task MTL regimes against the single-task model of that task."""
    out: list[BootDiff] = []
    for task in TASKS[1:]:
        single = f"st_{task}"
        if single not in res.regimes:
            continue
        out += [bootstrap_diff(res, r, single, task, cfg) for r in regimes if r in res.regimes]
    return out


def emotion_recall(res: Results, regime: str) -> F64:
    """[4] per-class recall of the emotion task, averaged over seeds."""
    m = res.covered
    recs = []
    for p in res.preds[(regime, "emotion")]:
        cm = np.bincount(res.emotion[m] * 4 + p[m].argmax(axis=1), minlength=16).reshape(4, 4).astype(np.float64)
        recs.append(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))
    return np.mean(recs, axis=0)


def mean_cosines(res: Results, regime: str) -> F64 | None:
    """Mean over training epochs of the task-gradient cosine matrix on the shared trunk, [T, T] (None for 1 task)."""
    h = res.histories.get(regime)
    if h is None or not h.cosines:
        return None
    return np.mean(np.stack(h.cosines), axis=0)


def conflict_share(res: Results, regime: str) -> float | None:
    """Share of task pairs with negative gradient cosine, averaged over epochs (probe batch, no update)."""
    h = res.histories.get(regime)
    if h is None or not h.cosines:
        return None
    iu = np.triu_indices(h.cosines[0].shape[0], k=1)
    return float(np.mean([(c[iu] < 0).mean() for c in h.cosines]))


def sweep_rows(res: Results, names: tuple[str, ...]) -> list[tuple[float, MeanStd]]:
    """(auxiliary loss weight, emotion UA) for the regimes of the weight sweep that were run."""
    return [(REGIME_BY_NAME[r].aux_weight, metric(res, r, "emotion")) for r in names if r in res.regimes]
