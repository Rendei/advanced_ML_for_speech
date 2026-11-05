"""The experiment: every regime x every held-out session x every seed."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from .config import REGIME_BY_NAME, Config, is_regression, primary_metric, task_dim
from .features import Features
from .tasks import Standardizer, task_scores
from .tensors import Dataset, session_splits
from .train import History, predict, targets, trace_shapes, train_regime

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass
class Results:
    """Out-of-fold predictions of every (regime, task) and everything needed for the report."""

    emotion: I64  # [N] ground truth, kept so that analysis needs no dataset
    gender: I64  # [N]
    vad: F64  # [N, 3] on the original scale
    session: I64  # [N]
    dialogue: I64  # [N] bootstrap cluster of every utterance
    covered: npt.NDArray[np.bool_]  # [N] utterances that were in some test fold
    regimes: tuple[str, ...]
    seeds: tuple[int, ...]
    preds: dict[tuple[str, str], F64]  # (regime, task) -> [S, N, C] probabilities or [S, N] values
    fold_scores: dict[tuple[str, str], list[list[dict[str, float]]]]  # (regime, task) -> [seed][fold] metrics
    histories: dict[str, History]  # first fold, first seed
    shapes: list[tuple[str, str]]
    n_params: dict[str, int]
    seconds: float


def run_experiment(cfg: Config, feats: Features, device: torch.device) -> Results:
    t0 = time.time()
    ds = Dataset(feats).to(device)
    meta = feats.meta
    n, n_seeds = len(ds), len(cfg.seeds)
    splits = session_splits(meta.session, cfg.n_folds)
    regimes = [REGIME_BY_NAME[name] for name in cfg.regimes]

    preds: dict[tuple[str, str], F64] = {}
    fold_scores: dict[tuple[str, str], list[list[dict[str, float]]]] = {}
    for r in regimes:
        for t in r.tasks:
            shape = (n_seeds, n) if is_regression(t) else (n_seeds, n, task_dim(t))
            preds[(r.name, t)] = np.zeros(shape)
            fold_scores[(r.name, t)] = [[] for _ in cfg.seeds]
    histories: dict[str, History] = {}
    n_params: dict[str, int] = {}
    shapes: list[tuple[str, str]] = []
    covered = np.zeros(n, dtype=bool)

    for sp in splits:
        covered[sp.test] = True
        print(
            f"[fold {sp.fold + 1}/{len(splits)}] test session {int(meta.session[sp.test[0]])}: "
            f"train {len(sp.train)} / val {len(sp.val)} / test {len(sp.test)}",
            flush=True,
        )
        std = Standardizer.fit(ds.vad[torch.as_tensor(sp.train, device=ds.vad.device)])
        for si, seed in enumerate(cfg.seeds):
            for r in regimes:
                model, hist = train_regime(r, cfg, ds, sp, std, seed, device, feats.dim)
                out = predict(model, ds, sp.test, device, std)
                line = []
                for t in r.tasks:
                    preds[(r.name, t)][si, sp.test] = out[t]
                    sc = task_scores(t, targets(ds, sp.test, t), out[t])
                    fold_scores[(r.name, t)][si].append(sc)
                    line.append(f"{t[:3]} {primary_metric(t)} {sc[primary_metric(t)]:.3f}")
                print(
                    f"    seed {seed} {r.name:16s} "
                    + " | ".join(line)
                    + f"  (best epoch {hist.best_epoch + 1}/{len(hist.val_score)})",
                    flush=True,
                )
                if sp.fold == 0 and si == 0:
                    histories[r.name] = hist
                    n_params[r.name] = sum(p.numel() for p in model.parameters())
                    if r.name == "mtl_equal" or (not shapes and r.is_multitask):
                        shapes = trace_shapes(model, ds.batch(sp.test[:8]).to(device))
    return Results(
        emotion=meta.emotion,
        gender=meta.gender,
        vad=meta.vad,
        session=meta.session,
        dialogue=meta.dialogue,
        covered=covered,
        regimes=tuple(r.name for r in regimes),
        seeds=cfg.seeds,
        preds=preds,
        fold_scores=fold_scores,
        histories=histories,
        shapes=shapes,
        n_params=n_params,
        seconds=time.time() - t0,
    )
