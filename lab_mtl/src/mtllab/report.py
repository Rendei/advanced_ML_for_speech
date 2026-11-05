"""Runtime reports: ``eda.html`` (data only) and ``report.html`` (EDA + experiment) from this run's numbers."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import platform
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.figure import Figure

from . import viz
from .analysis import (
    BootDiff,
    MeanStd,
    aux_gains,
    bootstrap_diff,
    conflict_share,
    emotion_gains,
    mean_cosines,
    metric,
    regimes_with,
    sweep_rows,
)
from .config import (
    BASELINE,
    EMOTION_CLASSES,
    REGIME_BY_NAME,
    TASK_TITLES,
    TASKS,
    VAD_TASKS,
    WEIGHT_SWEEP,
    Config,
    primary_metric,
)
from .data import Meta
from .experiment import Results

TEMPLATES = Path(__file__).parent


def _png(fig: Figure, name: str, out: Path | None) -> str:
    """Base64 PNG for embedding into the HTML; also saved to ``out/<name>.png`` when ``out`` is given."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.png").write_bytes(buf.getvalue())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _verdict(d: BootDiff) -> str:
    if d.lo > 0:
        return "значимо лучше"
    if d.hi < 0:
        return "значимо хуже"
    return "различие не значимо"


def _diff_row(d: BootDiff, digits: int = 1) -> dict[str, str]:
    k = 100 if primary_metric(d.task) == "ua" else 1
    unit = "п.п." if k == 100 else "CCC"
    return {
        "a": REGIME_BY_NAME[d.a].title,
        "b": REGIME_BY_NAME[d.b].title,
        "task": TASK_TITLES[d.task],
        "diff": f"{d.diff * k:+.{digits + (1 if k == 1 else 0)}f}",
        "lo": f"{d.lo * k:+.{digits + (1 if k == 1 else 0)}f}",
        "hi": f"{d.hi * k:+.{digits + (1 if k == 1 else 0)}f}",
        "unit": unit,
        "p": f"{d.p_not_better:.3f}",
        "verdict": _verdict(d),
    }


# ---- EDA ----------
def eda_context(meta: Meta, fig_dir: Path | None) -> dict[str, Any]:
    """Numbers and figures of the data section (shared by ``eda.html`` and ``report.html``)."""
    figs = {
        "overview": _png(viz.eda_overview(meta), "eda_overview", fig_dir),
        "vad": _png(viz.eda_vad_distributions(meta), "eda_vad_distributions", fig_dir),
        "space": _png(viz.eda_va_space(meta), "eda_va_space", fig_dir),
        "box": _png(viz.eda_class_vad(meta), "eda_class_vad", fig_dir),
        "corr": _png(viz.eda_correlations(meta), "eda_correlations", fig_dir),
    }
    counts = np.bincount(meta.emotion, minlength=4)
    class_rows = [
        {
            "name": EMOTION_CLASSES[c],
            "ru": viz.EMOTION_RU[EMOTION_CLASSES[c]],
            "n": int(counts[c]),
            "share": f"{100 * counts[c] / counts.sum():.1f}",
            "vad": [f"{meta.vad[meta.emotion == c, i].mean():.2f}" for i in range(3)],
        }
        for c in range(4)
    ]
    vad_corr: npt.NDArray[np.float64] = np.asarray(np.corrcoef(meta.vad.T))
    return {
        "n": len(meta),
        "hours": float(meta.seconds.sum() / 3600),
        "n_dialogues": len(np.unique(meta.dialogue)),
        "n_female": int((meta.gender == 0).sum()),
        "n_male": int((meta.gender == 1).sum()),
        "class_rows": class_rows,
        "corr_va": f"{vad_corr[0, 1]:+.2f}",
        "corr_ad": f"{vad_corr[1, 2]:+.2f}",
        "corr_vd": f"{vad_corr[0, 2]:+.2f}",
        "edafigs": figs,
    }


def _environment(cfg: Config) -> dict[str, Any]:
    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cfg": cfg,
        "env": f"Python {platform.python_version()}, torch {torch.__version__}, "
        f"{'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        "limited": cfg.limit is not None,
    }


def _render(template: str, ctx: dict[str, Any], out: Path) -> Path:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.get_template(template).render(**ctx), encoding="utf-8")
    return out


def build_eda(cfg: Config, meta: Meta) -> Path:
    """``eda.html``: exploratory analysis of the data only (no training, no GPU)."""
    ctx = {**_environment(cfg), **eda_context(meta, cfg.out_dir / "figures")}
    return _render("eda.html.j2", ctx, cfg.out_dir / "eda.html")


# ---- full report ----------
def build_report(cfg: Config, meta: Meta, res: Results) -> Path:
    """Render ``outputs/report.html`` (+ figures/ and metrics.json) from the finished experiment."""
    fig_dir = cfg.out_dir / "figures"
    ctx: dict[str, Any] = {**_environment(cfg), **eda_context(meta, fig_dir)}

    # ---- emotion: every regime against the single-task baseline ----------
    gains = emotion_gains(res, cfg)
    em_regimes = regimes_with(res, "emotion")
    gain_by = {d.a: d for d in gains}
    emotion_rows = []
    for r in em_regimes:
        d = gain_by.get(r)
        emotion_rows.append(
            {
                "name": r,
                "title": REGIME_BY_NAME[r].title,
                "ua": metric(res, r, "emotion").fmt(),
                "wa": metric(res, r, "emotion", "wa").fmt(),
                "f1": metric(res, r, "emotion", "f1").fmt(),
                "params": f"{res.n_params.get(r, 0) / 1e3:.0f}k",
                "base": r == BASELINE,
                "delta": f"{d.diff * 100:+.1f}" if d else "—",
                "ci": f"[{d.lo * 100:+.1f}; {d.hi * 100:+.1f}]" if d else "—",
                "verdict": _verdict(d) if d else "baseline",
            }
        )
    mtl_gains = [d for d in gains if REGIME_BY_NAME[d.a].is_multitask]
    best = max(mtl_gains, key=lambda d: d.diff) if mtl_gains else None

    extra: list[BootDiff] = []  # balancing strategies against plain equal weights
    for a in ("mtl_uncertainty", "mtl_pcgrad"):
        if a in res.regimes and "mtl_equal" in res.regimes:
            extra.append(bootstrap_diff(res, a, "mtl_equal", "emotion", cfg))

    # ---- auxiliary tasks ----------
    aux = aux_gains(res, cfg)
    aux_rows = []
    for t in TASKS[1:]:
        single = f"st_{t}"
        if single not in res.regimes:
            continue
        row: dict[str, Any] = {
            "task": TASK_TITLES[t],
            "metric": "CCC" if primary_metric(t) == "ccc" else "UA, %",
            "pct": primary_metric(t) == "ua",
            "single": _fmt_task(metric(res, single, t), t),
        }
        for r in ("mtl_equal", "mtl_uncertainty", "mtl_pcgrad"):
            d = next((x for x in aux if x.task == t and x.a == r), None)
            row[r] = (
                (_fmt_task(metric(res, r, t), t), _diff_row(d)["diff"] if d else "", _verdict(d) if d else "")
                if r in res.regimes
                else ("", "", "")
            )
        aux_rows.append(row)

    # ---- gradient conflicts ----------
    cos_regimes = [r for r in ("mtl_equal", "mtl_uncertainty", "mtl_pcgrad") if mean_cosines(res, r) is not None]
    pair_rows = []
    for i, j in combinations(range(len(TASKS)), 2):
        cells = []
        for r in cos_regimes:
            c = mean_cosines(res, r)
            assert c is not None
            cells.append(f"{c[i, j]:+.2f}")
        pair_rows.append({"pair": f"{TASK_TITLES[TASKS[i]]} — {TASK_TITLES[TASKS[j]]}", "cells": cells})
    shares = {r: conflict_share(res, r) for r in cos_regimes}

    sweep = sweep_rows(res, WEIGHT_SWEEP)
    h_unc = res.histories.get("mtl_uncertainty")
    unc_final = {
        TASK_TITLES[t]: f"{w:.2f}"
        for t, w in (h_unc.uncertainty_weights[-1].items() if h_unc and h_unc.uncertainty_weights else [])
    }
    gender_acc = metric(res, "st_gender", "gender") if "st_gender" in res.regimes else None

    figs = ctx["edafigs"]
    figs["arch"] = _png(viz.architecture(cfg.trunk), "architecture", fig_dir)
    figs["emotion"] = _png(viz.emotion_bars(res), "emotion_regimes", fig_dir)
    figs["gains"] = _png(
        viz.gains_forest(gains, "Выигрыш MTL над single-task моделью эмоций", "п.п."), "emotion_gains", fig_dir
    )
    if aux:
        figs["aux_gains"] = _png(
            viz.gains_forest(aux, "Выигрыш MTL для вспомогательных задач", "CCC / доля"), "aux_gains", fig_dir
        )
    figs["aux"] = _png(viz.aux_tasks(res), "aux_tasks", fig_dir)
    figs["sweep"] = _png(viz.weight_sweep(res), "weight_sweep", fig_dir)
    for key, fig, name in (
        ("unc", viz.uncertainty_weights(res), "uncertainty_weights"),
        ("cos", viz.gradient_cosines(res), "gradient_cosines"),
        ("vadscatter", viz.vad_scatter(res), "vad_scatter"),
    ):
        figs[key] = _png(fig, name, fig_dir) if fig is not None else ""
    figs["curves"] = _png(viz.training_curves(res), "training_curves", fig_dir)
    figs["recall"] = _png(viz.recall_by_class(res), "recall_by_class", fig_dir)

    ctx.update(
        {
            "figs": figs,
            "titles": {r: REGIME_BY_NAME[r].title for r in res.regimes},
            "emotion_rows": emotion_rows,
            "baseline_ua": metric(res, BASELINE, "emotion").fmt(),
            "best": _diff_row(best) if best else None,
            "best_regime": REGIME_BY_NAME[best.a].title if best else "",
            "best_significant": bool(best and best.lo > 0),
            "gain_rows": [_diff_row(d) for d in gains],
            "extra_rows": [_diff_row(d) for d in extra],
            "aux_rows": aux_rows,
            "cos_regimes": [REGIME_BY_NAME[r].title for r in cos_regimes],
            "pair_rows": pair_rows,
            "shares": {
                REGIME_BY_NAME[r].title: (f"{100 * s:.0f}" if s is not None else "—") for r, s in shares.items()
            },
            "sweep": [{"w": f"{w:g}", "ua": m.fmt()} for w, m in sweep],
            "unc_final": unc_final,
            "gender_acc": gender_acc.fmt() if gender_acc else None,
            "shapes": res.shapes,
            "minutes": res.seconds / 60,
            "seeds": ", ".join(map(str, res.seeds)),
            "n_regimes": len(res.regimes),
            "tasks": [TASK_TITLES[t] for t in TASKS],
            "vad_names": [TASK_TITLES[t] for t in VAD_TASKS],
            "vad_gain": next((d for d in gains if d.a == "mtl_vad"), None),
            "gender_gain": next((d for d in gains if d.a == "mtl_gender"), None),
            "vad_gain_row": next((_diff_row(d) for d in gains if d.a == "mtl_vad"), None),
            "gender_gain_row": next((_diff_row(d) for d in gains if d.a == "mtl_gender"), None),
        }
    )
    path = _render("report.html.j2", ctx, cfg.out_dir / "report.html")

    metrics: dict[str, Any] = {
        "config": {k: str(v) for k, v in dataclasses.asdict(cfg).items()},
        "emotion": {
            r: {m: dataclasses.asdict(metric(res, r, "emotion", m)) for m in ("ua", "wa", "f1")} for r in em_regimes
        },
        "tasks": {
            f"{r}|{t}": dataclasses.asdict(metric(res, r, t)) for r in res.regimes for t in REGIME_BY_NAME[r].tasks
        },
        "emotion_gains": [dataclasses.asdict(d) for d in gains],
        "aux_gains": [dataclasses.asdict(d) for d in aux],
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _fmt_task(m: MeanStd, task: str) -> str:
    return m.fmt(1, pct=True) if primary_metric(task) == "ua" else m.fmt(3, pct=False)
