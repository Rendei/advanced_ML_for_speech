"""Runtime report: figures + numbers of *this* run are rendered into a self-contained HTML file."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import platform
from datetime import datetime
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
    bootstrap_diff,
    complementarity,
    default_summary,
    example_rows,
    fold_table,
    shuffle_summary,
    summary,
    tuning_rows,
)
from .config import CLASSES, MODEL_NAMES, MODEL_TITLES, Config
from .experiment import Results
from .features import Features

TEMPLATE = Path(__file__).with_name("report.html.j2")


def _png(fig: Figure, name: str, out: Path | None) -> str:
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


def _row_similarity(att: npt.NDArray[np.float64]) -> float | None:
    """Mean pairwise cosine similarity of the token rows of an attention map (special tokens dropped).

    Close to 1: every word looks at the same audio frames (no word-level alignment);
    low: different words attend to different moments of the signal.
    """
    rows = att[1:-1] if att.shape[0] > 3 else att
    k = rows.shape[0]
    if k < 2:
        return None
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    return float((sim.sum() - k) / (k * (k - 1)))


def build_report(cfg: Config, feats: Features, res: Results) -> Path:
    """Render ``outputs/report.html`` (+ figures/ and metrics.json) from the finished experiment."""
    fig_dir = cfg.out_dir / "figures"
    summ = summary(res)
    folds = fold_table(res)
    shuf = shuffle_summary(res)
    default_summ = default_summary(res)

    unimodal_best = max(("audio", "text"), key=lambda k: summ[k]["ua"].mean)
    fusion_best = max(("cross", "bicross"), key=lambda k: summ[k]["ua"].mean)
    pairs = [(fusion_best, unimodal_best), (fusion_best, "audio" if unimodal_best == "text" else "text"),
             ("cross", "concat"), ("bicross", "concat"), ("cross", "text"), ("cross", "audio")]
    seen: set[tuple[str, str]] = set()
    diffs: list[BootDiff] = []
    for a, b in pairs:
        if (a, b) not in seen:
            seen.add((a, b))
            diffs.append(bootstrap_diff(res, a, b, cfg))
    main_diff = diffs[0]
    cross_vs_concat = next(d for d in diffs if (d.a, d.b) == ("cross", "concat"))
    att_sim = _row_similarity(res.attention.attention) if res.attention is not None else None

    figs: dict[str, str] = {}
    figs["overview"] = _png(viz.data_overview(feats), "data_overview", fig_dir)
    figs["examples"] = _png(viz.examples(feats), "spectrograms", fig_dir)
    figs["space"] = _png(viz.feature_space(feats), "feature_space", fig_dir)
    figs["arch"] = _png(viz.architecture(), "architecture", fig_dir)
    figs["curves"] = _png(viz.training_curves(res), "training_curves", fig_dir)
    figs["main"] = _png(viz.main_results(summ, folds), "main_results", fig_dir)
    figs["gains"] = _png(viz.gains(diffs), "gains", fig_dir)
    figs["conf"] = _png(viz.confusions(res), "confusions", fig_dir)
    figs["perclass"] = _png(viz.per_class(res), "per_class_recall", fig_dir)
    figs["reliance"] = _png(viz.reliance(summ, shuf), "reliance", fig_dir)
    tune_fig = viz.tuning_plot(res)
    figs["tuning"] = _png(tune_fig, "tuning", fig_dir) if tune_fig is not None else ""
    att = viz.attention_map(res, feats)
    figs["attention"] = _png(att, "cross_attention", fig_dir) if att is not None else ""

    counts = np.bincount(feats.meta.label, minlength=len(CLASSES))
    table = [
        {
            "key": k,
            "title": MODEL_TITLES[k],
            "ua": summ[k]["ua"].fmt(),
            "wa": summ[k]["wa"].fmt(),
            "f1": summ[k]["f1"].fmt(),
            "params": f"{res.n_params.get(k, 0) / 1e3:.0f}k",
            "fusion": k in {"concat", "cross", "bicross"},
            "best": k == max(MODEL_NAMES, key=lambda q: summ[q]["ua"].mean),
        }
        for k in MODEL_NAMES
    ]
    ctx: dict[str, Any] = {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cfg": cfg,
        "n": len(feats.meta),
        "hours": float(feats.meta.seconds.sum() / 3600),
        "class_rows": [(CLASSES[c], viz.CLASS_RU[CLASSES[c]], int(counts[c]), f"{100 * counts[c] / counts.sum():.1f}")
                       for c in range(len(CLASSES))],
        "n_sessions": len(np.unique(feats.meta.session)),
        "table": table,
        "titles": MODEL_TITLES,
        "diffs": [
            {"a": MODEL_TITLES[d.a], "b": MODEL_TITLES[d.b], "diff": f"{d.diff * 100:+.1f}", "lo": f"{d.lo * 100:+.1f}",
             "hi": f"{d.hi * 100:+.1f}", "p": f"{d.p_not_better:.3f}", "verdict": _verdict(d)}
            for d in diffs
        ],
        "main": {
            "fusion": MODEL_TITLES[fusion_best], "uni": MODEL_TITLES[unimodal_best],
            "fusion_ua": summ[fusion_best]["ua"].fmt(), "uni_ua": summ[unimodal_best]["ua"].fmt(),
            "diff": f"{main_diff.diff * 100:+.1f}",
            "lo": f"{main_diff.lo * 100:+.1f}",
            "hi": f"{main_diff.hi * 100:+.1f}",
            "verdict": _verdict(main_diff), "significant": main_diff.lo > 0,
        },
        "fold_rows": [
            {"title": MODEL_TITLES[k], "vals": [f"{m.mean * 100:.1f}" for m in folds[k]]} for k in MODEL_NAMES
        ],
        "n_folds": len(folds["audio"]),
        "shuffle": [
            {"title": MODEL_TITLES[k], "full": summ[k]["ua"].fmt(), "audio": shuf[(k, "audio")].fmt(),
             "text": shuf[(k, "text")].fmt()}
            for k in ("concat", "cross", "bicross")
        ],
        "comp": {k: f"{v * 100:.1f}" for k, v in complementarity(res).items()},
        "examples_rows": example_rows(res, feats),
        "shapes": res.shapes,
        "figs": figs,
        "att_sim": None if att_sim is None else f"{att_sim:.2f}",
        "cross_vs_concat": {
            "diff": f"{cross_vs_concat.diff * 100:+.1f}",
            "lo": f"{cross_vs_concat.lo * 100:+.1f}",
            "hi": f"{cross_vs_concat.hi * 100:+.1f}",
            "verdict": _verdict(cross_vs_concat),
            "better": cross_vs_concat.lo > 0,
        },
        "shuf_cross": {"full": summ["cross"]["ua"].fmt(), "audio": shuf[("cross", "audio")].fmt(),
                       "text": shuf[("cross", "text")].fmt()},
        "minutes": res.seconds / 60,
        "env": f"Python {platform.python_version()}, torch {torch.__version__}, "
               f"{'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        "seeds": ", ".join(map(str, res.seeds)),
        "tuned": bool(res.tuning),
        "tune_rows": tuning_rows(res),
        "default_vs_tuned": _default_vs_tuned(summ, default_summ),
    }
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE.parent), autoescape=True)
    html = env.get_template(TEMPLATE.name).render(**ctx)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / "report.html"
    path.write_text(html, encoding="utf-8")

    metrics = {
        "config": {k: str(v) for k, v in dataclasses.asdict(cfg).items()},
        "summary": {k: {m: {"mean": s.mean, "std": s.std} for m, s in v.items()} for k, v in summ.items()},
        "bootstrap": [dataclasses.asdict(d) for d in diffs],
        "shuffle_probe_ua": {f"{k}|{m}": s.mean for (k, m), s in shuf.items()},
        "fold_ua": {k: [m.mean for m in v] for k, v in folds.items()},
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _default_vs_tuned(
    tuned: dict[str, dict[str, MeanStd]], default: dict[str, dict[str, MeanStd]] | None
) -> list[dict[str, str]]:
    """Test UA/F1 of every model with default vs. tuned hyperparameters (empty without a search)."""
    if default is None:
        return []
    return [
        {
            "title": MODEL_TITLES[k],
            "ua_default": default[k]["ua"].fmt(),
            "ua_tuned": tuned[k]["ua"].fmt(),
            "delta": f"{(tuned[k]['ua'].mean - default[k]['ua'].mean) * 100:+.1f}",
            "f1_default": default[k]["f1"].fmt(),
            "f1_tuned": tuned[k]["f1"].fmt(),
        }
        for k in MODEL_NAMES
    ]
