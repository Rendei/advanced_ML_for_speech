"""All figures of the report (matplotlib, Agg backend). Every function returns a ``Figure``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .analysis import BootDiff, emotion_recall, mean_cosines, metric, regimes_with, truth
from .config import (
    BASELINE,
    EMOTION_CLASSES,
    REGIME_BY_NAME,
    TASKS,
    VAD_TASKS,
    WEIGHT_SWEEP,
    primary_metric,
)
from .data import Meta
from .experiment import Results

F64 = npt.NDArray[np.float64]

INK, MUTED = "#0b0b0b", "#52514e"
# One colour per entity for the whole report: a regime keeps its colour on every figure.
REGIME_COLORS = {
    "st_emotion": "#8f8e88",
    "st_valence": "#8f8e88",
    "st_arousal": "#8f8e88",
    "st_dominance": "#8f8e88",
    "st_gender": "#8f8e88",
    "mtl_vad": "#2a78d6",
    "mtl_gender": "#eb6834",
    "mtl_w0.1": "#9fd8c4",
    "mtl_w0.3": "#5fc4a1",
    "mtl_equal": "#1baf7a",
    "mtl_uncertainty": "#6250d6",
    "mtl_pcgrad": "#e87ba4",
}
TASK_COLORS = {
    "emotion": "#2f2e2b",
    "valence": "#2a78d6",
    "arousal": "#eb6834",
    "dominance": "#1baf7a",
    "gender": "#6250d6",
}
EMOTION_COLORS = ("#8f8e88", "#eda100", "#2a78d6", "#e34948")
EMOTION_RU = {"neutral": "нейтр.", "happy": "радость", "sad": "грусть", "angry": "злость"}
SHORT = {
    "st_emotion": "single\nэмоция",
    "st_valence": "single\nvalence",
    "st_arousal": "single\narousal",
    "st_dominance": "single\ndominance",
    "st_gender": "single\nпол",
    "mtl_vad": "MTL\n+VAD",
    "mtl_gender": "MTL\n+пол",
    "mtl_w0.1": "MTL\nw=0.1",
    "mtl_w0.3": "MTL\nw=0.3",
    "mtl_equal": "MTL\nравные",
    "mtl_uncertainty": "MTL\nuncert.",
    "mtl_pcgrad": "MTL\nPCGrad",
}
TASK_SHORT = {
    "emotion": "эмоция",
    "valence": "valence",
    "arousal": "arousal",
    "dominance": "dominance",
    "gender": "пол",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.dpi": 110,
            "savefig.dpi": 140,
        }
    )


_style()


# ============================ EDA (needs only the metadata) =====================================================
def eda_overview(meta: Meta) -> Figure:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    sessions = np.arange(1, 6)
    bottom = np.zeros(5)
    for c, name in enumerate(EMOTION_CLASSES):
        cnt = np.array([((meta.session == s) & (meta.emotion == c)).sum() for s in sessions])
        ax[0].bar(
            sessions,
            cnt,
            bottom=bottom,
            color=EMOTION_COLORS[c],
            label=EMOTION_RU[name],
            width=0.7,
            edgecolor="white",
            linewidth=1.5,
        )
        bottom += cnt
    ax[0].set(xlabel="сессия (пара дикторов)", ylabel="число реплик", title="Эмоции по сессиям")
    ax[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    w = 0.38
    for g, (name, col) in enumerate((("женщины", "#e87ba4"), ("мужчины", "#2a78d6"))):
        share = [
            100 * ((meta.gender == g) & (meta.emotion == c)).sum() / max((meta.gender == g).sum(), 1) for c in range(4)
        ]
        ax[1].bar(np.arange(4) + (g - 0.5) * w, share, w * 0.92, color=col, label=name)
    ax[1].set_xticks(range(4), [EMOTION_RU[c] for c in EMOTION_CLASSES])
    ax[1].set(ylabel="доля реплик диктора, %", title="Эмоции у женщин и мужчин")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].hist(meta.seconds, bins=40, color="#2a78d6", edgecolor="white")
    ax[2].set(xlabel="длительность, с", ylabel="число реплик", title="Длительность реплик")
    fig.tight_layout()
    return fig


def eda_vad_distributions(meta: Meta) -> Figure:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for i, task in enumerate(VAD_TASKS):
        ax[i].hist(meta.vad[:, i], bins=np.arange(0.75, 5.8, 0.5), color=TASK_COLORS[task], edgecolor="white")
        ax[i].axvline(meta.vad[:, i].mean(), color=INK, lw=1.2, ls="--")
        ax[i].set(
            xlabel="оценка (1–5)",
            title=f"{TASK_SHORT[task]}: среднее {meta.vad[:, i].mean():.2f}, σ {meta.vad[:, i].std():.2f}",
        )
    ax[0].set_ylabel("число реплик")
    fig.tight_layout()
    return fig


def eda_va_space(meta: Meta) -> Figure:
    """Utterances in the valence-arousal and arousal-dominance planes: the emotion classes occupy different regions."""
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.2, 0.2, size=meta.vad.shape)  # ratings lie on a 0.5 grid; jitter only for visibility
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for a, (i, j) in zip(ax, ((0, 1), (1, 2))):
        for c in range(len(EMOTION_CLASSES)):
            m = meta.emotion == c
            a.scatter(
                meta.vad[m, i] + jit[m, i],
                meta.vad[m, j] + jit[m, j],
                s=6,
                alpha=0.25,
                linewidths=0,
                color=EMOTION_COLORS[c],
            )
        for c, name in enumerate(EMOTION_CLASSES):
            m = meta.emotion == c
            a.scatter(
                meta.vad[m, i].mean(),
                meta.vad[m, j].mean(),
                s=110,
                color=EMOTION_COLORS[c],
                edgecolor="white",
                linewidth=2,
                zorder=3,
                label=EMOTION_RU[name],
            )
        a.set(xlabel=TASK_SHORT[VAD_TASKS[i]], ylabel=TASK_SHORT[VAD_TASKS[j]])
    ax[0].set_title("Valence × arousal")
    ax[1].set_title("Arousal × dominance")
    ax[1].legend(frameon=False, fontsize=8, title="центры классов", title_fontsize=8)
    fig.tight_layout()
    return fig


def eda_class_vad(meta: Meta) -> Figure:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for i, task in enumerate(VAD_TASKS):
        data = [meta.vad[meta.emotion == c, i] for c in range(4)]
        bp = ax[i].boxplot(
            data,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": INK},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
        for patch, col in zip(bp["boxes"], EMOTION_COLORS):
            patch.set(facecolor=col, alpha=0.65, edgecolor=MUTED)
        ax[i].set_xticks(range(1, 5), [EMOTION_RU[c] for c in EMOTION_CLASSES])
        ax[i].set_title(TASK_SHORT[task])
    ax[0].set_ylabel("оценка (1–5)")
    fig.tight_layout()
    return fig


def eda_correlations(meta: Meta) -> Figure:
    """Pearson correlation between all targets of the lab (point-biserial for the binary ones)."""
    cols: dict[str, F64] = {TASK_SHORT[t]: meta.vad[:, i] for i, t in enumerate(VAD_TASKS)}
    cols["пол (муж.)"] = meta.gender.astype(np.float64)
    for c, name in enumerate(EMOTION_CLASSES):
        cols[f"эмоция: {EMOTION_RU[name]}"] = (meta.emotion == c).astype(np.float64)
    names = list(cols)
    corr: F64 = np.asarray(np.corrcoef(np.stack([cols[n] for n in names])))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(corr[i, j]) > 0.6 else INK,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, label="корреляция")
    ax.set_title("Связи между целевыми переменными")
    fig.tight_layout()
    return fig


# ============================ method ========================================================================
def architecture(trunk: str) -> Figure:
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5)

    def box(x: float, y: float, w: float, h: float, text: str, color: str, alpha: float = 0.18) -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=alpha))
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=color, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.3})

    box(0.1, 3.3, 1.9, 1.0, "аудио: WavLM\n(заморожен)\n[B, Ta, 768]", "#2a78d6")
    box(0.1, 0.7, 1.9, 1.0, "текст: RoBERTa\n(заморожен)\n[B, Tt, 768]", "#eb6834")
    box(
        2.5,
        1.0,
        3.6,
        3.0,
        f"ОБЩИЙ ТРАНК (fusion «{trunk}»)\nэнкодеры + cross-attention\nQ=текст ↔ K/V=аудио\n→ вектор z [B, z_dim]",
        "#8f8e88",
        0.25,
    )
    arrow(2.0, 3.8, 2.5, 3.3)
    arrow(2.0, 1.2, 2.5, 1.9)
    names = [(t, TASK_SHORT[t]) for t in TASKS]
    outs = {
        "emotion": "[B, 4] логиты",
        "valence": "[B, 1]",
        "arousal": "[B, 1]",
        "dominance": "[B, 1]",
        "gender": "[B, 2] логиты",
    }
    for k, (t, ru) in enumerate(names):
        y = 4.1 - k * 0.85
        box(7.0, y - 0.3, 2.3, 0.6, f"голова: {ru}", TASK_COLORS[t], 0.22)
        box(9.9, y - 0.3, 2.4, 0.6, f"{outs[t]}\n{'CE' if t in ('emotion', 'gender') else 'MSE'}", TASK_COLORS[t], 0.1)
        arrow(6.1, 2.5, 7.0, y)
        arrow(9.3, y, 9.9, y)
    ax.text(8.4, 4.85, "по одной небольшой голове на задачу", ha="center", fontsize=8.5, color=MUTED)
    ax.text(3.0, 0.25, "градиенты всех задач встречаются в общем транке", fontsize=8.5, color=MUTED)
    return fig


# ============================ results =======================================================================
def _label(regime: str) -> str:
    return SHORT.get(regime, regime)


def emotion_bars(res: Results) -> Figure:
    regimes = regimes_with(res, "emotion")
    ms = [metric(res, r, "emotion") for r in regimes]
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    x = np.arange(len(regimes))
    ua = np.array([m.mean for m in ms]) * 100
    sd = np.array([m.std for m in ms]) * 100
    ax.bar(x, ua, yerr=sd, color=[REGIME_COLORS[r] for r in regimes], width=0.62, capsize=3, error_kw={"lw": 1.2})
    for xi, v in zip(x, ua):
        ax.text(float(xi), float(v) + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    if BASELINE in regimes:
        ax.axhline(ua[regimes.index(BASELINE)], color=MUTED, lw=1, ls="--", label="single-task baseline")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xticks(x, [_label(r) for r in regimes], fontsize=8.5)
    ax.set(
        ylabel="UA эмоций, % (leave-one-session-out)",
        title="Распознавание эмоций: single-task против MTL",
        ylim=(max(0, ua.min() - 6), ua.max() + 3),
    )
    fig.tight_layout()
    return fig


def gains_forest(diffs: list[BootDiff], title: str, unit: str) -> Figure:
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(diffs) + 1.5))
    y = np.arange(len(diffs))[::-1]
    k = 100 if unit == "п.п." else 1
    for yi, d in zip(y, diffs):
        col = "#1baf7a" if d.lo > 0 else ("#e34948" if d.hi < 0 else "#8f8e88")
        ax.plot([d.lo * k, d.hi * k], [yi, yi], color=col, lw=3, solid_capstyle="round")
        ax.plot(d.diff * k, yi, "o", color=col, ms=8, mec="white", mew=1.5)
        ax.text(
            d.hi * k + 0.02 * (1 if k == 1 else 25),
            yi,
            f"{d.diff * k:+.2f} [{d.lo * k:+.2f}; {d.hi * k:+.2f}]",
            va="center",
            fontsize=8,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(
        y, [f"{TASK_SHORT[d.task]}: {SHORT.get(d.a, d.a).replace(chr(10), ' ')}" for d in diffs], fontsize=8.5
    )
    ax.set(xlabel=f"разница, {unit} (95 % интервал, bootstrap по диалогам)", title=title)
    lo, hi = min(d.lo for d in diffs) * k, max(d.hi * k for d in diffs)
    pad = (hi - lo) * 0.45 + 0.1
    ax.set_xlim(min(lo - 0.1 * pad, -0.1 * pad), hi + pad)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def aux_tasks(res: Results) -> Figure:
    """Do the auxiliary tasks themselves benefit from sharing? Single-task model vs the all-task MTL regimes."""
    tasks = TASKS[1:]
    mtl = [r for r in ("mtl_equal", "mtl_uncertainty", "mtl_pcgrad") if r in res.regimes]
    fig, ax = plt.subplots(1, len(tasks), figsize=(13, 3.4))
    for a, t in zip(ax, tasks):
        regimes = [f"st_{t}", *mtl]
        regimes = [r for r in regimes if r in res.regimes]
        ms = [metric(res, r, t) for r in regimes]
        a.bar(
            range(len(regimes)),
            [m.mean for m in ms],
            yerr=[m.std for m in ms],
            capsize=3,
            color=[REGIME_COLORS[r] for r in regimes],
            width=0.62,
            error_kw={"lw": 1.1},
        )
        for i, m in enumerate(ms):
            a.text(
                i, m.mean + 0.01, f"{m.mean:.3f}" if t != "gender" else f"{m.mean * 100:.1f}", ha="center", fontsize=8
            )
        a.set_xticks(range(len(regimes)), [SHORT[r].replace("single\n" + t, "single") for r in regimes], fontsize=7.5)
        a.set_title(f"{TASK_SHORT[t]}\n({'CCC' if primary_metric(t) == 'ccc' else 'UA'})", fontsize=9.5)
        lo = min(m.mean for m in ms)
        a.set_ylim(max(0, lo - 0.12) if t != "gender" else max(0.5, lo - 0.05), max(m.mean for m in ms) + 0.05)
    fig.tight_layout()
    return fig


def weight_sweep(res: Results) -> Figure:
    names = [r for r in WEIGHT_SWEEP if r in res.regimes]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = [REGIME_BY_NAME[r].aux_weight for r in names]
    ms = [metric(res, r, "emotion") for r in names]
    ax.errorbar(
        xs,
        [m.mean * 100 for m in ms],
        yerr=[m.std * 100 for m in ms],
        fmt="-o",
        color="#1baf7a",
        lw=2,
        ms=7,
        mec="white",
        capsize=3,
        label="MTL, все задачи",
    )
    if BASELINE in res.regimes:
        b = metric(res, BASELINE, "emotion")
        ax.axhspan((b.mean - b.std) * 100, (b.mean + b.std) * 100, color="#8f8e88", alpha=0.25, lw=0)
        ax.axhline(b.mean * 100, color=MUTED, lw=1.2, ls="--", label="single-task baseline")
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.set(
        xlabel="вес вспомогательных потерь (вес потери эмоций = 1)",
        ylabel="UA эмоций, %",
        title="Чувствительность к весу вспомогательных задач",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def uncertainty_weights(res: Results) -> Figure | None:
    h = res.histories.get("mtl_uncertainty")
    if h is None or not h.uncertainty_weights:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ep = np.arange(1, len(h.uncertainty_weights) + 1)
    for t in TASKS:
        ax.plot(ep, [w[t] for w in h.uncertainty_weights], color=TASK_COLORS[t], lw=2, label=TASK_SHORT[t])
    ax.set_yscale("log")
    ax.set(
        xlabel="эпоха",
        ylabel="вес потери exp(−s)/2 (лог. шкала)",
        title="Веса задач, выученные uncertainty weighting (фолд 1, seed 0)",
    )
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return fig


def gradient_cosines(res: Results) -> Figure | None:
    regimes = [r for r in ("mtl_equal", "mtl_uncertainty", "mtl_pcgrad") if mean_cosines(res, r) is not None]
    if not regimes:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.8 * len(regimes), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regimes):
        cos = mean_cosines(res, r)
        assert cos is not None
        im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.grid(False)
        tasks = REGIME_BY_NAME[r].tasks
        ax.set_xticks(range(len(tasks)), [TASK_SHORT[t] for t in tasks], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tasks)), [TASK_SHORT[t] for t in tasks], fontsize=8)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                ax.text(
                    j,
                    i,
                    f"{cos[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(cos[i, j]) > 0.6 else INK,
                )
        ax.set_title(SHORT[r].replace("\n", " "), fontsize=10)
    fig.colorbar(im, ax=list(axes), fraction=0.03, label="косинус градиентов")
    return fig


def training_curves(res: Results) -> Figure:
    regimes = [r for r in regimes_with(res, "emotion") if r in res.histories]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for r in regimes:
        h = res.histories[r]
        ep = np.arange(1, len(h.train_loss) + 1)
        ax[0].plot(ep, h.train_loss, color=REGIME_COLORS[r], lw=1.8, label=SHORT[r].replace("\n", " "))
        ax[1].plot(ep, h.val_score, color=REGIME_COLORS[r], lw=1.8)
        ax[1].plot(h.best_epoch + 1, h.val_score[h.best_epoch], "o", color=REGIME_COLORS[r], ms=5, mec="white")
    ax[0].set(xlabel="эпоха", ylabel="сумма потерь задач", title="Функция потерь (обучение; фолд 1, seed 0)")
    ax[1].set(xlabel="эпоха", ylabel="UA эмоций на валидации", title="Качество на валидации (● — выбранная эпоха)")
    ax[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def vad_scatter(res: Results, regime: str = "mtl_equal") -> Figure | None:
    if regime not in res.regimes:
        return None
    m = res.covered
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    for a, t in zip(ax, VAD_TASKS):
        y = truth(res, t)[m]
        p = res.preds[(regime, t)].mean(axis=0)[m]
        hb = a.hexbin(y, p, gridsize=22, cmap="Blues", mincnt=1, extent=(0.5, 5.5, 0.5, 5.5))
        a.plot([0.5, 5.5], [0.5, 5.5], color=MUTED, lw=1, ls="--")
        a.grid(False)
        a.set(
            xlabel="истинная оценка",
            ylabel="предсказание",
            title=f"{TASK_SHORT[t]}: CCC {metric(res, regime, t).mean:.2f}",
        )
    fig.colorbar(hb, ax=list(ax), fraction=0.025, label="число реплик")
    return fig


def recall_by_class(res: Results) -> Figure:
    names = [r for r in (BASELINE, "mtl_vad", "mtl_equal", "mtl_pcgrad") if r in res.regimes]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(names)
    x = np.arange(4)
    for j, r in enumerate(names):
        ax.bar(
            x + (j - (len(names) - 1) / 2) * w,
            emotion_recall(res, r) * 100,
            w * 0.92,
            color=REGIME_COLORS[r],
            label=SHORT[r].replace("\n", " "),
        )
    ax.set_xticks(x, [EMOTION_RU[c] for c in EMOTION_CLASSES])
    ax.set(ylabel="recall, %", title="Полнота по классам эмоций")
    ax.legend(frameon=False, fontsize=8, ncol=len(names))
    fig.tight_layout()
    return fig
