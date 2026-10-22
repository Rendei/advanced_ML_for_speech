"""All figures of the report (matplotlib, Agg backend). Every function returns a ``Figure``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .analysis import BootDiff, MeanStd, mean_confusion, per_class_recall
from .config import CLASSES, MODEL_NAMES
from .experiment import Results
from .features import Features

F64 = npt.NDArray[np.float64]

# One colour per entity, fixed for the whole report (blue/orange: unimodal; green/violet: cross-attention).
COLORS = {"audio": "#2a78d6", "text": "#eb6834", "concat": "#8f8e88", "cross": "#1baf7a", "bicross": "#6250d6"}
CLASS_COLORS = ("#8f8e88", "#eda100", "#2a78d6", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"
CLASS_RU = {"neutral": "нейтр.", "happy": "радость", "sad": "грусть", "angry": "злость"}
SHORT = {"audio": "аудио", "text": "текст", "concat": "конкат.", "cross": "cross-attn", "bicross": "bi-cross"}


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


def data_overview(feats: Features) -> Figure:
    meta = feats.meta
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    sessions = np.arange(1, 6)
    bottom = np.zeros(5)
    for c, name in enumerate(CLASSES):
        cnt = np.array([((meta.session == s) & (meta.label == c)).sum() for s in sessions])
        ax[0].bar(sessions, cnt, bottom=bottom, color=CLASS_COLORS[c], label=CLASS_RU[name], width=0.7,
                  edgecolor="white", linewidth=1.5)
        bottom += cnt
    ax[0].set(xlabel="сессия (пара дикторов)", ylabel="число реплик", title="Классы по сессиям")
    ax[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax[1].hist(feats.meta.seconds, bins=40, color="#2a78d6", edgecolor="white")
    ax[1].set(xlabel="длительность, с", ylabel="число реплик", title="Длительность реплик")
    ax[2].hist(feats.text_len, bins=40, color="#eb6834", edgecolor="white")
    ax[2].set(xlabel="токенов RoBERTa", ylabel="число реплик", title="Длина текста")
    fig.tight_layout()
    return fig


def examples(feats: Features) -> Figure:
    """Spectrogram + transcript of one utterance per class."""
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(13, 3.0), sharey=True)
    for c, ax in enumerate(axes):
        wav = feats.examples.get(c)
        ax.grid(False)
        if wav is not None:
            ax.specgram(wav, NFFT=512, Fs=16_000, noverlap=384, cmap="magma", vmin=-120, vmax=-40)
        ax.set_title(CLASS_RU[CLASSES[c]], fontsize=11)
        ax.set_xlabel("время, с")
    axes[0].set_ylabel("частота, Гц")
    fig.tight_layout()
    return fig


def feature_space(feats: Features, n_max: int = 2500) -> Figure:
    """PCA of mean-pooled frozen features: how separable is each modality before any training?"""
    rng = np.random.default_rng(0)
    n = len(feats.meta)
    idx = rng.choice(n, min(n, n_max), replace=False)
    a_off = np.concatenate([[0], np.cumsum(feats.audio_len)])
    t_off = np.concatenate([[0], np.cumsum(feats.text_len)])
    xa = np.stack([feats.audio_flat[a_off[i] : a_off[i + 1]].astype(np.float32).mean(0) for i in idx])
    xt = np.stack([feats.text_flat[t_off[i] : t_off[i + 1]].astype(np.float32).mean(0) for i in idx])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    panels = ((ax[0], xa, "Аудио: WavLM (среднее по кадрам)"), (ax[1], xt, "Текст: RoBERTa (среднее по токенам)"))
    for a, x, title in panels:
        x = x - x.mean(0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        z = x @ vt[:2].T
        for c, name in enumerate(CLASSES):
            m = feats.meta.label[idx] == c
            a.scatter(z[m, 0], z[m, 1], s=7, alpha=0.55, color=CLASS_COLORS[c], label=CLASS_RU[name], linewidths=0)
        a.set(title=title, xlabel="PC1", ylabel="PC2")
    ax[1].legend(frameon=False, markerscale=2.5, fontsize=8)
    fig.tight_layout()
    return fig


def architecture() -> Figure:
    """Block diagram of the main method (text queries attend to audio)."""
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.18))
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=color, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.4})

    box(0.1, 2.7, 1.9, 0.9, "аудио: WavLM-base+\n(заморожен)\n[B, Ta, 768]", COLORS["audio"])
    box(0.1, 0.4, 1.9, 0.9, "текст: RoBERTa-base\n(заморожен)\n[B, Tt, 768]", COLORS["text"])
    box(2.6, 2.7, 1.6, 0.9, "Linear+PE+\nTransformer\n[B, Ta, d]", COLORS["audio"])
    box(2.6, 0.4, 1.6, 0.9, "Linear+PE+\nTransformer\n[B, Tt, d]", COLORS["text"])
    box(5.0, 1.2, 3.4, 1.6, "Cross-attention ×2\nQ = текстовые токены\nK, V = аудио-кадры\n→ [B, Tt, d]",
        COLORS["cross"])
    box(9.0, 1.4, 1.3, 1.2, "attention\npooling\n[B, d]", MUTED)
    box(10.6, 1.4, 1.3, 1.2, "Linear\n[B, 4]\nэмоция", MUTED)
    arrow(2.0, 3.15, 2.6, 3.15)
    arrow(2.0, 0.85, 2.6, 0.85)
    arrow(4.2, 3.15, 5.4, 2.8)
    arrow(4.2, 0.85, 5.4, 1.2)
    arrow(8.4, 2.0, 9.0, 2.0)
    arrow(10.3, 2.0, 10.6, 2.0)
    ax.text(6.7, 3.3, "аудио → ключи/значения (K, V)", ha="center", fontsize=8, color=MUTED)
    ax.text(6.7, 0.7, "текст → запросы (Q)", ha="center", fontsize=8, color=MUTED)
    return fig


def training_curves(res: Results) -> Figure:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for kind in MODEL_NAMES:
        h = res.histories[kind]
        ep = np.arange(1, len(h.train_loss) + 1)
        ax[0].plot(ep, h.train_loss, color=COLORS[kind], lw=2, label=SHORT[kind])
        ax[1].plot(ep, h.val_ua, color=COLORS[kind], lw=2)
        ax[1].plot(h.best_epoch + 1, h.val_ua[h.best_epoch], "o", color=COLORS[kind], ms=6, mec="white", mew=1.5)
    ax[0].set(xlabel="эпоха", ylabel="train loss", title="Функция потерь (обучение)")
    ax[1].set(xlabel="эпоха", ylabel="UA на валидации", title="Качество на валидации (● — выбранная эпоха)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def main_results(summary: dict[str, dict[str, MeanStd]], folds: dict[str, list[MeanStd]]) -> Figure:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(MODEL_NAMES))
    ua = np.array([summary[k]["ua"].mean for k in MODEL_NAMES]) * 100
    sd = np.array([summary[k]["ua"].std for k in MODEL_NAMES]) * 100
    ax[0].bar(x, ua, yerr=sd, color=[COLORS[k] for k in MODEL_NAMES], width=0.62, capsize=3, error_kw={"lw": 1.2})
    for xi, v in zip(x, ua):
        ax[0].text(xi, v + 1.0, f"{v:.1f}", ha="center", fontsize=9)
    ax[0].set_xticks(x, [SHORT[k] for k in MODEL_NAMES])
    ax[0].set(ylabel="UA, % (leave-one-session-out)", title="Качество по всем сессиям",
              ylim=(max(0, ua.min() - 12), ua.max() + 6))
    n_f = len(next(iter(folds.values())))
    xs = np.arange(1, n_f + 1)
    for k in MODEL_NAMES:
        ax[1].plot(xs, [m.mean * 100 for m in folds[k]], "-o", color=COLORS[k], lw=2, ms=6, mec="white", label=SHORT[k])
    ax[1].set_xticks(xs)
    ax[1].set(xlabel="тестовая сессия", ylabel="UA, %", title="Качество по тестовым сессиям")
    ax[1].legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return fig


def gains(diffs: list[BootDiff]) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(diffs) + 1.4))
    y = np.arange(len(diffs))[::-1]
    for yi, d in zip(y, diffs):
        col = "#1baf7a" if d.lo > 0 else "#8f8e88"
        ax.plot([d.lo * 100, d.hi * 100], [yi, yi], color=col, lw=3, solid_capstyle="round")
        ax.plot(d.diff * 100, yi, "o", color=col, ms=8, mec="white", mew=1.5)
        label = f"{d.diff * 100:+.1f} [{d.lo * 100:+.1f}; {d.hi * 100:+.1f}]"
        ax.text(d.hi * 100 + 0.25, yi, label, va="center", fontsize=8.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [f"{SHORT[d.a]} − {SHORT[d.b]}" for d in diffs])
    ax.set(xlabel="разница UA, п.п. (95 % bootstrap-интервал)", title="Выигрыш фузирования")
    lo, hi = min(d.lo for d in diffs) * 100, max(d.hi for d in diffs) * 100
    ax.set_xlim(min(lo - 1, -1), hi + 5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def confusions(res: Results) -> Figure:
    kinds = ("audio", "text", "cross")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    labels = [CLASS_RU[c] for c in CLASSES]
    for ax, kind in zip(axes, kinds):
        cm = mean_confusion(res, kind)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.55 else INK, fontsize=10)
        ax.set_xticks(range(len(CLASSES)), labels, fontsize=8)
        ax.set_yticks(range(len(CLASSES)), labels, fontsize=8)
        ax.set(xlabel="предсказано", title=SHORT[kind])
    axes[0].set_ylabel("истинный класс (доля, %)")
    fig.tight_layout()
    return fig


def per_class(res: Results) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    kinds = ("audio", "text", "cross")
    w = 0.26
    x = np.arange(len(CLASSES))
    for j, k in enumerate(kinds):
        r = per_class_recall(res, k) * 100
        ax.bar(x + (j - 1) * w, r, w * 0.92, color=COLORS[k], label=SHORT[k])
    ax.set_xticks(x, [CLASS_RU[c] for c in CLASSES])
    ax.set(ylabel="recall, %", title="Полнота по классам")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def reliance(summary: dict[str, dict[str, MeanStd]], shuffled: dict[tuple[str, str], MeanStd]) -> Figure:
    """UA with intact inputs vs with audio / text replaced by another utterance's."""
    kinds = ("concat", "cross", "bicross")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(kinds))
    w = 0.26
    series = [
        ("обе модальности", [summary[k]["ua"].mean for k in kinds], "#2f2e2b"),
        ("аудио подменено", [shuffled[(k, "audio")].mean for k in kinds], COLORS["audio"]),
        ("текст подменён", [shuffled[(k, "text")].mean for k in kinds], COLORS["text"]),
    ]
    for j, (name, vals, col) in enumerate(series):
        ax.bar(x + (j - 1) * w, np.array(vals) * 100, w * 0.92, color=col, label=name)
    ax.axhline(25, color=MUTED, lw=1, ls="--", label="случайный выбор (25 %)")
    ax.set_xticks(x, [SHORT[k] for k in kinds])
    ax.set(ylabel="UA, %", title="На какую модальность опирается модель")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def attention_map(res: Results, feats: Features) -> Figure | None:
    ex = res.attention
    if ex is None:
        return None
    toks = [t.replace("Ġ", "") for t in feats.tokens[ex.index]][: ex.attention.shape[0]]
    dt = 0.02 * 4  # WavLM frame (20 ms) x pooling factor 4
    t_end = ex.attention.shape[1] * dt
    fig, ax = plt.subplots(figsize=(10, 0.38 * len(toks) + 1.6))
    im = ax.imshow(ex.attention, aspect="auto", cmap="viridis", extent=(0, t_end, len(toks) - 0.5, -0.5))
    ax.set_yticks(range(len(toks)), toks)
    ax.grid(False)
    true = CLASSES[int(res.y[ex.index])]
    title = f"Cross-attention: токены текста (Q) → аудио-кадры (K/V); класс «{CLASS_RU[true]}»"
    ax.set(xlabel="время в аудио, с", title=title)
    fig.colorbar(im, ax=ax, label="вес внимания", fraction=0.04)
    fig.tight_layout()
    return fig


def tuning_plot(res: Results) -> Figure | None:
    """Validation UA of every trial per model and fold: grey = tried, hollow = default, filled = chosen."""
    if not res.tuning:
        return None
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(13, 3.4), sharey=True)
    for ax, kind in zip(axes, MODEL_NAMES):
        trials = [t for t in res.tuning if t.kind == kind]
        ax.scatter([t.fold + 1 for t in trials], [t.val_ua * 100 for t in trials], s=18, color="#c9c8c2", zorder=2)
        d = [t for t in trials if t.trial == 0]
        ax.scatter([t.fold + 1 for t in d], [t.val_ua * 100 for t in d], s=60, facecolor="white",
                   edgecolor=COLORS[kind], linewidth=1.8, zorder=3, label="по умолчанию")
        c = [t for t in trials if t.chosen]
        ax.scatter([t.fold + 1 for t in c], [t.val_ua * 100 for t in c], s=34, color=COLORS[kind], zorder=4,
                   label="выбрано")
        ax.set_title(SHORT[kind])
        ax.set_xlabel("внешний фолд")
        ax.set_xticks(sorted({t.fold + 1 for t in trials}))
    axes[0].set_ylabel("UA на валидации, %")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig
