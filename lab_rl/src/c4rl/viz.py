"""All figures of the report (matplotlib, Agg backend). Every function returns a ``Figure``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import itertools

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from .eda import FirstMoves, PlyCounts, RandomPlay
from .evaluation import PositionSet, Tournament
from .game import CELLS, HEIGHT, WIDTH, from_moves, to_grid
from .history import TrainLog

F64 = npt.NDArray[np.float64]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
X_COLOR, O_COLOR = "#e34948", "#eda100"
OUTCOME_COLOR = {1: "#1baf7a", 0: "#8f8e88", -1: "#e34948"}
OUTCOME_NAME = {1: "победа", 0: "ничья", -1: "поражение"}
# one colour per entity throughout the report
RUN_COLOR = {"az": "#1baf7a", "az_lite": "#6250d6", "dqn": "#eb6834"}
RUN_LABEL = {"az": "AlphaZero-lite", "az_lite": "AlphaZero-lite, слабый поиск в self-play", "dqn": "DQN"}
FAMILY_COLOR = {
    "baseline": "#8f8e88",
    "uct": "#2a78d6",
    "dqn": "#eb6834",
    "az": "#1baf7a",
    "az-lite": "#6250d6",
    "perfect": "#0b0b0b",
}
FAMILY_RU = {
    "baseline": "простой бейзлайн",
    "uct": "UCT (случайные прогоны)",
    "dqn": "DQN",
    "az": "AlphaZero-lite",
    "az-lite": "AlphaZero-lite, слабый self-play",
    "perfect": "идеальный игрок",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": 110,
            "legend.frameon": False,
        }
    )


_style()


def _fig(w: float, h: float, nrows: int = 1, ncols: int = 1) -> tuple[Figure, npt.NDArray[np.object_]]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
    return fig, axes


# ---------------------------------------------------------------------------------------------
# board drawing
# ---------------------------------------------------------------------------------------------
def draw_board(ax: Axes, moves: str, mark: int | None = None, best: list[bool] | None = None) -> None:
    cur, mask = from_moves(moves)
    grid = to_grid(cur, mask)
    ax.set_xlim(-0.6, WIDTH - 0.4)
    ax.set_ylim(-0.9, HEIGHT - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-0.5, -0.5), WIDTH, HEIGHT, fc="#2a78d6", ec="none", zorder=0))
    for r in range(HEIGHT):
        for c in range(WIDTH):
            v = grid[r][c]
            ax.add_patch(Circle((c, r), 0.4, fc="white" if v == 0 else X_COLOR if v == 1 else O_COLOR, ec="none"))
    for c in range(WIDTH):
        good = best is not None and best[c]
        ax.text(
            c,
            -0.75,
            str(c + 1),
            ha="center",
            va="center",
            fontsize=9,
            color="#1baf7a" if good else MUTED,
            fontweight="bold" if good else "normal",
        )
    if mark is not None:
        ax.add_patch(Rectangle((mark - 0.5, -0.5), 1, HEIGHT, fc="none", ec=INK, lw=1.4, ls="--"))


# ---------------------------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------------------------
def eda_state_space(counts: PlyCounts) -> Figure:
    fig, axes = _fig(9, 3.8, 1, 2)
    ax = axes[0, 0]
    plies = np.arange(len(counts.total))
    ax.semilogy(plies, counts.total, "o-", color="#2a78d6", label="все позиции")
    ax.semilogy(plies, counts.up_to_mirror, "s--", color="#1baf7a", label="с точностью до зеркала")
    term = np.array(counts.terminal, dtype=np.float64)
    ax.semilogy(plies[term > 0], term[term > 0], "^:", color="#e34948", label="партия уже закончилась")
    ax.set_xlabel("число сделанных ходов")
    ax.set_ylabel("различных позиций")
    ax.set_title("Рост пространства состояний")
    ax.legend()
    ax = axes[0, 1]
    ratio = np.array(counts.total[1:]) / np.array(counts.total[:-1])
    ax.bar(plies[1:], ratio, color="#2a78d6")
    ax.axhline(7, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("число сделанных ходов")
    ax.set_ylabel("во сколько раз больше позиций, чем на ход раньше")
    ax.set_title("Эффективный коэффициент ветвления")
    fig.tight_layout()
    return fig


def eda_random_play(rp: RandomPlay) -> Figure:
    fig, axes = _fig(9, 3.6, 1, 2)
    ax = axes[0, 0]
    ax.hist(rp.length, bins=np.arange(6.5, CELLS + 1.5, 1), color="#2a78d6")
    ax.set_xlabel("длина партии, ходов")
    ax.set_ylabel("партий")
    ax.set_title("Случайная игра: длина партий")
    ax = axes[0, 1]
    shares = [(rp.result == 1).mean(), (rp.result == 0).mean(), (rp.result == -1).mean()]
    ax.bar(["побеждает X\n(ходит первым)", "ничья", "побеждает O"], shares, color=[X_COLOR, "#8f8e88", O_COLOR])
    for i, s in enumerate(shares):
        ax.text(i, s + 0.01, f"{100 * s:.1f}%", ha="center")
    ax.set_ylim(0, max(shares) * 1.15)
    ax.set_title("Случайная игра: исходы")
    fig.tight_layout()
    return fig


def eda_first_moves(first: FirstMoves, rp: RandomPlay) -> Figure:
    fig, axes = _fig(9, 3.6, 1, 2)
    ax = axes[0, 0]
    ax.bar(np.arange(1, WIDTH + 1), np.ones(WIDTH), color=[OUTCOME_COLOR[int(o)] for o in first.outcome])
    for c, o in enumerate(first.outcome):
        ax.text(
            c + 1, 0.5, OUTCOME_NAME[int(o)], ha="center", va="center", color="white", rotation=90, fontweight="bold"
        )
    ax.set_yticks([])
    ax.set_xlabel("первый ход X в колонку")
    ax.set_title("Точный результат после первого хода (солвер)")
    ax.grid(False)
    ax = axes[0, 1]
    ax.bar(np.arange(1, WIDTH + 1), rp.first_move_wins, color="#2a78d6")
    ax.set_ylim(0.4, 0.7)
    ax.set_xlabel("первый ход X в колонку")
    ax.set_ylabel("доля побед X при случайной игре")
    ax.set_title("То же, но для случайной игры")
    fig.tight_layout()
    return fig


def eda_sets(sets: dict[str, PositionSet]) -> Figure:
    fig, axes = _fig(10, 7, 2, 2)
    edges = [4, 10, 16, 22, 28, 35]
    labels = [f"{a}–{b - 1}" for a, b in itertools.pairwise(edges)]
    for col, (name, ps) in enumerate(sets.items()):
        ax = axes[0, col]
        bottom = np.zeros(len(labels))
        for v in (1, 0, -1):
            share = np.array(
                [
                    ((ps.value == v) & (ps.ply >= lo) & (ps.ply < hi)).sum()
                    / max(((ps.ply >= lo) & (ps.ply < hi)).sum(), 1)
                    for lo, hi in itertools.pairwise(edges)
                ]
            )
            ax.bar(labels, share, bottom=bottom, color=OUTCOME_COLOR[v], label=OUTCOME_NAME[v])
            bottom += share
        ax.set_title(f"Набор «{name}»: результат позиции для ходящего")
        ax.set_xlabel("число ходов в позиции")
        ax.set_ylim(0, 1)
        if col == 0:
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        ax = axes[1, col]
        nt = [
            ps.nontrivial[(ps.ply >= lo) & (ps.ply < hi)].mean() if ((ps.ply >= lo) & (ps.ply < hi)).any() else np.nan
            for lo, hi in itertools.pairwise(edges)
        ]
        nb = [
            ps.n_best[(ps.ply >= lo) & (ps.ply < hi)].mean() / 7 if ((ps.ply >= lo) & (ps.ply < hi)).any() else np.nan
            for lo, hi in itertools.pairwise(edges)
        ]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, nt, 0.4, color="#6250d6", label="доля «нетривиальных» позиций")
        ax.bar(x + 0.2, nb, 0.4, color="#9fd8c4", label="доля оптимальных ходов среди 7 колонок")
        ax.set_xticks(x, labels)
        ax.set_xlabel("число ходов в позиции")
        ax.set_ylim(0, 1.05)
        if col == 0:
            ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# method
# ---------------------------------------------------------------------------------------------
def pipeline_diagram() -> Figure:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    boxes = [
        (0.1, 1.9, "Self-play\n256 партий параллельно", "#9fd8c4"),
        (2.7, 1.9, "Батчевый MCTS\nN симуляций на ход", "#2a78d6"),
        (5.3, 1.9, "Replay-буфер\n(поз., π, z)", "#eda100"),
        (7.9, 1.9, "Сеть f(s) = (p, v)\nSGD по π и z", "#6250d6"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.0, 1.0, boxstyle="round,pad=0.05", fc=c, ec="none", alpha=0.35))
        ax.text(x + 1.0, y + 0.5, t, ha="center", va="center", fontsize=9)
    for x0, x1, y in ((2.15, 2.65, 2.4), (4.75, 5.25, 2.4), (7.35, 7.85, 2.4)):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color=MUTED))
    ax.add_patch(
        FancyArrowPatch(
            (8.9, 1.85), (1.1, 1.85), arrowstyle="-|>", mutation_scale=14, color=MUTED, connectionstyle="arc3,rad=0.35"
        )
    )
    ax.text(5.0, 0.55, "новые веса сети → следующая итерация self-play", ha="center", color=MUTED, fontsize=9)
    ax.text(5.0, 3.15, "Цикл AlphaZero-lite", ha="center", fontsize=11, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def learning_curves(logs: dict[str, TrainLog]) -> Figure:
    fig, axes = _fig(13, 4, 1, 3)
    panels = (
        (
            axes[0, 0],
            "acc_nt_expert",
            "games",
            "число партий self-play",
            "точность на «нетривиальных» позициях, набор expert",
        ),
        (axes[0, 1], "acc_nt_expert", "seconds", "время обучения, с", "то же, по времени"),
        (axes[0, 2], "acc_nt_random", "games", "число партий self-play", "набор random"),
    )
    for ax, key, xk, xl, yl in panels:
        for name, log in logs.items():
            if not log.metrics:
                continue
            x = np.array(log.games if xk == "games" else log.seconds, dtype=np.float64)
            ax.plot(x, log.series(key), "o-", ms=3, color=RUN_COLOR[name], label=RUN_LABEL[name])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Качество чистой политики (без поиска)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def training_diagnostics(logs: dict[str, TrainLog]) -> Figure:
    fig, axes = _fig(13, 3.8, 1, 3)
    for name in ("az", "az_lite"):
        log = logs[name]
        x = np.array(log.loss_x)
        axes[0, 0].plot(
            x, [m["policy"] for m in log.losses], color=RUN_COLOR[name], label=f"{RUN_LABEL[name]}: политика"
        )
        axes[0, 0].plot(x, [m["value"] for m in log.losses], "--", color=RUN_COLOR[name], label="значение")
        axes[0, 1].plot(x, [m["game_len"] for m in log.losses], color=RUN_COLOR[name], label=RUN_LABEL[name])
        axes[0, 2].plot(x, [m["draw"] for m in log.losses], color=RUN_COLOR[name], label="ничьи")
        axes[0, 2].plot(x, [m["first_win"] for m in log.losses], "--", color=RUN_COLOR[name], label="победы X")
    log = logs["dqn"]
    if log.losses:
        ax2 = axes[0, 0].twinx()
        ax2.plot(log.loss_x, [m["td"] for m in log.losses], color=RUN_COLOR["dqn"], alpha=0.8)
        ax2.set_ylabel("DQN: TD-ошибка (Huber)", color=RUN_COLOR["dqn"])
        ax2.grid(False)
    axes[0, 0].set_xlabel("число партий self-play")
    axes[0, 0].set_ylabel("AZ: потеря политики / значения")
    axes[0, 0].set_title("Потери", loc="left", fontsize=10)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_xlabel("число партий self-play")
    axes[0, 1].set_ylabel("средняя длина партии self-play, ходов")
    axes[0, 1].set_title("Длина партий", loc="left", fontsize=10)
    axes[0, 2].set_xlabel("число партий self-play")
    axes[0, 2].set_ylabel("доля партий")
    axes[0, 2].set_title("Исходы self-play (сплошная — ничьи)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------
def agents_bars(
    names: list[str],
    families: list[str],
    summary: dict[str, dict[str, dict[str, float]]],
    ci: dict[str, dict[str, dict[str, tuple[float, float]]]],
    metric: str,
    title: str,
) -> Figure:
    fig, axes = _fig(11, 0.36 * len(names) + 1.3, 1, 2)
    for col, set_name in enumerate(("random", "expert")):
        ax = axes[0, col]
        vals = np.array([summary[n][set_name][metric] for n in names])
        order = np.argsort(vals) if metric.startswith("acc") else np.argsort(-vals)
        y = np.arange(len(names))
        for k, i in enumerate(order):
            lo, hi = ci[names[i]][set_name][metric]
            ax.barh(k, vals[i], color=FAMILY_COLOR[families[i]], alpha=0.9)
            ax.plot([lo, hi], [k, k], color=INK, lw=1.2)
            ax.text(min(vals[i] + 0.01, 0.97), k, f"{100 * vals[i]:.1f}", va="center", fontsize=8)
        ax.set_yticks(y, [names[i] for i in order])
        ax.set_xlim(0, 1.05)
        ax.set_title(f"набор «{set_name}»", loc="left", fontsize=10)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def sims_sweep(points: list[tuple[str, int, float, float]], n_ref: dict[str, float]) -> Figure:
    """``points``: (net, sims, accuracy on non-trivial expert positions, seconds per position)."""
    fig, axes = _fig(10, 3.8, 1, 2)
    style = {
        "final": ("#1baf7a", "обученная сеть + MCTS"),
        "init": ("#8f8e88", "необученная сеть + MCTS"),
        "uct": ("#2a78d6", "UCT: случайные прогоны"),
    }
    for net, (color, label) in style.items():
        pts = sorted((s, a, t) for n, s, a, t in points if n == net)
        if not pts:
            continue
        xs = [max(s, 0.7) for s, _, _ in pts]
        axes[0, 0].plot(xs, [a for _, a, _ in pts], "o-", color=color, label=label)
        axes[0, 1].plot(xs, [1000 * t for _, _, t in pts], "o-", color=color, label=label)
    for a in axes[0]:
        a.set_xscale("log")
        a.set_xlabel("симуляций MCTS на ход (0 — чистая политика)")
    axes[0, 0].set_ylabel("точность, «нетривиальные» позиции, набор expert")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("мс на позицию (батч из всех позиций)")
    fig.tight_layout()
    return fig


def elo_forest(names: list[str], families: list[str], elo: F64, lo: F64, hi: F64) -> Figure:
    order = np.argsort(elo)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.2))
    for k, i in enumerate(order):
        ax.plot([lo[i], hi[i]], [k, k], color=INK, lw=1.2)
        ax.plot(elo[i], k, "o", color=FAMILY_COLOR[families[i]], ms=8)
        ax.text(hi[i] + 15, k, f"{elo[i]:.0f}", va="center", fontsize=8)
    ax.set_yticks(range(len(names)), [names[i] for i in order])
    ax.set_xlabel("Elo (random = 0), 95% bootstrap-интервал")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def tournament_matrix(t: Tournament, elo: F64) -> Figure:
    score, games = t.points()
    order = np.argsort(-elo)
    share = np.where(games > 0, score / np.maximum(games, 1), np.nan)[np.ix_(order, order)]
    p = len(order)
    fig, ax = plt.subplots(figsize=(0.55 * p + 2.4, 0.55 * p + 1.6))
    im = ax.imshow(share, vmin=0, vmax=1, cmap="RdYlGn")
    for i in range(p):
        for j in range(p):
            if i != j and not np.isnan(share[i, j]):
                ax.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(p), [t.names[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(p), [t.names[i] for i in order], fontsize=8)
    ax.grid(False)
    ax.set_title("Доля очков игрока в строке против игрока в столбце, %", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def elo_progress(names: list[str], elo: F64, lo: F64, hi: F64, total_games: int) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    pts = [("az-init", 0.0), ("az-25%", 0.25), ("az-50%", 0.5), ("az-raw", 1.0)]
    idx = [names.index(n) for n, _ in pts if n in names]
    xs = [f * total_games for n, f in pts if n in names]
    ax.errorbar(
        xs,
        elo[idx],
        yerr=[elo[idx] - lo[idx], hi[idx] - elo[idx]],
        fmt="o-",
        color="#1baf7a",
        capsize=3,
        label="чистая политика",
    )
    for n, c in (("az+mcts25", "#5fc4a1"), ("az+mcts100", "#6250d6")):
        if n in names:
            i = names.index(n)
            ax.errorbar(
                [total_games],
                [elo[i]],
                yerr=[[elo[i] - lo[i]], [hi[i] - elo[i]]],
                fmt="s",
                color=c,
                capsize=3,
                label=f"финальная сеть, {n.split('mcts')[1]} симуляций",
            )
    ax.set_xlabel("число партий self-play")
    ax.set_ylabel("Elo (random = 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def value_head(class_mean: dict[int, float], class_std: dict[int, float], edges: list[int], mae: list[float]) -> Figure:
    fig, axes = _fig(9.5, 3.6, 1, 2)
    ax = axes[0, 0]
    cs = sorted(class_mean)
    ax.bar(
        [OUTCOME_NAME[c] for c in cs],
        [class_mean[c] for c in cs],
        yerr=[class_std[c] for c in cs],
        color=[OUTCOME_COLOR[c] for c in cs],
        capsize=4,
    )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("выход value-головы (для ходящего)")
    ax.set_xlabel("истинный результат позиции (солвер)")
    ax.set_title("Что «думает» value-голова", loc="left", fontsize=10)
    ax = axes[0, 1]
    labels = [f"{a}–{b - 1}" for a, b in itertools.pairwise(edges)]
    ax.bar(labels, mae, color="#6250d6")
    ax.set_xlabel("число ходов в позиции")
    ax.set_ylabel("|value − истинный результат|")
    ax.set_title("Ошибка value-головы по стадии партии", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def value_traces(traces: list[tuple[str, list[int], list[float], list[int]]]) -> Figure:
    n = max(len(traces), 1)
    fig, axes = _fig(4.3 * n, 3.4, 1, n)
    for k, (title, plies, nv, tv) in enumerate(traces):
        ax = axes[0, k]
        ax.step(plies, tv, where="post", color=INK, lw=1.5, label="солвер")
        ax.plot(plies, nv, "o-", ms=3, color="#1baf7a", label="value-голова")
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("номер хода")
        if k == 0:
            ax.set_ylabel("оценка для первого игрока (X)")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def policy_examples(examples: list[tuple[str, list[float], list[float], list[bool], int, int]]) -> Figure:
    """Rows: (moves, raw policy, MCTS visit shares, optimal columns, raw choice, MCTS choice)."""
    n = max(len(examples), 1)
    fig, axes = _fig(9.5, 3.0 * n, n, 2)
    for k, (moves, raw, vis, best, raw_c, mcts_c) in enumerate(examples):
        draw_board(axes[k, 0], moves, best=best)
        axes[k, 0].set_title(f"после {len(moves)} ходов, ходит {'X' if len(moves) % 2 == 0 else 'O'}", fontsize=9)
        ax = axes[k, 1]
        x = np.arange(WIDTH)
        ax.bar(x - 0.2, raw, 0.4, color="#2a78d6", label="чистая политика")
        ax.bar(x + 0.2, vis, 0.4, color="#1baf7a", label="доля визитов MCTS (100)")
        for c in range(WIDTH):
            if best[c]:
                ax.text(c, -0.07, "★", ha="center", color="#eda100", fontsize=12, clip_on=False)
        ax.set_xticks(x, [str(c + 1) for c in x])
        ax.set_ylim(0, 1)
        ax.set_title(f"политика выбирает {raw_c + 1}, MCTS выбирает {mcts_c + 1} (★ — оптимальные ходы)", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
