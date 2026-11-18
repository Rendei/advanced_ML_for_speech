"""Evaluation against ground truth: solver-labelled test positions, round-robin tournaments and Elo ratings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .config import EvalConfig
from .game import CELLS, I64, WIDTH, BoolArr, from_moves, legal_cols, legal_moves, play_int, to_moves
from .players import Player, play_games, random_openings
from .solver import Solver

F64 = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


# ---------------------------------------------------------------------------------------------
# labelled positions
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PositionSet:
    """Positions labelled by the perfect solver.

    ``outcome[i, c]`` is the game-theoretic result (+1 win, 0 draw, -1 loss for the player to move) after playing
    column ``c`` in position ``i`` (``-2`` for a full column); ``best`` marks the columns that keep the best result.
    """

    name: str
    moves: list[str]
    cur: I64
    mask: I64
    ply: IntArr
    outcome: IntArr  # (N, 7)
    best: BoolArr  # (N, 7)
    value: IntArr  # (N,) result of the position itself

    def __len__(self) -> int:
        return len(self.moves)

    @property
    def legal(self) -> BoolArr:
        return legal_moves(self.mask)

    @property
    def n_best(self) -> IntArr:
        return self.best.sum(axis=1).astype(np.int64)

    @property
    def nontrivial(self) -> BoolArr:
        """Some legal move would worsen the result, i.e. there is something to get wrong."""
        return self.n_best < self.legal.sum(axis=1)


def _label(name: str, moves: list[str], solver: Solver) -> PositionSet:
    cur = np.zeros(len(moves), dtype=np.int64)
    mask = np.zeros(len(moves), dtype=np.int64)
    out = np.zeros((len(moves), WIDTH), dtype=np.int64)
    for i, m in enumerate(moves):
        cur[i], mask[i] = from_moves(m)
        out[i] = solver.outcomes(m)
    value = out.max(axis=1)
    return PositionSet(
        name, moves, cur, mask, np.array([len(m) for m in moves], dtype=np.int64), out, out == value[:, None], value
    )


def _random_prefix(rng: np.random.Generator, ply: int) -> list[int] | None:
    cur = mask = 0
    cols: list[int] = []
    for _ in range(ply):
        c = int(rng.choice(legal_cols(mask)))
        cur, mask, won = play_int(cur, mask, c)
        cols.append(c)
        if won:
            return None
    return cols


def make_position_sets(cfg: EvalConfig, solver: Solver, seed: int) -> dict[str, PositionSet]:
    """Two test sets: ``random`` (uniformly random play, mostly easy tactics) and ``expert`` (near-perfect play with
    a few random slips, the realistic and much harder distribution). Both exclude terminal positions."""
    rng = np.random.default_rng(10_000 + seed)
    random_moves: list[str] = []
    seen: set[str] = set()
    while len(random_moves) < cfg.n_positions:
        ply = int(rng.integers(cfg.min_ply, cfg.max_ply + 1))
        cols = _random_prefix(rng, ply)
        if cols is None:
            continue
        s = to_moves(cols)
        if s not in seen and legal_cols(from_moves(s)[1]):
            seen.add(s)
            random_moves.append(s)

    expert_moves: list[str] = []
    seen = set()
    while len(expert_moves) < cfg.n_positions:
        target = int(rng.integers(cfg.min_ply, cfg.max_ply + 1))
        cur = mask = 0
        cols = []
        ok = True
        for _ in range(target):
            legal = legal_cols(mask)
            if not legal:
                ok = False
                break
            if rng.random() < cfg.expert_eps:
                c = int(rng.choice(legal))
            else:
                best = solver.best_set(to_moves(cols))
                c = int(rng.choice([k for k in legal if best[k]]))
            cur, mask, won = play_int(cur, mask, c)
            cols.append(c)
            if won:
                ok = False
                break
        s = to_moves(cols)
        if ok and s not in seen and legal_cols(mask):
            seen.add(s)
            expert_moves.append(s)
    solver.flush()
    return {"random": _label("random", random_moves, solver), "expert": _label("expert", expert_moves, solver)}


@dataclass(frozen=True)
class PositionScores:
    """Per-position verdicts of one player on one set (kept so that bootstrap intervals can be computed)."""

    correct: BoolArr  # chose an optimal move
    blunder: BoolArr  # chose a move that worsens the game-theoretic result
    action: IntArr


def score_player(player: Player, ps: PositionSet) -> PositionScores:
    a = player.act(ps.cur, ps.mask, ps.moves).astype(np.int64)
    rows = np.arange(len(ps))
    chosen = ps.outcome[rows, a]
    return PositionScores(ps.best[rows, a], chosen < ps.value, a)


def summarise(ps: PositionSet, sc: PositionScores) -> dict[str, float]:
    nt = ps.nontrivial
    return {
        "acc": float(sc.correct.mean()),
        "acc_nt": float(sc.correct[nt].mean()) if nt.any() else float("nan"),
        "blunder": float(sc.blunder.mean()),
        "blunder_nt": float(sc.blunder[nt].mean()) if nt.any() else float("nan"),
    }


def bootstrap_ci(values: npt.NDArray[Any], n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values.astype(np.float64)[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------------------------
# tournament + Elo
# ---------------------------------------------------------------------------------------------
@dataclass
class Tournament:
    names: list[str]
    pair_scores: dict[tuple[int, int], F64] = field(default_factory=dict)  # i < j: score of i in every game
    sample_games: dict[tuple[int, int], list[tuple[list[int], int]]] = field(default_factory=dict)

    def points(self) -> tuple[F64, F64]:
        """``(score, games)`` matrices: ``score[i, j]`` = points of i against j (win 1, draw 0.5)."""
        p = len(self.names)
        score = np.zeros((p, p))
        games = np.zeros((p, p))
        for (i, j), s in self.pair_scores.items():
            score[i, j], score[j, i] = s.sum(), len(s) - s.sum()
            games[i, j] = games[j, i] = len(s)
        return score, games


def run_tournament(players: list[Player], games_per_pair: int, open_plies: int, rng: np.random.Generator) -> Tournament:
    """Round robin. Every opening is played twice with the colours swapped, so luck of the opening cancels out."""
    t = Tournament([p.name for p in players])
    n_open = max(games_per_pair // 2, 1)
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            openings = random_openings(n_open, open_plies, rng)
            g1 = play_games(players[i], players[j], openings)  # i plays X
            g2 = play_games(players[j], players[i], openings)  # j plays X
            s = np.concatenate([(g1.outcome + 1) / 2.0, (1 - g2.outcome) / 2.0])
            t.pair_scores[(i, j)] = s
            t.sample_games[(i, j)] = (
                list(zip(g1.moves, g1.outcome.tolist(), strict=True))[:3]
                + list(zip(g2.moves, (-g2.outcome).tolist(), strict=True))[:3]
            )
            print(f"  {players[i].name:>22} vs {players[j].name:<22} {s.mean():.2f}", flush=True)
    return t


def fit_elo(
    names: list[str], pair_scores: dict[tuple[int, int], F64], anchor: str | None = None, prior: float = 1.0
) -> F64:
    """Bradley-Terry ratings in Elo units. Every pair gets ``prior`` virtual drawn games, which keeps ratings finite
    when one player never scores. ``anchor`` (a player name) is pinned to 0 Elo."""
    p = len(names)
    scale = 400.0 / np.log(10.0)
    ii = np.array([i for (i, _j) in pair_scores])
    jj = np.array([j for (_i, j) in pair_scores])
    n = np.array([len(s) for s in pair_scores.values()], dtype=np.float64)
    wins = np.array([s.sum() for s in pair_scores.values()], dtype=np.float64)

    def nll(theta: F64) -> tuple[float, F64]:
        d = theta[ii] - theta[jj]
        pr = 1.0 / (1.0 + np.exp(-d))
        total = n + 2 * prior
        w = wins + prior
        loss = -(w * np.log(pr + 1e-12) + (total - w) * np.log(1 - pr + 1e-12)).sum()
        gd = -(w - total * pr)
        grad = np.zeros(p)
        np.add.at(grad, ii, gd)
        np.add.at(grad, jj, -gd)
        return float(loss), grad

    res = minimize(nll, np.zeros(p), jac=True, method="L-BFGS-B")
    elo = res.x * scale
    ref = elo[names.index(anchor)] if anchor in names else elo.mean()
    return np.asarray(elo - ref, dtype=np.float64)


def elo_with_ci(t: Tournament, anchor: str, n_boot: int, rng: np.random.Generator) -> tuple[F64, F64, F64]:
    """Point estimate and 95% bootstrap interval (games resampled within each pair) of every player's Elo."""
    point = fit_elo(t.names, t.pair_scores, anchor)
    boots = np.zeros((n_boot, len(t.names)))
    for b in range(n_boot):
        resampled = {k: s[rng.integers(0, len(s), size=len(s))] for k, s in t.pair_scores.items()}
        boots[b] = fit_elo(t.names, resampled, anchor)
    return point, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


__all__ = [
    "CELLS",
    "PositionScores",
    "PositionSet",
    "Tournament",
    "bootstrap_ci",
    "elo_with_ci",
    "fit_elo",
    "make_position_sets",
    "run_tournament",
    "score_player",
    "summarise",
]
