"""Players (anything that picks a column for a batch of positions) and a batched game runner.

A player gets ``cur`` / ``mask`` bitboards of every game in which it has to move, plus the move histories as
solver-style digit strings (only the perfect player needs those) and returns one column per game.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch

from .game import (
    BOARD_MASK,
    CELLS,
    HEIGHT,
    I64,
    WIDTH,
    legal_cols,
    legal_moves,
    play,
    play_int,
    to_moves,
)
from .mcts import BatchMCTS, Evaluator, policy_from_visits
from .net import NetEvaluator, PVNet, QNet, to_tensor
from .solver import Solver

IntArr = npt.NDArray[np.int64]


class Player(Protocol):
    name: str

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr: ...


def _pick(scores: npt.NDArray[np.floating], legal: npt.NDArray[np.bool_], rng: np.random.Generator) -> IntArr:
    """Argmax over legal columns with random tie-breaking."""
    s = np.where(legal, scores, -np.inf) + rng.random(scores.shape) * 1e-9
    return s.argmax(axis=1).astype(np.int64)


class RandomPlayer:
    def __init__(self, rng: np.random.Generator, name: str = "random") -> None:
        self.rng, self.name = rng, name

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        legal = legal_moves(mask)
        return _pick(np.zeros(legal.shape), legal, self.rng)


class GreedyPlayer:
    """Takes a winning move if there is one, otherwise blocks the opponent's winning move, otherwise random."""

    def __init__(self, rng: np.random.Generator, name: str = "greedy") -> None:
        self.rng, self.name = rng, name

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        legal = legal_moves(mask)
        score = np.zeros(legal.shape)
        opp = mask ^ cur
        for c in range(WIDTH):
            a = np.full(len(cur), c, dtype=np.int64)
            ok = legal[:, c]
            _, _, win = play(cur, mask, a)
            _, _, opp_win = play(opp, mask, a)  # the opponent moving into the same column
            score[:, c] = np.where(ok & win, 2.0, np.where(ok & opp_win, 1.0, 0.0))
        return _pick(score, legal, self.rng)


# ---------------------------------------------------------------------------------------------
# alpha-beta minimax with a window-counting heuristic (classical non-learned baseline)
# ---------------------------------------------------------------------------------------------
def _windows() -> list[int]:
    out: list[int] = []
    h1 = HEIGHT + 1
    for c in range(WIDTH):
        for r in range(HEIGHT):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in cells):
                    out.append(sum(1 << (x * h1 + y) for x, y in cells))
    return out


WINDOWS = _windows()  # the 69 lines of four
WINDOW_WEIGHT = (0, 1, 4, 16, 0)
COL_ORDER = (3, 2, 4, 1, 5, 0, 6)
WIN_SCORE = 100_000


def _heuristic(cur: int, mask: int) -> int:
    opp = mask ^ cur
    s = 0
    for w in WINDOWS:
        if not w & opp:
            s += WINDOW_WEIGHT[(w & cur).bit_count()]
        elif not w & cur:
            s -= WINDOW_WEIGHT[(w & opp).bit_count()]
    return s


def _negamax(cur: int, mask: int, depth: int, alpha: int, beta: int) -> int:
    cols = [c for c in COL_ORDER if not mask & (1 << (c * (HEIGHT + 1) + HEIGHT - 1))]
    if not cols:
        return 0
    if depth == 0:
        return _heuristic(cur, mask)
    children = []
    for c in cols:
        ncur, nmask, won = play_int(cur, mask, c)
        if won:
            return WIN_SCORE + depth  # faster wins are better
        children.append((ncur, nmask))
    best = -WIN_SCORE * 2
    for ncur, nmask in children:
        v = -_negamax(ncur, nmask, depth - 1, -beta, -alpha)
        best = max(best, v)
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best


class MinimaxPlayer:
    def __init__(self, depth: int, rng: np.random.Generator) -> None:
        self.depth, self.rng = depth, rng
        self.name = f"minimax-{depth}"

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        out = np.empty(len(cur), dtype=np.int64)
        for i, (c0, m0) in enumerate(zip(cur.tolist(), mask.tolist(), strict=True)):
            scores: dict[int, int] = {}
            for c in legal_cols(m0):
                ncur, nmask, won = play_int(c0, m0, c)
                scores[c] = WIN_SCORE + self.depth if won else -_negamax(ncur, nmask, self.depth - 1, -(10**9), 10**9)
            top = max(scores.values())
            out[i] = self.rng.choice([c for c, v in scores.items() if v == top])
        return out


# ---------------------------------------------------------------------------------------------
# learned players
# ---------------------------------------------------------------------------------------------
class PolicyPlayer:
    """The raw policy head, no search (0 simulations)."""

    def __init__(self, model: PVNet, device: torch.device, rng: np.random.Generator, name: str) -> None:
        self.model, self.device, self.rng, self.name = model, device, rng, name

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        self.model.eval()
        with torch.inference_mode():
            logits, _ = self.model(to_tensor(cur, mask, self.device))
        return _pick(logits.cpu().numpy(), legal_moves(mask), self.rng)


class MCTSPlayer:
    """Plays the most visited move of a batched MCTS (no exploration noise)."""

    def __init__(
        self, evaluator: Evaluator, sims: int, rng: np.random.Generator, name: str, c_puct: float = 1.5
    ) -> None:
        self.mcts = BatchMCTS(evaluator, sims, rng, c_puct=c_puct)
        self.rng, self.name = rng, name

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        visits = self.mcts.search(cur, mask, noise=False).visits
        return policy_from_visits(visits, 0.0, self.rng).argmax(axis=1).astype(np.int64)


class QPlayer:
    def __init__(self, model: QNet, device: torch.device, rng: np.random.Generator, name: str) -> None:
        self.model, self.device, self.rng, self.name = model, device, rng, name

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        self.model.eval()
        with torch.inference_mode():
            q = self.model(to_tensor(cur, mask, self.device))
        return _pick(q.cpu().numpy(), legal_moves(mask), self.rng)


class PerfectPlayer:
    """Asks the solver; picks uniformly among the moves that keep the best game-theoretic result."""

    def __init__(self, solver: Solver, rng: np.random.Generator, name: str = "perfect") -> None:
        self.solver, self.rng, self.name = solver, rng, name

    def act(self, cur: I64, mask: I64, hist: Sequence[str]) -> IntArr:
        best = np.stack([self.solver.best_set(h) for h in hist]).astype(np.float64)
        return _pick(best, legal_moves(mask), self.rng)


def mcts_net_player(
    model: PVNet, device: torch.device, sims: int, rng: np.random.Generator, name: str | None = None
) -> Player:
    return MCTSPlayer(NetEvaluator(model, device), sims, rng, name or f"az+mcts{sims}")


# ---------------------------------------------------------------------------------------------
# batched game runner
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchResult:
    outcome: IntArr  # (G,) from the point of view of the first player (X): +1 win, 0 draw, -1 loss
    moves: list[list[int]]  # full move lists (columns 0..6), openings included


def random_openings(n: int, plies: int, rng: np.random.Generator) -> list[list[int]]:
    """``n`` random, legal, non-terminal opening move lists of length ``plies``."""
    out: list[list[int]] = []
    while len(out) < n:
        cur = mask = 0
        cols: list[int] = []
        ok = True
        for _ in range(plies):
            legal = legal_cols(mask)
            c = int(rng.choice(legal))
            cur, mask, won = play_int(cur, mask, c)
            cols.append(c)
            if won:
                ok = False
                break
        if ok:
            out.append(cols)
    return out


def play_games(first: Player, second: Player, openings: list[list[int]]) -> MatchResult:
    """Play one game per opening; ``first`` owns the stones that move first (X) after the opening if the opening
    has even length. All games advance in lock-step so each player receives a batch of positions per ply."""
    g = len(openings)
    cur = np.zeros(g, dtype=np.int64)
    mask = np.zeros(g, dtype=np.int64)
    moves: list[list[int]] = [[] for _ in range(g)]
    for i, opening in enumerate(openings):
        c, m = 0, 0
        for col in opening:
            c, m, _ = play_int(c, m, col)
            moves[i].append(col)
        cur[i], mask[i] = c, m
    outcome = np.zeros(g, dtype=np.int64)
    alive = np.arange(g)
    ply = len(openings[0])
    while alive.size:
        x_moves = ply % 2 == 0
        player = first if x_moves else second
        hist = [to_moves(moves[i]) for i in alive]
        a = player.act(cur[alive], mask[alive], hist)
        ncur, nmask, won = play(cur[alive], mask[alive], a)
        for k, gi in enumerate(alive):
            moves[gi].append(int(a[k]))
        cur[alive], mask[alive] = ncur, nmask
        outcome[alive[won]] = 1 if x_moves else -1
        full = ~won & (nmask == BOARD_MASK)
        alive = alive[~won & ~full]
        ply += 1
        if ply > CELLS:
            break
    return MatchResult(outcome, moves)
