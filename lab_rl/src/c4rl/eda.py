"""Exploratory analysis of the game itself: size of the state space, random-play statistics, solver values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .game import BOARD_MASK, CELLS, I64, WIDTH, has_four, legal_moves, mirror_bits, play
from .solver import Solver

IntArr = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PlyCounts:
    """Distinct positions reachable after exactly ``ply`` moves (ply 0..max)."""

    total: list[int]  # all distinct positions, including those where the last move won
    terminal: list[int]  # of which the last move completed four in a row
    up_to_mirror: list[int]  # distinct positions when a board and its mirror image are identified


def _unique_rows(cur: I64, mask: I64) -> tuple[I64, I64]:
    keys = np.stack([mask, cur], axis=1)
    u = np.unique(keys, axis=0)
    return u[:, 1].copy(), u[:, 0].copy()


def count_positions(max_ply: int) -> PlyCounts:
    """Exhaustive breadth-first enumeration (numpy-vectorised) of the positions after 0..``max_ply`` moves."""
    cur = np.zeros(1, dtype=np.int64)
    mask = np.zeros(1, dtype=np.int64)
    total, terminal, mirrored = [1], [0], [1]
    for _ in range(max_ply):
        cs, ms = [], []
        legal = legal_moves(mask)
        for c in range(WIDTH):
            ok = legal[:, c]
            nc, nm, _ = play(cur[ok], mask[ok], np.full(int(ok.sum()), c, dtype=np.int64))
            cs.append(nc)
            ms.append(nm)
        cur, mask = _unique_rows(np.concatenate(cs), np.concatenate(ms))
        won = has_four(mask ^ cur)  # the stones of the player who has just moved
        total.append(len(cur))
        terminal.append(int(won.sum()))
        mc, mm = mirror_bits(cur), mirror_bits(mask)
        keep = (mask < mm) | ((mask == mm) & (cur <= mc))  # one representative per {board, mirror image}
        mirrored.append(int(keep.sum()))
        cur, mask = cur[~won], mask[~won]  # finished games are not expanded
    return PlyCounts(total, terminal, mirrored)


@dataclass(frozen=True)
class RandomPlay:
    """Statistics of uniformly random self-play."""

    length: IntArr  # (n,) number of moves of every game
    result: IntArr  # (n,) +1 first player won, 0 draw, -1 second player won
    first_move_wins: npt.NDArray[np.float64]  # (7,) share of games won by X per first column


def random_play(n_games: int, rng: np.random.Generator) -> RandomPlay:
    cur = np.zeros(n_games, dtype=np.int64)
    mask = np.zeros(n_games, dtype=np.int64)
    first = np.zeros(n_games, dtype=np.int64)
    length = np.zeros(n_games, dtype=np.int64)
    result = np.zeros(n_games, dtype=np.int64)
    alive = np.arange(n_games)
    ply = 0
    while alive.size:
        legal = legal_moves(mask[alive])
        r = rng.random(legal.shape)
        r[~legal] = -1.0
        a = r.argmax(axis=1).astype(np.int64)
        if ply == 0:
            first[alive] = a
        nc, nm, won = play(cur[alive], mask[alive], a)
        cur[alive], mask[alive] = nc, nm
        result[alive[won]] = 1 if ply % 2 == 0 else -1
        full = ~won & (nm == BOARD_MASK)
        length[alive[won | full]] = ply + 1
        alive = alive[~won & ~full]
        ply += 1
    fm = np.array([(result[first == c] == 1).mean() if (first == c).any() else np.nan for c in range(WIDTH)])
    return RandomPlay(length, result, fm)


@dataclass(frozen=True)
class FirstMoves:
    """Game-theoretic value of every first move for the first player (solver): +1 win, 0 draw, -1 loss."""

    outcome: IntArr  # (7,)


def first_moves(solver: Solver) -> FirstMoves:
    return FirstMoves(solver.outcomes("").astype(np.int64))


@dataclass(frozen=True)
class EdaResults:
    counts: PlyCounts
    random: RandomPlay
    first: FirstMoves
    max_len: int = CELLS
