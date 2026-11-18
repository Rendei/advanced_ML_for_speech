"""Batched Monte-Carlo tree search, fully vectorised in numpy.

Instead of one Python tree per game, ``G`` independent trees live in dense arrays (``G x nodes x 7`` edge
statistics). One call of ``_simulate`` advances **every** game by one simulation: all games descend by PUCT in
lock-step (one numpy operation per tree level), the new leaves of all games are evaluated by *one* network call,
and the values are backed up with one numpy operation per level. This makes the search ~50x faster than a
node-object implementation and keeps the GPU busy with large batches.

Conventions: the statistics of an edge ``(node, a)`` are from the point of view of the player to move at ``node``;
a position's value is always for the player to move there (so a position the opponent has just won is -1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .game import BOARD_MASK, CELLS, F32, I64, WIDTH, legal_moves, play

UNEXPLORED = -1
TERMINAL_WIN = -2  # the move completes four in a row (edge value +1 for the mover)
TERMINAL_DRAW = -3  # the move fills the board


class Evaluator(Protocol):
    """``(cur, mask) -> (priors (N, 7), value (N,))``; the value is for the player to move."""

    def __call__(self, cur: I64, mask: I64) -> tuple[F32, F32]: ...


class RolloutEvaluator:
    """Classical UCT: uniform priors and the result of ``n`` uniformly random playouts as the value."""

    def __init__(self, rng: np.random.Generator, n_rollouts: int = 1) -> None:
        self.rng = rng
        self.n = n_rollouts

    def __call__(self, cur: I64, mask: I64) -> tuple[F32, F32]:
        legal = legal_moves(mask)
        priors = (legal / np.maximum(legal.sum(1, keepdims=True), 1)).astype(np.float32)
        total = np.zeros(len(cur), dtype=np.float32)
        for _ in range(self.n):
            total += self._playout(cur, mask)
        return priors, (total / self.n).astype(np.float32)

    def _playout(self, cur: I64, mask: I64) -> F32:
        cur, mask = cur.copy(), mask.copy()
        n = len(cur)
        sign = np.ones(n, dtype=np.float32)  # +1: the side that moves now is the one we evaluate for
        result = np.zeros(n, dtype=np.float32)
        alive = np.ones(n, dtype=np.bool_)
        for _ in range(CELLS):
            legal = legal_moves(mask)
            alive &= legal.any(axis=1)
            if not alive.any():
                break
            r = self.rng.random((n, WIDTH))
            r[~legal] = -1.0
            a = r.argmax(axis=1)
            ncur, nmask, won = play(cur, mask, a)
            won &= alive
            result[won] = sign[won]
            alive &= ~won
            cur = np.where(alive, ncur, cur)
            mask = np.where(alive, nmask, mask)
            sign = -sign
        return result


@dataclass(frozen=True)
class SearchResult:
    visits: F32  # (G, 7) root visit counts
    value: F32  # (G,) mean value of the root for the player to move


class BatchMCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        sims: int,
        rng: np.random.Generator,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 1.0,
        dirichlet_eps: float = 0.25,
    ) -> None:
        if sims < 1:
            raise ValueError("sims must be >= 1 (use the raw policy for 0 simulations)")
        self.evaluator = evaluator
        self.sims = sims
        self.rng = rng
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps

    # -------------------------------------------------------------------------------------
    def search(self, cur: I64, mask: I64, noise: bool = False) -> SearchResult:
        """Run ``sims`` simulations from each of the (non-terminal) root positions."""
        g, m = len(cur), self.sims + 1
        self._child = np.full((g, m, WIDTH), UNEXPLORED, dtype=np.int32)
        self._n = np.zeros((g, m, WIDTH), dtype=np.float32)
        self._w = np.zeros((g, m, WIDTH), dtype=np.float32)
        self._p = np.zeros((g, m, WIDTH), dtype=np.float32)
        self._cur = np.zeros((g, m), dtype=np.int64)
        self._mask = np.zeros((g, m), dtype=np.int64)
        self._count = np.ones(g, dtype=np.int64)
        self._cur[:, 0], self._mask[:, 0] = cur, mask

        priors, _ = self.evaluator(cur, mask)
        root_p = self._normalise(priors, legal_moves(mask))
        if noise:
            gamma = self.rng.gamma(self.alpha, size=(g, WIDTH)).astype(np.float32)
            gamma *= legal_moves(mask)
            dirichlet = gamma / np.maximum(gamma.sum(1, keepdims=True), 1e-12)
            root_p = ((1 - self.eps) * root_p + self.eps * dirichlet).astype(np.float32)
        self._p[:, 0] = root_p

        for _ in range(self.sims):
            self._simulate()
        visits = self._n[:, 0].copy()
        total = np.maximum(visits.sum(1), 1.0)
        return SearchResult(visits, (self._w[:, 0].sum(1) / total).astype(np.float32))

    @staticmethod
    def _normalise(priors: F32, legal: npt.NDArray[np.bool_]) -> F32:
        p = priors * legal
        s = p.sum(1, keepdims=True)
        uniform = legal / np.maximum(legal.sum(1, keepdims=True), 1)
        return np.where(s > 1e-12, p / np.maximum(s, 1e-12), uniform).astype(np.float32)

    # -------------------------------------------------------------------------------------
    def _simulate(self) -> None:
        g = len(self._count)
        path_node = np.zeros((g, CELLS + 1), dtype=np.int32)
        path_act = np.zeros((g, CELLS + 1), dtype=np.int32)
        plen = np.zeros(g, dtype=np.int32)
        leaf_val = np.zeros(g, dtype=np.float32)  # value of the last edge, for the player who played it
        eval_node = np.full(g, -1, dtype=np.int64)  # node to evaluate with the network, or -1
        node = np.zeros(g, dtype=np.int64)
        active = np.arange(g)

        while active.size:
            nd = node[active]
            n = self._n[active, nd]
            w = self._w[active, nd]
            p = self._p[active, nd]
            legal = legal_moves(self._mask[active, nd])
            total = n.sum(axis=1, keepdims=True)
            q = np.divide(w, n, out=np.zeros_like(w), where=n > 0)
            score = q + self.c_puct * p * np.sqrt(total + 1.0) / (1.0 + n)
            score[~legal] = -np.inf
            a = score.argmax(axis=1)

            path_node[active, plen[active]] = nd
            path_act[active, plen[active]] = a
            plen[active] += 1

            ch = self._child[active, nd, a]
            go_on = ch >= 0
            node[active[go_on]] = ch[go_on]

            done_term = ch <= TERMINAL_WIN
            leaf_val[active[done_term]] = np.where(ch[done_term] == TERMINAL_WIN, 1.0, 0.0)

            new = ch == UNEXPLORED
            if new.any():
                gi, nd_n, a_n = active[new], nd[new], a[new]
                ncur, nmask, won = play(self._cur[gi, nd_n], self._mask[gi, nd_n], a_n)
                full = ~won & (nmask == BOARD_MASK)
                alive = ~won & ~full
                self._child[gi[won], nd_n[won], a_n[won]] = TERMINAL_WIN
                leaf_val[gi[won]] = 1.0
                self._child[gi[full], nd_n[full], a_n[full]] = TERMINAL_DRAW
                gl = gi[alive]
                nid = self._count[gl]
                self._count[gl] += 1
                self._child[gl, nd_n[alive], a_n[alive]] = nid
                self._cur[gl, nid] = ncur[alive]
                self._mask[gl, nid] = nmask[alive]
                eval_node[gl] = nid

            active = active[go_on]

        ev = np.nonzero(eval_node >= 0)[0]
        if ev.size:
            nid = eval_node[ev]
            cur, mask = self._cur[ev, nid], self._mask[ev, nid]
            priors, value = self.evaluator(cur, mask)
            self._p[ev, nid] = self._normalise(priors, legal_moves(mask))
            leaf_val[ev] = -value  # the edge belongs to the player who moved, the network speaks for the opponent

        for d in range(int(plen.max())):
            sel = np.nonzero(plen > d)[0]
            sign = np.where((plen[sel] - 1 - d) % 2 == 0, 1.0, -1.0).astype(np.float32)
            pn, pa = path_node[sel, d], path_act[sel, d]
            self._n[sel, pn, pa] += 1.0
            self._w[sel, pn, pa] += sign * leaf_val[sel]


def policy_from_visits(visits: F32, temperature: float, rng: np.random.Generator | None = None) -> F32:
    """Visit counts -> move distribution. ``temperature`` 0 = most visited (ties broken at random)."""
    if temperature <= 1e-6:
        noise = (rng.random(visits.shape) * 1e-3) if rng is not None else 0.0
        best = (visits + noise).argmax(axis=1)
        out = np.zeros_like(visits)
        out[np.arange(len(visits)), best] = 1.0
        return out
    v = visits ** (1.0 / temperature)
    return np.asarray(v / np.maximum(v.sum(1, keepdims=True), 1e-12), dtype=np.float32)


def sample_actions(probs: F32, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    cdf = probs.cumsum(axis=1)
    u = rng.random((len(probs), 1)) * cdf[:, -1:]
    return np.minimum((cdf < u).sum(axis=1), WIDTH - 1).astype(np.int64)
