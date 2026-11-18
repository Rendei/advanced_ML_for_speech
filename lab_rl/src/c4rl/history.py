"""Training logs shared by the AlphaZero-lite and DQN trainers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from torch import nn

from .game import F32, I64, WIDTH, mirror_action, mirror_bits

EvalFn = Callable[[nn.Module], dict[str, float]]  # evaluates the model being trained, returns named metrics


@dataclass
class TrainLog:
    """Everything the report needs about one training run."""

    name: str
    games: list[int] = field(default_factory=list)  # cumulative self-play games at each evaluation point
    seconds: list[float] = field(default_factory=list)
    metrics: list[dict[str, float]] = field(default_factory=list)
    loss_x: list[int] = field(default_factory=list)  # iteration / gradient step of every loss record
    losses: list[dict[str, float]] = field(default_factory=list)
    checkpoints: dict[str, Path] = field(default_factory=dict)  # "init", "p25", "p50", "final" -> file
    n_params: int = 0
    total_seconds: float = 0.0
    selfplay_seconds: float = 0.0
    total_games: int = 0
    total_positions: int = 0

    def series(self, key: str) -> npt.NDArray[np.float64]:
        return np.array([m.get(key, np.nan) for m in self.metrics], dtype=np.float64)


class ReplayBuffer:
    """Ring buffer of (position, MCTS policy, outcome) triples stored as bitboards (a position is two int64)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cur = np.zeros(capacity, dtype=np.int64)
        self.mask = np.zeros(capacity, dtype=np.int64)
        self.pi = np.zeros((capacity, WIDTH), dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self._head = 0

    def add(self, cur: I64, mask: I64, pi: F32, z: F32) -> None:
        cur, mask, pi, z = cur[-self.capacity :], mask[-self.capacity :], pi[-self.capacity :], z[-self.capacity :]
        idx = (self._head + np.arange(len(cur))) % self.capacity
        self.cur[idx], self.mask[idx], self.pi[idx], self.z[idx] = cur, mask, pi, z
        self._head = (self._head + len(cur)) % self.capacity
        self.size = min(self.size + len(cur), self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[I64, I64, F32, F32]:
        """Random batch with a left-right mirror applied to a random half (Connect-4 is mirror symmetric)."""
        idx = rng.integers(0, self.size, size=n)
        cur, mask, pi, z = self.cur[idx], self.mask[idx], self.pi[idx].copy(), self.z[idx]
        flip = rng.random(n) < 0.5
        cur = np.where(flip, mirror_bits(cur), cur)
        mask = np.where(flip, mirror_bits(mask), mask)
        pi[flip] = pi[flip][:, ::-1]
        return cur, mask, pi, z


def mirror_batch(
    cur: I64, mask: I64, a: npt.NDArray[np.int64], ncur: I64, nmask: I64, flip: npt.NDArray[np.bool_]
) -> tuple[I64, I64, npt.NDArray[np.int64], I64, I64]:
    """Mirror the transitions selected by ``flip`` (states, action and next states together)."""
    return (
        np.where(flip, mirror_bits(cur), cur),
        np.where(flip, mirror_bits(mask), mask),
        np.where(flip, mirror_action(a), a),
        np.where(flip, mirror_bits(ncur), ncur),
        np.where(flip, mirror_bits(nmask), nmask),
    )
