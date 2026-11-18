"""Experiment configuration: three profiles (``smoke`` for CI, ``standard`` for the report, ``full`` for long)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SOLVER_REF = "d6ba50d8aaf2308c769d9bf2abd42d90f34baf41"  # PascalPons/connect4, pinned for reproducibility
SOLVER_REPO = "https://github.com/PascalPons/connect4"
BOOK_URL = "https://github.com/PascalPons/connect4/releases/download/book/7x6.book"


@dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    blocks: int = 5


@dataclass(frozen=True)
class AZConfig:
    """AlphaZero-lite: self-play with batched MCTS -> replay buffer -> supervised update of the policy/value net."""

    iterations: int = 80
    games_per_iter: int = 256  # played in parallel as one batch
    sims: int = 200  # MCTS simulations per move during self-play
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    temp_moves: int = 10  # sample from visit counts for the first moves, then play the most visited
    buffer: int = 150_000  # positions
    train_steps: int = 150  # gradient steps per iteration
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class DQNConfig:
    """Negamax double DQN trained from self-play."""

    games: int = 61_440  # 3x the AZ games: a DQN game is far cheaper (no search)
    envs: int = 64
    buffer: int = 200_000
    batch: int = 256
    updates_per_step: float = 1.0  # gradient steps per environment step (one step = one move in every env)
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.6  # fraction of the games over which epsilon decays linearly
    target_every: int = 500  # gradient steps


@dataclass(frozen=True)
class EvalConfig:
    n_positions: int = 1000  # per test set
    min_ply: int = 4
    max_ply: int = 34
    expert_eps: float = 0.15  # chance of a random move while generating the "expert" set
    games_per_pair: int = 60  # tournament games per pair of players (half with each colour)
    open_plies: int = 2  # random opening plies of every tournament game (shared by both colours)
    sims_sweep: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 400)
    curve_every: int = 4  # evaluate the test sets every N iterations / N*games_per_iter games
    bootstrap: int = 400


@dataclass(frozen=True)
class Config:
    profile: str
    out_dir: Path
    cache_dir: Path
    seed: int = 0
    net: NetConfig = field(default_factory=NetConfig)
    az: AZConfig = field(default_factory=AZConfig)
    az_lite: AZConfig = field(default_factory=lambda: AZConfig(sims=50))  # ablation: less search during self-play
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ev: EvalConfig = field(default_factory=EvalConfig)
    minimax_depths: tuple[int, ...] = (2, 4)
    mcts_rollout_sims: int = 200
    eda_depth: int = 9  # exhaustive position counting up to this ply
    eda_random_games: int = 20_000


def make_config(profile: str, out_dir: Path, cache_dir: Path) -> Config:
    if profile == "smoke":
        return Config(
            profile,
            out_dir,
            cache_dir,
            net=NetConfig(32, 2),
            az=AZConfig(iterations=3, games_per_iter=32, sims=16, train_steps=20, batch=64, buffer=20_000),
            az_lite=AZConfig(iterations=3, games_per_iter=32, sims=8, train_steps=20, batch=64, buffer=20_000),
            dqn=DQNConfig(games=300, envs=16, batch=64, buffer=20_000, target_every=50),
            ev=EvalConfig(
                n_positions=60, games_per_pair=6, sims_sweep=(0, 10), curve_every=1, bootstrap=50, max_ply=30
            ),
            minimax_depths=(2,),
            mcts_rollout_sims=20,
            eda_depth=6,
            eda_random_games=1_000,
        )
    if profile == "standard":
        return Config(profile, out_dir, cache_dir)
    if profile == "full":
        return Config(
            profile,
            out_dir,
            cache_dir,
            az=AZConfig(iterations=160, sims=400),
            az_lite=AZConfig(iterations=160, sims=100),
            dqn=DQNConfig(games=122_880),
            ev=EvalConfig(n_positions=1000, games_per_pair=100, curve_every=8),
            minimax_depths=(2, 4, 6),
        )
    raise ValueError(f"unknown profile {profile!r}")
