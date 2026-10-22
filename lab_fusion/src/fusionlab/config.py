"""Experiment configuration: one frozen dataclass, four named profiles (smoke, standard, tuned, full)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Four-class IEMOCAP protocol used in most papers: "excited" is merged into "happy".
CLASSES: tuple[str, ...] = ("neutral", "happy", "sad", "angry")
RAW_TO_CLASS: dict[str, int] = {"neu": 0, "hap": 1, "exc": 1, "sad": 2, "ang": 3}

# Fusion methods compared in the lab (names are used as keys everywhere).
MODEL_NAMES: tuple[str, ...] = ("audio", "text", "concat", "cross", "bicross")
MODEL_TITLES: dict[str, str] = {
    "audio": "Только аудио (WavLM)",
    "text": "Только текст (RoBERTa)",
    "concat": "Конкатенация (reference)",
    "cross": "Cross-attention: Q=текст, K/V=аудио",
    "bicross": "Двунаправленный cross-attention",
}


@dataclass(frozen=True)
class Config:
    # data -------------------------------------------------------------
    dataset_repo: str = "Ar4ikov/iemocap_audio_text"
    limit: int | None = None  # use only the first N utterances (smoke profile)
    # pretrained encoders (frozen) --------------------------------------
    audio_model: str = "microsoft/wavlm-base-plus"
    text_model: str = "roberta-base"
    audio_layers: tuple[int, int] = (4, 9)  # mean of hidden states [lo, hi)
    text_last_layers: int = 4  # mean of the last K hidden states
    audio_pool: int = 4  # 50 Hz frames -> 12.5 Hz
    max_audio_seconds: float = 12.0
    max_text_tokens: int = 64
    # trainable heads ---------------------------------------------------
    d_model: int = 192
    n_heads: int = 4
    n_cross_layers: int = 2
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-2
    patience: int = 8
    # protocol -----------------------------------------------------------
    n_folds: int = 5  # leave-one-session-out (sessions 1..5)
    seeds: tuple[int, ...] = (0, 1, 2)
    n_bootstrap: int = 2000
    # hyperparameter search (0 = off): trial 0 is always the default config, the rest are random draws
    tune_trials: int = 0
    tune_seed: int = 100  # training seed used only for the search (the final runs use `seeds`)
    # paths ---------------------------------------------------------------
    cache_dir: Path = Path("cache")
    out_dir: Path = Path("outputs")

    @property
    def n_classes(self) -> int:
        return len(CLASSES)


def make_config(profile: str, out_dir: Path | None = None, cache_dir: Path | None = None) -> Config:
    """Profiles: smoke (minutes, tiny subset), standard (the lab), tuned (+ hyperparameter search), full (5 seeds)."""
    base = Config()
    if profile == "smoke":
        cfg = replace(base, limit=600, epochs=4, n_folds=2, seeds=(0,), n_bootstrap=200, tune_trials=3)
    elif profile == "standard":
        cfg = base
    elif profile == "tuned":
        cfg = replace(base, tune_trials=8)
    elif profile == "full":
        cfg = replace(base, seeds=(0, 1, 2, 3, 4))
    else:
        raise ValueError(f"unknown profile {profile!r}")
    if out_dir is not None:
        cfg = replace(cfg, out_dir=out_dir)
    if cache_dir is not None:
        cfg = replace(cfg, cache_dir=cache_dir)
    return cfg


# Hyperparameters searched per (model, outer fold); everything else (epochs, batch size, ...) stays fixed.
SEARCH_SPACE: dict[str, tuple[float, ...]] = {
    "lr": (2e-4, 5e-4, 1e-3),
    "d_model": (128, 192, 256),  # all divisible by n_heads = 4
    "dropout": (0.1, 0.2, 0.3, 0.4),
    "weight_decay": (1e-3, 1e-2, 1e-1),
    "n_cross_layers": (1, 2, 3),
}
CROSS_KINDS: frozenset[str] = frozenset({"cross", "bicross"})  # the only kinds with cross-attention layers


def sample_trials(base: Config, kind: str, n_trials: int, seed: int = 2024) -> list[Config]:
    """``n_trials`` distinct configs for ``kind``: the default first, then random draws from ``SEARCH_SPACE``.

    ``n_cross_layers`` only exists in cross-attention models, so it is not varied for the other kinds
    (otherwise they would be trained several times with identical settings).
    """
    rng = np.random.default_rng(seed)
    space = {k: v for k, v in SEARCH_SPACE.items() if k != "n_cross_layers" or kind in CROSS_KINDS}
    trials = [base]
    seen = {tuple(getattr(base, k) for k in space)}
    for _ in range(50 * n_trials):  # the space is small; give up gracefully if it is exhausted
        if len(trials) >= n_trials:
            break
        draw = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(draw.values())
        if key in seen:
            continue
        seen.add(key)
        trials.append(replace(base, **{k: type(getattr(base, k))(v) for k, v in draw.items()}))
    return trials
