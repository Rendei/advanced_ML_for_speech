"""Tasks, training regimes and the experiment configuration (one frozen dataclass, three profiles)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# ---- tasks ----------
# emotion  : 4-way classification (the main task of the lab)
# valence / arousal / dominance : regression on the 1..5 ratings given to every utterance (related auxiliary tasks)
# gender   : speaker gender, binary classification (an auxiliary task that is *unrelated* to emotion)
TASKS: tuple[str, ...] = ("emotion", "valence", "arousal", "dominance", "gender")
VAD_TASKS: tuple[str, ...] = ("valence", "arousal", "dominance")
EMOTION_CLASSES: tuple[str, ...] = ("neutral", "happy", "sad", "angry")
GENDER_CLASSES: tuple[str, ...] = ("female", "male")
RAW_TO_EMOTION: dict[str, int] = {"neu": 0, "hap": 1, "exc": 1, "sad": 2, "ang": 3}  # "excited" is merged into "happy"
CLASS_NAMES: dict[str, tuple[str, ...]] = {"emotion": EMOTION_CLASSES, "gender": GENDER_CLASSES}
TASK_TITLES: dict[str, str] = {
    "emotion": "эмоция",
    "valence": "valence (знак эмоции)",
    "arousal": "arousal (активация)",
    "dominance": "dominance (доминантность)",
    "gender": "пол диктора",
}


def is_regression(task: str) -> bool:
    return task in VAD_TASKS


def task_dim(task: str) -> int:
    """Number of outputs of a task head: 1 for regression, number of classes for classification."""
    return 1 if is_regression(task) else len(CLASS_NAMES[task])


def primary_metric(task: str) -> str:
    """The one number that represents a task: UA (balanced accuracy) for classification, CCC for regression."""
    return "ccc" if is_regression(task) else "ua"


# ---- training regimes ----------
@dataclass(frozen=True)
class Regime:
    """One way of training the shared model.

    ``tasks``      which tasks the model is trained on (one task = the single-task baseline)
    ``balancing``  how the losses are combined: ``equal`` (fixed weights), ``uncertainty`` (learned weights,
                   Kendall et al. 2018) or ``pcgrad`` (gradient surgery on the shared part, Yu et al. 2020)
    ``aux_weight`` weight of every non-emotion loss for ``equal`` / ``pcgrad``; the emotion loss has weight 1
    """

    name: str
    title: str
    tasks: tuple[str, ...]
    balancing: str = "equal"
    aux_weight: float = 1.0

    @property
    def is_multitask(self) -> bool:
        return len(self.tasks) > 1

    @property
    def selection_task(self) -> str:
        """The task whose validation score picks the best epoch: emotion whenever it is trained, else the only task."""
        return "emotion" if "emotion" in self.tasks else self.tasks[0]


REGIMES: tuple[Regime, ...] = (
    # single-task baselines: one model per task (the point of comparison for every MTL regime)
    Regime("st_emotion", "Single-task: эмоция", ("emotion",)),
    Regime("st_valence", "Single-task: valence", ("valence",)),
    Regime("st_arousal", "Single-task: arousal", ("arousal",)),
    Regime("st_dominance", "Single-task: dominance", ("dominance",)),
    Regime("st_gender", "Single-task: пол", ("gender",)),
    # which auxiliary tasks help the emotion task?
    Regime("mtl_vad", "MTL: эмоция + VAD", ("emotion", *VAD_TASKS)),
    Regime("mtl_gender", "MTL: эмоция + пол", ("emotion", "gender")),
    # all five tasks, how much weight should the auxiliary losses get?
    Regime("mtl_w0.1", "MTL (все задачи), вес aux = 0.1", TASKS, aux_weight=0.1),
    Regime("mtl_w0.3", "MTL (все задачи), вес aux = 0.3", TASKS, aux_weight=0.3),
    Regime("mtl_equal", "MTL (все задачи), равные веса", TASKS),
    # all five tasks, smarter balancing
    Regime("mtl_uncertainty", "MTL (все задачи), uncertainty weighting", TASKS, balancing="uncertainty"),
    Regime("mtl_pcgrad", "MTL (все задачи), PCGrad", TASKS, balancing="pcgrad"),
)
REGIME_BY_NAME: dict[str, Regime] = {r.name: r for r in REGIMES}
BASELINE = "st_emotion"  # the single-task emotion model every emotion MTL regime is compared with
WEIGHT_SWEEP: tuple[str, ...] = ("mtl_w0.1", "mtl_w0.3", "mtl_equal")


# ---- experiment configuration ----------
@dataclass(frozen=True)
class Config:
    # data ----------
    dataset_repo: str = "Ar4ikov/iemocap_audio_text"
    limit: int | None = None  # use only N utterances (smoke profile)
    # frozen pretrained encoders (identical to lab 1, so the extracted features are reusable) ----------
    audio_model: str = "microsoft/wavlm-base-plus"
    text_model: str = "roberta-base"
    audio_layers: tuple[int, int] = (4, 9)  # mean of hidden states [lo, hi)
    text_last_layers: int = 4
    audio_pool: int = 4  # 50 Hz frames -> 12.5 Hz
    max_audio_seconds: float = 12.0
    max_text_tokens: int = 64
    # shared trunk + heads ----------
    trunk: str = "bicross"  # concat | cross | bicross  (the fusion block of lab 1)
    d_model: int = 192
    n_heads: int = 4
    n_cross_layers: int = 2
    head_hidden: int = 64  # every task head is a small MLP on top of the shared vector
    dropout: float = 0.2
    # optimisation ----------
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-2
    logvar_lr_mult: float = 10.0  # uncertainty weighting: the learned log-variances move faster than the network
    patience: int = 8
    probe_size: int = 256  # utterances of the fixed batch used to measure gradient conflicts every epoch
    # protocol ----------
    n_folds: int = 5  # leave-one-session-out (sessions 1..5)
    seeds: tuple[int, ...] = (0, 1, 2)
    n_bootstrap: int = 2000
    regimes: tuple[str, ...] = tuple(r.name for r in REGIMES)
    # paths ----------
    cache_dir: Path = Path("cache")
    out_dir: Path = Path("outputs")


def make_config(profile: str, out_dir: Path | None = None, cache_dir: Path | None = None) -> Config:
    """Profiles: smoke (minutes, tiny subset), standard (the lab), full (5 seeds)."""
    base = Config()
    if profile == "smoke":
        cfg = replace(base, limit=600, epochs=4, n_folds=2, seeds=(0,), n_bootstrap=200, probe_size=64)
    elif profile == "standard":
        cfg = base
    elif profile == "full":
        cfg = replace(base, seeds=(0, 1, 2, 3, 4))
    else:
        raise ValueError(f"unknown profile {profile!r}")
    if out_dir is not None:
        cfg = replace(cfg, out_dir=out_dir)
    if cache_dir is not None:
        cfg = replace(cfg, cache_dir=cache_dir)
    return cfg
