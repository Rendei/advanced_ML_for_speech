"""Task losses, target standardisation and metrics.

Classification tasks (emotion, gender) are scored with accuracy-type metrics, regression tasks (valence, arousal,
dominance) with the concordance correlation coefficient, the standard metric for dimensional emotion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import VAD_TASKS, is_regression, primary_metric
from .tensors import Batch

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Standardizer:
    """Per-dimension z-scoring of the (valence, arousal, dominance) targets, fitted on the training utterances only.

    Regression heads predict z-scores (all three dimensions then have the same loss scale); predictions are mapped
    back to the original 1..5 scale before any metric is computed.
    """

    mean: Tensor  # [3]
    std: Tensor  # [3]

    @staticmethod
    def fit(vad: Tensor) -> Standardizer:
        return Standardizer(vad.mean(dim=0), vad.std(dim=0).clamp_min(1e-6))

    def encode(self, vad: Tensor) -> Tensor:
        return (vad - self.mean) / self.std

    def decode(self, z: Tensor) -> Tensor:
        return z * self.std + self.mean


def task_losses(
    outputs: dict[str, Tensor], batch: Batch, std: Standardizer, emotion_weight: Tensor | None
) -> dict[str, Tensor]:
    """One scalar loss per task: cross-entropy (class-weighted, label-smoothed for emotion) or MSE on z-scores."""
    losses: dict[str, Tensor] = {}
    z = std.encode(batch.vad)  # [B, 3]
    for task, out in outputs.items():
        if task == "emotion":
            losses[task] = F.cross_entropy(out, batch.emotion, weight=emotion_weight, label_smoothing=0.05)
        elif task == "gender":
            losses[task] = F.cross_entropy(out, batch.gender)
        else:
            losses[task] = F.mse_loss(out[:, 0], z[:, VAD_TASKS.index(task)])
    return losses


def decode_outputs(outputs: dict[str, Tensor], std: Standardizer) -> dict[str, Tensor]:
    """Logits -> class probabilities [B, C]; regression outputs -> predictions on the original scale [B]."""
    decoded: dict[str, Tensor] = {}
    for task, out in outputs.items():
        if is_regression(task):
            i = VAD_TASKS.index(task)
            decoded[task] = out[:, 0] * std.std[i] + std.mean[i]
        else:
            decoded[task] = torch.softmax(out, dim=-1)
    return decoded


# ---- metrics ----------
def confusion(y: I64, pred: I64, n_classes: int) -> I64:
    return np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def class_scores(y: I64, pred: I64, n_classes: int) -> dict[str, float]:
    """WA = accuracy, UA = unweighted accuracy (macro recall), F1 = macro F1; absent classes are ignored."""
    cm = confusion(y, pred, n_classes).astype(np.float64)
    tp = np.diag(cm)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = cm.sum(axis=1) > 0
    return {"wa": float(tp.sum() / cm.sum()), "ua": float(recall[present].mean()), "f1": float(f1[present].mean())}


def ccc(y: F64, p: F64) -> float:
    """Lin's concordance correlation coefficient: 2 cov / (var_y + var_p + (mean_y - mean_p)^2)."""
    my, mp = y.mean(), p.mean()
    cov = ((y - my) * (p - mp)).mean()
    return float(2 * cov / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def regression_scores(y: F64, p: F64) -> dict[str, float]:
    pearson = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else 0.0
    return {"ccc": ccc(y, p), "pearson": pearson, "mae": float(np.abs(y - p).mean())}


def task_scores(task: str, y: F64 | I64, pred: F64) -> dict[str, float]:
    """All metrics of one task; ``pred`` is [n, C] probabilities for classification, [n] values for regression."""
    if is_regression(task):
        return regression_scores(np.asarray(y, dtype=np.float64), pred)
    return class_scores(np.asarray(y, dtype=np.int64), pred.argmax(axis=1), pred.shape[1])


def metric_value(task: str, y: F64 | I64, pred: F64, metric: str | None = None) -> float:
    """One number of one task (its primary metric unless ``metric`` is given)."""
    return task_scores(task, y, pred)[metric or primary_metric(task)]
