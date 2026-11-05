"""How the per-task losses are combined into one update of the shared trunk.

* ``equal``       L = sum_t w_t L_t with fixed weights (w_emotion = 1, other tasks ``aux_weight``)
* ``uncertainty`` L = sum_t ( exp(-s_t) / 2 * L_t + s_t / 2 ), s_t = log sigma_t^2 is learned   (Kendall et al., 2018)
                  a noisy task (large sigma) is down-weighted; the +s_t term stops sigma from growing forever
* ``pcgrad``      per-task gradients g_t of the shared trunk are made non-conflicting before they are summed
                  (Yu et al., 2020): if <g_i, g_j> < 0, g_i <- g_i - <g_i, g_j> / |g_j|^2 * g_j
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

F64 = npt.NDArray[np.float64]


class UncertaintyWeighting(nn.Module):
    """Learned homoscedastic-uncertainty weights, one log-variance ``s_t`` per task."""

    def __init__(self, tasks: tuple[str, ...]) -> None:
        super().__init__()
        self.tasks = tasks
        self.log_var = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        total = torch.zeros((), device=self.log_var.device)
        for i, t in enumerate(self.tasks):
            total = total + 0.5 * torch.exp(-self.log_var[i]) * losses[t] + 0.5 * self.log_var[i]
        return total

    def weights(self) -> dict[str, float]:
        """Effective loss weight of every task, exp(-s_t) / 2."""
        w = (0.5 * torch.exp(-self.log_var)).detach().cpu().tolist()
        return {t: float(w[i]) for i, t in enumerate(self.tasks)}


def fixed_weights(tasks: tuple[str, ...], aux_weight: float) -> dict[str, float]:
    """Emotion has weight 1; every other task ``aux_weight`` (irrelevant for a single-task model)."""
    return {t: 1.0 if t == "emotion" or len(tasks) == 1 else aux_weight for t in tasks}


def weighted_sum(losses: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
    total = torch.zeros((), device=next(iter(losses.values())).device)
    for t, loss in losses.items():
        total = total + weights[t] * loss
    return total


def shared_task_grads(shared: list[nn.Parameter], losses: dict[str, Tensor], tasks: tuple[str, ...]) -> Tensor:
    """Gradient of every task loss w.r.t. the shared parameters, flattened: [T, P]. The graph is kept."""
    rows: list[Tensor] = []
    for t in tasks:
        grads = torch.autograd.grad(losses[t], shared, retain_graph=True, allow_unused=True)
        rows.append(
            torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, shared)])
        )
    return torch.stack(rows)


def cosine_matrix(grads: Tensor) -> F64:
    """Pairwise cosine similarity of the task gradients, [T, T]; negative = the tasks pull the trunk apart."""
    unit = grads / grads.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos: F64 = (unit @ unit.T).detach().cpu().numpy().astype(np.float64)
    return cos


def pcgrad_merge(grads: Tensor, rng: np.random.Generator) -> Tensor:
    """Project every task gradient away from the gradients it conflicts with, then sum: [T, P] -> [P]."""
    n = grads.shape[0]
    merged = grads.clone()
    for i in range(n):
        for j in rng.permutation([k for k in range(n) if k != i]):
            dot = torch.dot(merged[i], grads[j])
            if dot < 0:
                merged[i] = merged[i] - dot / grads[j].dot(grads[j]).clamp_min(1e-12) * grads[j]
    return merged.sum(dim=0)


def assign_flat_grad(params: list[nn.Parameter], flat: Tensor) -> None:
    """Write a flat gradient vector back into ``p.grad`` of every parameter."""
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n
