"""Training of one regime (one set of tasks + one loss-balancing strategy) and prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from .balancing import (
    UncertaintyWeighting,
    assign_flat_grad,
    cosine_matrix,
    fixed_weights,
    pcgrad_merge,
    shared_task_grads,
    weighted_sum,
)
from .config import VAD_TASKS, Config, Regime, is_regression, primary_metric
from .models import MTLNet
from .tasks import Standardizer, decode_outputs, metric_value, task_losses
from .tensors import Batch, Dataset, Split

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class History:
    """What happened during one training run (kept for the first fold and seed only)."""

    train_loss: list[float] = field(default_factory=list)  # sum of the raw task losses, per epoch
    val_score: list[float] = field(default_factory=list)  # primary metric of the selection task, per epoch
    best_epoch: int = 0
    uncertainty_weights: list[dict[str, float]] = field(default_factory=list)  # exp(-s_t)/2 per epoch
    cosines: list[F64] = field(default_factory=list)  # [T, T] task-gradient cosines on a fixed probe batch, per epoch
    pcgrad_conflict: list[float] = field(default_factory=list)  # share of conflicting task pairs among training steps


def targets(ds: Dataset, idx: I64, task: str) -> F64 | I64:
    """Ground truth of one task for the utterances ``idx`` (class indices or values on the original scale)."""
    i = torch.as_tensor(idx, dtype=torch.long, device=ds.emotion.device)
    if task == "emotion":
        return ds.emotion[i].cpu().numpy().astype(np.int64)
    if task == "gender":
        return ds.gender[i].cpu().numpy().astype(np.int64)
    return ds.vad[i, VAD_TASKS.index(task)].cpu().numpy().astype(np.float64)


@torch.inference_mode()
def predict(
    model: MTLNet, ds: Dataset, idx: I64, device: torch.device, std: Standardizer, batch_size: int = 256
) -> dict[str, F64]:
    """{task: probabilities [n, C]} for classification, {task: values [n]} (original scale) for regression."""
    model.eval()
    chunks: dict[str, list[Tensor]] = {t: [] for t in model.tasks}
    for s in range(0, len(idx), batch_size):
        decoded = decode_outputs(model(ds.batch(idx[s : s + batch_size]).to(device)), std)
        for t, v in decoded.items():
            chunks[t].append(v.cpu())
    return {t: torch.cat(v).numpy().astype(np.float64) for t, v in chunks.items()}


def probe_cosines(model: MTLNet, batch: Batch, std: Standardizer, emotion_weight: Tensor | None) -> F64:
    """Cosine similarity between the per-task gradients on the shared trunk for one fixed batch (no update)."""
    was_training = model.training
    model.eval()  # no dropout: the probe measures the tasks, not the noise
    losses = task_losses(model(batch), batch, std, emotion_weight)
    cos = cosine_matrix(shared_task_grads(model.shared_parameters(), losses, model.tasks))
    model.train(was_training)
    return cos


def _selection_score(
    model: MTLNet, regime: Regime, ds: Dataset, split: Split, device: torch.device, std: Standardizer
) -> float:
    task = regime.selection_task
    pred = predict(model, ds, split.val, device, std)[task]
    return metric_value(task, targets(ds, split.val, task), pred, primary_metric(task))


def train_regime(
    regime: Regime,
    cfg: Config,
    ds: Dataset,
    split: Split,
    std: Standardizer,
    seed: int,
    device: torch.device,
    d_in: int,
) -> tuple[MTLNet, History]:
    """Train on ``split.train``, pick the epoch by the validation score of the selection task, restore the best."""
    set_seed(seed)
    tasks = regime.tasks
    model = MTLNet(tasks, cfg, d_in).to(device)
    shared = model.shared_parameters()
    head_params = [p for h in model.heads.values() for p in h.parameters()]
    uncertainty = UncertaintyWeighting(tasks).to(device) if regime.balancing == "uncertainty" else None
    weights = fixed_weights(tasks, regime.aux_weight)

    emotion_weight: Tensor | None = None
    if "emotion" in tasks:  # inverse-frequency class weights: emotion classes are imbalanced
        freq = np.bincount(np.asarray(targets(ds, split.train, "emotion"), dtype=np.int64), minlength=4).astype(
            np.float64
        )
        emotion_weight = torch.tensor(freq.sum() / (4 * np.maximum(freq, 1)), dtype=torch.float32, device=device)

    groups: list[dict[str, object]] = [
        {"params": list(model.parameters()), "lr": cfg.lr, "weight_decay": cfg.weight_decay}
    ]
    max_lr = [cfg.lr]
    if uncertainty is not None:
        groups.append(
            {"params": list(uncertainty.parameters()), "lr": cfg.lr * cfg.logvar_lr_mult, "weight_decay": 0.0}
        )
        max_lr.append(cfg.lr * cfg.logvar_lr_mult)
    opt = torch.optim.AdamW(groups)
    steps = cfg.epochs * int(np.ceil(len(split.train) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=0.15)

    rng = np.random.default_rng(seed)
    probe_idx = rng.choice(split.train, size=min(cfg.probe_size, len(split.train)), replace=False)
    probe = ds.batch(probe_idx).to(device)

    hist, best, best_state, bad = History(), -np.inf, copy.deepcopy(model.state_dict()), 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(split.train)
        total, conflicts, n_steps = 0.0, 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            b = ds.batch(order[s : s + cfg.batch_size]).to(device)
            losses = task_losses(model(b), b, std, emotion_weight)
            opt.zero_grad(set_to_none=True)
            if regime.balancing == "pcgrad" and regime.is_multitask:
                scaled = {t: weights[t] * losses[t] for t in tasks}
                grads = shared_task_grads(shared, scaled, tasks)  # [T, P], graph kept for the heads
                cos = cosine_matrix(grads)
                iu = np.triu_indices(len(tasks), k=1)
                conflicts += float((cos[iu] < 0).mean())
                head_loss = weighted_sum(scaled, {t: 1.0 for t in tasks})
                torch.autograd.backward(head_loss, inputs=head_params)  # heads: own task only
                assign_flat_grad(shared, pcgrad_merge(grads, rng))  # trunk: de-conflicted sum
            else:
                loss = uncertainty(losses) if uncertainty is not None else weighted_sum(losses, weights)
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(sum(v.detach() for v in losses.values())) * len(b)
            n_steps += 1
        score = _selection_score(model, regime, ds, split, device, std)
        hist.train_loss.append(total / len(order))
        hist.val_score.append(score)
        if uncertainty is not None:
            hist.uncertainty_weights.append(uncertainty.weights())
        if regime.is_multitask:
            hist.cosines.append(probe_cosines(model, probe, std, emotion_weight))
            if regime.balancing == "pcgrad":
                hist.pcgrad_conflict.append(conflicts / max(n_steps, 1))
        if score > best:
            best, bad, hist.best_epoch = score, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), hist


def trace_shapes(model: MTLNet, batch: Batch) -> list[tuple[str, str]]:
    """Output shape of the main modules during one forward pass (shown in the report)."""
    rows: list[tuple[str, str]] = [
        ("input.audio", str(tuple(batch.audio.shape))),
        ("input.text", str(tuple(batch.text.shape))),
    ]
    hooks = []
    wanted = {
        "trunk.enc_a",
        "trunk.enc_t",
        "trunk.t_from_a.0",
        "trunk.a_from_t.0",
        "trunk.pool_t",
        "trunk.pool_a",
        "trunk",
    } | {f"heads.{t}" for t in model.tasks}

    def make(name: str):  # type: ignore[no-untyped-def]
        def hook(_m: nn.Module, _i: object, out: object) -> None:
            assert isinstance(out, Tensor)
            rows.append((name, str(tuple(out.shape))))

        return hook

    for name, mod in model.named_modules():
        if name in wanted:
            hooks.append(mod.register_forward_hook(make(name)))
    with torch.inference_mode():
        model.eval()(batch)
    for h in hooks:
        h.remove()
    return rows


def is_cls_output(task: str) -> bool:
    return not is_regression(task)
