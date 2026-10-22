"""Training, prediction and metrics."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from .config import Config
from .models import Model
from .tensors import Batch, Dataset, Split

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion(y: I64, pred: I64, n_classes: int) -> I64:
    return np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def scores(y: I64, pred: I64, n_classes: int) -> dict[str, float]:
    """WA = accuracy, UA = unweighted accuracy (macro recall), F1 = macro F1."""
    cm = confusion(y, pred, n_classes).astype(np.float64)
    tp = np.diag(cm)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = cm.sum(axis=1) > 0  # classes absent from y (tiny test subsets) do not count
    return {"wa": float(tp.sum() / cm.sum()), "ua": float(recall[present].mean()), "f1": float(f1[present].mean())}


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_ua: list[float] = field(default_factory=list)
    best_epoch: int = 0


@torch.inference_mode()
def predict(
    model: Model,
    ds: Dataset,
    idx: I64,
    device: torch.device,
    shuffle: str | None = None,
    seed: int = 0,
    batch_size: int = 256,
) -> F64:
    """Class probabilities [len(idx), C].

    ``shuffle`` = "audio" | "text" replaces that modality by another utterance's (a probe that the
    model really uses the modality: a unimodal-text model would not notice shuffled audio).
    """
    model.eval()
    perm = np.random.default_rng(seed).permutation(len(idx))
    out: list[Tensor] = []
    for s in range(0, len(idx), batch_size):
        sl = slice(s, s + batch_size)
        b = ds.batch(idx[sl])
        if shuffle is not None:
            other = ds.batch(idx[perm[sl]])
            if shuffle == "audio":
                b = Batch(other.audio, other.audio_mask, b.text, b.text_mask, b.y)
            else:
                b = Batch(b.audio, b.audio_mask, other.text, other.text_mask, b.y)
        out.append(torch.softmax(model(b.to(device)), dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train_model(
    kind: str, cfg: Config, ds: Dataset, split: Split, seed: int, device: torch.device, d_in: int
) -> tuple[Model, History]:
    """Train on ``split.train``, select the epoch by validation UA, return the best weights."""
    set_seed(seed)
    model = Model(kind, cfg, d_in).to(device)
    y_train = ds.y[torch.as_tensor(split.train, device=ds.y.device)].cpu().numpy()
    freq = np.bincount(y_train, minlength=cfg.n_classes).astype(np.float64)
    weight = torch.tensor(freq.sum() / (cfg.n_classes * np.maximum(freq, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(split.train) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(seed)
    y_val = ds.y[torch.as_tensor(split.val, device=ds.y.device)].cpu().numpy()
    hist, best_ua, best_state, bad = History(), -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(split.train)
        total = 0.0
        for s in range(0, len(order), cfg.batch_size):
            b = ds.batch(order[s : s + cfg.batch_size]).to(device)
            loss = loss_fn(model(b), b.y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        ua = scores(y_val, predict(model, ds, split.val, device).argmax(1), cfg.n_classes)["ua"]
        hist.train_loss.append(total / len(order))
        hist.val_ua.append(ua)
        if ua > best_ua:
            best_ua, bad, hist.best_epoch = ua, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), hist


def trace_shapes(model: Model, batch: Batch) -> list[tuple[str, str]]:
    """Output shape of every leaf-ish module during one forward pass (for the report)."""
    rows: list[tuple[str, str]] = [
        ("input.audio", str(tuple(batch.audio.shape))),
        ("input.text", str(tuple(batch.text.shape))),
    ]
    hooks = []
    wanted = ("enc_a", "enc_t", "t_from_a.0", "t_from_a.1", "a_from_t.0", "a_from_t.1", "pool_a", "pool_t", "head")

    def make(name: str) -> Callable[[nn.Module, object, object], None]:
        def hook(_m: nn.Module, _i: object, out: object) -> None:
            o = out[0] if isinstance(out, tuple) else out
            assert isinstance(o, Tensor)
            rows.append((name, str(tuple(o.shape))))

        return hook

    for name, mod in model.named_modules():
        if name in wanted:
            hooks.append(mod.register_forward_hook(make(name)))
    with torch.inference_mode():
        model.eval()(batch)
    for h in hooks:
        h.remove()
    return rows
