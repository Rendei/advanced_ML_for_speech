"""Padded batches of frozen features and the leave-one-session-out protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from .features import Features


@dataclass(frozen=True)
class Batch:
    """audio [B, Ta, D], audio_mask [B, Ta] (True = real frame), text [B, Tt, D], text_mask [B, Tt], y [B]."""

    audio: Tensor
    audio_mask: Tensor
    text: Tensor
    text_mask: Tensor
    y: Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(*(t.to(device) for t in (self.audio, self.audio_mask, self.text, self.text_mask, self.y)))

    def __len__(self) -> int:
        return int(self.y.shape[0])


def _pad(flat: npt.NDArray[np.float16], lens: npt.NDArray[np.int64]) -> tuple[Tensor, Tensor]:
    n, d, t = len(lens), flat.shape[1], int(lens.max())
    out = np.zeros((n, t, d), dtype=np.float16)
    start = np.concatenate([[0], np.cumsum(lens)])
    for i in range(n):
        out[i, : lens[i]] = flat[start[i] : start[i + 1]]
    mask = np.arange(t)[None, :] < lens[:, None]
    return torch.from_numpy(out), torch.from_numpy(mask)


class Dataset:
    """The whole corpus as padded fp16 tensors; ``batch(idx)`` gathers any subset of utterances as a ``Batch``."""

    def __init__(self, feats: Features) -> None:
        self.audio, self.audio_mask = _pad(feats.audio_flat, feats.audio_len)
        self.text, self.text_mask = _pad(feats.text_flat, feats.text_len)
        self.y = torch.from_numpy(feats.meta.label)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def to(self, device: torch.device) -> Dataset:
        """Move all tensors (fp16, ≈1.5 GB for IEMOCAP) to the device so that batching is a GPU gather."""
        self.audio, self.audio_mask = self.audio.to(device), self.audio_mask.to(device)
        self.text, self.text_mask = self.text.to(device), self.text_mask.to(device)
        self.y = self.y.to(device)
        return self

    def batch(self, idx: npt.NDArray[np.int64] | Tensor) -> Batch:
        i = torch.as_tensor(idx, dtype=torch.long, device=self.y.device)
        return Batch(self.audio[i].float(), self.audio_mask[i], self.text[i].float(), self.text_mask[i], self.y[i])


@dataclass(frozen=True)
class Split:
    fold: int
    train: npt.NDArray[np.int64]
    val: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]


def session_splits(session: npt.NDArray[np.int64], n_folds: int) -> list[Split]:
    """Leave-one-session-out: test = session k, validation = the next session (cyclic), train = the rest.

    Sessions are disjoint in speakers, so this is the speaker-independent protocol used in the literature.
    """
    present = sorted(int(s) for s in np.unique(session))
    splits: list[Split] = []
    for f, s_test in enumerate(present[:n_folds]):
        s_val = present[(present.index(s_test) + 1) % len(present)]
        test = np.flatnonzero(session == s_test)
        val = np.flatnonzero(session == s_val)
        train = np.flatnonzero((session != s_test) & (session != s_val))
        splits.append(Split(f, train, val, test))
    return splits
