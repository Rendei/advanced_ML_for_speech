"""Networks: a small residual CNN trunk with a policy/value head (AlphaZero-lite) or a Q head (DQN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .config import NetConfig
from .game import CELLS, F32, HEIGHT, I64, WIDTH, planes

IN_CHANNELS = 3


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(y)))


class _Trunk(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, cfg.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[_ResBlock(cfg.channels) for _ in range(cfg.blocks)])

    def forward(self, x: Tensor) -> Tensor:  # (B, 3, 6, 7) -> (B, C, 6, 7)
        out: Tensor = self.blocks(self.stem(x))
        return out


class PVNet(nn.Module):
    """``(B, 3, 6, 7) -> (policy logits (B, 7), value (B,) in [-1, 1])``; the value is for the player to move."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = _Trunk(cfg)
        self.policy = nn.Sequential(
            nn.Conv2d(cfg.channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * CELLS, WIDTH),
        )
        self.value = nn.Sequential(
            nn.Conv2d(cfg.channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * CELLS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)


class QNet(nn.Module):
    """``(B, 3, 6, 7) -> Q-values of the 7 columns (B, 7)`` for the player to move."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = _Trunk(cfg)
        self.head = nn.Sequential(
            nn.Conv2d(cfg.channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * CELLS, 128),
            nn.ReLU(),
            nn.Linear(128, WIDTH),
        )

    def forward(self, x: Tensor) -> Tensor:
        q: Tensor = self.head(self.trunk(x))
        return q


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_tensor(cur: I64, mask: I64, device: torch.device) -> Tensor:
    return torch.from_numpy(planes(cur, mask)).to(device)


class NetEvaluator:
    """Adapter that lets a ``PVNet`` act as the evaluator of ``BatchMCTS`` (priors = softmax of the logits)."""

    def __init__(self, model: PVNet, device: torch.device, chunk: int = 4096) -> None:
        self.model = model
        self.device = device
        self.chunk = chunk

    def __call__(self, cur: I64, mask: I64) -> tuple[F32, F32]:
        self.model.eval()
        probs: list[F32] = []
        values: list[F32] = []
        with torch.inference_mode():
            for i in range(0, len(cur), self.chunk):
                x = to_tensor(cur[i : i + self.chunk], mask[i : i + self.chunk], self.device)
                logits, v = self.model(x)
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                values.append(v.cpu().numpy())
        return np.concatenate(probs), np.concatenate(values)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def load_pv(path: Path, cfg: NetConfig, device: torch.device) -> PVNet:
    net = PVNet(cfg)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval()


def load_q(path: Path, cfg: NetConfig, device: torch.device) -> QNet:
    net = QNet(cfg)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval()


__all__ = [
    "HEIGHT",
    "NetEvaluator",
    "PVNet",
    "QNet",
    "load_pv",
    "load_q",
    "n_parameters",
    "save_checkpoint",
    "to_tensor",
]
