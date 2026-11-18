"""Registry of the trained networks that can be played against (shipped in ``weights/``, plus any fresh checkpoints)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .config import NetConfig
from .net import PVNet, QNet, load_pv, load_q

Kind = Literal["pv", "q"]
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    kind: Kind  # pv = policy + value head (AlphaZero-lite), q = Q head (DQN)
    path: Path
    net: NetConfig
    elo: float | None = None  # rating of the raw network in the report's tournament (random = 0)
    elo_search: float | None = None  # rating with 100 MCTS simulations, when it was measured
    group: str = "Нейросети"


def infer_architecture(path: Path) -> tuple[Kind, NetConfig]:
    """Read the architecture off the weights themselves, so any checkpoint of any profile can be loaded."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    kind: Kind = "pv" if any(k.startswith("policy.") for k in state) else "q"
    channels = int(state["trunk.stem.0.weight"].shape[0])
    blocks = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("trunk.blocks."))
    return kind, NetConfig(channels, blocks)


def discover(weights_dir: Path, checkpoints_dir: Path | None = None) -> dict[str, ModelInfo]:
    """Models listed in ``weights/manifest.json`` first, then any other ``*.pt`` found (e.g. outputs/checkpoints)."""
    found: dict[str, ModelInfo] = {}
    manifest = weights_dir / MANIFEST
    if manifest.is_file():
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            path = weights_dir / entry["file"]
            if path.is_file():
                kind, net = infer_architecture(path)
                found[entry["id"]] = ModelInfo(
                    entry["id"], entry["label"], kind, path, net, entry.get("elo"), entry.get("elo_search")
                )
    for directory in (weights_dir, checkpoints_dir):
        if directory is None or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pt")):
            if path.stem not in found:
                kind, net = infer_architecture(path)
                found[path.stem] = ModelInfo(path.stem, path.stem, kind, path, net, group="Прочие чекпоинты")
    return found


def load_model(info: ModelInfo, device: torch.device) -> PVNet | QNet:
    return load_pv(info.path, info.net, device) if info.kind == "pv" else load_q(info.path, info.net, device)


# model id -> (checkpoint stem, label, tournament name of the raw net, tournament name with search)
SHIPPED: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("az_init", "az_init", "AlphaZero-lite: необученная (0%)", "az-init", None),
    ("az_p25", "az_p25", "AlphaZero-lite: 25% обучения", "az-25%", None),
    ("az_p50", "az_p50", "AlphaZero-lite: 50% обучения", "az-50%", None),
    ("az_final", "az_final", "AlphaZero-lite: финальная", "az-raw", "az+mcts100"),
    ("az_lite_final", "az_lite_final", "AlphaZero-lite: слабый поиск в self-play", "azlite-raw", "azlite+mcts100"),
    ("dqn_final", "dqn_final", "DQN: финальная", "dqn", None),
)


def export_weights(checkpoints_dir: Path, weights_dir: Path, elo: dict[str, float]) -> list[Path]:
    """Copy the chosen checkpoints into the versioned ``weights/`` folder and write the manifest with their Elo."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    copied: list[Path] = []
    for model_id, stem, label, raw_name, search_name in SHIPPED:
        src = checkpoints_dir / f"{stem}.pt"
        if not src.is_file():
            continue
        shutil.copyfile(src, weights_dir / src.name)
        copied.append(weights_dir / src.name)
        entries.append(
            {
                "id": model_id,
                "file": src.name,
                "label": label,
                "elo": round(elo[raw_name]) if raw_name in elo else None,
                "elo_search": round(elo[search_name]) if search_name and search_name in elo else None,
            }
        )
    (weights_dir / MANIFEST).write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return copied
