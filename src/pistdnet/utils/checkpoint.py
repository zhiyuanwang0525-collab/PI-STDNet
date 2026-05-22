"""Checkpoint load/save helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, **metadata: Any) -> None:
    """Save a model state dict with optional metadata."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), **metadata}
    torch.save(payload, ckpt_path)


def load_checkpoint(path: str | Path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint into a model and return the checkpoint payload."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=map_location)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return payload if isinstance(payload, dict) else {"model_state_dict": state_dict}

