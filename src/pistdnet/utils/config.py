"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    inherited = config.get("inherits")
    if inherited:
        base_config = load_config(config_path.parent / inherited)
        config = _deep_merge(base_config, {k: v for k, v in config.items() if k != "inherits"})
    return config


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the model section with common top-level compatibility keys."""
    model_cfg = dict(config.get("model", {}))
    for key in ["input_shape", "n_frames", "channels_per_frame", "n_sweeps"]:
        if key in config and key not in model_cfg:
            model_cfg[key] = config[key]
    return model_cfg


def resolve_catalog_path(config: dict[str, Any]) -> Path:
    """Resolve catalog_path, defaulting to dataset_root/catalog.csv."""
    data_cfg = config.get("data", {})
    if data_cfg.get("catalog_path"):
        return Path(data_cfg["catalog_path"])
    return Path(data_cfg.get("dataset_root", "/path/to/TorNet")) / "catalog.csv"
