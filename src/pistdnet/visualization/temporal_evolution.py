"""Temporal evolution visualization placeholder."""

from __future__ import annotations

from pathlib import Path


def visualize_temporal_case(config: dict, case_id: str, checkpoint: str | None = None) -> None:
    """Validate inputs and explain where the full temporal figure workflow lives."""
    data_root = Path(config.get("data", {}).get("dataset_root", "/path/to/TorNet"))
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if checkpoint is not None and not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Temporal visualization requested for case_id={case_id}.")
    print("The publication figure scripts are preserved under tools/legacy/visualization_scripts/.")
    print("TODO: Port the full Figure 11 rendering workflow into this module after public data paths are finalized.")

