"""TorNet dataset wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import load_tornet_file
from .transforms import V8Augmentation


class TorNetDataset(Dataset):
    """Load TorNet catalog rows and return tensors, binary labels, and category ids."""

    def __init__(
        self,
        catalog_path: str | Path,
        root_dir: str | Path,
        mode: str = "train",
        n_sweeps: int = 2,
        n_frames: int = 4,
        channels_per_frame: int = 11,
    ):
        self.catalog_path = Path(catalog_path)
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.n_sweeps = n_sweeps
        self.n_frames = n_frames
        self.channels_per_frame = channels_per_frame

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog file not found: {self.catalog_path}. See docs/data_preparation.md for setup instructions."
            )
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root_dir}. Set dataset_root in the config; see docs/data_preparation.md."
            )

        df = pd.read_csv(self.catalog_path)
        if "type" not in df or "category" not in df or "filename" not in df:
            raise ValueError("Catalog must contain at least type, category, and filename columns.")

        self.catalog = df[df["type"] == mode].reset_index(drop=True)
        if len(self.catalog) == 0:
            raise ValueError(f"No rows with type == {mode!r} found in {self.catalog_path}.")

        self.labels = (self.catalog["category"] == "TOR").astype(int).values
        self.cat_ids = np.zeros(len(self.catalog), dtype=int)
        self.cat_ids[self.catalog["category"] == "TOR"] = 2
        wrn_mask = (self.catalog["category"] == "WRN") | (self.catalog["category"] == "Tornado Warning")
        self.cat_ids[wrn_mask] = 1
        self.augmentor = V8Augmentation(mode, n_sweeps, n_frames, channels_per_frame)

    def __len__(self) -> int:
        return len(self.catalog)

    def _resolve_sample_path(self, row: pd.Series) -> Path:
        year = pd.to_datetime(row["start_time"]).year if "start_time" in row and pd.notna(row["start_time"]) else None
        candidates = []
        if year is not None:
            candidates.append(self.root_dir / f"tornet_{year}" / row["filename"])
        candidates.append(self.root_dir / row["filename"])
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find sample {row['filename']} under {self.root_dir}. See docs/data_preparation.md."
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.catalog.iloc[idx]
        data = load_tornet_file(self._resolve_sample_path(row), self.n_sweeps, self.n_frames, self.channels_per_frame).numpy()
        data = self.augmentor(data.copy())
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cat_id = torch.tensor(self.cat_ids[idx], dtype=torch.int8)
        return torch.from_numpy(data), label, cat_id

