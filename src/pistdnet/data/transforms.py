"""Lightweight augmentations for TorNet tensors."""

from __future__ import annotations

import random

import numpy as np


class V8Augmentation:
    """Apply the augmentation policy used by the original V8 training script."""

    def __init__(self, mode: str = "train", n_sweeps: int = 2, n_frames: int = 4, channels_per_frame: int = 11):
        self.mode = mode
        self.n_sweeps = n_sweeps
        self.n_frames = n_frames
        self.channels_per_frame = channels_per_frame

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.mode != "train":
            return data
        _, height, width = data.shape
        if random.random() > 0.5:
            data = np.flip(data, axis=2).copy()
        if random.random() > 0.5:
            data = np.flip(data, axis=1).copy()

        safe_channels = [0, 2, 3, 4, 5]
        if random.random() > 0.5:
            for sweep in range(self.n_sweeps):
                for frame in range(self.n_frames):
                    base = sweep * (self.n_frames * self.channels_per_frame) + frame * self.channels_per_frame
                    for channel_offset in safe_channels:
                        channel = base + channel_offset
                        data[channel] = data[channel] * random.uniform(0.95, 1.05) + random.uniform(-0.02, 0.02)
        if random.random() > 0.5:
            erase_h, erase_w = random.randint(10, 25), random.randint(20, 50)
            erase_y = random.randint(0, max(1, height - erase_h))
            erase_x = random.randint(0, max(1, width - erase_w))
            data[:, erase_y : erase_y + erase_h, erase_x : erase_x + erase_w] = 0.0
        return data

