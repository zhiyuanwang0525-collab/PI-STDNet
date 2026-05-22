"""Evaluation loop for PI-STDNet."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pistdnet.metrics import evaluate_view


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect probabilities, labels, and category ids."""
    model.eval()
    all_probs, all_labels, all_cats = [], [], []
    amp_enabled = use_amp and device.type == "cuda"
    for images, labels, cats in tqdm(loader, desc="Evaluating"):
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            cls_logit, aux_logit, _, _ = model(images.to(device, non_blocking=True))
        probs = 0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_cats.extend(cats.numpy())
    return np.asarray(all_probs), np.asarray(all_labels), np.asarray(all_cats)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float | None = None, use_amp: bool = True) -> dict[str, dict]:
    """Evaluate NC and WC views."""
    probs, labels, cats = collect_predictions(model, loader, device, use_amp)
    return {
        "NC": evaluate_view(probs, labels, threshold=threshold),
        "WC": evaluate_view(probs, labels, mask=(cats == 2) | (cats == 1), threshold=threshold),
    }

