"""Minimal PI-STDNet inference example."""

from __future__ import annotations

from pathlib import Path

import torch

from pistdnet.data import load_tornet_file
from pistdnet.models import PI_STDNet
from pistdnet.utils.checkpoint import load_checkpoint


def main() -> None:
    sample_path = Path("/path/to/TorNet/tornet_YYYY/sample.nc")
    checkpoint_path = Path("/path/to/checkpoint.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = load_tornet_file(sample_path).unsqueeze(0).to(device)
    model = PI_STDNet().to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    with torch.no_grad():
        cls_logit, aux_logit, _, _ = model(x)
        prob = 0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)
    print(f"tornado_probability={prob.item():.6f}")


if __name__ == "__main__":
    main()

