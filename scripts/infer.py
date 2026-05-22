"""Minimal single-sample inference entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PI-STDNet inference for one TorNet NetCDF sample.")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config.")
    parser.add_argument("--input", required=True, help="Path to one TorNet-style .nc sample.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch

        from pistdnet.data import load_tornet_file
        from pistdnet.models import PI_STDNet
        from pistdnet.utils.checkpoint import load_checkpoint
        from pistdnet.utils.config import get_model_config, load_config
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("Missing dependency: torch. Install the environment with environment.yml or requirements.txt.") from exc
        raise

    config = load_config(args.config)
    model_cfg = get_model_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        x = load_tornet_file(args.input, model_cfg.get("n_sweeps", 2), model_cfg.get("n_frames", 4), model_cfg.get("channels_per_frame", 11)).unsqueeze(0).to(device)
        model = PI_STDNet(model_cfg).to(device)
        if args.checkpoint:
            load_checkpoint(args.checkpoint, model, map_location=device)
        else:
            print("Warning: no checkpoint was provided; output is from randomly initialized weights.")
        model.eval()
        with torch.no_grad():
            cls_logit, aux_logit, _, _ = model(x)
            prob = 0.6 * torch.sigmoid(cls_logit) + 0.4 * torch.sigmoid(aux_logit)
        print(f"tornado_probability={prob.item():.6f}")
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nPlease provide a real sample path; see docs/data_preparation.md.") from exc


if __name__ == "__main__":
    main()
