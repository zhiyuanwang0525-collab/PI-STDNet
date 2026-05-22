"""Command-line evaluation entry point for PI-STDNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PI-STDNet on TorNet-style data.")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from pistdnet.data import TorNetDataset
        from pistdnet.engine import evaluate_model
        from pistdnet.models import PI_STDNet
        from pistdnet.utils.checkpoint import load_checkpoint
        from pistdnet.utils.config import get_model_config, load_config, resolve_catalog_path
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("Missing dependency: torch. Install the environment with environment.yml or requirements.txt.") from exc
        raise

    config = load_config(args.config)
    model_cfg = get_model_config(config)
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dataset = TorNetDataset(
            resolve_catalog_path(config),
            Path(data_cfg.get("dataset_root", "/path/to/TorNet")),
            mode=data_cfg.get("test_split", "test"),
            n_sweeps=model_cfg.get("n_sweeps", 2),
            n_frames=model_cfg.get("n_frames", 4),
            channels_per_frame=model_cfg.get("channels_per_frame", 11),
        )
        loader = DataLoader(dataset, batch_size=int(eval_cfg.get("batch_size", 32)), shuffle=False, num_workers=int(eval_cfg.get("num_workers", 4)))
        model = PI_STDNet(model_cfg).to(device)
        load_checkpoint(args.checkpoint, model, map_location=device)
        results = evaluate_model(model, loader, device, threshold=eval_cfg.get("threshold"), use_amp=bool(eval_cfg.get("use_amp", True)))
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nPlease prepare required files as described in docs/data_preparation.md and docs/evaluation.md.") from exc

    output_dir = Path(eval_cfg.get("output_dir", "./outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "metrics_summary.npz", **{f"{view}_{key}": value for view, metrics in results.items() for key, value in metrics.items()})
    for view, metrics in results.items():
        print(f"[{view}] CSI={metrics['csi']:.4f} POD={metrics['pod']:.4f} FAR={metrics['far']:.4f} AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} threshold={metrics['threshold']:.2f}")


if __name__ == "__main__":
    main()
