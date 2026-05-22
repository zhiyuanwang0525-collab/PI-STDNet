"""Temporal case visualization entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pistdnet.visualization.temporal_evolution import visualize_temporal_case
from pistdnet.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a temporal TorNet case.")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config.")
    parser.add_argument("--case-id", required=True, help="Case identifier from the TorNet catalog.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    try:
        visualize_temporal_case(config, args.case_id, checkpoint=args.checkpoint)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nPlease prepare data/checkpoints as described in docs/data_preparation.md and docs/evaluation.md.") from exc


if __name__ == "__main__":
    main()

