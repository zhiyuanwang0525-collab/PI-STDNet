"""Command-line training entry point for PI-STDNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-STDNet on TorNet-style data.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pistdnet.engine import train_model
        from pistdnet.utils.config import load_config
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("Missing dependency: torch. Install the environment with environment.yml or requirements.txt.") from exc
        raise

    config = load_config(args.config)
    try:
        train_model(config)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nPlease prepare the dataset as described in docs/data_preparation.md.") from exc


if __name__ == "__main__":
    main()
