"""Helper for launching the documented training/evaluation workflow."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the PI-STDNet reproducibility workflow.")
    parser.parse_args()
    print("1. Train NC: python scripts/train.py --config configs/train_nc.yaml")
    print("2. Train WC: python scripts/train.py --config configs/train_wc.yaml")
    print("3. Evaluate: python scripts/evaluate.py --config configs/eval.yaml --checkpoint /path/to/checkpoint.pth")
    print("Experiment identifiers will be updated upon publication.")


if __name__ == "__main__":
    main()
