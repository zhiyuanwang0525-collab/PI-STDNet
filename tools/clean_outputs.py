"""Move local generated artifacts into _local_artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


DEFAULT_PATTERNS = ["*.pth", "*.pt", "*.ckpt", "*.log", "*.png", "*.pdf", "*.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Move generated artifacts out of the public source tree.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    target = root / "_local_artifacts" / "manual_cleanup"
    for pattern in DEFAULT_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            dest = target / path.name
            print(f"{path.relative_to(root)} -> {dest.relative_to(root)}")
            if not args.dry_run:
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))


if __name__ == "__main__":
    main()

