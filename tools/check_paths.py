"""Scan the repository for local paths, secrets, and large-file patterns."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


RISK_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/][^\s'\"<>]+)"),
    "linux_home_or_mount_path": re.compile(r"(?<![A-Za-z0-9_])(/home/[^\s'\"<>]+|/root/[^\s'\"<>]+|/mnt/[^\s'\"<>]+|/data/[^\s'\"<>]+)"),
    "onedrive_path": re.compile(r"OneDrive", re.IGNORECASE),
    "secret_keyword": re.compile(r"(api[_-]?key|token|password|passwd|secret|ssh[_-]?key|wandb[_-]?key)", re.IGNORECASE),
}

LARGE_FILE_SUFFIXES = {".pth", ".pt", ".ckpt", ".onnx", ".h5", ".hdf5", ".nc", ".npy", ".npz", ".parquet"}
SKIP_DIRS = {".git", "__pycache__", "_local_artifacts", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".cff", ".toml", ".json", ".ini", ".sh", ".ps1"}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        rel = path.relative_to(root)
        if rel == Path("tools/check_paths.py"):
            continue
        if path.suffix.lower() in LARGE_FILE_SUFFIXES:
            findings.append(f"large_file_suffix: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in RISK_PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                if value.startswith("/path/to/"):
                    continue
                findings.append(f"{name}: {rel}: {value[:120]}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan repository for paths, secrets, and large-file suffixes.")
    parser.add_argument("path", nargs="?", default=None, help="Repository root to scan.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    args = parser.parse_args()
    scan_root = args.path or args.root
    findings = scan(Path(scan_root).resolve())
    if findings:
        print("\n".join(findings))
        raise SystemExit(1)
    print("No path/secret/large-file risks found in scanned files.")


if __name__ == "__main__":
    main()
