#!/usr/bin/env python3
"""
Print a compact summary from dataset_info.json.
"""

import argparse
import json
from pathlib import Path


def print_dataset_info(info_path: Path):
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset info file not found: {info_path}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    print(f"  Total samples: {info.get('total_samples', 'unknown')}")
    print(f"  Downloaded:    {info.get('download_time', 'unknown')}")

    splits = info.get("splits")
    if isinstance(splits, dict):
        for split_name, split_size in splits.items():
            print(f"  Split {split_name}: {split_size} samples")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print dataset_info.json summary.")
    parser.add_argument(
        "--info-path",
        required=True,
        help="Path to dataset_info.json",
    )
    args = parser.parse_args()

    print_dataset_info(Path(args.info_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
