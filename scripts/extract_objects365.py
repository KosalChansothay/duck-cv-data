#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.objects365 import Objects365DuckExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract Duck data from Objects365")
    parser.add_argument("--split", choices=["val", "train"], default="val", help="Dataset split to extract (default: val)")
    parser.add_argument("--dir", default=".", help="Root directory for dataset (default: current directory)")
    parser.add_argument("--max-patches", type=int, default=None, help="Maximum number of patches to process")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of duck images to extract (optional cap)")
    parser.add_argument("--skip-images", action="store_true", help="Skip image extraction and only generate labels/summary")
    args = parser.parse_args()

    extractor = Objects365DuckExtractor(root_dir=args.dir)
    extractor.run_pipeline(
        split=args.split,
        max_images=args.max_images,
        max_patches=args.max_patches,
        skip_images=args.skip_images
    )


if __name__ == "__main__":
    main()
