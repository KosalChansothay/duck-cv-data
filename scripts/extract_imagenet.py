#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.imagenet import ImageNetDuckExtractor, IMAGENET_DUCK_SYNSETS


def main():
    parser = argparse.ArgumentParser(description="Extract Duck Images from ImageNet (Index 97 & Broad Synset)")
    parser.add_argument("--wnid", default="n01847000", choices=list(IMAGENET_DUCK_SYNSETS.keys()), help="WordNet ID (n01847000 is Index 97)")
    parser.add_argument("--max-images", type=int, default=500, help="Quota of images to download")
    parser.add_argument("--output-dir", default="images/imagenet_duck", help="Destination folder")
    parser.add_argument("--workers", type=int, default=16, help="Download threads")
    parser.add_argument("--timeout", type=int, default=5, help="HTTP timeout seconds")
    args = parser.parse_args()

    extractor = ImageNetDuckExtractor(output_dir=args.output_dir, num_workers=args.workers, timeout=args.timeout)
    extractor.extract(wnid=args.wnid, max_images=args.max_images)


if __name__ == "__main__":
    main()
