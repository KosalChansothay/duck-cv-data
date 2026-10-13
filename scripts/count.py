#!/usr/bin/env python3
import sys
import argparse
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.counter import DuckCounter


def main():
    parser = argparse.ArgumentParser(description="Run Duck Counting Inference on Image or Directory")
    parser.add_argument("--source", required=True, help="Path to image file or directory of images")
    parser.add_argument("--weights", default="yolo11s.pt", help="Path to model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--save-dir", default="outputs", help="Directory to save annotated image results")
    args = parser.parse_args()

    counter = DuckCounter(weights=args.weights, conf=args.conf, iou=args.iou, device=args.device)
    src_path = Path(args.source)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        res = counter.count_image(src_path, return_annotated=True)
        print(f"\n{'='*50}")
        print(f"Image:          {src_path.name}")
        print(f"Ducks Counted:  {res.count}")
        print(f"Confidences:    {[round(c, 2) for c in res.confidences]}")
        print(f"{'='*50}")

        if res.annotated_image is not None:
            out_file = save_dir / f"counted_{src_path.name}"
            cv2.imwrite(str(out_file), res.annotated_image)
            print(f"Saved annotated image: {out_file}\n")

    elif src_path.is_dir():
        results = counter.count_directory(src_path)
        print(f"\n{'='*50}")
        print(f"Processed Directory: {src_path}")
        print(f"Total Images:        {len(results)}")
        print(f"Total Ducks Counted: {sum(r.count for r in results)}")
        print(f"{'='*50}")
        for r in results[:20]:
            print(f"  • {Path(r.image_path).name}: {r.count} ducks")
        if len(results) > 20:
            print(f"  ... and {len(results)-20} more images.")
    else:
        raise FileNotFoundError(f"Source not found: {src_path}")


if __name__ == "__main__":
    main()
