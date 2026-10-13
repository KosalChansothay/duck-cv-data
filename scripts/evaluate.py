#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluator import evaluate_duck_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Model on Duck Detection & Counting")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", default="configs/duck.yaml", help="Dataset YAML path")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit on images to evaluate")
    args = parser.parse_args()

    evaluate_duck_model(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        max_eval_images=args.max_images
    )


if __name__ == "__main__":
    main()
