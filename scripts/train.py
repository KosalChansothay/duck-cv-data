#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trainer import train_duck_model


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Model for Duck Counting")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config YAML")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model weights (yolo11n.pt, yolo11s.pt, etc.)")
    parser.add_argument("--data", default="configs/duck.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (640 or 1024)")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--project", default="runs/train", help="Output project directory")
    parser.add_argument("--name", default="duck_counter", help="Experiment name")
    args = parser.parse_args()

    train_duck_model(
        config_path=args.config if Path(args.config).exists() else None,
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
