import yaml
from pathlib import Path
from typing import Any, Dict

def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_default_train_config() -> Dict[str, Any]:
    """Return default training hyperparameters for YOLO duck counting."""
    return {
        "model": "yolo11s.pt",
        "data": "configs/duck.yaml",
        "epochs": 50,
        "batch": 16,
        "imgsz": 640,
        "device": 0,
        "workers": 4,
        "patience": 15,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "project": "runs/train",
        "name": "duck_counter"
    }
