import os
from pathlib import Path
from typing import Dict, Any, Optional

from ultralytics import YOLO
from src.utils.config import load_yaml_config, get_default_train_config


def train_duck_model(
    config_path: Optional[str | Path] = None,
    model_name: Optional[str] = None,
    data_yaml: Optional[str | Path] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[int | str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a YOLO model on the Duck dataset.
    
    Args:
        config_path: Path to YAML training configuration file.
        model_name: Base model weights (e.g., 'yolo11n.pt', 'yolo11s.pt', 'yolov8s.pt').
        data_yaml: Dataset YAML path (e.g., 'configs/duck.yaml').
        epochs: Number of training epochs.
        batch: Batch size.
        imgsz: Training image size (640 or 1024).
        device: Device ID (0, 'cpu', etc.).
        project: Output directory.
        name: Experiment name.
        **kwargs: Additional hyperparameters passed to model.train().
        
    Returns:
        Dictionary containing trained model, results dict, and best checkpoint path.
    """
    # 1. Base config
    cfg = get_default_train_config()
    if config_path and Path(config_path).exists():
        cfg.update(load_yaml_config(config_path))

    # 2. Override with explicit function arguments
    if model_name: cfg["model"] = model_name
    if data_yaml: cfg["data"] = str(data_yaml)
    if epochs: cfg["epochs"] = epochs
    if batch: cfg["batch"] = batch
    if imgsz: cfg["imgsz"] = imgsz
    if device is not None: cfg["device"] = device
    if project: cfg["project"] = project
    if name: cfg["name"] = name
    cfg.update(kwargs)

    # 3. Verify dataset YAML
    data_path = Path(cfg["data"])
    if not data_path.exists():
        # Fallback to root duck.yaml if configs/duck.yaml not found
        if Path("duck.yaml").exists():
            cfg["data"] = "duck.yaml"
        else:
            raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    print(f"\n{'='*70}")
    print(f"🦆 Starting Duck Counting Model Training")
    print(f"Base Model:    {cfg['model']}")
    print(f"Dataset YAML:  {cfg['data']}")
    print(f"Epochs:        {cfg['epochs']}")
    print(f"Batch Size:    {cfg['batch']}")
    print(f"Image Size:    {cfg['imgsz']}")
    print(f"Device:        {cfg['device']}")
    print(f"Output:        {cfg['project']}/{cfg['name']}")
    print(f"{'='*70}\n")

    # 4. Initialize model
    model = YOLO(cfg["model"])

    # 5. Extract training parameters
    train_args = {k: v for k, v in cfg.items() if k not in ["model"]}

    # 6. Execute training
    results = model.train(**train_args)

    best_checkpoint = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    print(f"\n[Training Complete] Best weights saved to: {best_checkpoint}")

    return {
        "model": model,
        "results": results,
        "best_weights": str(best_checkpoint) if best_checkpoint.exists() else None
    }
