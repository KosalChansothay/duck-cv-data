import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from ultralytics import YOLO
from tqdm.auto import tqdm
from src.utils.metrics import calculate_counting_metrics


def evaluate_duck_model(
    weights_path: str | Path,
    data_yaml: str | Path = "configs/duck.yaml",
    split: str = "val",
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_eval_images: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation: both standard YOLO mAP metrics AND
    direct duck counting metrics (MAE, RMSE, exact match rate).
    
    Args:
        weights_path: Path to trained weights (e.g., 'runs/train/duck_counter/weights/best.pt').
        data_yaml: Dataset config path.
        split: Split to evaluate on ('val' or 'test').
        imgsz: Image resolution.
        conf_thresh: Confidence threshold for counting.
        iou_thresh: IOU threshold for NMS.
        max_eval_images: Optional cap on images to evaluate.
        
    Returns:
        Dictionary with mAP metrics and counting metrics.
    """
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    data_path = Path(data_yaml)
    if not data_path.exists() and Path("duck.yaml").exists():
        data_path = Path("duck.yaml")

    model = YOLO(str(weights))

    print(f"\n{'='*70}")
    print(f" Step 1: Evaluating YOLO Detection Metrics (mAP)")
    print(f"{'='*70}")
    val_results = model.val(data=str(data_path), split=split, imgsz=imgsz)
    
    map50 = float(val_results.box.map50) if hasattr(val_results.box, "map50") else 0.0
    map50_95 = float(val_results.box.map) if hasattr(val_results.box, "map") else 0.0
    precision = float(val_results.box.mp) if hasattr(val_results.box, "mp") else 0.0
    recall = float(val_results.box.mr) if hasattr(val_results.box, "mr") else 0.0

    print(f"\n{'='*70}")
    print(f" Step 2: Evaluating Duck Counting Accuracy (MAE & RMSE)")
    print(f"{'='*70}")

    root_dir = Path(".").resolve()
    val_images_dir = root_dir / "images" / split
    val_labels_dir = root_dir / "labels" / split

    gt_counts: List[int] = []
    pred_counts: List[int] = []

    if val_images_dir.exists() and val_labels_dir.exists():
        image_files = sorted(list(val_images_dir.glob("*.jpg")))
        if max_eval_images:
            image_files = image_files[:max_eval_images]

        print(f"Evaluating duck counting across {len(image_files):,} {split} images...")
        for img_p in tqdm(image_files, desc="Counting Evaluation"):
            lbl_p = val_labels_dir / f"{img_p.stem}.txt"
            
            # Ground truth count from label file
            gt = 0
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    gt = sum(1 for line in f if line.strip())

            # Prediction count
            preds = model.predict(str(img_p), conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, verbose=False)[0]
            pred_count = len(preds.boxes)

            gt_counts.append(gt)
            pred_counts.append(pred_count)

        counting_metrics = calculate_counting_metrics(gt_counts, pred_counts)
    else:
        counting_metrics = {"error": f"Images or labels directory not found for split: {split}"}

    summary = {
        "model_weights": str(weights),
        "split": split,
        "detection_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        },
        "counting_metrics": counting_metrics
    }

    print(f"\n{'='*70}")
    print(f"           FINAL EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Detection mAP@50:       {map50:.4f}")
    print(f"Detection mAP@50:95:    {map50_95:.4f}")
    if "mae" in counting_metrics:
        print(f"Counting MAE:           {counting_metrics['mae']} ducks / image")
        print(f"Counting RMSE:          {counting_metrics['rmse']}")
        print(f"Exact Match Rate:       {counting_metrics['exact_match_percent']}%")
        print(f"Within ±1 Duck Rate:    {counting_metrics['within_one_percent']}%")
        print(f"Correlation (r):        {counting_metrics['pearson_r']}")
    print(f"{'='*70}\n")

    return summary
