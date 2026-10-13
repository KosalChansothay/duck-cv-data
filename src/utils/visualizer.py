import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Dict, Any, Optional

def draw_duck_detections(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    confidences: Optional[Sequence[float]] = None,
    count_text: Optional[str] = None,
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Draw duck bounding boxes and a prominent count banner on the image.
    
    Args:
        image: BGR or RGB numpy image
        boxes: List of [x1, y1, x2, y2]
        confidences: Optional list of confidence scores
        count_text: Custom text banner (default: 'Ducks Counted: N')
        box_color: Bounding box RGB/BGR color
        text_color: Text color
        
    Returns:
        Annotated copy of image
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # Draw individual duck boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        
        if confidences is not None and i < len(confidences):
            label = f"{confidences[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Banner header for duck count
    total_ducks = len(boxes)
    banner_str = count_text if count_text else f"🦆 Ducks Counted: {total_ducks}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.7, min(1.2, w / 900.0))
    thickness = 2
    (bw, bh), baseline = cv2.getTextSize(banner_str, font, scale, thickness)
    
    # Semi-transparent top overlay banner
    banner_h = bh + 24
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    
    cv2.putText(annotated, banner_str, (16, bh + 10), font, scale, text_color, thickness, cv2.LINE_AA)
    return annotated


def plot_count_distribution(distribution_dict: Dict[str, int], title: str = "Duck Density Distribution", save_path: Optional[str | Path] = None):
    """Plot a bar chart of duck count density buckets."""
    labels = list(distribution_dict.keys())
    values = list(distribution_dict.values())
    
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color="#2b5c8f", edgecolor="#1a3b5c", alpha=0.9, width=0.55)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Ducks per Image", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + max(values) * 0.015, f"{h:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_error_analysis(gt_counts: Sequence[float], pred_counts: Sequence[float], save_path: Optional[str | Path] = None):
    """Plot scatter plot of Ground Truth vs Predicted duck counts with identity line."""
    plt.figure(figsize=(7, 7))
    plt.scatter(gt_counts, pred_counts, alpha=0.5, color="#1f77b4", edgecolors="none", s=25, label="Images")
    
    max_val = max(max(gt_counts), max(pred_counts)) + 5
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect Match (GT = Pred)")
    
    plt.title("Duck Count: Ground Truth vs Predicted", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Ground Truth Ducks", fontsize=11)
    plt.ylabel("Predicted Ducks", fontsize=11)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
