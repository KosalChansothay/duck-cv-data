import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.visualizer import draw_duck_detections


@dataclass
class DuckCountResult:
    """Result of duck counting inference."""
    count: int
    boxes: List[List[float]]          # [x1, y1, x2, y2]
    confidences: List[float]
    image_path: Optional[str] = None
    annotated_image: Optional[np.ndarray] = None


class DuckCounter:
    """
    High-level duck counting and detection inference pipeline.
    """

    def __init__(
        self,
        weights: str | Path = "yolo11s.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[int | str] = None
    ):
        self.weights_path = Path(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(str(self.weights_path))

    def count_image(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        return_annotated: bool = True
    ) -> DuckCountResult:
        """
        Detect and count ducks in a single image.
        
        Args:
            source: Image path, numpy array (BGR/RGB), or PIL Image.
            conf: Optional confidence threshold override.
            iou: Optional IOU threshold override.
            return_annotated: Whether to generate an annotated visualization.
            
        Returns:
            DuckCountResult containing total count, boxes, confidences, and annotated image.
        """
        c = conf if conf is not None else self.conf
        i = iou if iou is not None else self.iou

        preds = self.model.predict(
            source,
            conf=c,
            iou=i,
            device=self.device,
            verbose=False
        )[0]

        boxes = []
        confidences = []

        if len(preds.boxes) > 0:
            for b in preds.boxes:
                coords = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu().numpy())
                boxes.append(coords)
                confidences.append(score)

        count = len(boxes)
        img_path = str(source) if isinstance(source, (str, Path)) else None

        annotated_img = None
        if return_annotated:
            orig_img = preds.orig_img
            annotated_img = draw_duck_detections(
                orig_img,
                boxes,
                confidences,
                count_text=f"🦆 Ducks Counted: {count}"
            )

        return DuckCountResult(
            count=count,
            boxes=boxes,
            confidences=confidences,
            image_path=img_path,
            annotated_image=annotated_img
        )

    def count_directory(
        self,
        dir_path: str | Path,
        extensions: tuple = (".jpg", ".jpeg", ".png")
    ) -> List[DuckCountResult]:
        """Count ducks across all images in a directory."""
        dir_p = Path(dir_path)
        img_files = sorted([p for p in dir_p.rglob("*") if p.suffix.lower() in extensions])
        
        results = []
        for p in img_files:
            res = self.count_image(p, return_annotated=False)
            results.append(res)
        return results
