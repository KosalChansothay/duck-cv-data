import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def convert_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO format bounding box [x_min, y_min, width, height]
    to normalized YOLO format [x_center, y_center, width, height].
    Clamps coordinates strictly within [0.0, 1.0].
    """
    x, y, w, h = bbox
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return round(xc, 6), round(yc, 6), round(nw, 6), round(nh, 6)


def write_yolo_labels(
    duck_images: Dict[int, Dict[str, Any]],
    duck_annotations: Dict[int, List[Dict[str, Any]]],
    split: str,
    output_dir: str | Path,
    class_id: int = 0
) -> Dict[int, int]:
    """
    Write single-class YOLO .txt label files for all duck images.
    Returns:
        dict mapping img_id -> duck count in that image
    """
    out_path = Path(output_dir)
    labels_dir = out_path / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    for img_id, img_info in duck_images.items():
        base_name = Path(img_info["file_name"]).stem
        label_file = labels_dir / f"{base_name}.txt"
        
        w = img_info["width"]
        h = img_info["height"]
        anns = duck_annotations.get(img_id, [])
        counts[img_id] = len(anns)
        
        lines = []
        for ann in anns:
            xc, yc, nw, nh = convert_bbox_to_yolo(ann["bbox"], w, h)
            if nw > 0 and nh > 0:
                lines.append(f"{class_id} {xc} {yc} {nw} {nh}\n")
                
        with open(label_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
    return counts


def compute_density_distribution(counts: Dict[Any, int]) -> Dict[str, int]:
    """Group image counts into standard density bins."""
    dist = {
        "1 duck": 0,
        "2 ducks": 0,
        "3-5 ducks": 0,
        "6-10 ducks": 0,
        "10+ ducks": 0
    }
    for c in counts.values():
        if c == 1:
            dist["1 duck"] += 1
        elif c == 2:
            dist["2 ducks"] += 1
        elif 3 <= c <= 5:
            dist["3-5 ducks"] += 1
        elif 6 <= c <= 10:
            dist["6-10 ducks"] += 1
        elif c > 10:
            dist["10+ ducks"] += 1
    return dist


def verify_dataset_integrity(root_dir: str | Path) -> Dict[str, Any]:
    """
    Verify 1-to-1 matching between images and label files across splits.
    Checks bounding box validity and returns verification metrics.
    """
    root = Path(root_dir)
    results = {}
    
    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        
        if not img_dir.exists():
            results[split] = {"status": "missing_directory", "images": 0, "labels": 0}
            continue
            
        imgs = set(p.stem for p in img_dir.glob("*.jpg"))
        lbls = set(p.stem for p in lbl_dir.glob("*.txt")) if lbl_dir.exists() else set()
        
        matched = imgs & lbls
        missing_lbl = imgs - lbls
        missing_img = lbls - imgs
        
        # Count total instances
        total_boxes = 0
        corrupt_boxes = 0
        if lbl_dir.exists():
            for lp in lbl_dir.glob("*.txt"):
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            total_boxes += 1
                            coords = [float(x) for x in parts[1:]]
                            if any(c < 0.0 or c > 1.0 for c in coords):
                                corrupt_boxes += 1
                                
        results[split] = {
            "total_images": len(imgs),
            "total_labels": len(lbls),
            "matched_pairs": len(matched),
            "missing_labels": len(missing_lbl),
            "missing_images": len(missing_img),
            "total_boxes": total_boxes,
            "corrupt_boxes": corrupt_boxes,
            "is_valid": len(missing_lbl) == 0 and corrupt_boxes == 0 and len(imgs) > 0
        }
    return results
