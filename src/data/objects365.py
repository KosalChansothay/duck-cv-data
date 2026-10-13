import os
import sys
import json
import time
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from .dataset import write_yolo_labels, compute_density_distribution

BASE_URL = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"

ANNOTATION_URLS = {
    "val": f"{BASE_URL}/val/zhiyuan_objv2_val.json",
    "train": f"{BASE_URL}/train/zhiyuan_objv2_train.tar.gz",
}

DUCK_CATEGORY_ID = 137  # 'Duck' category in Objects365 v2


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024, max_retries: int = 10, timeout: int = 30):
    """Download a file with HTTP Range resume and auto-retry on drops."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = 0
    try:
        head_req = Request(url, headers={"User-Agent": "DuckExtractor/2.0"}, method="HEAD")
        with urlopen(head_req, timeout=15) as head_resp:
            total_size = int(head_resp.headers.get("content-length", 0))
    except Exception:
        pass

    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        if total_size > 0 and existing_size == total_size:
            print(f"[Info] File already complete: {dest_path.name} ({existing_size / (1024*1024):.1f} MB)")
            return
        elif total_size > 0 and existing_size > total_size:
            dest_path.unlink()

    retries = 0
    while retries < max_retries:
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        if total_size > 0 and existing_size >= total_size:
            return

        headers = {"User-Agent": "DuckExtractor/2.0"}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            req = Request(url, headers=headers)
            mode = "ab" if existing_size > 0 else "wb"
            with urlopen(req, timeout=timeout) as resp, open(dest_path, mode) as f:
                if total_size == 0:
                    total_size = existing_size + int(resp.headers.get("content-length", 0))
                downloaded = existing_size
                last_reported_mb = downloaded / (1024 * 1024)
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    mb = downloaded / (1024 * 1024)
                    if mb - last_reported_mb >= 25 or (total_size and downloaded >= total_size):
                        pct = (downloaded / total_size * 100) if total_size else 0
                        tot_mb = total_size / (1024 * 1024) if total_size else 0.0
                        print(f"  Downloaded: {mb:.1f} / {tot_mb:.1f} MB ({pct:.1f}%)")
                        last_reported_mb = mb

            final_size = dest_path.stat().st_size
            if total_size == 0 or final_size >= total_size:
                print(f"[Download] Finished: {dest_path.name} ({final_size / (1024*1024):.1f} MB)")
                return
        except Exception as e:
            retries += 1
            print(f"\n[Warning] Connection dropped ({e}). Retrying ({retries}/{max_retries})...")
            time.sleep(3)

    raise ConnectionError(f"Failed to download {url} after {max_retries} retries.")


def get_patch_url(file_name: str, split: str = "val") -> Tuple[str, str]:
    """Resolve archive patch name and download URL from Objects365 file path."""
    fn = file_name.replace("\\", "/")
    if split == "val":
        if "v1/patch" in fn:
            part = fn.split("v1/patch")[1].split("/")[0]
            return f"v1_patch{part}", f"{BASE_URL}/val/images/v1/patch{part}.tar.gz"
        elif "v2/patch" in fn:
            part = fn.split("v2/patch")[1].split("/")[0]
            return f"v2_patch{part}", f"{BASE_URL}/val/images/v2/patch{part}.tar.gz"
        else:
            parts = fn.split("/")
            for p in parts:
                if p.startswith("patch"):
                    num = p.replace("patch", "")
                    return f"patch{num}", f"{BASE_URL}/val/images/v1/patch{num}.tar.gz"
    else:  # train
        parts = fn.split("/")
        for p in parts:
            if p.startswith("patch"):
                num = p.replace("patch", "")
                return f"patch{num}", f"{BASE_URL}/train/patch{num}.tar.gz"

    raise ValueError(f"Could not resolve patch for file_name: {file_name}")


class Objects365DuckExtractor:
    """Modular extractor for Objects365 Duck dataset."""

    def __init__(self, root_dir: str | Path = "."):
        self.root_dir = Path(root_dir).resolve()
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.temp_dir = self.root_dir / "temp_downloads"
        self.annotations_dir = self.root_dir / "data" / "annotations"
        
        for d in [self.images_dir, self.labels_dir, self.temp_dir, self.annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def find_annotation_file(self, split: str) -> Path:
        """Locate annotation file in data/annotations or root directory."""
        candidates = [
            self.annotations_dir / f"zhiyuan_objv2_{split}.json",
            self.root_dir / f"zhiyuan_objv2_{split}.json",
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 1000:
                return c
        return candidates[0]

    def ensure_annotations(self, split: str) -> Path:
        """Ensure annotation JSON file is available locally."""
        ann_file = self.find_annotation_file(split)
        if ann_file.exists():
            print(f"[Info] Found existing annotations: {ann_file}")
            return ann_file

        ann_url = ANNOTATION_URLS[split]
        if split == "val":
            print(f"[Download] Fetching validation annotations...")
            download_file(ann_url, ann_file)
        else:
            ann_tar = self.annotations_dir / f"zhiyuan_objv2_{split}.tar.gz"
            download_file(ann_url, ann_tar)
            print(f"[Extract] Unpacking {ann_tar.name}...")
            with tarfile.open(ann_tar, "r:gz") as tar:
                tar.extractall(path=self.annotations_dir)
        return ann_file

    def filter_ducks(self, ann_file: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
        """
        Filter instances matching Duck category.
        Dynamically resolves 'Duck' by name from categories list to prevent 0-index vs 1-index errors.
        """
        print(f"[Filter] Reading annotations from {ann_file.name}...")
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Dynamic category resolution
        categories = data.get("categories", [])
        duck_cat = None
        duck_idx = None
        for idx, cat in enumerate(categories):
            if cat.get("name", "").strip().lower() == "duck":
                duck_cat = cat
                duck_idx = idx
                break

        if duck_cat:
            target_cat_id = duck_cat["id"]
            print(f"[Filter] Resolved category '{duck_cat['name']}': ID={target_cat_id} (0-indexed position={duck_idx})")
        else:
            target_cat_id = DUCK_CATEGORY_ID
            print(f"[Filter] Fallback category ID: {target_cat_id}")

        all_images = {img["id"]: img for img in data["images"]}
        duck_annotations = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("category_id") == target_cat_id:
                duck_annotations[ann["image_id"]].append(ann)

        duck_images = {img_id: all_images[img_id] for img_id in duck_annotations if img_id in all_images}
        total_inst = sum(len(a) for a in duck_annotations.values())
        print(f"[Filter] Found {len(duck_images):,} duck images ({total_inst:,} duck instances).")
        return duck_images, duck_annotations

    def extract_images(
        self,
        duck_images: Dict[int, Dict[str, Any]],
        split: str,
        max_patches: Optional[int] = None,
        keep_archives: bool = False
    ):
        """Selectively extract only duck images from required patch archives using streaming iterator."""
        split_img_dir = self.images_dir / split
        split_img_dir.mkdir(parents=True, exist_ok=True)

        patch_to_images = defaultdict(dict)
        for img_id, img_info in duck_images.items():
            base_name = Path(img_info["file_name"]).name
            if (split_img_dir / base_name).exists():
                continue
            try:
                patch_key, patch_url = get_patch_url(img_info["file_name"], split=split)
                patch_to_images[patch_key]["url"] = patch_url
                if "files" not in patch_to_images[patch_key]:
                    patch_to_images[patch_key]["files"] = set()
                patch_to_images[patch_key]["files"].add(base_name)
            except Exception:
                pass

        total_needed = sum(len(p["files"]) for p in patch_to_images.values())
        print(f"[Extract] {total_needed} images need downloading across {len(patch_to_images)} patches.")

        processed = 0
        for patch_key, info in sorted(patch_to_images.items()):
            if max_patches and processed >= max_patches:
                print(f"[Extract] Reached max patches limit ({max_patches}).")
                break

            patch_url = info["url"]
            needed_files = info["files"]
            tar_name = Path(patch_url).name
            possible_tar_paths = [
                self.temp_dir / f"{patch_key}.tar.gz",
                self.temp_dir / f"{split}_{patch_key}.tar.gz",
                self.temp_dir / tar_name,
                self.temp_dir / f"{split}_{patch_key}_{tar_name}",
            ]
            tar_path = next((p for p in possible_tar_paths if p.exists() and p.stat().st_size > 1000), possible_tar_paths[1])

            print(f"\n--- [Patch {processed+1}/{len(patch_to_images)}] {patch_key} ({len(needed_files)} duck images) ---")
            download_file(patch_url, tar_path)

            extracted_count = 0
            # Use streaming generator 'for member in tar' instead of getmembers()
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar:
                    m_base = Path(member.name).name
                    if m_base in needed_files:
                        member.name = m_base
                        tar.extract(member, path=split_img_dir)
                        extracted_count += 1
                        if extracted_count >= len(needed_files):
                            break  # Early exit once all needed duck images in this patch are found!

            print(f"[Extract] Extracted {extracted_count}/{len(needed_files)} duck images into {split_img_dir.name}/")
            if not keep_archives and tar_path.exists():
                tar_path.unlink()
            processed += 1

    def run_pipeline(
        self,
        split: str = "val",
        max_images: Optional[int] = None,
        max_patches: Optional[int] = None,
        skip_images: bool = False
    ) -> Dict[str, Any]:
        """Execute full extraction pipeline: download annotations -> extract images -> write YOLO labels -> summary."""
        ann_file = self.ensure_annotations(split)
        duck_images, duck_annotations = self.filter_ducks(ann_file)

        if max_images and len(duck_images) > max_images:
            selected_ids = list(duck_images.keys())[:max_images]
            duck_images = {k: duck_images[k] for k in selected_ids}
            duck_annotations = {k: duck_annotations[k] for k in selected_ids}
            print(f"[Filter] Capped extraction at {max_images:,} images.")

        if not skip_images:
            self.extract_images(duck_images, split=split, max_patches=max_patches)

        counts = write_yolo_labels(duck_images, duck_annotations, split=split, output_dir=self.root_dir)
        density_dist = compute_density_distribution(counts)

        summary = {
            "dataset": "Objects365 v2",
            "split": split,
            "category_name": "Duck",
            "category_id": DUCK_CATEGORY_ID,
            "total_duck_images": len(duck_images),
            "total_duck_instances": sum(counts.values()),
            "average_ducks_per_image": round(sum(counts.values()) / max(1, len(duck_images)), 2),
            "max_ducks_in_single_image": max(counts.values()) if counts else 0,
            "distribution": density_dist
        }

        # Save summary to reports/ and root
        reports_dir = self.root_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_file = reports_dir / f"duck_counting_{split}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[Done] Summary written to {summary_file}")
        return summary
