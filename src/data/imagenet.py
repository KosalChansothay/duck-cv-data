import io
import time
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import requests
from PIL import Image
from tqdm.auto import tqdm

# ImageNet WordNet IDs for ducks
IMAGENET_DUCK_SYNSETS = {
    "n01847000": {"name": "drake_duck", "description": "ImageNet-1K Index 97: Drake (Duck)"},
    "n01846331": {"name": "duck_general", "description": "Full WordNet Duck Synset"}
}


class ImageNetDuckExtractor:
    """Modular extractor for ImageNet Duck classes (Index 97: n01847000 & broad synsets)."""

    def __init__(
        self,
        output_dir: str | Path = "images/imagenet_duck",
        num_workers: int = 16,
        timeout: int = 5,
        min_dim: int = 100
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.timeout = timeout
        self.min_dim = min_dim

    def fetch_urls(self, wnid: str = "n01847000") -> List[str]:
        """Fetch image URLs for a WordNet ID from ImageNet API."""
        api_url = f"http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageNetDuckExtractor/2.0"}
        resp = requests.get(api_url, headers=headers, timeout=15)
        resp.raise_for_status()
        return [line.strip() for line in resp.text.splitlines() if line.strip()]

    def download_image(self, url: str, save_path: Path) -> Optional[Dict[str, Any]]:
        """Download and verify image integrity."""
        if not url.startswith(("http://", "https://")):
            return None
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageNetDuckExtractor/2.0"}
        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            if r.status_code != 200 or len(r.content) < 2048:
                return None

            img = Image.open(io.BytesIO(r.content))
            img.verify()

            img = Image.open(io.BytesIO(r.content))
            w, h = img.size
            if w < self.min_dim or h < self.min_dim:
                return None

            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(save_path, format="JPEG", quality=95)
            return {
                "url": url,
                "file": save_path.name,
                "width": w,
                "height": h,
                "size_bytes": save_path.stat().st_size
            }
        except Exception:
            return None

    def extract(self, wnid: str = "n01847000", max_images: int = 500) -> List[Dict[str, Any]]:
        """Extract verified duck images up to max_images quota."""
        cls_info = IMAGENET_DUCK_SYNSETS.get(wnid, {"name": wnid, "description": wnid})
        cls_folder = self.output_dir / cls_info["name"]
        cls_folder.mkdir(parents=True, exist_ok=True)

        print(f"[ImageNet] Extracting {cls_info['description']} ({wnid}) into {cls_folder.name}/")
        urls = self.fetch_urls(wnid)
        # Prioritize Flickr URLs
        flickr_urls = [u for u in urls if "flickr.com" in u]
        other_urls = [u for u in urls if "flickr.com" not in u]
        ordered_urls = flickr_urls + other_urls

        existing = set(f.name for f in cls_folder.glob("*.jpg"))
        count = len(existing)
        metadata = []

        pbar = tqdm(total=max_images, initial=count, desc=f"ImageNet {cls_info['name']}")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_url = {}
            url_iter = iter(ordered_urls)

            for _ in range(self.num_workers * 3):
                try:
                    u = next(url_iter)
                    uhash = hashlib.md5(u.encode("utf-8")).hexdigest()[:12]
                    save_path = cls_folder / f"{wnid}_{uhash}.jpg"
                    if save_path.name in existing:
                        continue
                    future_to_url[executor.submit(self.download_image, u, save_path)] = (u, save_path)
                except StopIteration:
                    break

            while future_to_url and count < max_images:
                done = [f for f in list(future_to_url.keys()) if f.done()]
                if not done:
                    time.sleep(0.05)
                    continue

                for f in done:
                    u, save_path = future_to_url.pop(f)
                    res = f.result()
                    if res:
                        count += 1
                        pbar.update(1)
                        metadata.append(res)
                        if count >= max_images:
                            break

                    try:
                        next_u = next(url_iter)
                        next_hash = hashlib.md5(next_u.encode("utf-8")).hexdigest()[:12]
                        next_path = cls_folder / f"{wnid}_{next_hash}.jpg"
                        future_to_url[executor.submit(self.download_image, next_u, next_path)] = (next_u, next_path)
                    except StopIteration:
                        pass

        pbar.close()
        print(f"[ImageNet] Done. Extracted {count:,} images for {wnid}.")
        return metadata
