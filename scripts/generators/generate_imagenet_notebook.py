import json
from pathlib import Path

def create_imagenet_duck_notebook():
    cells = []

    # Markdown Cell 1: Header & Context
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#  ImageNet Duck Class Extractor\n",
            "\n",
            "This notebook extracts the **Duck** class from **ImageNet** using the approach from the repository:\n",
            " **[mf1024/ImageNet-datasets-downloader](https://github.com/mf1024/ImageNet-datasets-downloader)**\n",
            "\n",
            "---\n",
            "\n",
            "###  Class Identification & Index 97\n",
            "In standard ImageNet (ILSVRC 1000 classes, 0-indexed):\n",
            "- **Index 96:** `n01843383` (`toucan`)\n",
            "- **Index 97:** `n01847000` (`drake` / `duck` - male duck, canonical duck category in ImageNet-1K)\n",
            "- **General Duck Synset:** `n01846331` (`duck` - broad WordNet synset encompassing all duck species)\n",
            "\n",
            "| Index | Synset ID (WNID) | Class Name | Total URLs | Flickr URLs |\n",
            "|---|---|---|---|---|\n",
            "| **97 (ILSVRC)** | `n01847000` | **drake (duck)** | 1,266 | 1,257 |\n",
            "| Broad Synset | `n01846331` | **duck** | 1,642 | 1,622 |\n",
            "| Related | `n01847170` | duckling | 1,257 | 1,198 |\n",
            "| Related | `n01851895` | wild duck | 916 | 823 |\n",
            "| Related | `n01852400` | mandarin duck | 1,314 | 1,268 |\n",
            "| Related | `n01852671` | muscovy duck | 1,413 | 1,400 |\n",
            "\n",
            "### ️ Pipeline Highlights\n",
            "1. **Direct API Fetching**: Queries the ImageNet synset API (`http://www.image-net.org/api/imagenet.synset.geturls?wnid=...`) to retrieve image URLs.\n",
            "2. **Built-in Concurrent Downloader**: Uses multithreading (`ThreadPoolExecutor`) with non-blocking timeouts to bypass dead URLs quickly.\n",
            "3. **Automatic Image Verification**: Discards corrupted responses, placeholder 404 images, and non-image payloads using `PIL`.\n",
            "4. **CLI Alternative**: Demonstrates running the original repository's `downloader.py` command line interface.\n",
            "5. **Interactive Gallery & Stats**: Visualizes downloaded duck images and summarizes dataset metrics."
        ]
    })

    # Code Cell 2: Imports & Environment
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import io\n",
            "import time\n",
            "import json\n",
            "import hashlib\n",
            "from pathlib import Path\n",
            "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
            "from urllib.request import Request, urlopen\n",
            "\n",
            "import requests\n",
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "from tqdm.auto import tqdm\n",
            "\n",
            "WORK_DIR = Path(\".\").resolve()\n",
            "print(f\"Project Directory: {WORK_DIR}\")\n",
            "print(f\"Python Executable: {sys.executable}\")"
        ]
    })

    # Markdown Cell 3: Configuration
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Download Configuration\n",
            "\n",
            "Configure the extraction parameters below:\n",
            "- `TARGET_WNIDS`: List of WordNet IDs to download. Defaults to `['n01847000']` (Index 97: drake / duck). You can also add `'n01846331'` (general duck).\n",
            "- `IMAGES_PER_CLASS`: Number of verified duck images to download (set to `500` or `None` for all available).\n",
            "- `OUTPUT_DIR`: Directory where images will be saved.\n",
            "- `NUM_WORKERS`: Number of parallel download threads (e.g., 16-32 for fast I/O).\n",
            "- `REQUEST_TIMEOUT`: Seconds before giving up on slow/dead links."
        ]
    })

    # Code Cell 4: Config Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Index 97 in ImageNet-1K is 'n01847000' (drake / duck)\n",
            "# 'n01846331' is the broader 'duck' synset\n",
            "TARGET_CLASSES = {\n",
            "    \"n01847000\": {\"name\": \"drake_duck\", \"description\": \"ImageNet-1K Index 97: Drake (Duck)\"},\n",
            "    \"n01846331\": {\"name\": \"duck_general\", \"description\": \"Full WordNet Duck Synset\"}\n",
            "}\n",
            "\n",
            "# Select which synsets to download (defaults to Index 97)\n",
            "SELECTED_WNIDS = [\"n01847000\"]  # Set to [\"n01847000\", \"n01846331\"] for combined extraction\n",
            "\n",
            "# Download quotas and directory\n",
            "IMAGES_PER_CLASS = 500  # Target number of images per class (or None for all)\n",
            "OUTPUT_DIR = WORK_DIR / \"images\" / \"imagenet_duck\"\n",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Network settings\n",
            "NUM_WORKERS = 16\n",
            "REQUEST_TIMEOUT = 5  # seconds\n",
            "MIN_IMAGE_DIM = 100  # discard images smaller than 100x100 px (placeholders/icons)\n",
            "\n",
            "print(f\"Selected Classes: {[TARGET_CLASSES[w]['description'] for w in SELECTED_WNIDS]}\")\n",
            "print(f\"Target Quota: {IMAGES_PER_CLASS} images per class\")\n",
            "print(f\"Output Directory: {OUTPUT_DIR}\")"
        ]
    })

    # Markdown Cell 5: Metadata verification
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Verify Index 97 in ImageNet Metadata\n",
            "\n",
            "We fetch `imagenet_class_info.json` directly from the `mf1024/ImageNet-datasets-downloader` repository to inspect class metadata."
        ]
    })

    # Code Cell 6: Metadata verification code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "REPO_INFO_URL = \"https://raw.githubusercontent.com/mf1024/ImageNet-datasets-downloader/master/imagenet_class_info.json\"\n",
            "\n",
            "try:\n",
            "    print(\"Fetching imagenet_class_info.json from repository...\")\n",
            "    resp = requests.get(REPO_INFO_URL, timeout=10)\n",
            "    resp.raise_for_status()\n",
            "    class_info = resp.json()\n",
            "    print(f\"Successfully loaded metadata for {len(class_info):,} ImageNet classes.\")\n",
            "    \n",
            "    print(\"\\nMetadata for target classes:\")\n",
            "    for wnid in TARGET_CLASSES:\n",
            "        if wnid in class_info:\n",
            "            info = class_info[wnid]\n",
            "            print(f\"  • {wnid} ({TARGET_CLASSES[wnid]['description']}):\")\n",
            "            print(f\"      Class Name: '{info.get('class_name')}'\")\n",
            "            print(f\"      Total Image URLs: {info.get('img_url_count'):,}\")\n",
            "            print(f\"      Flickr Image URLs: {info.get('flickr_img_url_count'):,}\")\n",
            "except Exception as e:\n",
            "    print(f\"Warning: Could not fetch remote metadata ({e}). Using offline defaults.\")"
        ]
    })

    # Markdown Cell 7: In-notebook Downloader
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. High-Performance In-Notebook Downloader\n",
            "\n",
            "This built-in extractor queries the ImageNet URL endpoint:\n",
            "```\n",
            "http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}\n",
            "```\n",
            "and concurrently downloads and validates images using a multithreaded worker pool."
        ]
    })

    # Code Cell 8: Downloader Implementation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def fetch_synset_urls(wnid: str) -> list[str]:\n",
            "    \"\"\"Fetch all image URLs for a given WordNet ID from ImageNet API.\"\"\"\n",
            "    api_url = f\"http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}\"\n",
            "    headers = {\"User-Agent\": \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageNetDuckExtractor/1.0\"}\n",
            "    \n",
            "    resp = requests.get(api_url, headers=headers, timeout=15)\n",
            "    resp.raise_for_status()\n",
            "    \n",
            "    urls = [line.strip() for line in resp.text.splitlines() if line.strip()]\n",
            "    return urls\n",
            "\n",
            "\n",
            "def download_and_verify_image(url: str, save_path: Path, min_dim: int = 100, timeout: int = 5) -> dict | None:\n",
            "    \"\"\"\n",
            "    Download a single image URL, verify image integrity and dimensions,\n",
            "    and save to disk if valid.\n",
            "    \"\"\"\n",
            "    if not url.startswith((\"http://\", \"https://\")):\n",
            "        return None\n",
            "    \n",
            "    headers = {\"User-Agent\": \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageNetDuckExtractor/1.0\"}\n",
            "    try:\n",
            "        r = requests.get(url, headers=headers, timeout=timeout)\n",
            "        if r.status_code != 200:\n",
            "            return None\n",
            "        \n",
            "        content = r.content\n",
            "        if len(content) < 2048:  # Skip tiny payloads (e.g. 1x1 pixels or error pages)\n",
            "            return None\n",
            "        \n",
            "        # Validate image format and dimensions using PIL\n",
            "        img = Image.open(io.BytesIO(content))\n",
            "        img.verify()  # Check for header/byte corruption\n",
            "        \n",
            "        # Re-open to read dimensions and convert if necessary\n",
            "        img = Image.open(io.BytesIO(content))\n",
            "        w, h = img.size\n",
            "        if w < min_dim or h < min_dim:\n",
            "            return None\n",
            "        \n",
            "        # Convert non-RGB (e.g. RGBA or P) to RGB for standardized JPEG saving\n",
            "        if img.mode != \"RGB\":\n",
            "            img = img.convert(\"RGB\")\n",
            "        \n",
            "        img.save(save_path, format=\"JPEG\", quality=95)\n",
            "        return {\n",
            "            \"url\": url,\n",
            "            \"file\": save_path.name,\n",
            "            \"width\": w,\n",
            "            \"height\": h,\n",
            "            \"size_bytes\": save_path.stat().st_size\n",
            "        }\n",
            "    except Exception:\n",
            "        return None"
        ]
    })

    # Markdown Cell 9: Extraction Run Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Run Duck Extraction\n",
            "\n",
            "Executes multithreaded extraction for the selected duck synsets.\n",
            "- Prioritizes reliable Flickr URLs (which have high retention).\n",
            "- Stops as soon as the quota (`IMAGES_PER_CLASS`) is satisfied."
        ]
    })

    # Code Cell 10: Extraction Execution
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "extraction_results = {}\n",
            "metadata_log = []\n",
            "\n",
            "for wnid in SELECTED_WNIDS:\n",
            "    cls_info = TARGET_CLASSES.get(wnid, {\"name\": wnid, \"description\": wnid})\n",
            "    cls_name = cls_info[\"name\"]\n",
            "    cls_folder = OUTPUT_DIR / cls_name\n",
            "    cls_folder.mkdir(parents=True, exist_ok=True)\n",
            "    \n",
            "    print(f\"\\n{'='*70}\")\n",
            "    print(f\"[Extracting] {cls_info['description']} (wnid: {wnid})\")\n",
            "    print(f\"Target Directory: {cls_folder}\")\n",
            "    print(f\"{'='*70}\")\n",
            "    \n",
            "    all_urls = fetch_synset_urls(wnid)\n",
            "    print(f\"Retrieved {len(all_urls):,} total URLs from ImageNet API.\")\n",
            "    \n",
            "    # Sort to prioritize Flickr URLs (significantly higher survival rate)\n",
            "    flickr_urls = [u for u in all_urls if \"flickr.com\" in u]\n",
            "    other_urls = [u for u in all_urls if \"flickr.com\" not in u]\n",
            "    ordered_urls = flickr_urls + other_urls\n",
            "    print(f\"Prioritizing {len(flickr_urls):,} Flickr URLs + {len(other_urls):,} general URLs.\")\n",
            "    \n",
            "    # Determine target count\n",
            "    target_count = IMAGES_PER_CLASS if IMAGES_PER_CLASS else len(ordered_urls)\n",
            "    \n",
            "    # Track existing images\n",
            "    existing_files = set(f.name for f in cls_folder.glob(\"*.jpg\"))\n",
            "    downloaded_count = len(existing_files)\n",
            "    print(f\"Already downloaded: {downloaded_count:,} images.\")\n",
            "    \n",
            "    pbar = tqdm(total=target_count, initial=downloaded_count, desc=f\"Downloading {cls_name}\")\n",
            "    \n",
            "    # Multithreaded worker queue\n",
            "    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:\n",
            "        future_to_url = {}\n",
            "        url_iter = iter(ordered_urls)\n",
            "        \n",
            "        # Initial pool fill\n",
            "        for _ in range(NUM_WORKERS * 3):\n",
            "            try:\n",
            "                u = next(url_iter)\n",
            "                url_hash = hashlib.md5(u.encode('utf-8')).hexdigest()[:12]\n",
            "                save_path = cls_folder / f\"{wnid}_{url_hash}.jpg\"\n",
            "                if save_path.name in existing_files:\n",
            "                    continue\n",
            "                future_to_url[executor.submit(download_and_verify_image, u, save_path, MIN_IMAGE_DIM, REQUEST_TIMEOUT)] = (u, save_path)\n",
            "            except StopIteration:\n",
            "                break\n",
            "        \n",
            "        # Process results and keep pool saturated\n",
            "        while future_to_url and downloaded_count < target_count:\n",
            "            # Grab completed tasks\n",
            "            done_futures = [f for f in list(future_to_url.keys()) if f.done()]\n",
            "            if not done_futures:\n",
            "                time.sleep(0.05)\n",
            "                continue\n",
            "            \n",
            "            for f in done_futures:\n",
            "                u, save_path = future_to_url.pop(f)\n",
            "                res = f.result()\n",
            "                if res is not None:\n",
            "                    downloaded_count += 1\n",
            "                    pbar.update(1)\n",
            "                    metadata_log.append(res)\n",
            "                    if downloaded_count >= target_count:\n",
            "                        break\n",
            "                \n",
            "                # Submit next URL if available\n",
            "                try:\n",
            "                    next_u = next(url_iter)\n",
            "                    next_hash = hashlib.md5(next_u.encode('utf-8')).hexdigest()[:12]\n",
            "                    next_path = cls_folder / f\"{wnid}_{next_hash}.jpg\"\n",
            "                    future_to_url[executor.submit(download_and_verify_image, next_u, next_path, MIN_IMAGE_DIM, REQUEST_TIMEOUT)] = (next_u, next_path)\n",
            "                except StopIteration:\n",
            "                    pass\n",
            "    \n",
            "    pbar.close()\n",
            "    extraction_results[wnid] = downloaded_count\n",
            "    print(f\"[Done] Extracted {downloaded_count:,} verified duck images for {wnid}.\")"
        ]
    })

    # Markdown Cell 11: Metadata Export
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Metadata Export\n",
            "\n",
            "Saves the extraction log with image dimensions, source URLs, and file paths to `imagenet_duck_metadata.json`."
        ]
    })

    # Code Cell 12: Metadata Save Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "metadata_file = OUTPUT_DIR / \"imagenet_duck_metadata.json\"\n",
            "with open(metadata_file, \"w\", encoding=\"utf-8\") as f:\n",
            "    json.dump({\n",
            "        \"total_images\": len(metadata_log),\n",
            "        \"classes\": TARGET_CLASSES,\n",
            "        \"records\": metadata_log\n",
            "    }, f, indent=2)\n",
            "\n",
            "print(f\"Saved dataset metadata: {metadata_file} ({len(metadata_log):,} records)\")"
        ]
    })

    # Markdown Cell 13: Visual Inspection Gallery
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Visual Inspection Gallery\n",
            "\n",
            "Displays a 4x4 sample grid of the downloaded duck images to verify image quality and content."
        ]
    })

    # Code Cell 14: Visual Gallery Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import random\n",
            "\n",
            "all_downloaded = list(OUTPUT_DIR.rglob(\"*.jpg\"))\n",
            "print(f\"Total downloaded duck images found: {len(all_downloaded):,}\")\n",
            "\n",
            "if all_downloaded:\n",
            "    sample_images = random.sample(all_downloaded, min(16, len(all_downloaded)))\n",
            "    fig, axes = plt.subplots(4, 4, figsize=(14, 14))\n",
            "    axes = axes.flatten()\n",
            "    \n",
            "    for i, img_path in enumerate(sample_images):\n",
            "        img = Image.open(img_path)\n",
            "        axes[i].imshow(img)\n",
            "        axes[i].set_title(f\"{img_path.name[:18]}...\\n{img.size[0]}x{img.size[1]}\", fontsize=9)\n",
            "        axes[i].axis(\"off\")\n",
            "        \n",
            "    for j in range(len(sample_images), len(axes)):\n",
            "        axes[j].axis(\"off\")\n",
            "        \n",
            "    plt.suptitle(\"Sample ImageNet Duck Images (Index 97: Drake / Duck)\", fontsize=15, fontweight=\"bold\", y=0.92)\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"No images found yet to display.\")"
        ]
    })

    # Markdown Cell 15: CLI Integration Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Alternative: Running via `mf1024/ImageNet-datasets-downloader` CLI\n",
            "\n",
            "If you prefer to run the command-line utility directly from the original GitHub repository:\n",
            "\n",
            "```bash\n",
            "# 1. Clone the repository\n",
            "git clone https://github.com/mf1024/ImageNet-datasets-downloader.git\n",
            "\n",
            "# 2. Install dependencies (requests, numpy)\n",
            "pip install requests numpy\n",
            "\n",
            "# 3. Run downloader for Index 97 (drake / duck: n01847000)\n",
            "python ./ImageNet-datasets-downloader/downloader.py \\\n",
            "    -data_root ./images/imagenet_raw \\\n",
            "    -use_class_list True \\\n",
            "    -class_list n01847000 \\\n",
            "    -images_per_class 500 \\\n",
            "    -multiprocessing_workers 16\n",
            "```\n",
            "\n",
            ">  **Tip:** In Windows PowerShell, run without backslashes or use backticks (`` ` ``) for line continuation."
        ]
    })

    # Markdown Cell 16: Integration with YOLO / Duck Counting Project
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Summary & Integration with Duck Counting Dataset\n",
            "\n",
            "| Split / Source | Description | Path |\n",
            "|---|---|---|\n",
            "| **Objects365 Val** | 1,547 duck images with YOLO labels | `images/val`, `labels/val` |\n",
            "| **Objects365 Train** | 5,000 duck images with YOLO labels | `images/train`, `labels/train` |\n",
            "| **ImageNet Duck (Index 97)** | Verified high-res duck photos | `images/imagenet_duck/` |\n",
            "\n",
            "These images can be added to your training pool (`images/train`) or used for pre-training / classification!"
        ]
    })

    # Code Cell 17: Summary Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "duck_files = list(OUTPUT_DIR.rglob(\"*.jpg\"))\n",
            "total_size_mb = sum(f.stat().st_size for f in duck_files) / (1024 * 1024)\n",
            "\n",
            "print(f\"\\n{'='*60}\")\n",
            "print(\"           IMAGENET DUCK EXTRACTION SUMMARY\")\n",
            "print(f\"{'='*60}\")\n",
            "print(f\"Total Images Extracted:  {len(duck_files):,}\")\n",
            "print(f\"Total Disk Space:        {total_size_mb:.2f} MB\")\n",
            "print(f\"Output Directory:        {OUTPUT_DIR}\")\n",
            "print(f\"{'='*60}\")"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_path = Path("extract_imagenet_duck.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Generated notebook successfully: {output_path} ({len(cells)} cells)")

if __name__ == "__main__":
    create_imagenet_duck_notebook()
