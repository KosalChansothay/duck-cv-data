import json
from pathlib import Path

def create_notebook():
    cells = []

    # Markdown Cell 1: Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Objects365 Duck Dataset Extractor for Duck Counting\n",
            "\n",
            "This notebook extracts **only Duck instances** from the [Objects365 dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml) for a specialized computer vision task: **Duck Counting**.\n",
            "\n",
            "### Dataset Breakdown & Selective Pipeline\n",
            "The full Objects365 dataset is massive (>700 GB, 1.7M images across 51 train and 44 val patches).\n",
            "To build an optimal duck counting dataset without downloading hundreds of gigabytes of irrelevant data:\n",
            "\n",
            "1. **Validation Split (1,547 images, 12,221 duck instances)**: Extracted into `images/val` and `labels/val` as the evaluation benchmark.\n",
            "2. **Training Split (+5,000 duck images)**: Filtered from `zhiyuan_objv2_train.json` and selectively extracted from only the initial patches needed to fulfill the 5,000 duck quota into `images/train` and `labels/train`.\n",
            "3. **Single-Class YOLO Labels**: All bounding boxes mapped to Class `0: duck` (`0 xc yc w h`).\n",
            "4. **Duck Counting Analysis**: Duck density distribution (1 duck, pairs, small groups, dense flocks up to 99 ducks).\n",
            "5. **Ultralytics YOLO Training & Counting**: Configured via `duck.yaml` for training YOLOv8 / YOLO11."
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
            "import json\n",
            "import shutil\n",
            "import tarfile\n",
            "from pathlib import Path\n",
            "from collections import defaultdict\n",
            "from urllib.request import Request, urlopen\n",
            "\n",
            "import cv2\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
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
            "## 1. Global Configuration\n",
            "\n",
            "- `BASE_URL`: Megvii / KSYun repository for Objects365 v2.\n",
            "- `DUCK_CATEGORY_ID`: `137` is the official Objects365 v2 category ID for `Duck`.\n",
            "- Target directories: `images/val`, `labels/val`, `images/train`, `labels/train`."
        ]
    })

    # Code Cell 4: Config Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "BASE_URL = \"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86\"\n",
            "DUCK_CATEGORY_ID = 137  # Objects365 v2 Duck category\n",
            "\n",
            "# Directory structure\n",
            "VAL_IMAGES = WORK_DIR / \"images\" / \"val\"\n",
            "VAL_LABELS = WORK_DIR / \"labels\" / \"val\"\n",
            "TRAIN_IMAGES = WORK_DIR / \"images\" / \"train\"\n",
            "TRAIN_LABELS = WORK_DIR / \"labels\" / \"train\"\n",
            "TEMP_DIR = WORK_DIR / \"temp_downloads\"\n",
            "\n",
            "for p in [VAL_IMAGES, VAL_LABELS, TRAIN_IMAGES, TRAIN_LABELS, TEMP_DIR]:\n",
            "    p.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(\"Directories initialized:\")\n",
            "print(f\"  Val Images:   {VAL_IMAGES}\")\n",
            "print(f\"  Train Images: {TRAIN_IMAGES}\")"
        ]
    })

    # Markdown Cell 5: Helper Functions
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Helper Functions: Download, Patch Resolution & YOLO Formatting\n",
            "\n",
            "These utility functions handle:\n",
            "1. Chunked HTTP downloads with progress tracking.\n",
            "2. Resolving which patch archive (`.tar.gz`) contains each image.\n",
            "3. Converting COCO `[x, y, w, h]` bounding boxes to normalized YOLO `[0, xc, yc, w, h]` coordinates."
        ]
    })

    # Code Cell 6: Helper Functions Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def download_with_progress(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):\n",
            "    \"\"\"Download file with live MB progress reporting.\"\"\"\n",
            "    dest_path.parent.mkdir(parents=True, exist_ok=True)\n",
            "    if dest_path.exists():\n",
            "        size_mb = dest_path.stat().st_size / (1024 * 1024)\n",
            "        print(f\"[Exists] {dest_path.name} ({size_mb:.1f} MB)\")\n",
            "        return\n",
            "\n",
            "    print(f\"[Downloading] {url} -> {dest_path.name}...\")\n",
            "    req = Request(url, headers={\"User-Agent\": \"Objects365-Duck/1.0\"})\n",
            "    with urlopen(req) as resp, open(dest_path, \"wb\") as f:\n",
            "        total = int(resp.headers.get(\"content-length\", 0))\n",
            "        downloaded = 0\n",
            "        last_mb = 0\n",
            "        while True:\n",
            "            chunk = resp.read(chunk_size)\n",
            "            if not chunk:\n",
            "                break\n",
            "            f.write(chunk)\n",
            "            downloaded += len(chunk)\n",
            "            mb = downloaded / (1024 * 1024)\n",
            "            if mb - last_mb >= 20 or downloaded == total:\n",
            "                pct = (downloaded / total * 100) if total else 0\n",
            "                print(f\"  {mb:.1f} / {total / (1024 * 1024):.1f} MB ({pct:.1f}%)\")\n",
            "                last_mb = mb\n",
            "    print(f\"[Completed] {dest_path.name}\")\n",
            "\n",
            "\n",
            "def resolve_patch(file_name: str, split: str = 'val'):\n",
            "    \"\"\"Determine which patch tar.gz archive holds the image.\"\"\"\n",
            "    fn = file_name.replace(\"\\\\\", \"/\")\n",
            "    if split == 'val':\n",
            "        if \"v1/patch\" in fn:\n",
            "            p_num = fn.split(\"v1/patch\")[1].split(\"/\")[0]\n",
            "            return f\"v1_patch{p_num}\", f\"{BASE_URL}/val/images/v1/patch{p_num}.tar.gz\"\n",
            "        elif \"v2/patch\" in fn:\n",
            "            p_num = fn.split(\"v2/patch\")[1].split(\"/\")[0]\n",
            "            return f\"v2_patch{p_num}\", f\"{BASE_URL}/val/images/v2/patch{p_num}.tar.gz\"\n",
            "    else:  # train\n",
            "        parts = fn.split(\"/\")\n",
            "        for p in parts:\n",
            "            if p.startswith(\"patch\"):\n",
            "                p_num = p.replace(\"patch\", \"\")\n",
            "                return f\"patch{p_num}\", f\"{BASE_URL}/train/patch{p_num}.tar.gz\"\n",
            "    raise ValueError(f\"Could not resolve patch for: {file_name}\")\n",
            "\n",
            "\n",
            "def extract_patch_images(patch_key: str, patch_url: str, needed_files: set, dest_dir: Path):\n",
            "    \"\"\"Download single patch archive, extract ONLY duck images, and delete archive.\"\"\"\n",
            "    # Filter out files already downloaded\n",
            "    remaining = {f for f in needed_files if not (dest_dir / f).exists()}\n",
            "    if not remaining:\n",
            "        return 0\n",
            "\n",
            "    tar_path = TEMP_DIR / f\"{patch_key}.tar.gz\"\n",
            "    download_with_progress(patch_url, tar_path)\n",
            "\n",
            "    extracted = 0\n",
            "    with tarfile.open(tar_path, \"r:gz\") as tar:\n",
            "        for member in tar.getmembers():\n",
            "            base_name = Path(member.name).name\n",
            "            if base_name in remaining:\n",
            "                f_obj = tar.extractfile(member)\n",
            "                if f_obj:\n",
            "                    with open(dest_dir / base_name, \"wb\") as out_f:\n",
            "                        shutil.copyfileobj(f_obj, out_f)\n",
            "                    extracted += 1\n",
            "\n",
            "    if tar_path.exists():\n",
            "        tar_path.unlink()  # free disk space immediately\n",
            "\n",
            "    print(f\"  [{patch_key}] Extracted {extracted}/{len(remaining)} duck images. Cleaned up archive.\")\n",
            "    return extracted\n",
            "\n",
            "\n",
            "def write_yolo_labels(duck_images: dict, duck_anns: dict, labels_dir: Path):\n",
            "    \"\"\"Convert COCO bounding boxes to normalized YOLO .txt label format.\"\"\"\n",
            "    labels_dir.mkdir(parents=True, exist_ok=True)\n",
            "    counts = {}\n",
            "    for img_id, img_info in duck_images.items():\n",
            "        base_name = Path(img_info[\"file_name\"]).name\n",
            "        txt_name = Path(base_name).with_suffix(\".txt\").name\n",
            "        w, h = img_info[\"width\"], img_info[\"height\"]\n",
            "        anns = duck_anns.get(img_id, [])\n",
            "        counts[base_name] = len(anns)\n",
            "        lines = []\n",
            "        for a in anns:\n",
            "            x, y, bw, bh = a[\"bbox\"]\n",
            "            xc = max(0.0, min(1.0, (x + bw / 2.0) / w))\n",
            "            yc = max(0.0, min(1.0, (y + bh / 2.0) / h))\n",
            "            wn = max(0.0, min(1.0, bw / w))\n",
            "            hn = max(0.0, min(1.0, bh / h))\n",
            "            lines.append(f\"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\\n\")\n",
            "        with open(labels_dir / txt_name, \"w\", encoding=\"utf-8\") as f:\n",
            "            f.writelines(lines)\n",
            "    return counts"
        ]
    })

    # Markdown Cell 7: Validation Split Processing
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Validation Split (1,547 Duck Images / 12,221 Ducks)\n",
            "\n",
            "The validation annotations `zhiyuan_objv2_val.json` (~269 MB) provide a ready-to-use evaluation split.\n",
            "Let's load the annotations and verify the validation duck dataset."
        ]
    })

    # Code Cell 8: Validation Split Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "val_ann_file = WORK_DIR / \"zhiyuan_objv2_val.json\"\n",
            "download_with_progress(f\"{BASE_URL}/val/zhiyuan_objv2_val.json\", val_ann_file)\n",
            "\n",
            "print(\"Loading validation JSON annotations...\")\n",
            "with open(val_ann_file, \"r\", encoding=\"utf-8\") as f:\n",
            "    val_data = json.load(f)\n",
            "\n",
            "# Filter validation duck annotations\n",
            "val_all_imgs = {img[\"id\"]: img for img in val_data[\"images\"]}\n",
            "val_duck_anns = defaultdict(list)\n",
            "for a in val_data[\"annotations\"]:\n",
            "    if a.get(\"category_id\") == DUCK_CATEGORY_ID:\n",
            "        val_duck_anns[a[\"image_id\"]].append(a)\n",
            "\n",
            "val_duck_images = {img_id: val_all_imgs[img_id] for img_id in val_duck_anns if img_id in val_all_imgs}\n",
            "val_counts = write_yolo_labels(val_duck_images, val_duck_anns, VAL_LABELS)\n",
            "\n",
            "print(f\"=== Validation Split Ready ===\")\n",
            "print(f\"  Total Duck Images:     {len(val_duck_images):,}\")\n",
            "print(f\"  Total Duck Instances:  {sum(val_counts.values()):,}\")\n",
            "print(f\"  YOLO Labels Generated: {len(list(VAL_LABELS.glob('*.txt'))):,}\")"
        ]
    })

    # Markdown Cell 9: Extract Val Images
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.1 Download Validation Duck Images\n",
            "Run this cell to extract duck images for the validation split.\n",
            "You can set `MAX_VAL_PATCHES = None` to extract all 44 validation patches, or set a number (e.g. `5`) to download selectively."
        ]
    })

    # Code Cell 10: Extract Val Images Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "MAX_VAL_PATCHES = None  # None = download all validation patches\n",
            "\n",
            "val_patch_map = defaultdict(lambda: {\"url\": \"\", \"files\": set()})\n",
            "for img_id, img_info in val_duck_images.items():\n",
            "    base_name = Path(img_info[\"file_name\"]).name\n",
            "    p_key, p_url = resolve_patch(img_info[\"file_name\"], split='val')\n",
            "    val_patch_map[p_key][\"url\"] = p_url\n",
            "    val_patch_map[p_key][\"files\"].add(base_name)\n",
            "\n",
            "processed = 0\n",
            "for p_key, p_info in sorted(val_patch_map.items()):\n",
            "    if MAX_VAL_PATCHES and processed >= MAX_VAL_PATCHES:\n",
            "        break\n",
            "    extract_patch_images(p_key, p_info[\"url\"], p_info[\"files\"], VAL_IMAGES)\n",
            "    processed += 1\n",
            "\n",
            "print(f\"Total validation images downloaded: {len(list(VAL_IMAGES.glob('*.jpg')))} / {len(val_duck_images)}\")"
        ]
    })

    # Markdown Cell 11: Training Split Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 4. Training Split: Extracting +5,000 Duck Images\n",
            "\n",
            "The full Objects365 training set contains 1.7 million images across 51 patches (~345 GB).\n",
            "\n",
            "### 🎯 Our Smart Quota Strategy\n",
            "Instead of downloading all 51 training patches:\n",
            "1. We set `TARGET_TRAIN_DUCKS = 5000` (quota of 5,000 duck images).\n",
            "2. We unpack `zhiyuan_objv2_train.tar.gz` (~1.33 GB) and find all training duck images.\n",
            "3. We select the first 5,000 duck images.\n",
            "4. We determine **only the specific patches needed** to fulfill those 5,000 images (usually only ~7 to 8 patches out of 51!).\n",
            "5. We download only those required patches, extract strictly the duck images to `images/train/`, and delete the archive immediately."
        ]
    })

    # Code Cell 12: Download & Parse Train Annotations
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "TARGET_TRAIN_DUCKS = 5000  # Target quota of duck images from training split\n",
            "\n",
            "train_tar_file = WORK_DIR / \"zhiyuan_objv2_train.tar.gz\"\n",
            "train_ann_file = WORK_DIR / \"zhiyuan_objv2_train.json\"\n",
            "\n",
            "# Step 1: Download and unpack training annotations (1.33 GB compressed)\n",
            "if not train_ann_file.exists():\n",
            "    print(\"Downloading training annotations tar.gz (~1.33 GB)... This may take 5-10 minutes.\")\n",
            "    download_with_progress(f\"{BASE_URL}/train/zhiyuan_objv2_train.tar.gz\", train_tar_file)\n",
            "    print(\"Extracting zhiyuan_objv2_train.json from archive...\")\n",
            "    with tarfile.open(train_tar_file, \"r:gz\") as tar:\n",
            "        tar.extractall(path=WORK_DIR)\n",
            "    if train_tar_file.exists():\n",
            "        train_tar_file.unlink()  # free 1.3 GB archive space\n",
            "    print(\"Training annotation JSON is ready!\")\n",
            "else:\n",
            "    print(f\"Training annotation file already exists: {train_ann_file.name}\")"
        ]
    })

    # Markdown Cell 13: Filter & Sample 5000 Train Ducks
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.1 Filter and Select 5,000 Duck Images\n",
            "\n",
            "We parse the training JSON, locate Duck annotations, and select 5,000 duck images."
        ]
    })

    # Code Cell 14: Filter Train Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Loading training JSON annotations (this takes ~30-60s due to file size)...\")\n",
            "with open(train_ann_file, \"r\", encoding=\"utf-8\") as f:\n",
            "    train_data = json.load(f)\n",
            "\n",
            "train_all_imgs = {img[\"id\"]: img for img in train_data[\"images\"]}\n",
            "train_duck_anns = defaultdict(list)\n",
            "for a in train_data[\"annotations\"]:\n",
            "    if a.get(\"category_id\") == DUCK_CATEGORY_ID:\n",
            "        train_duck_anns[a[\"image_id\"]].append(a)\n",
            "\n",
            "all_train_duck_images = {img_id: train_all_imgs[img_id] for img_id in train_duck_anns if img_id in train_all_imgs}\n",
            "print(f\"Total Duck Images available in Train set: {len(all_train_duck_images):,}\")\n",
            "\n",
            "# Select up to TARGET_TRAIN_DUCKS\n",
            "selected_train_duck_images = {}\n",
            "for img_id, img_info in all_train_duck_images.items():\n",
            "    selected_train_duck_images[img_id] = img_info\n",
            "    if len(selected_train_duck_images) >= TARGET_TRAIN_DUCKS:\n",
            "        break\n",
            "\n",
            "selected_instances = sum(len(train_duck_anns[img_id]) for img_id in selected_train_duck_images)\n",
            "print(f\"Selected Quota: {len(selected_train_duck_images):,} Images containing {selected_instances:,} Duck Instances!\")\n",
            "\n",
            "# Generate YOLO labels for training split\n",
            "train_counts = write_yolo_labels(selected_train_duck_images, train_duck_anns, TRAIN_LABELS)\n",
            "print(f\"Wrote {len(train_counts):,} YOLO label files into: {TRAIN_LABELS}\")"
        ]
    })

    # Markdown Cell 15: Selective Extraction of 5,000 Ducks
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 Selectively Extract 5,000 Duck Images from Minimum Patches\n",
            "\n",
            "Notice how few patches we need to download to get 5,000 duck images:"
        ]
    })

    # Code Cell 16: Extract 5000 Ducks Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_patch_map = defaultdict(lambda: {\"url\": \"\", \"files\": set()})\n",
            "for img_id, img_info in selected_train_duck_images.items():\n",
            "    base_name = Path(img_info[\"file_name\"]).name\n",
            "    p_key, p_url = resolve_patch(img_info[\"file_name\"], split='train')\n",
            "    train_patch_map[p_key][\"url\"] = p_url\n",
            "    train_patch_map[p_key][\"files\"].add(base_name)\n",
            "\n",
            "print(f\"5,000 duck images are contained in only {len(train_patch_map)} patches (out of 51 total!):\")\n",
            "for p_key in sorted(train_patch_map.keys()):\n",
            "    print(f\"  {p_key}: {len(train_patch_map[p_key]['files'])} duck images needed\")\n",
            "\n",
            "# Download and extract selectively\n",
            "print(\"\\nStarting selective extraction for training patches...\")\n",
            "total_extracted = 0\n",
            "for p_key, p_info in sorted(train_patch_map.items()):\n",
            "    print(f\"\\nProcessing {p_key}...\")\n",
            "    n = extract_patch_images(p_key, p_info[\"url\"], p_info[\"files\"], TRAIN_IMAGES)\n",
            "    total_extracted += n\n",
            "\n",
            "print(f\"\\nExtraction finished! Total training duck images: {len(list(TRAIN_IMAGES.glob('*.jpg'))):,}\")"
        ]
    })

    # Markdown Cell 17: Duck Counting Distribution Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Duck Counting Analysis & Density Distribution\n",
            "\n",
            "Duck counting requires training on diverse densities (isolated ducks vs dense flocks).\n",
            "Let's visualize the distribution of ducks per image across our combined dataset."
        ]
    })

    # Code Cell 18: Distribution Plot Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "all_counts = list(val_counts.values()) + list(train_counts.values())\n",
            "total_ducks = sum(all_counts)\n",
            "total_imgs = len(all_counts)\n",
            "\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.hist(all_counts, bins=range(1, min(60, max(all_counts) + 2)), color='#0284c7', edgecolor='black', alpha=0.85)\n",
            "plt.title(f\"Duck Counting Distribution (Total: {total_ducks:,} Ducks across {total_imgs:,} Images)\", fontsize=13)\n",
            "plt.xlabel(\"Ducks in Image\", fontsize=11)\n",
            "plt.ylabel(\"Image Count\", fontsize=11)\n",
            "plt.grid(axis='y', linestyle='--', alpha=0.6)\n",
            "plt.show()\n",
            "\n",
            "print(f\"=== Combined Duck Counting Dataset Statistics ===\")\n",
            "print(f\"Total Images:            {total_imgs:,}\")\n",
            "print(f\"Total Duck Annotations:  {total_ducks:,}\")\n",
            "print(f\"Average Ducks per Image: {total_ducks / total_imgs:.2f}\")\n",
            "print(f\"Max Ducks in One Image:  {max(all_counts)}\")\n",
            "print(f\"1 duck:     {sum(1 for c in all_counts if c == 1):5d} images\")\n",
            "print(f\"2 ducks:    {sum(1 for c in all_counts if c == 2):5d} images\")\n",
            "print(f\"3-5 ducks:  {sum(1 for c in all_counts if 3 <= c <= 5):5d} images\")\n",
            "print(f\"6-10 ducks: {sum(1 for c in all_counts if 6 <= c <= 10):5d} images\")\n",
            "print(f\"10+ ducks:  {sum(1 for c in all_counts if c > 10):5d} images\")"
        ]
    })

    # Markdown Cell 19: Visualizing Samples
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Visualize Duck Annotations with Ground Truth Counts\n",
            "\n",
            "Let's display sample duck images overlaying ground truth bounding boxes and count badges."
        ]
    })

    # Code Cell 20: Visualizing Samples Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_files = list(VAL_IMAGES.glob(\"*.jpg\"))[:3]\n",
            "if not sample_files:\n",
            "    sample_files = list(TRAIN_IMAGES.glob(\"*.jpg\"))[:3]\n",
            "\n",
            "if sample_files:\n",
            "    fig, axes = plt.subplots(1, len(sample_files), figsize=(16, 6))\n",
            "    if len(sample_files) == 1:\n",
            "        axes = [axes]\n",
            "\n",
            "    for ax, img_path in zip(axes, sample_files):\n",
            "        img = cv2.imread(str(img_path))\n",
            "        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "        h, w = img.shape[:2]\n",
            "        lbl_path = VAL_LABELS / img_path.with_suffix(\".txt\").name\n",
            "        if not lbl_path.exists():\n",
            "            lbl_path = TRAIN_LABELS / img_path.with_suffix(\".txt\").name\n",
            "\n",
            "        cnt = 0\n",
            "        if lbl_path.exists():\n",
            "            with open(lbl_path, \"r\") as f:\n",
            "                for line in f:\n",
            "                    parts = line.strip().split()\n",
            "                    if parts:\n",
            "                        cnt += 1\n",
            "                        xc, yc, bw, bh = map(float, parts[1:])\n",
            "                        x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)\n",
            "                        x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)\n",
            "                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (34, 197, 94), 2)\n",
            "\n",
            "        ax.imshow(img_rgb)\n",
            "        ax.set_title(f\"{img_path.name}\\nGround Truth: {cnt} Ducks\", fontsize=12, fontweight='bold')\n",
            "        ax.axis(\"off\")\n",
            "    plt.tight_layout()\n",
            "    plt.show()"
        ]
    })

    # Markdown Cell 21: duck.yaml config
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Generate `duck.yaml` Dataset Configuration\n",
            "\n",
            "Configured with both `train` (5,000 images) and `val` (1,547 images)."
        ]
    })

    # Code Cell 22: duck.yaml config code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "yaml_content = f\"\"\"# Ultralytics YOLO Duck Counting Dataset\n",
            "path: {WORK_DIR.as_posix()}\n",
            "train: images/train  # 5,000 training images\n",
            "val: images/val      # 1,547 validation images\n",
            "\n",
            "names:\n",
            "  0: duck\n",
            "\"\"\"\n",
            "\n",
            "yaml_file = WORK_DIR / \"duck.yaml\"\n",
            "with open(yaml_file, \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(yaml_content)\n",
            "\n",
            "print(f\"Generated: {yaml_file}\")\n",
            "print(yaml_content)"
        ]
    })

    # Markdown Cell 23: YOLO Training & Counting
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Train YOLO Model & Run Duck Counting Inference\n",
            "\n",
            "Train with the Ultralytics CLI or Python API:\n",
            "```bash\n",
            "pip install ultralytics\n",
            "yolo train data=duck.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=16 name=duck_counter\n",
            "```\n",
            "\n",
            "Use the function below for counting ducks in test images or videos."
        ]
    })

    # Code Cell 24: YOLO Training & Counting Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def predict_and_count_ducks(image_path: str, model_weights: str = 'yolo11n.pt', conf: float = 0.25):\n",
            "    \"\"\"\n",
            "    Load model, detect all ducks, and return the exact duck count with visualization.\n",
            "    \"\"\"\n",
            "    try:\n",
            "        from ultralytics import YOLO\n",
            "        model = YOLO(model_weights)\n",
            "        results = model(image_path, conf=conf)[0]\n",
            "        \n",
            "        duck_count = len(results.boxes)\n",
            "        print(f\"Image: {image_path} -> Counted {duck_count} ducks!\")\n",
            "        \n",
            "        # Annotated image with bounding boxes\n",
            "        annotated_img = results.plot()\n",
            "        return duck_count, annotated_img\n",
            "    except ImportError:\n",
            "        print(\"Ultralytics is not installed. Run: pip install ultralytics\")\n",
            "        return None, None\n",
            "\n",
            "print(\"Duck counting inference function ready!\")"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (.venv - Duck Counting)",
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
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = Path("d:/DataNLP/duck/duck_counting_dataset.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook successfully updated at: {out_path}")

if __name__ == "__main__":
    create_notebook()
