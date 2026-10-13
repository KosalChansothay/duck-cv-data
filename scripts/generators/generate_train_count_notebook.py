import json
from pathlib import Path

def create_train_count_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#  Duck Counting: YOLO Model Training & Counting Pipeline\n",
                "\n",
                "This notebook provides the complete end-to-end workflow to:\n",
                "1. **Verify Dataset Integrity**: Validate YOLO bounding boxes and image-label pairs.\n",
                "2. **Train YOLO Detector**: Fine-tune a YOLOv8 / YOLO11 model on your duck dataset using `configs/train.yaml`.\n",
                "3. **Evaluate Counting Accuracy**: Compute both detection mAP and direct duck counting metrics (MAE, RMSE, exact match rate).\n",
                "4. **Run Live Inference & Counting**: Test single images and folders with visual count overlays."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Ensure project root is in python path\n",
                "WORK_DIR = Path('.').resolve()\n",
                "if str(WORK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORK_DIR))\n",
                "\n",
                "import cv2\n",
                "import matplotlib.pyplot as plt\n",
                "from src.data import verify_dataset_integrity\n",
                "from src.models import train_duck_model, evaluate_duck_model, DuckCounter\n",
                "from src.utils import plot_count_distribution, load_yaml_config\n",
                "\n",
                "print(f'Project root: {WORK_DIR}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Verify Dataset Integrity\n",
                "\n",
                "Before launching training, we verify that image and label files match 1-to-1 without coordinate errors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "integrity = verify_dataset_integrity(WORK_DIR)\n",
                "for split, info in integrity.items():\n",
                "    print(f'\\n[{split.upper()} SPLIT]')\n",
                "    print(f'  • Total Images:   {info[\"total_images\"]:,}')\n",
                "    print(f'  • Total Labels:   {info[\"total_labels\"]:,}')\n",
                "    print(f'  • Matched Pairs:  {info[\"matched_pairs\"]:,}')\n",
                "    print(f'  • Total Boxes:    {info.get(\"total_boxes\", 0):,}')\n",
                "    print(f'  • Integrity Valid: {info.get(\"is_valid\", False)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Train YOLO Duck Counting Model\n",
                "\n",
                "Trains YOLO using the hyperparameters configured in `configs/train.yaml`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute model training\n",
                "# train_results = train_duck_model(\n",
                "#     config_path='configs/train.yaml',\n",
                "#     model_name='yolo11s.pt',\n",
                "#     epochs=50,\n",
                "#     batch=16,\n",
                "#     imgsz=640,\n",
                "#     device=0\n",
                "# )\n",
                "print('Ready to train. Hyperparameters loaded from configs/train.yaml.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Comprehensive Evaluation (Detection & Counting)\n",
                "\n",
                "Evaluates both object detection (mAP@50) and crowd counting metrics (Mean Absolute Error)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "weights_path = WORK_DIR / 'runs/train/duck_counter/weights/best.pt'\n",
                "if not weights_path.exists():\n",
                "    weights_path = 'yolo11s.pt'  # Base pretrained weights fallback\n",
                "\n",
                "print(f'Evaluating weights: {weights_path}')\n",
                "# eval_summary = evaluate_duck_model(weights_path=weights_path, split='val', max_eval_images=50)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Interactive Duck Counting Inference\n",
                "\n",
                "Run `DuckCounter` on a sample image and view the visual count banner."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "counter = DuckCounter(weights='yolo11s.pt', conf=0.25, iou=0.45)\n",
                "sample_images = list((WORK_DIR / 'images/val').glob('*.jpg'))\n",
                "if sample_images:\n",
                "    test_img = sample_images[0]\n",
                "    result = counter.count_image(test_img, return_annotated=True)\n",
                "    print(f'Image: {test_img.name}')\n",
                "    print(f'Ducks Counted: {result.count}')\n",
                "    if result.annotated_image is not None:\n",
                "        plt.figure(figsize=(10, 8))\n",
                "        plt.imshow(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB))\n",
                "        plt.axis('off')\n",
                "        plt.title(f'Ducks Counted: {result.count}', fontsize=14, fontweight='bold')\n",
                "        plt.show()\n",
                "else:\n",
                "    print('No sample images found in images/val')"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_p = Path("notebooks/04_train_and_count.ipynb")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {out_p}")

if __name__ == "__main__":
    create_train_count_notebook()
