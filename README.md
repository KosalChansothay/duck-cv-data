# 🦆 Automatic Duck Counting System for Smallholder Farms
### *Digital Image Processing & Computer Vision for Cambodian Poultry Farming*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-11%20%7C%20v8-00FFFF.svg)](https://docs.ultralytics.com/)

> **Project Overview**: An automated computer vision and digital image processing pipeline for poultry flock detection, spatial distribution analysis, and density counting on smallholder duck farms.

---

## 🌾 Real-World Motivation: Empowering Smallholder Duck Farmers

Duck farming is a cornerstone of household livelihoods and food security in rural Cambodia, especially in lowland poultry-producing provinces such as **Takeo, Kampong Cham**, and surrounding rural regions.

### The Problem
* **Labor-Intensive & Error-Prone**: Farmers typically manage flocks numbering in the hundreds or thousands. Counting them manually by eye during feeding, health checks, or market sales is slow and inaccurate.
* **Fast Movement & Clustering**: Ducks constantly move, overlap, and cluster together on ground and water, leading to severe under- and over-counting.
* **Lack of Accessible Tools**: Smallholder farmers do not have access to expensive drone swarms or high-end industrial vision hardware.

### The Objective
Develop an accessible, fast, and automated desktop counting tool that smallholder farmers and agricultural technicians can run on a standard laptop with photos or videos taken from a smartphone or pole-mounted camera (top-down or near-top-down perspective), targeting **at least 90% counting accuracy** ($\ge 90\%$).

---

## 🎯 Bridging the Research Gap: Classical DIP vs. Modern YOLO

In traditional automated animal counting, the main barrier preventing the adoption of Deep Learning (YOLO) was the **absence of a curated, labeled duck dataset** and the perception that deep learning required cloud GPUs. 

This repository **solves both challenges**:
1. **Curated Farm-Realistic Dataset**: Dense annotations extracted directly from **Objects365 v2** (featuring multi-duck flocks with up to 99 ducks per image) and **ImageNet**.
2. **Ultra-Lightweight Edge Architecture**: Fine-tuned **YOLO11n** (2.6M parameters, ~5 MB) and **YOLO11s** run in real-time (>30 FPS) on a standard laptop CPU without needing a GPU or internet connection.
3. **Dual-Track Benchmarking**: Enables a side-by-side comparative study between **Classical DIP** (Otsu + Watershed) and **Deep Learning** (YOLO11), evaluating trade-offs in accuracy, occlusions, and compute efficiency.

```
                              [Captured Farm Image]
                     (Top-down Smartphone / Pole Camera)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   [Track A: Classical DIP]                             [Track B: Deep Learning]
   • Image Sampling & Resolution Reduction              • Pretrained YOLO11 Backbone
   • 8-Adjacency & Connected Components                 • Trained on Objects365 Duck Split
   • HSV Color Thresholding (Otsu)                      • Resilient to Muddy Water & Occlusion
   • Distance Transform + Watershed Segmentation        • Outputs Bounding Boxes & Count
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        [Evaluation & Comparison]
              • Precision, Recall, and Counting Accuracy (≥ 90%)
              • Mean Absolute Error (MAE) & Root Mean Square Error (RMSE)
                                      │
                                      ▼
                            [Java Desktop GUI]
                   Export to ONNX (Offline Desktop Deployment)
```

---

## 📊 Dataset Overview

This project uses a targeted dataset built specifically for high-density duck counting under varied outdoor farm lighting (morning, midday, evening) over soil, grass, and pond water:

| Split / Source | Images | Instances (Ducks) | Avg Ducks / Image | Annotation Type |
| :--- | :---: | :---: | :---: | :--- |
| **Objects365 v2 (Duck ID 137)** | **591+** | **4,908+** | **8.3** (up to 99) | Dense BBoxes `(0 x y w h)` |
| **Objects365 Total Available** | 1,547 | 12,221 | 7.9 | Dense BBoxes |
| **ImageNet (Index 97: Drake)** | 1,264 URLs | Single-object | 1.0 | Classification / Synset |

### Density Distribution in Extracted Farm Scenes
* **Single / Pairs (1–2 ducks)**: ~15%
* **Small Groups (3–5 ducks)**: ~25%
* **Medium Flocks (6–10 ducks)**: ~35%
* **Dense Flocks (10–99 ducks)**: ~25%

---

## 🗓️ 10-Week Implementation Roadmap

| Week | Phase | Activity & Focus | Deliverables |
| :---: | :---: | :--- | :--- |
| **Wk 1** | Scoping | Problem formulation, requirements, environment setup | Architecture plan |
| **Wk 2** | Acquisition | Farm image acquisition and dataset structuring | Data pipeline established |
| **Wk 3** | Preprocessing | Noise reduction (Gaussian/median filter), HSV color space | Color separation baseline |
| **Wk 4** | Detection | Segmentation: Otsu thresholding & 8-connected components | Initial blob detection |
| **Wk 5** | Refinement | Cluster separation & morphological cleanup | Intermediate validation |
| **Wk 6** | Clustered DIP | Distance transform ($D_e$) & Watershed segmentation | Watershed separation |
| **Wk 7** | Deep Learning | Fine-tune YOLO11 on curated duck dataset | Trained detector checkpoint |
| **Wk 8** | Benchmarking | Multi-condition testing (Otsu vs. YOLO11) | Accuracy evaluation ($\ge 90\%$) |
| **Wk 9** | Deployment | Java Desktop GUI development & ONNX model integration | Offline Desktop App |
| **Wk 10** | Release | Full system validation, documentation, and reporting | Final codebase & benchmarks |

---

## 📁 Repository Structure

```text
duck/
├── configs/                         # Training & dataset configurations
│   ├── duck.yaml                    # YOLO dataset definition (path, train, val, classes)
│   └── train.yaml                   # Hyperparameters (epochs, batch, imgsz, lr, mosaic)
│
├── notebooks/                       # Step-by-step interactive Jupyter Notebooks
│   ├── 01_duck_counting_dataset.ipynb       # Objects365 duck validation benchmark & exploration
│   ├── 02_extract_imagenet_duck.ipynb       # ImageNet Index 97 (Drake / Mallard) synset extractor
│   └── 03_train_and_count.ipynb             # End-to-end model training, evaluation & counting demo
│
├── scripts/                         # Command-line entry points
│   ├── train.py                     # Train YOLO detector & counter
│   ├── evaluate.py                  # Dual evaluation (Detection mAP + Counting MAE/RMSE)
│   ├── count.py                     # Live duck counting on single image or folder
│   ├── extract_objects365.py        # Stream-based Objects365 duck extractor
│   └── extract_imagenet.py          # Multithreaded ImageNet downloader
│
├── src/                             # Modular Python package
│   ├── data/                        # Dataset extraction, YOLO bboxes, integrity checks
│   │   ├── objects365.py            # Stream-based tar extractor with early exit
│   │   ├── imagenet.py              # Multithreaded ImageNet downloader
│   │   └── dataset.py               # BBox normalization & split verification
│   ├── models/                      # Deep learning & counting logic
│   │   ├── trainer.py               # Ultralytics YOLO training wrapper
│   │   ├── evaluator.py             # Dual detection + counting metric evaluator
│   │   └── counter.py               # High-level DuckCounter inference engine
│   └── utils/                       # Metrics, visualization & configuration
│       ├── visualizer.py            # Bounding box & counting banner overlay
│       ├── metrics.py               # MAE, RMSE, exact match, counting accuracy
│       └── config.py                # YAML loader
│
├── images/                          # Extracted high-resolution duck images
│   └── val/                         # 591+ verified duck images
│
├── labels/                          # YOLO format labels (0 x_center y_center w h)
│   └── val/                         # 1,547 label files (12,221 instances)
│
├── reports/                         # Dataset metrics and summary reports
│   └── duck_counting_val_summary.json
│
├── pyproject.toml                   # Standard packaging (pip install -e .)
└── requirements.txt                 # Project dependencies
```

---

## ⚡ Quick Start

### 1. Environment Setup
Using `uv` (recommended) or standard `pip`:

```bash
# Clone the repository
git clone https://github.com/KosalChansothay/duck-cv-data.git
cd duck-cv-data

# Create and activate virtual environment
uv venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e .
```

---

## 🚀 Running the System

### 1. Verify Dataset Integrity
Before training, verify that images and YOLO label files match:
```bash
python -c "from src.data import verify_dataset_integrity; import pprint; pprint.pprint(verify_dataset_integrity('.'))"
```

### 2. Train the YOLO Duck Counting Model
Fine-tune a lightweight YOLO11 model using the configuration in [`configs/train.yaml`](file:///d:/DataNLP/duck/configs/train.yaml):

```bash
# Standard training (auto GPU/CPU)
python scripts/train.py --config configs/train.yaml

# Fast baseline test (30 epochs, YOLO11 nano)
python scripts/train.py --model yolo11n.pt --epochs 30 --batch 16 --imgsz 640
```

### 3. Evaluate Counting Accuracy ($\ge 90\%$ Metric)
Evaluate both **Object Detection mAP** and **Crowd Counting Error**:

```bash
python scripts/evaluate.py --weights runs/train/duck_counter/weights/best.pt --split val
```

This outputs:
* **Detection Metrics**: Precision, Recall, $\text{mAP@50}$, $\text{mAP@50-95}$
* **Counting Metrics**:
  * $\text{Mean Absolute Error (MAE)} = \frac{1}{N}\sum |Count_{\text{pred}} - Count_{\text{gt}}|$
  * $\text{Root Mean Square Error (RMSE)}$
  * $\text{Average Counting Accuracy} = 1 - \frac{|Count_{\text{pred}} - Count_{\text{gt}}|}{Count_{\text{gt}}}$
  * $\text{Exact Match Rate}$ (images where predicted count equals ground truth exactly)

### 4. Count Ducks in Any Farm Image
Run inference on a single image or an entire folder of photos:

```bash
# Count ducks in a single photograph
python scripts/count.py --weights runs/train/duck_counter/weights/best.pt --source test_farm_duck.jpg --save

# Count ducks across a folder
python scripts/count.py --weights runs/train/duck_counter/weights/best.pt --source images/val/ --conf 0.35 --save
```

---

## ☕ Java Desktop Application Integration (Offline Deployment)

To build a standalone **Java Desktop Application (Swing / JavaFX)** that smallholder farmers can run offline without Python:

### Step 1: Export Trained Model to ONNX
```bash
yolo export model=runs/train/duck_counter/weights/best.pt format=onnx imgsz=640
```
This produces `best.onnx` (~6 MB for YOLO11n).

### Step 2: Load Model in Java via OpenCV (JavaCV)
In your Java desktop application, load and run the model using OpenCV's DNN module:

```java
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.Imgcodecs;

public class DuckCounterApp {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        // 1. Load exported ONNX model
        Net net = Dnn.readNetFromONNX("models/best.onnx");
        
        // 2. Read farm image
        Mat image = Imgcodecs.imread("farm_photo.jpg");
        
        // 3. Preprocess image into 640x640 blob
        Mat blob = Dnn.blobFromImage(image, 1.0 / 255.0, new Size(640, 640), new Scalar(0), true, false);
        net.setInput(blob);
        
        // 4. Run forward inference
        Mat output = net.forward();
        
        // 5. Parse bounding boxes and count total ducks!
        int duckCount = parseDetections(output, 0.40); // 40% confidence threshold
        System.out.println("Total Ducks Counted: " + duckCount);
    }
}
```

---

## 🔬 Classical DIP vs. YOLO Comparison

### Performance Comparison Matrix

| Challenge | Classical Pipeline (Otsu + Watershed) | Deep Learning Pipeline (YOLO11) |
| :--- | :--- | :--- |
| **Clustered / Touching Ducks** | Watershed often over-segments single ducks or merges touching bodies | Bounding boxes cleanly resolve overlapping instances with NMS |
| **Muddy Water / Grass Reflection** | Color thresholding (HSV) picks up muddy soil or vegetation of similar hue | Convolutional feature extraction distinguishes duck texture from pond water |
| **Variable Outdoor Sunlight** | Harsh shadows cause thresholding artifacts | Mosaic augmentation trains the model to be illumination-invariant |
| **Speed & Edge Usability** | Multi-pass morphological filtering takes 1–3s per image | YOLO11n runs in ~25ms on CPU (real-time >30 FPS) |

---

## 📚 References Cited

1. **Verykokou, S., & Ioannidis, C. (2023)**. An overview on image-based and scanner-based 3D modeling technologies. *Sensors*, 23(2), 596.
2. **Chabot, D., & Francis, C. M. (2016)**. Computer-automated bird detection and counts in high-resolution aerial images: a review. *Journal of Field Ornithology*, 87(4), 343–359.
3. **Cao, L., Xiao, Z., Liao, X., et al. (2021)**. Automated chicken counting in surveillance camera environments based on point supervision: LC-DenseFCN. *Agriculture*, 11(6), 493.
4. **Gonzalez, R. C., & Woods, R. E. (2008)**. *Digital Image Processing* (3rd ed.). Prentice Hall.
5. **Otsu, N. (1979)**. A threshold selection method from gray-level histograms. *IEEE TSMC*, 9(1), 62–66.
6. **Vincent, L., & Soille, P. (1991)**. Watersheds in digital spaces: an efficient algorithm based on immersion simulations. *IEEE TPAMI*, 13(6), 583–598.
7. **Birhanu, M. Y., et al. (2021)**. Poultry production, marketing and consumption in Cambodia: a review of literature. *ILRI Research Report 81*, Nairobi, Kenya.

---

## 🤝 Contributing & Support

Contributions, issues, and feature requests are welcome! For questions or fine-tuning assistance, feel free to open an issue or submit a pull request.
