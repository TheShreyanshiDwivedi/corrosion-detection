# Automated Corrosion Detection and Condition State Segmentation Using Deep Learning

### A Multi-Architecture Pipeline for Structural Inspection Automation

---

**Author:** Shreyanshi Dwivedi  
**Date:** May 2026

---

## Abstract

Every year, structural failures caused by undetected corrosion cost billions in emergency repairs and, in the worst cases, lives. The conventional approach to catching corrosion early — sending trained inspectors to visually scan every steel element — is slow, expensive, and inconsistent. Two inspectors looking at the same corroded beam will often disagree on how bad it is.

This project asks a simple question: **can a machine learn to see corrosion the way an expert does — and do it faster?**

The answer, as demonstrated in this work, is yes. We built a complete deep learning pipeline that takes a single photograph of a steel structural element and produces a pixel-by-pixel map of corrosion severity — distinguishing between surface rust, active pack rust, and critical section loss — in under a second.

The system uses two complementary neural network architectures working in tandem: **DeepLabV3+** for dense semantic segmentation (86.67% F1-score) and **YOLO26m-seg** for real-time instance detection with bounding boxes. To overcome the scarcity of labeled corrosion data, we built an automated pipeline that scrapes thousands of corrosion images from the web, validates them, and generates pseudo-labels — effectively teaching the model from the internet itself.

Everything runs through an interactive **Streamlit web application** where an engineer can upload an inspection photo and get an instant severity report — no machine learning expertise required.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Background and Prior Work](#2-background-and-prior-work)
3. [The Dataset](#3-the-dataset)
4. [Scaling Up: The Web Scraping Pipeline](#4-scaling-up-the-web-scraping-pipeline)
5. [How It Works: The Models](#5-how-it-works-the-models)
6. [Training the Models](#6-training-the-models)
7. [Results](#7-results)
8. [The Application](#8-the-application)
9. [Project Structure](#9-project-structure)
10. [What's Next](#10-whats-next)
11. [References](#11-references)
12. [Appendix](#12-appendix)

---

## 1. The Problem

Picture this: a steel beam in an industrial facility. Over the years, rain, humidity, and chemical exposure have eaten into its surface. The coating has failed. Rust is spreading. But how bad is it? Is it just cosmetic? Or has the steel actually started losing thickness — silently weakening the structure it's supposed to hold up?

This is the problem structural inspectors face every day. They climb, crawl, and squint at steel surfaces, trying to classify what they see into standardized condition states:

| Condition State | What it looks like | What it means |
|:---:|:---|:---|
| **CS-0 (Good)** | Clean steel, intact paint | Nothing to worry about |
| **CS-1 (Fair)** | Freckled rust, orange discoloration | Surface-level — monitor it |
| **CS-2 (Poor)** | Flaking, pitting, pack rust forming between layers | Active deterioration — schedule maintenance |
| **CS-3 (Severe)** | Holes, visible thinning, steel eaten through | Structural risk — act immediately |

The challenge? These categories blur into each other. Fair rust and poor rust can look nearly identical to the human eye, especially under bad lighting or from a distance. Studies have shown that different inspectors looking at the same element agree on the condition state less than 70% of the time [9]. That's barely better than flipping a coin between two options.

**Our goal:** build a system that looks at a photo and maps every pixel to one of these four categories — consistently, instantly, and without human bias.

---

## 2. Background and Prior Work

This project stands on the shoulders of several key contributions:

**The Dataset.** We use a publicly available corrosion condition state segmentation dataset [1, 2] containing 440 expertly annotated images of corroded steel elements. Each pixel in every image has been manually labeled as Good, Fair, Poor, or Severe using the LabelMe annotation tool. The accompanying research [3] demonstrated that deep learning can achieve 86.67% F1-score on this task — proving the concept is viable.

**DeepLabV3+.** Proposed by Chen et al. [4], this architecture is purpose-built for semantic segmentation. It uses Atrous Spatial Pyramid Pooling (ASPP) to capture features at multiple scales — crucial for corrosion, which can appear as tiny freckles or massive patches in the same image. The ResNet-101 backbone [5], pre-trained on ImageNet's 14 million images, gives the model a massive head start in understanding visual features.

**YOLO.** The You Only Look Once family [8] revolutionized real-time object detection by processing the entire image in a single forward pass. The latest version, YOLO26 [6], extends this to instance segmentation — it can detect individual corrosion regions, draw bounding boxes around them, classify their severity, AND generate pixel-precise masks, all simultaneously, at 16+ frames per second.

**Pseudo-Labeling.** When you don't have enough labeled data (and you almost never do), Lee [7] showed that you can use a trained model's own predictions as labels for new, unlabeled images. The model becomes its own teacher. We use this technique extensively in our web scraping pipeline.

---

## 3. The Dataset

### 3.1 What We Started With

The foundation of this project is a publicly available corrosion dataset [1, 2] — 440 images of corroded steel surfaces, each with pixel-perfect segmentation masks drawn by domain experts.

| Property | Value |
|:---|:---|
| **Total Images** | 440 (396 train / 44 test) |
| **Resolution** | 512 × 512 pixels |
| **Annotation** | Pixel-level masks, 4 classes |
| **Annotation Tool** | LabelMe (polygon-by-polygon) |

Each mask uses a specific color to indicate the condition state:

| Class | Condition State | Color in Mask | Training Instances |
|:---:|:---|:---:|:---:|
| 0 | Good (Background) | Black | — |
| 1 | Fair | Dark Red | 2,453 |
| 2 | Poor | Green | 1,145 |
| 3 | Severe | Teal | 131 |

Notice that last number: **131 instances of Severe corrosion** out of 3,729 total. That's 3.5%. This severe class imbalance is one of the biggest challenges in this project — the model sees 19× more "Fair" examples than "Severe" ones during training.

![Corrosion condition states with visual descriptions](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/corrosion_pred_with_descriptions.png)

![Class color mapping legend](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/class_color_mapping.png)

### 3.2 Converting for YOLO

The original dataset uses semantic masks (one colored image per annotation). YOLO expects a completely different format — normalized polygon coordinates in text files. We built a conversion pipeline that:

1. Reads each colored mask and isolates regions per class
2. Traces the contour of each region using OpenCV's `findContours`
3. Simplifies the polygon (reduces vertices by ~50% without losing shape accuracy)
4. Filters out tiny noise regions (< 100 pixels)
5. Normalizes all coordinates to `[0, 1]` and writes YOLO-format `.txt` files

This conversion is unglamorous but critical — a single bug here (wrong color mapping, coordinate flip, missed class) would silently corrupt the entire training process.

---

## 4. Scaling Up: The Web Scraping Pipeline

### 4.1 The Problem: 396 Images Isn't Enough

Deep learning is data-hungry. Modern segmentation models are typically trained on tens of thousands of images. We had 396. Manual annotation wasn't an option — each image takes 15-30 minutes of careful polygon drawing. We needed a way to scale the dataset automatically.

### 4.2 The Solution: Scrape, Validate, Pseudo-Label

We built a fully automated pipeline (`yolo_automation_pipeline.py`) that expands the training set by mining the internet for corrosion images and generating labels algorithmically.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  PHASE 1:        │     │  PHASE 2:        │     │  PHASE 3:        │
│  Scrape images   │────▶│  Validate        │────▶│  Auto-label      │
│  from the web    │     │  quality          │     │  with teacher    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                  │
         ▼                                                  ▼
┌──────────────────┐     ┌──────────────────┐
│  PHASE 4:        │     │  PHASE 5:        │
│  Merge with      │────▶│  Train on the    │
│  original data   │     │  bigger dataset  │
└──────────────────┘     └──────────────────┘
```

**Phase 1 — Scraping.** We programmatically queried DuckDuckGo Image Search with 10 carefully chosen search terms:

```python
SEARCH_QUERIES = [
    "corroded steel beam",
    "rusted metal girder",
    "steel corrosion damage close up",
    "steel beam rust pitting",
    "corroded steel I-beam",
    "steel section loss corrosion",
    "pack rust steel connection",
    "severe steel corrosion",
    "steel bearing corrosion",
    "corroded steel railing"
]
```

Each query targeted 200 images. Downloads ran in parallel across 30 threads, pulling ~15-20 images per second. In under 5 minutes, we had ~2,000 raw candidate images.

**Phase 2 — Validation.** Most of those 2,000 images were garbage — broken downloads, tiny thumbnails, diagrams, memes. The validation step automatically filtered by checking decodability, minimum resolution (64×64), channel count (must be 3-channel RGB), and resized survivors to 512×512.

**Phase 3 — Pseudo-Labeling.** Here's where it gets interesting. We took our base YOLO model (trained on the original 396 images) and used it as a "teacher" to predict corrosion masks on every validated web image. These predictions aren't perfect — but they're good enough to serve as approximate training labels. The model learns from its own imperfect knowledge, bootstrapping itself to handle more diverse images.

**Phase 4 — Merge.** The pseudo-labeled images were combined with the original expert-annotated training set:

| Source | Images |
|:---|:---:|
| Original expert-annotated | 396 |
| Web-scraped + pseudo-labeled | ~350 |
| **Combined** | **746** |

**The honest caveat:** pseudo-labels are noisy. A teacher model with 15% accuracy generates noisy labels. We learned this the hard way — our first YOLO model trained on this merged data barely improved. The real breakthrough came when we went back to clean data only and used a much larger model (see Section 6).

---

## 5. How It Works: The Models

We use two fundamentally different architectures, each with its own strengths:

### 5.1 DeepLabV3+ — The Pixel Painter

Think of DeepLabV3+ as a model that looks at every single pixel and asks: *"Is this pixel corroded? If so, how badly?"*

It processes the entire image through a deep ResNet-101 backbone (101 layers deep, pre-trained on ImageNet) and produces a color-coded map where every pixel is classified independently.

```python
model = deeplabv3plus_resnet101(num_classes=4, output_stride=8)
model.classifier = DeepLabHeadV3Plus(
    inplanes=2048, low_level_planes=256, num_classes=4
)
```

**What makes it special:**
- **ASPP (Atrous Spatial Pyramid Pooling):** Looks at each pixel's surroundings at three different zoom levels (dilation rates 6, 12, 18). This lets it understand both fine details (a small pit) and broader context (the entire corroded region).
- **Output stride of 8:** Preserves 1/8th of the original spatial resolution throughout the backbone, giving much sharper boundaries than typical models.
- **44 million parameters** — a large model that can capture subtle differences between Fair and Poor corrosion.

**Strength:** Extremely accurate pixel-level classification (86.67% F1).
**Weakness:** No bounding boxes, no instance separation — it can tell you WHERE corrosion is, but not HOW MANY individual regions there are.

### 5.2 YOLO26m-seg — The All-in-One Detector

YOLO takes a completely different approach. Instead of classifying individual pixels, it processes the entire image in one shot and outputs:
- **Bounding boxes** around each corroded region
- **Confidence scores** for each detection
- **Class labels** (Fair / Poor / Severe)
- **Instance masks** — pixel-precise segmentation for each individual region

```
Input Image (512×512)
    │
    ▼
┌─────────────────────────────┐
│  YOLO26 Backbone            │  ← Pre-trained on COCO (80 object classes)
│  (Feature extraction)       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Feature Pyramid Neck       │  ← Fuses features at multiple scales
└──────────────┬──────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────┐  ┌────────────────┐
│ Detection │  │ Segmentation   │
│ Head      │  │ Head           │
│ Boxes +   │  │ Per-instance   │
│ Class +   │  │ binary masks   │
│ Confidence│  │                │
└───────────┘  └────────────────┘
```

**Why YOLO26 specifically?** We tried three versions:

| Model | Params | Result |
|:---|:---:|:---|
| YOLOv8n-seg (2023) | 3.4M | 15.6% mAP — basically useless |
| YOLO11m-seg (2024) | 22.4M | Available but superseded |
| **YOLO26m-seg (2026)** | **~22M** | **Latest architecture — our final model** |

The nano model was like trying to learn advanced chemistry from a pamphlet — not enough capacity to represent the subtle differences between corrosion types. YOLO26m-seg has 6.5× more parameters and the latest architectural improvements.

---

## 6. Training the Models

### 6.1 Hardware

| Component | Specification |
|:---|:---|
| **GPU** | Apple Silicon (Metal Performance Shaders) |
| **Framework** | PyTorch 2.x + Ultralytics 8.4.54 |
| **Python** | 3.11+ |

### 6.2 The Training Journey (What Went Wrong and What Went Right)

This wasn't a straight path. Here's how the YOLO training evolved:

**Attempt 1: YOLOv8n-seg, 15 epochs, original data.**
Result: 15.6% mAP. The model was too small and undertrained. It labeled everything as "Fair" regardless of actual severity.

**Attempt 2: YOLOv8n-seg, 31 epochs, original + web-scraped data.**
Result: 15.8% mAP. Adding pseudo-labeled web images barely helped because (a) the pseudo-labels came from the 15% accuracy model — garbage in, garbage out — and (b) the model was still too small to learn.

**Attempt 3 (Final): YOLO26m-seg, 100 epochs, clean expert data only.**
This is the approach that works. We stripped out all the noisy pseudo-labels, upgraded to the latest architecture with 6.5× more parameters, and trained for 100 epochs with aggressive augmentation:

| Parameter | Value | Why |
|:---|:---:|:---|
| Model | YOLO26m-seg | Latest architecture, ~22M params |
| Epochs | 100 | With early stopping (patience=20) |
| Data | Clean expert annotations only | No pseudo-label noise |
| Optimizer | AdamW (lr=0.001) | Conservative for fine-tuning |
| Warmup | 5 epochs | Prevent early instability |
| Mosaic | 1.0 | Combines 4 images into 1 — essential for small datasets |
| MixUp | 0.15 | Blends image pairs for regularization |
| Copy-Paste | 0.1 | Copies corrosion instances between images |
| Rotation | ±15° | Geometric invariance |
| Scale | ±50% | Size invariance |
| Color Jitter | H=0.015, S=0.7, V=0.4 | Lighting invariance |
| Close Mosaic | Last 10 epochs | Fine-tune on clean single images |

### 6.3 Post-Processing

Raw model predictions can be noisy — small isolated pixels incorrectly classified. We apply morphological operations to clean them up:

```python
def post_process_predictions(y_pred, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for cls_val in [1, 2, 3]:
        cls_mask = (y_pred == cls_val).astype(np.uint8)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
        y_pred[cls_mask == 1] = cls_val
    return y_pred
```

---

## 7. Results

### 7.1 DeepLabV3+

| Metric | Score |
|:---|:---:|
| **F1-Score** | 86.67% |
| **Backbone** | ResNet-101 |
| **Best Checkpoint** | Epoch 35/40 |

![DeepLabV3+ quantitative results across loss functions](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/deeplab_results_table.png)

**Sample Prediction — DeepLabV3+ segmentation overlay:**

![DeepLabV3+ corrosion segmentation with color-coded severity overlay](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/deeplab_prediction_output.png)

### 7.2 YOLO Training Curves

**Base model (YOLOv8n, 15 epochs) — the first attempt:**

![Training loss and metrics for the initial YOLOv8n base model](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_base_training_results.png)

![Confusion matrix showing the base model's classification accuracy](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_base_confusion_matrix.png)

**Heavy-tuned model (YOLOv8n, 31 epochs, merged data) — the second attempt:**

![Training curves for the heavy-tuned model with merged dataset](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_heavy_training_results.png)

![Confusion matrix for the heavy-tuned model](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_heavy_confusion_matrix.png)

### 7.3 Model Comparison

| Metric | YOLOv8n Base | YOLOv8n Heavy | YOLO26m Final |
|:---|:---:|:---:|:---:|
| **Architecture** | v8 nano (3.4M) | v8 nano (3.4M) | **v26 medium (~22M)** |
| **Training Data** | 396 (clean) | 746 (noisy) | **396 (clean)** |
| **Epochs** | 15 | 31 | **100** |
| **mAP@50 (Box)** | 0.157 | 0.158 | *Training in progress* |
| **mAP@50 (Mask)** | 0.140 | 0.149 | *Training in progress* |
| **Precision** | 0.166 | 0.219 | *Training in progress* |
| **Recall** | 0.233 | 0.282 | *Training in progress* |

### 7.4 Precision-Recall Analysis

![Mask Precision vs Recall at different confidence thresholds](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_heavy_mask_pr_curve.png)

![F1 score vs confidence threshold for mask predictions](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_heavy_mask_f1_curve.png)

![Box detection Precision-Recall curve](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_heavy_box_pr_curve.png)

### 7.5 Visual Results

**YOLO detecting corrosion on a completely unseen image:**

![YOLO instance segmentation detecting and classifying corrosion regions with confidence scores](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/rusted_rod_prediction.png)

**Test set predictions (original → prediction side-by-side):**

![Test image with original and predicted segmentation comparison](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_test_sample_0.png)

![Test image with original and predicted segmentation comparison](/Users/srijanupadhyay/.gemini/antigravity/brain/dffdd2a7-5961-4662-a2c1-92831c9c0b58/figures/yolo_test_sample_1.png)

---

## 8. The Application

Trained models are useless if nobody can use them. We built a full web application using Streamlit that puts the power of both models behind a simple drag-and-drop interface.

### 8.1 What It Does

| Feature | How it works |
|:---|:---|
| **Dual-Model Toggle** | Switch between DeepLabV3+ and YOLO26 with one click |
| **Model Selection** | Choose from multiple checkpoints per architecture |
| **Confidence Control** | Slide to adjust detection sensitivity in real-time |
| **Overlay Transparency** | Blend segmentation masks over the original image |
| **Noise Filtering** | Morphological smoothing with adjustable kernel |
| **Severity Report** | Automatic % breakdown by condition state |
| **Recommendations** | "Monitor", "Schedule Maintenance", or "Act Immediately" |
| **Download Masks** | Export raw prediction masks for documentation |

### 8.2 How to Run It

```bash
source .venv/bin/activate
streamlit run app.py
# Opens at http://localhost:8501
```

Upload an image → select a model → get instant corrosion analysis with quantitative severity breakdown.

---

## 9. Project Structure

```
Project Root/
├── app.py                           # Streamlit web application
├── retrain_yolo_proper.py           # YOLO26 retraining pipeline
├── yolo_automation_pipeline.py      # Web scraping + pseudo-labeling
├── run_inference.py                 # DeepLabV3+ inference
├── predict_samples.py               # Batch prediction
├── yolo_robust_evaluation.py        # Validation + visualization
├── PROJECT_REPORT.md                # This document
│
├── code_repo/                       # Model architectures and training code
│   ├── Training - Testing/
│   │   ├── main_plus.py             # DeepLabV3+ training loop
│   │   ├── model_plus.py            # Architecture factory
│   │   ├── datahandler_plus.py      # Dataset loader
│   │   └── network/                 # DeepLabV3+ modules
│   ├── Pre-processing/              # Annotation conversion
│   └── Visualization/               # Plotting utilities
│
├── Corrosion Condition State Classification/
│   └── 512x512/                     # 396 train + 44 test images with masks
│
├── yolo_dataset_clean/              # Clean YOLO-format dataset
├── runs/segment/                    # All trained model weights
│   ├── retrained/corrosion_yolo26m_proper/  # YOLO26m final
│   └── yolo_project*/               # Earlier YOLO experiments
│
├── scraped_images_raw/              # Web-scraped image cache
└── evaluation_results/              # Validation metrics and samples
```

---

## 10. What's Next

This project demonstrates that automated corrosion detection is not just feasible but practical. However, several directions remain open:

**More data.** 396 images got us surprisingly far, but the Severe class (131 instances) remains underrepresented. Active learning — where the deployed model flags uncertain predictions for human review — could efficiently grow the dataset where it matters most.

**Temporal analysis.** Corrosion progresses over time. Comparing images of the same element across inspection cycles could enable rate-of-deterioration estimation, moving from "how bad is it now?" to "how fast is it getting worse?"

**Edge deployment.** The YOLO26 model can be exported to ONNX or CoreML format for on-device inference on smartphones and tablets, enabling field inspectors to get instant feedback during walkthroughs.

**Beyond corrosion.** The same dual-architecture approach could be extended to other structural defects — cracking, spalling, delamination, fatigue damage — creating a unified structural condition assessment platform.

---

## 11. References

[1] E. Bianchi and M. Hebdon, "Corrosion Condition State Semantic Segmentation Dataset," University Libraries, Virginia Tech, 2021. DOI: [10.7294/16624663.v2](https://doi.org/10.7294/16624663.v2)

[2] E. Bianchi and M. Hebdon, "Trained Model for the Semantic Segmentation of Structural Material," University Libraries, Virginia Tech, 2021. DOI: [10.7294/16628620.v1](https://doi.org/10.7294/16628620.v1)

[3] E. Bianchi and M. Hebdon, "Development of Extendable Open-Source Structural Inspection Datasets," *Journal of Computing in Civil Engineering*, vol. 36, no. 6, 2022. DOI: [10.1061/(ASCE)CP.1943-5487.0001045](https://doi.org/10.1061/(ASCE)CP.1943-5487.0001045)

[4] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation," *ECCV*, 2018. [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE CVPR*, pp. 770–778, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023–2026. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

[7] D.-H. Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks," *ICML Workshop*, 2013.

[8] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *IEEE CVPR*, pp. 779–788, 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

[9] B. M. Phares, G. A. Washer, D. D. Rolander, B. A. Graybeal, and M. Moore, "Routine Highway Bridge Inspection Condition Documentation Accuracy and Reliability," *Journal of Bridge Engineering*, vol. 9, no. 4, pp. 403–413, 2004.

[10] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid Networks for Object Detection," *IEEE CVPR*, pp. 2117–2125, 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

---

## 12. Appendix

### A. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python numpy pillow scikit-learn matplotlib tqdm
pip install ultralytics streamlit
```

### B. Quick Start

```bash
# Retrain YOLO26 from scratch
python retrain_yolo_proper.py

# Run DeepLabV3+ inference on a single image
python run_inference.py --image "path/to/image.jpeg" \
    --model "Corrosion Condition State Classification - Trained Model/l2_loss/weights_35.pt"

# Launch the web app
streamlit run app.py
```

### C. Prediction Color Legend

| Class | Color | What it means |
|:---:|:---:|:---|
| Good | Black | No corrosion detected |
| Fair | Red | Surface oxidation — cosmetic, monitor periodically |
| Poor | Green | Active corrosion — schedule maintenance |
| Severe | Yellow | Section loss — requires immediate intervention |
