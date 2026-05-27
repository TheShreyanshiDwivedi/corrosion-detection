# 🔬 Automated Corrosion Detection & Condition State Segmentation

A deep learning pipeline for pixel-level corrosion severity classification on steel structural elements, combining **DeepLabV3+** semantic segmentation with **YOLO26m-seg** real-time instance detection.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)
![YOLO](https://img.shields.io/badge/YOLO26-Instance_Seg-green)
![DeepLab](https://img.shields.io/badge/DeepLabV3+-Semantic_Seg-purple)

---

## 🎯 What It Does

Upload a photo of a corroded steel element → get an instant, color-coded severity map with quantitative analysis.

| Input | Segmentation Output |
|:---:|:---:|
| Original steel image | Color-coded mask (Fair / Poor / Severe) |

**Condition States Detected:**

| State | Color | Description |
|:---:|:---:|:---|
| Good | ⬛ Black | No corrosion |
| Fair | 🔴 Red | Surface oxidation |
| Poor | 🟢 Green | Pack rust, pitting |
| Severe | 🟡 Yellow | Section loss |

---

## 🏗️ Architecture

### Dual-Model Pipeline

```
                    ┌──────────────────────────┐
                    │    Input Image (512×512)  │
                    └──────────┬───────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
    ┌────────────────────┐      ┌────────────────────┐
    │   DeepLabV3+       │      │   YOLO26m-seg      │
    │   ResNet-101       │      │   Instance Seg     │
    │   (44M params)     │      │   (~22M params)    │
    │                    │      │                    │
    │   ✅ 86.67% F1     │      │   ✅ Detection     │
    │   ✅ Dense masks   │      │   ✅ Bounding boxes│
    │                    │      │   ✅ Confidence    │
    └────────┬───────────┘      └────────┬───────────┘
             │                           │
             ▼                           ▼
    ┌────────────────────────────────────────────────┐
    │          Streamlit Web Application              │
    │   Severity Report + Recommendations             │
    └────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/<your-username>/corrosion-detection.git
cd corrosion-detection

python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python numpy pillow scikit-learn matplotlib tqdm ultralytics streamlit
```

### 2. Run the Web App
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 3. Command-Line Inference
```bash
# DeepLabV3+
python run_inference.py --image path/to/image.jpg

# Retrain YOLO from scratch
python retrain_yolo_proper.py
```

---

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── retrain_yolo_proper.py          # YOLO26 retraining pipeline
├── yolo_automation_pipeline.py     # Web scraping + pseudo-labeling
├── run_inference.py                # DeepLabV3+ single-image inference
├── predict_samples.py              # Batch DeepLabV3+ prediction
├── predict_samples_yolo.py         # Batch YOLO prediction
├── yolo_robust_evaluation.py       # Validation metrics + visualization
├── PROJECT_REPORT.md               # Full research-style documentation
├── README.md                       # This file
│
├── code_repo/                      # Model architectures & training
│   ├── Training - Testing/
│   │   ├── main_plus.py            # DeepLabV3+ training loop
│   │   ├── model_plus.py           # Architecture factory
│   │   ├── trainer_plus.py         # Train/val logic
│   │   ├── datahandler_plus.py     # Dataset loader
│   │   └── network/                # DeepLabV3+ modules
│   ├── Pre-processing/             # Annotation conversion
│   └── Visualization/              # Plotting utilities
│
├── Corrosion Condition State Classification/
│   └── 512x512/                    # Dataset (396 train + 44 test)
│
├── yolo_dataset_clean/             # Clean YOLO-format dataset
│
├── runs/segment/                   # Trained model weights
│   └── retrained/corrosion_yolo26m_proper/
│       └── weights/best.pt         # Final YOLO26m model
│
└── evaluation_results/             # Validation outputs
```

---

## 📊 Results

| Model | mAP@50 (Box) | mAP@50 (Mask) | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **DeepLabV3+ (ResNet-101)** | — | — | — | **86.67%** |
| **YOLO26m-seg** | **0.229** | **0.220** | **0.586** | — |
| YOLOv8n-seg (baseline) | 0.157 | 0.140 | 0.166 | — |

---

## 📚 References

1. Bianchi & Hebdon, "Corrosion Condition State Semantic Segmentation Dataset," Virginia Tech, 2021
2. Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation," ECCV 2018
3. Ultralytics, "YOLO," 2023–2026
4. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016

---

## 📄 License

This project uses publicly available datasets and pre-trained models. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for full citations and acknowledgments.
