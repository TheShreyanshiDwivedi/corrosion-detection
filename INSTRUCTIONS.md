# Corrosion Condition State Classification

This guide describes how to set up the environment, run inference (predictions), evaluate metrics, and train new models for the **Corrosion Condition State Classification** system.

---

## 1. Environment Setup

A virtual environment (`.venv`) has already been created in this repository with all required dependencies installed:
- PyTorch & Torchvision
- OpenCV
- NumPy & Pillow
- Scikit-Learn
- Matplotlib
- TQDM

To activate the environment in your terminal, run:
```bash
source .venv/bin/activate
```

---

## 2. Running Inference (Predictions)

A script named `run_inference.py` has been created in the root directory to run predictions on single images using any of the trained model weights.

### Run Single Image Prediction
To predict the corrosion condition states for a test image (e.g., `8.jpeg` using the `l2_loss` model):
```bash
python run_inference.py --image "Corrosion Condition State Classification/512x512/Test/images_512/8.jpeg" --model "Corrosion Condition State Classification - Trained Model/l2_loss/weights_35.pt" --output prediction_output.png
```

This will save two output files:
- `prediction_output.png`: The original image with prediction mask overlay (30% alpha blended).
- `prediction_output_mask.png`: The raw color-coded prediction mask.

### Run Multiple Images
A script named `predict_samples.py` has been created in the root directory to automatically run inference on a batch of test images:
```bash
python predict_samples.py
```
This generates overlays and masks for the first 5 images in the test folder and saves them to the `predictions/` directory.

### Prediction Color Codes
The predicted masks are color-coded based on the standard bridge inspection guidelines:
*   **Class 0 (Good / Background)**: Black `(0, 0, 0)` - No visible corrosion.
*   **Class 1 (Fair)**: Red `(0, 0, 255)` - Surface corrosion / exposed steel.
*   **Class 2 (Poor)**: Green `(0, 255, 0)` - Pack rust / deeper corrosion.
*   **Class 3 (Severe)**: Yellow `(0, 255, 255)` - Section loss / structural deterioration.

---

## 3. Running the Streamlit Web Application

We have developed a premium Streamlit Web UI (`app.py`) in the root directory. This allows you to:
- Select which model checkpoint to load.
- Manually upload inspection images from your local device.
- Adjust segmentation overlay transparency dynamically.
- View and download color-coded prediction masks and overlayed results.

To run the Streamlit app:
```bash
streamlit run app.py
```
This will automatically open the app in your default browser at `http://localhost:8501`.

---

## 4. Running Model Training

To train or fine-tune models, use the `main_plus.py` script inside the `code_repo/Training - Testing` directory.

```bash
python "code_repo/Training - Testing/main_plus.py" -data_directory "Corrosion Condition State Classification/512x512" -exp_directory "Corrosion Condition State Classification - Trained Model/my_new_experiment" --epochs 40 --batchsize 2
```

### Key Training Arguments:
*   `-data_directory`: Path to directory containing `Train` and `Test` folders with `images_512` and `mask_512` subdirectories.
*   `-exp_directory`: Target directory to save checkpoints and metrics logs.
*   `--epochs`: Number of training runs (default: 10).
*   `--batchsize`: Training batch size (default: 2).
*   `--pretrained`: (Optional) Path to a pre-trained `.pt` weight file to start or resume from.

---

## 5. Running Metrics Evaluation

To calculate test metrics (F1 score, IoU, confusion matrix) over the entire test set:

1. Edit the directory paths and model paths inside `code_repo/Training - Testing/run_metrics_evaluation.py`.
2. Run the evaluation script:
```bash
PYTHONPATH="code_repo/Training - Testing" python "code_repo/Training - Testing/run_metrics_evaluation.py"
```

---

## 6. Directory Structure Reference

*   📁 **`code_repo/`**: Cloned training/prediction codebase.
    *   📂 `Training - Testing/`: Model definition, training loop (`main_plus.py`), and evaluation code.
    *   📂 `Pre-processing/`: Scripts to convert LabelMe annotations to segmentation masks.
    *   📂 `Visualization/`: Scripts to overlay and plot outputs.
*   📁 **`Corrosion Condition State Classification/`**: Dataset containing raw and rescaled images and labels.
*   📁 **`Corrosion Condition State Classification - Trained Model/`**: High-performance pre-trained PyTorch weight files (`.pt` checkpoints).
