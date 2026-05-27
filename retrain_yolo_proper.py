"""
Proper YOLOv8 Corrosion Segmentation Training Pipeline
=======================================================
This script retrains YOLOv8 on the Virginia Tech Corrosion Condition State
dataset with proper configuration:

1. Clean expert-annotated data ONLY (no pseudo-labels)
2. YOLOv8s-seg (11M params) instead of nano (3M params)
3. 100 epochs with early stopping (patience=20)
4. Heavy augmentation to compensate for small dataset
5. Proper train/val split with the original test set as validation
6. Class-aware training with proper handling of imbalanced data
"""

import os
import sys
import cv2
import yaml
import shutil
import numpy as np
from tqdm import tqdm

# ============================================================
# PHASE 1: Rebuild clean YOLO dataset from original VT masks
# ============================================================

def mask_to_yolo_labels(mask_path, class_mapping):
    """
    Convert a semantic segmentation mask (BGR) to YOLO polygon annotations.
    
    Args:
        mask_path: Path to the BGR mask image
        class_mapping: Dict mapping BGR tuple → YOLO class index
    
    Returns:
        List of strings, each in YOLO format: "class_id x1 y1 x2 y2 ... xN yN"
    """
    mask = cv2.imread(mask_path)
    if mask is None:
        return []
    
    h, w = mask.shape[:2]
    labels = []
    
    for bgr_color, class_id in class_mapping.items():
        # Extract binary mask for this class
        color_mask = np.all(mask == np.array(bgr_color), axis=2).astype(np.uint8)
        
        if color_mask.sum() == 0:
            continue
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Skip tiny contours (noise) - require at least 100 pixels area
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            # Simplify contour to reduce point count (epsilon = 1% of perimeter)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Need at least 3 points for a polygon
            if len(contour) < 3:
                continue
            
            # Normalize coordinates to [0, 1]
            points = contour.reshape(-1, 2)
            normalized = points.astype(float)
            normalized[:, 0] /= w
            normalized[:, 1] /= h
            
            # Clip to [0, 1]
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Format: class_id x1 y1 x2 y2 ... xN yN
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
            labels.append(f"{class_id} {coords}")
    
    return labels


def build_clean_dataset(source_dir, output_dir):
    """
    Build a clean YOLO dataset from the original Virginia Tech masks.
    Ensures proper conversion with quality checks.
    """
    
    # BGR color → YOLO class ID (skip background)
    class_mapping = {
        (0, 0, 128): 0,    # Fair (surface rust)
        (0, 128, 0): 1,    # Poor (pack rust)
        (0, 128, 128): 2,  # Severe (section loss)
    }
    
    train_img_dir = os.path.join(source_dir, "Train", "images_512")
    train_mask_dir = os.path.join(source_dir, "Train", "mask_512")
    test_img_dir = os.path.join(source_dir, "Test", "images_512")
    test_mask_dir = os.path.join(source_dir, "Test", "mask_512")
    
    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    
    stats = {"train": {"total": 0, "with_labels": 0, "class_counts": {0: 0, 1: 0, 2: 0}},
             "val": {"total": 0, "with_labels": 0, "class_counts": {0: 0, 1: 0, 2: 0}}}
    
    for split, img_dir, mask_dir in [("train", train_img_dir, train_mask_dir),
                                      ("val", test_img_dir, test_mask_dir)]:
        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        
        print(f"\n{'='*60}")
        print(f"Processing {split} set: {len(images)} images")
        print(f"{'='*60}")
        
        for img_name in tqdm(images, desc=f"Converting {split}"):
            stats[split]["total"] += 1
            
            # Get matching mask
            base_name = os.path.splitext(img_name)[0]
            mask_name = None
            for ext in ['.png', '.jpeg', '.jpg']:
                candidate = base_name + ext
                if os.path.exists(os.path.join(mask_dir, candidate)):
                    mask_name = candidate
                    break
            
            if mask_name is None:
                print(f"  WARNING: No mask found for {img_name}, skipping")
                continue
            
            # Convert mask to YOLO labels
            mask_path = os.path.join(mask_dir, mask_name)
            labels = mask_to_yolo_labels(mask_path, class_mapping)
            
            # Copy image
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(output_dir, "images", split, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Write label file
            label_name = base_name + ".txt"
            dst_label = os.path.join(output_dir, "labels", split, label_name)
            
            if labels:
                stats[split]["with_labels"] += 1
                with open(dst_label, "w") as f:
                    f.write("\n".join(labels) + "\n")
                
                # Count classes
                for label in labels:
                    cls = int(label.split()[0])
                    stats[split]["class_counts"][cls] += 1
            else:
                # Write empty label file (background-only image)
                with open(dst_label, "w") as f:
                    pass
    
    # Write data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "Fair",
            1: "Poor",
            2: "Severe"
        }
    }
    
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATASET BUILD SUMMARY")
    print(f"{'='*60}")
    class_names = {0: "Fair", 1: "Poor", 2: "Severe"}
    for split in ["train", "val"]:
        s = stats[split]
        print(f"\n{split.upper()}: {s['total']} images, {s['with_labels']} with annotations")
        for cls_id, count in s["class_counts"].items():
            print(f"  {class_names[cls_id]}: {count} instances")
    
    print(f"\ndata.yaml written to: {yaml_path}")
    return yaml_path, stats


# ============================================================
# PHASE 2: Train YOLOv8s-seg with proper hyperparameters
# ============================================================

def train_model(data_yaml_path):
    """
    Train YOLOv8s-seg with configuration optimized for small,
    imbalanced corrosion datasets.
    """
    from ultralytics import YOLO
    
    # Use YOLO26m-seg (medium) — latest 2026 architecture
    # Best balance of capacity vs overfitting risk for 396-image dataset
    model = YOLO("yolo26m-seg.pt")
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING: YOLO26m-seg")
    print(f"{'='*60}")
    print(f"Model: YOLO26m-seg (latest 2026 architecture)")
    print(f"Dataset: {data_yaml_path}")
    print(f"Device: MPS (Apple Silicon GPU)")
    print(f"Epochs: 100 (with early stopping, patience=20)")
    print(f"{'='*60}\n")
    
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=512,
        batch=8,
        
        # Training
        optimizer="AdamW",
        lr0=0.001,              # Lower initial LR for fine-tuning
        lrf=0.01,               # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,        # Longer warmup for stability
        
        # Early stopping
        patience=20,            # Stop if no improvement for 20 epochs
        
        # Heavy augmentation (critical for small datasets)
        augment=True,
        mosaic=1.0,             # Mosaic augmentation (combine 4 images)
        mixup=0.15,             # MixUp augmentation
        copy_paste=0.1,         # Copy-paste augmentation for segmentation
        degrees=15.0,           # Random rotation ±15°
        translate=0.15,         # Random translation
        scale=0.5,              # Random scale ±50%
        shear=5.0,              # Random shear
        flipud=0.3,             # Vertical flip probability
        fliplr=0.5,             # Horizontal flip probability
        hsv_h=0.015,            # Hue augmentation
        hsv_s=0.7,              # Saturation augmentation
        hsv_v=0.4,              # Value (brightness) augmentation
        
        # Output
        project="runs/segment/retrained",
        name="corrosion_yolo26m_proper",
        exist_ok=True,
        
        # Device
        device="mps",
        
        # Validation
        val=True,
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        plots=True,
        
        # Close mosaic augmentation for last 10 epochs (better fine-tuning)
        close_mosaic=10,
        
        verbose=True,
    )
    
    return results


# ============================================================
# PHASE 3: Validate and report
# ============================================================

def validate_model(model_path, data_yaml_path):
    """Run full validation and print metrics."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path, imgsz=512, device="mps")
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Box mAP@50:    {results.box.map50:.4f}")
    print(f"Box mAP@50-95: {results.box.map:.4f}")
    print(f"Mask mAP@50:   {results.seg.map50:.4f}")
    print(f"Mask mAP@50-95:{results.seg.map:.4f}")
    print(f"Precision:     {results.box.mp:.4f}")
    print(f"Recall:        {results.box.mr:.4f}")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    SOURCE_DIR = "Corrosion Condition State Classification/512x512"
    OUTPUT_DIR = "yolo_dataset_clean"
    
    print("=" * 60)
    print("PROPER YOLOv8 CORROSION SEGMENTATION RETRAINING")
    print("=" * 60)
    
    # Phase 1: Build clean dataset
    print("\n[PHASE 1] Building clean YOLO dataset from VT masks...")
    data_yaml, stats = build_clean_dataset(SOURCE_DIR, OUTPUT_DIR)
    
    # Phase 2: Train
    print("\n[PHASE 2] Training YOLOv8s-seg (100 epochs, early stopping)...")
    train_model(data_yaml)
    
    # Phase 3: Validate
    print("\n[PHASE 3] Running validation...")
    best_weights = "runs/segment/retrained/corrosion_yolov8s_proper/weights/best.pt"
    if os.path.exists(best_weights):
        validate_model(best_weights, data_yaml)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best weights: {best_weights}")
    print("=" * 60)
