import os
import sys
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO

def main():
    model_path = "runs/segment/yolo_project_final/corrosion_heavy_seg/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return
        
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # 1. Run Ultralytics validation to get standard metrics
    print("Running YOLOv8 Validation on the Val Set...")
    metrics = model.val(data="yolo_dataset_heavy/dataset_heavy.yaml", split="val", verbose=True)
    
    # Extract metrics
    # metrics.box contains box metrics, metrics.seg contains segmentation mask metrics
    map50_seg = metrics.seg.map50
    map95_seg = metrics.seg.map
    precision_seg = metrics.seg.mp
    recall_seg = metrics.seg.mr
    
    print("\n--- YOLOv8 Segmentation Validation Metrics ---")
    print(f"Precision (Mask): {precision_seg:.4f}")
    print(f"Recall (Mask): {recall_seg:.4f}")
    print(f"mAP@50 (Mask): {map50_seg:.4f}")
    print(f"mAP@50-95 (Mask): {map95_seg:.4f}")
    print("----------------------------------------------\n")
    
    # Save metrics to JSON
    summary = {
        "precision": precision_seg,
        "recall": recall_seg,
        "map50": map50_seg,
        "map95": map95_seg,
        "speed": metrics.speed
    }
    
    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/metrics.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved validation metrics to evaluation_results/metrics.json")
    
    # 2. Run inference on custom sample images to show generalization
    print("Generating qualitative test predictions...")
    test_img_dir = "Corrosion Condition State Classification/512x512/Test/images_512/"
    if os.path.exists(test_img_dir):
        test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]
        for idx, filename in enumerate(test_images):
            img_path = os.path.join(test_img_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, verbose=False)
            res = results[0]
            
            # Draw predictions (only segmentation overlay, no boxes for clean visuals)
            plotted = res.plot(boxes=False, labels=True, probs=False)
            
            # Combine side-by-side: original vs predicted
            combined = np.hstack((img, plotted))
            cv2.imwrite(f"evaluation_results/test_sample_{idx}.png", combined)
            print(f"Saved visualization: evaluation_results/test_sample_{idx}.png")
            
    print("Robust evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
