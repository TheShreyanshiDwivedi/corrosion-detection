import os
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model_path = "runs/segment/yolo_project_final/corrosion_heavy_seg/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return
        
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    input_dir = "Corrosion Condition State Classification/512x512/Test/images_512/"
    output_dir = "predictions_yolo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]
    print(f"Running predictions on first {len(test_images)} test images...")
    
    for filename in test_images:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, verbose=False, conf=0.15)
        res = results[0]
        
        # Plot segmentation mask overlay using YOLO's default plotter
        plotted = res.plot(boxes=False, labels=True, probs=False)
        
        # Save output image
        out_path = os.path.join(output_dir, f"yolo_{filename}")
        cv2.imwrite(out_path, plotted)
        print(f"Saved prediction overlay: {out_path}")
        
    print("Done generating predictions!")

if __name__ == "__main__":
    main()
