import os
import sys

# Add the training directory to system path so PyTorch can locate the 'network' module
training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "code_repo", "Training - Testing"))
sys.path.append(training_dir)

import torch
import torch.hub
import torchvision.models

# Torchvision compatibility fix for older model files
import torchvision.models._utils as _utils
_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
sys.modules['torchvision.models.utils'] = _utils

import cv2
import numpy as np

def run_predictions_on_folder(input_dir, model_path, output_dir, limit=5):
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])
    files_to_process = all_files[:limit]
    
    print(f"Processing {len(files_to_process)} images from {input_dir}...")
    
    for filename in files_to_process:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        orig_h, orig_w = image.shape[:2]
        
        # Resize to 512x512 for inference
        resized_image = cv2.resize(image, (512, 512))
        img = resized_image.transpose(2, 0, 1)
        img = img.reshape(1, 3, 512, 512)
        
        # Forward pass
        with torch.no_grad():
            input_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
            output = model(input_tensor)
            
        pred = torch.argmax(output, dim=1)
        y_pred = pred.data.cpu().numpy().squeeze(0)
        
        # Map indices to BGR colors
        color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        color_mask[y_pred == 1] = [0, 0, 255]    # Red (Fair)
        color_mask[y_pred == 2] = [0, 255, 0]    # Green (Poor)
        color_mask[y_pred == 3] = [0, 255, 255]  # Yellow (Severe)
        
        # Resize back
        color_mask_resized = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(image, 0.7, color_mask_resized, 0.3, 0)
        
        # Save output images
        name_part = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_dir, f"{name_part}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(output_dir, f"{name_part}_mask.png"), color_mask_resized)
        print(f"Processed: {filename} -> Saved overlay and mask to {output_dir}")

if __name__ == "__main__":
    input_directory = "Corrosion Condition State Classification/512x512/Test/images_512/"
    model_checkpoint = "Corrosion Condition State Classification - Trained Model/l2_loss/weights_35.pt"
    output_directory = "predictions"
    
    run_predictions_on_folder(input_directory, model_checkpoint, output_directory, limit=5)
