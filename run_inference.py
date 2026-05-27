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

def run_prediction(image_path, model_path, output_path):
    print(f"Loading model from: {model_path}")
    # Load model on CPU or GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # We must set weights_only=False to allow loading arbitrary classes (network._deeplab.DeepLabV3)
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # Keep original dimensions for final output resize
    orig_h, orig_w = image.shape[:2]
    print(f"Original image shape: {image.shape}")
    
    # Resize to 512x512 as required by the model
    resized_image = cv2.resize(image, (512, 512))
    
    # Preprocess (channels first, normalized to FloatTensor)
    img = resized_image.transpose(2, 0, 1) # HWC to CHW
    img = img.reshape(1, 3, 512, 512)
    
    # Run inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
        output = model(input_tensor)
        
    # Get prediction class indices
    pred = torch.argmax(output, dim=1)
    y_pred = pred.data.cpu().numpy().squeeze(0) # Shape: (512, 512)
    
    # Map class indices to BGR colors:
    # 0 = background (black)
    # 1 = fair (red)
    # 2 = poor (green)
    # 3 = severe (yellow)
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    color_mask[y_pred == 1] = [0, 0, 255]    # Red (BGR)
    color_mask[y_pred == 2] = [0, 255, 0]    # Green (BGR)
    color_mask[y_pred == 3] = [0, 255, 255]  # Yellow (BGR)
    
    # Resize mask back to original image size
    color_mask_resized = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Overlay mask on original image (alpha blending)
    overlay = cv2.addWeighted(image, 0.7, color_mask_resized, 0.3, 0)
    
    # Save output files
    cv2.imwrite(output_path, overlay)
    
    # Also save the raw mask for completeness
    mask_output_path = output_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_output_path, color_mask_resized)
    
    print(f"Saved prediction overlay to: {output_path}")
    print(f"Saved raw color-coded mask to: {mask_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run corrosion detection model inference.")
    parser.add_argument("--image", default="Corrosion Condition State Classification/512x512/Test/images_512/8.jpeg", help="Path to input image")
    parser.add_argument("--model", default="Corrosion Condition State Classification - Trained Model/l2_loss/weights_35.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="prediction_output.png", help="Path to save result image")
    
    args = parser.parse_args()
    run_prediction(args.image, args.model, args.output)
