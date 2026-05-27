import os
import sys
import re
import urllib.parse
import shutil
import requests
import cv2
import numpy as np
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add training directory to path so PyTorch can locate the 'network' module
training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "code_repo", "Training - Testing"))
sys.path.append(training_dir)

# Torchvision compatibility fix for older model files
import torch.hub
import torchvision.models
import torchvision.models._utils as _utils
_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
sys.modules['torchvision.models.utils'] = _utils

from ultralytics import YOLO

# ----------------- PHASE 1: IMAGE SCRAPING -----------------

def get_vqd(query, headers):
    try:
        url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
        res = requests.get(url, headers=headers, timeout=10)
        vqd_match = re.search(r'vqd=["\']([\d-]+)["\']', res.text)
        if vqd_match:
            return vqd_match.group(1)
    except Exception as e:
        print(f"Error getting VQD token for query '{query}': {e}")
    return None

def scrape_duckduckgo_images(query, max_images=200):
    print(f"Crawling images for: '{query}'...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    vqd = get_vqd(query, headers)
    if not vqd:
        return []
        
    image_urls = []
    url = "https://duckduckgo.com/i.js"
    
    params = {
        "l": "wt-wt",
        "o": "json",
        "q": query,
        "vqd": vqd,
        "f": ",,,",
        "p": "1"
    }
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        data = res.json()
        results = data.get("results", [])
        for r in results:
            img_url = r.get("image")
            if img_url:
                image_urls.append(img_url)
                if len(image_urls) >= max_images:
                    break
    except Exception as e:
        print(f"Error requesting images for query '{query}': {e}")
        
    print(f"Found {len(image_urls)} candidate URLs.")
    return image_urls

def download_single_image(args):
    idx, url, save_dir, headers = args
    try:
        # Determine extension
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        elif ".webp" in url.lower():
            ext = ".webp"
            
        filename = f"scraped_{idx}{ext}"
        file_path = os.path.join(save_dir, filename)
        
        # Download file
        response = requests.get(url, headers=headers, timeout=4, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            return file_path
    except Exception:
        pass # Skip download failures
    return None

def download_images(urls, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    downloaded_paths = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    print(f"Downloading images to {save_dir} in parallel (30 threads)...")
    tasks = [(idx, url, save_dir, headers) for idx, url in enumerate(urls)]
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_single_image, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                downloaded_paths.append(res)
                
    print(f"Successfully downloaded {len(downloaded_paths)} images.")
    return downloaded_paths

# ----------------- PHASE 2: IMAGE VALIDATION -----------------

def validate_and_clean_images(image_paths):
    print("Validating downloaded images...")
    valid_paths = []
    
    for path in tqdm(image_paths):
        # 1. Check if file exists and has size
        if not os.path.exists(path) or os.path.getsize(path) < 1024: # ignore < 1KB
            if os.path.exists(path):
                os.remove(path)
            continue
            
        # 2. Check if OpenCV can open the image
        try:
            img = cv2.imread(path)
            if img is None or img.shape[0] < 100 or img.shape[1] < 100: # ignore very small images
                os.remove(path)
                continue
            valid_paths.append(path)
        except Exception:
            if os.path.exists(path):
                os.remove(path)
                
    print(f"Validation complete. {len(valid_paths)} images are valid and clean.")
    return valid_paths

# ----------------- PHASE 3: AUTO-LABELING (PSEUDO-LABELING) -----------------

def generate_pseudo_labels(image_paths, model_path, label_dir, conf_threshold=0.20):
    print(f"Loading base YOLO model from: {model_path} for auto-labeling...")
    model = YOLO(model_path)
    os.makedirs(label_dir, exist_ok=True)
    
    labeled_count = 0
    
    print("Generating labels using YOLO Instance Segmentation...")
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        name = os.path.splitext(filename)[0]
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Run inference (convert BGR to RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, verbose=False, conf=conf_threshold)
        result = results[0]
        
        yolo_lines = []
        if result.masks is not None and len(result.masks) > 0:
            for poly_pts, cls_idx in zip(result.masks.xy, result.boxes.cls):
                cls = int(cls_idx)
                
                # Normalize points
                normalized_points = []
                for pt in poly_pts:
                    nx = pt[0] / w
                    ny = pt[1] / h
                    # Clamp
                    nx = max(0.0, min(1.0, nx))
                    ny = max(0.0, min(1.0, ny))
                    normalized_points.append(f"{nx:.6f} {ny:.6f}")
                    
                if len(normalized_points) >= 3:
                    points_str = " ".join(normalized_points)
                    yolo_lines.append(f"{cls} {points_str}")
                    
        # If we successfully detected corrosion, save the label file
        if yolo_lines:
            label_file_path = os.path.join(label_dir, name + ".txt")
            with open(label_file_path, 'w') as lf:
                lf.write("\n".join(yolo_lines) + "\n")
            labeled_count += 1
            
    print(f"Auto-labeling complete. Generated labels for {labeled_count} images.")
    return labeled_count

# ----------------- PHASE 4: DATASET MERGING -----------------

def merge_datasets(scraped_img_dir, scraped_label_dir, base_yolo_dataset, merged_dataset_dir):
    print(f"Merging datasets into: {merged_dataset_dir}...")
    
    # Target directories
    dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for d in dirs:
        os.makedirs(os.path.join(merged_dataset_dir, d), exist_ok=True)
        
    # 1. Copy base dataset
    print("Copying base dataset...")
    for d in dirs:
        src = os.path.join(base_yolo_dataset, d)
        dest = os.path.join(merged_dataset_dir, d)
        if os.path.exists(src):
            for item in os.listdir(src):
                shutil.copy(os.path.join(src, item), os.path.join(dest, item))
                
    # 2. Copy auto-labeled scraped images to train set
    print("Merging scraped images and auto-labels...")
    copied_count = 0
    for label_file in os.listdir(scraped_label_dir):
        if not label_file.endswith('.txt'):
            continue
        name = os.path.splitext(label_file)[0]
        
        # Find corresponding image
        img_name = None
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp']:
            if os.path.exists(os.path.join(scraped_img_dir, name + ext)):
                img_name = name + ext
                break
                
        if img_name:
            # Copy label
            shutil.copy(
                os.path.join(scraped_label_dir, label_file),
                os.path.join(merged_dataset_dir, "labels/train", label_file)
            )
            # Copy image
            shutil.copy(
                os.path.join(scraped_img_dir, img_name),
                os.path.join(merged_dataset_dir, "images/train", img_name)
            )
            copied_count += 1
            
    print(f"Merged {copied_count} new images into training set.")

# ----------------- MAIN PIPELINE -----------------

def run_pipeline():
    # Setup directories
    temp_scrape_dir = "scraped_images_raw"
    temp_label_dir = "scraped_labels_raw"
    base_dataset = "yolo_dataset"
    merged_dataset = "yolo_dataset_heavy"
    base_model_path = "runs/segment/yolo_project/corrosion_seg/weights/best.pt"
    
    # 1. Scrape Images (10 search queries, max 100 images each = 1,000 images candidate)
    queries = [
        "rusted metal pipe", "corroded steel rod", "surface rust steel",
        "pack rust steel corrosion", "severe metal corrosion", "rusting iron beam",
        "bridge steel corrosion", "industrial rust corrosion", "corroded pipeline metal",
        "rusted steel sheet plate"
    ]
    
    all_urls = []
    for q in queries:
        urls = scrape_duckduckgo_images(q, max_images=100)
        all_urls.extend(urls)
        
    # Remove duplicates
    all_urls = list(set(all_urls))
    print(f"Total unique image URLs found: {len(all_urls)}")
    
    # Download images
    downloaded_files = download_images(all_urls, temp_scrape_dir)
    
    # 2. Validate Images
    valid_files = validate_and_clean_images(downloaded_files)
    
    if len(valid_files) == 0:
        print("Error: No valid images found after validation. Pipeline terminated.")
        return
        
    # 3. Generate Labels
    if not os.path.exists(base_model_path):
        print(f"Error: Base model checkpoint {base_model_path} not found. Cannot auto-label. Pipeline terminated.")
        return
        
    generate_pseudo_labels(valid_files, base_model_path, temp_label_dir, conf_threshold=0.15)
    
    # 4. Merge datasets
    merge_datasets(temp_scrape_dir, temp_label_dir, base_dataset, merged_dataset)
    
    # 5. Create YAML configuration
    yaml_content = f"""path: "{os.path.abspath(merged_dataset)}"
train: "images/train"
val: "images/val"

nc: 3
names:
  0: "Fair"
  1: "Poor"
  2: "Severe"
"""
    yaml_path = os.path.join(merged_dataset, "dataset_heavy.yaml")
    with open(yaml_path, 'w') as yf:
        yf.write(yaml_content)
        
    print(f"YAML configuration written to: {yaml_path}")
    
    # 6. Heavy Training (YOLOv8 Segmentation, 50 Epochs, MPS GPU Acceleration)
    print("Initializing Heavy Training on Merged Dataset...")
    model = YOLO("yolov8n-seg.pt")
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        
    print(f"Using training device: {device}")
    
    # Training parameters optimized for success and robust convergence
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=512,
        batch=16,
        device=device,
        project="yolo_project_final",
        name="corrosion_heavy_seg",
        patience=12,          # Early stopping patience to avoid overfitting
        optimizer="AdamW",     # Stable optimizer for custom segmentations
        lr0=0.001,             # Good starting learning rate
        weight_decay=0.0005,
        verbose=True
    )
    
    print("Heavy training complete!")
    print(f"Final best weights are saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    run_pipeline()
