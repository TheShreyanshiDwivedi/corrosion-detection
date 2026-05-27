import torch
from ultralytics import YOLO

def train_custom_yolo():
    print("Initializing YOLOv8 Nano Segmentation model...")
    # Load a pretrained YOLOv8n-seg model
    model = YOLO("yolov8n-seg.pt")
    
    # Determine the best device (mps, cuda, cpu)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        
    print(f"Using device: {device} for training...")
    
    # Train the model
    # imgsz=512, batch=16, epochs=15 for a relatively fast but meaningful training session
    results = model.train(
        data="yolo_dataset/dataset.yaml",
        epochs=15,
        imgsz=512,
        batch=16,
        device=device,
        project="yolo_project",
        name="corrosion_seg",
        verbose=True
    )
    print("Training complete!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    train_custom_yolo()
