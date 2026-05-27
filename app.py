import os
import sys

# Add training directory to path so PyTorch can locate the 'network' module
training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "code_repo", "Training - Testing"))
sys.path.append(training_dir)

import torch
import torch.hub
import torchvision.models

# Torchvision compatibility fix for older model files
import torchvision.models._utils as _utils
_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
sys.modules['torchvision.models.utils'] = _utils

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page layout to wide and add title
st.set_page_config(page_title="Bridge Corrosion Multi-Model AI UI", layout="wide", page_icon="🏗️")

# Custom CSS for modern visual design
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e6edf3;
    }
    .stHeader {
        background: linear-gradient(90deg, #1f6feb 0%, #11c2a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .custom-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 20px;
        font-weight: bold;
    }
    .legend-color {
        width: 15px;
        height: 15px;
        border-radius: 3px;
        margin-right: 8px;
        border: 1px solid #fff;
    }
    .report-stat-value {
        font-size: 24px;
        font-weight: 800;
        color: #1f6feb;
    }
    .report-stat-label {
        font-size: 14px;
        color: #8b949e;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_cached_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model with weights_only=False due to custom pickled class definitions
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_cached_yolo(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model

def post_process_predictions(y_pred, kernel_size=7):
    processed_mask = np.zeros_like(y_pred)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for c in [1, 2, 3]:
        class_mask = (y_pred == c).astype(np.uint8)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        processed_mask[class_mask == 1] = c
    return processed_mask

def run_deeplab_prediction(image_np, model, device, enable_smoothing=True, kernel_size=7):
    orig_h, orig_w = image_np.shape[:2]
    
    # Preprocess
    resized_image = cv2.resize(image_np, (512, 512))
    img = resized_image.transpose(2, 0, 1) # HWC to CHW
    img = img.reshape(1, 3, 512, 512)
    
    with torch.no_grad():
        input_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
        output = model(input_tensor)
        
    pred = torch.argmax(output, dim=1)
    y_pred = pred.data.cpu().numpy().squeeze(0) # Shape: (512, 512)
    
    if enable_smoothing:
        y_pred = post_process_predictions(y_pred, kernel_size)
    
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    color_mask[y_pred == 1] = [0, 0, 255]    # Red (BGR)
    color_mask[y_pred == 2] = [0, 255, 0]    # Green (BGR)
    color_mask[y_pred == 3] = [0, 255, 255]  # Yellow (BGR)
    
    color_mask_resized = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return color_mask_resized, y_pred, None

def run_yolo_prediction(image_np, model, show_boxes, show_masks, blend_alpha, yolo_conf, enable_smoothing=True, kernel_size=7):
    import tempfile
    h, w = image_np.shape[:2]
    
    # ---- EXACT same pipeline as rusted_rod_prediction.png ----
    # Save the image to a temp file and pass the FILE PATH to YOLO.
    # This eliminates all BGR/RGB numpy array ambiguity.
    # YOLO loads the file itself via its own loader, ensuring plot() returns
    # the image with correct colors — proven by rusted_rod_prediction.png.
    tmp_path = os.path.join(tempfile.gettempdir(), "streamlit_yolo_input.jpg")
    cv2.imwrite(tmp_path, image_np)
    
    results = model(tmp_path, verbose=False, conf=yolo_conf)
    result = results[0]
    
    class_names = {
        0: "Fair",
        1: "Poor",
        2: "Severe"
    }
    
    # ---- Use YOLO's native plot() renderer ----
    plotted = result.plot(
        boxes=show_boxes,
        masks=show_masks,
        labels=True,
        probs=False
    )
    # plotted is in the same format as rusted_rod_prediction.png
    # cv2.imwrite saves it correctly, so it's BGR
    output_image = plotted
    
    # ---- Build y_pred mask for pixel-level statistics ----
    y_pred = np.zeros((h, w), dtype=np.uint8)
    prediction_color_mask = np.zeros_like(image_np)
    
    class_colors = {
        0: [0, 0, 255],     # Red (BGR) - Fair
        1: [0, 255, 0],     # Green (BGR) - Poor
        2: [0, 255, 255]    # Yellow (BGR) - Severe
    }
    
    detections_summary = []
    
    if result.masks is not None and len(result.masks) > 0:
        for poly_pts, cls_idx in zip(result.masks.xy, result.boxes.cls):
            cls = int(cls_idx)
            poly_pts = poly_pts.astype(np.int32)
            color = class_colors.get(cls, [255, 255, 255])
            
            # Class index + 1 for coverage (1 = Fair, 2 = Poor, 3 = Severe)
            cv2.fillPoly(y_pred, [poly_pts], cls + 1)
            cv2.fillPoly(prediction_color_mask, [poly_pts], color)
            
        if enable_smoothing:
            y_pred = post_process_predictions(y_pred, kernel_size)
            prediction_color_mask.fill(0)
            for c in [1, 2, 3]:
                color = class_colors.get(c - 1, [255, 255, 255])
                prediction_color_mask[y_pred == c] = color
    
    # Build detections summary for the instance list
    if result.boxes is not None and len(result.boxes) > 0:
        for box, cls_idx, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            cls = int(cls_idx)
            conf_val = float(conf)
            name = class_names.get(cls, f"Class {cls}")
            x1, y1, x2, y2 = map(int, box)
            detections_summary.append({
                "class": name,
                "confidence": conf_val,
                "box": [x1, y1, x2, y2]
            })
        
    return output_image, y_pred, detections_summary, prediction_color_mask

# Streamlit App UI layout
st.markdown("<h1 class='stHeader'>🏗️ Bridge Corrosion Classifier, Detector & Segmenter</h1>", unsafe_allow_html=True)
st.markdown("Upload a structural inspection image to run DeepLabV3+ (semantic segmentation) or YOLOv8 (object detection, instance segmentation, and classification).")

# Sidebar Settings
st.sidebar.header("🔧 Settings & Model selection")

# Select Model Type
model_type = st.sidebar.radio("Select Model Framework", ["DeepLabV3+ (Semantic Segmentation)", "YOLOv8 (Segment, Detect & Classify)"], index=0)

available_deeplab_models = {
    "L2 Loss ResNet101 (Recommended - High Accuracy)": "Corrosion Condition State Classification - Trained Model/l2_loss/weights_35.pt",
    "Data Augmentation ResNet50 (Lightweight)": "Corrosion Condition State Classification - Trained Model/var_aug_batch_2_resnet50/var_aug_batch_2_resnet50_weights_18.pt",
    "L1 Loss ResNet101": "Corrosion Condition State Classification - Trained Model/l1_loss/weights_27.pt",
    "Original Weight ResNet101 (Epoch 40)": "Corrosion Condition State Classification - Trained Model/var_original_wbatch_2_plus/var_original_wbatch_2_plus_weights_40.pt"
}

available_yolo_models = {
    "YOLO26m-seg Retrained (Recommended — Latest Architecture)": "runs/segment/runs/segment/retrained/corrosion_yolo26m_proper/weights/best.pt",
    "Heavy-Tuned YOLOv8n-seg (Web-Scraped + Bridge Dataset)": "runs/segment/yolo_project_final/corrosion_heavy_seg/weights/best.pt",
    "Base YOLOv8n-seg (Original Bridge Dataset only)": "runs/segment/yolo_project/corrosion_seg/weights/best.pt"
}
model_loaded = False

if model_type == "DeepLabV3+ (Semantic Segmentation)":
    model_key = st.sidebar.selectbox("Select DeepLabV3+ Checkpoint", list(available_deeplab_models.keys()))
    selected_model_path = available_deeplab_models[model_key]
    blend_alpha = st.sidebar.slider("Overlay Transparency (Alpha)", 0.1, 1.0, 0.4, 0.05)
    
    try:
        model, device = load_cached_model(selected_model_path)
        st.sidebar.success(f"DeepLabV3+ loaded on {device}!")
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading DeepLab model: {e}")
else:
    yolo_key = st.sidebar.selectbox("Select YOLOv8 Checkpoint", list(available_yolo_models.keys()))
    selected_yolo_path = available_yolo_models[yolo_key]
    
    st.sidebar.warning("⚠️ YOLOv8 models have limited accuracy (~15% mAP). For reliable results, use **DeepLabV3+** (86.67% F1).")
    st.sidebar.subheader("YOLO Visualization Settings")
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes (Object Detection)", value=False)
    show_masks = st.sidebar.checkbox("Show Segmentation Masks (Instance Segmentation)", value=True)
    blend_alpha = st.sidebar.slider("Overlay Transparency (Alpha)", 0.1, 1.0, 0.4, 0.05)
    
    # Confidence threshold — 0.12 balances coverage vs noise for this model
    yolo_conf = st.sidebar.slider("YOLO Confidence Threshold", 0.01, 1.00, 0.12, 0.01)
    
    if os.path.exists(selected_yolo_path):
        try:
            model = load_cached_yolo(selected_yolo_path)
            st.sidebar.success(f"YOLOv8 Model loaded successfully!")
            model_loaded = True
        except Exception as e:
            st.sidebar.error(f"Error loading YOLO model: {e}")
    else:
        st.sidebar.warning(f"⚠️ Selected checkpoint not found (still training). Falling back to demo YOLOv8n-seg.")
        try:
            model = load_cached_yolo("yolov8n-seg.pt")
            st.sidebar.info("Demo YOLOv8n-seg loaded.")
            model_loaded = True
        except Exception as e:
            st.sidebar.error(f"Error loading demo YOLO model: {e}")

st.sidebar.write("---")
st.sidebar.subheader("🧼 Post-Processing Filters")
enable_smoothing = st.sidebar.checkbox("Enable Mask Smoothing (Clean Noise)", value=True)
kernel_size = st.sidebar.slider("Smoothing Filter Size (Kernel)", 3, 21, 7, 2)

# Color Legend Card
st.markdown("""
<div class='custom-card'>
    <h4>🎨 Classification Legend:</h4>
    <div class='legend-item'><div class='legend-color' style='background-color:#000000;'></div>Class 0: Good / Background (No corrosion)</div>
    <div class='legend-item'><div class='legend-color' style='background-color:#ff0000;'></div>Class 1: Fair (Surface rust/exposed steel)</div>
    <div class='legend-item'><div class='legend-color' style='background-color:#00ff00;'></div>Class 2: Poor (Pitting/Early section loss)</div>
    <div class='legend-item'><div class='legend-color' style='background-color:#ffff00;'></div>Class 3: Severe (Structural loss/cavities)</div>
</div>
""", unsafe_allow_html=True)

# Main Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image from local device...", 
        type=["jpg", "jpeg", "png", "webp", "tiff", "bmp", "gif"]
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1) # Load as BGR
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, caption="Uploaded Image", width="stretch")

with col2:
    st.subheader("🔍 Prediction & Analysis")
    if uploaded_file is not None and model_loaded:
        with st.spinner("Analyzing image for corrosion..."):
            
            # Run prediction based on framework
            if model_type == "DeepLabV3+ (Semantic Segmentation)":
                prediction_color_mask, y_pred, detections_summary = run_deeplab_prediction(
                    image, model, device, 
                    enable_smoothing=enable_smoothing, 
                    kernel_size=kernel_size
                )
                # Blend overlay
                overlay_bgr = cv2.addWeighted(image, 1.0 - blend_alpha, prediction_color_mask, blend_alpha, 0)
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            else:
                # YOLOv8 returns bounding box image, predicted class masks, and detected instances summary
                overlay_bgr, y_pred, detections_summary, prediction_color_mask = run_yolo_prediction(
                    image, model, 
                    show_boxes=show_boxes, 
                    show_masks=show_masks, 
                    blend_alpha=blend_alpha,
                    yolo_conf=yolo_conf,
                    enable_smoothing=enable_smoothing,
                    kernel_size=kernel_size
                )
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            
            # Display Prediction
            st.image(overlay_rgb, caption="Prediction Visual Output", width="stretch")
            
            # ---------------- Detailed Classification Analysis ----------------
            st.markdown("### 📊 Classification Summary")
            
            total_pixels = y_pred.size
            noise_threshold = 100 # ignore classes with less than 100 pixels to filter model noise
            
            c0 = np.sum(y_pred == 0)
            c1 = np.sum(y_pred == 1)
            c2 = np.sum(y_pred == 2)
            c3 = np.sum(y_pred == 3)
            
            # Apply noise threshold
            c1 = c1 if c1 > noise_threshold else 0
            c2 = c2 if c2 > noise_threshold else 0
            c3 = c3 if c3 > noise_threshold else 0
            c0 = total_pixels - (c1 + c2 + c3)
            
            p0 = (c0 / total_pixels) * 100
            p1 = (c1 / total_pixels) * 100
            p2 = (c2 / total_pixels) * 100
            p3 = (c3 / total_pixels) * 100
            
            total_corroded_pixels = c1 + c2 + c3
            corroded_percentage = (total_corroded_pixels / total_pixels) * 100
            
            # Determine overall state and display proper message
            if total_corroded_pixels == 0:
                st.success("✅ **STRUCTURALLY SOUND**: No corrosion detected! The steel component is in Good condition (Class 0).")
            else:
                st.error(f"⚠️ **CORROSION DETECTED**: Approximately **{corroded_percentage:.2f}%** of the analyzed surface area shows signs of corrosion.")
                
                # Determine maximum severity
                if c3 > 0:
                    st.markdown("""
                        <div style='background-color:#ffe3e3; border-left: 5px solid #d9383a; padding: 15px; border-radius: 4px; color: #5c0000; margin-bottom: 15px;'>
                            <strong>🚨 CRITICAL (Class 3 - Severe):</strong> Severe corrosion with section loss (holes/cavities) detected. Immediate structural assessment is recommended.
                        </div>
                    """, unsafe_allow_html=True)
                elif c2 > 0:
                    st.markdown("""
                        <div style='background-color:#fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 4px; color: #664d03; margin-bottom: 15px;'>
                            <strong>⚠️ WARNING (Class 2 - Poor):</strong> Poor condition state detected. Pack rust or early pitting is present. Maintenance action should be scheduled.
                        </div>
                    """, unsafe_allow_html=True)
                elif c1 > 0:
                    st.markdown("""
                        <div style='background-color:#e2f0d9; border-left: 5px solid #70ad47; padding: 15px; border-radius: 4px; color: #385723; margin-bottom: 15px;'>
                            <strong>ℹ️ NOTICE (Class 1 - Fair):</strong> Fair condition state detected. Surface rust or exposed steel is present. Monitor regularly.
                        </div>
                    """, unsafe_allow_html=True)
            
            # Detailed Stats Breakdown grid
            grid_c1, grid_c2, grid_c3, grid_c4 = st.columns(4)
            with grid_c1:
                st.markdown(f"<div class='custom-card'><div class='report-stat-value'>{p0:.2f}%</div><div class='report-stat-label'>CS-0 (Good)</div></div>", unsafe_allow_html=True)
            with grid_c2:
                st.markdown(f"<div class='custom-card'><div class='report-stat-value' style='color:#ff0000;'>{p1:.2f}%</div><div class='report-stat-label'>CS-1 (Fair)</div></div>", unsafe_allow_html=True)
            with grid_c3:
                st.markdown(f"<div class='custom-card'><div class='report-stat-value' style='color:#00ff00;'>{p2:.2f}%</div><div class='report-stat-label'>CS-2 (Poor)</div></div>", unsafe_allow_html=True)
            with grid_c4:
                st.markdown(f"<div class='custom-card'><div class='report-stat-value' style='color:#ffff00;'>{p3:.2f}%</div><div class='report-stat-label'>CS-3 (Severe)</div></div>", unsafe_allow_html=True)

            # Display individual YOLO detections list
            if model_type != "DeepLabV3+ (Semantic Segmentation)" and detections_summary:
                st.markdown("### 📋 Detected Instances (YOLO)")
                for idx, det in enumerate(detections_summary):
                    st.write(f"🔍 **Instance {idx+1}**: class **{det['class']}**, confidence **{det['confidence']:.2%}**, bounding box `{det['box']}`")

            # Download options
            st.write("---")
            st.write("📂 Export prediction files:")
            
            # Export BGR overlay & mask to byte buffer
            _, overlay_img_encoded = cv2.imencode('.png', overlay_bgr)
            _, mask_img_encoded = cv2.imencode('.png', prediction_color_mask)
            
            btn1, btn2 = st.columns(2)
            with btn1:
                st.download_button(
                    label="Download Overlay Image",
                    data=overlay_img_encoded.tobytes(),
                    file_name="corrosion_prediction_overlay.png",
                    mime="image/png"
                )
            with btn2:
                st.download_button(
                    label="Download Raw Mask",
                    data=mask_img_encoded.tobytes(),
                    file_name="corrosion_prediction_mask.png",
                    mime="image/png"
                )
    else:
        st.info("Please upload an image to see prediction and classification results.")
