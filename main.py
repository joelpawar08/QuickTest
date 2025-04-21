import streamlit as st
import cv2
import numpy as np
import uuid
import logging
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for responsive and modern UI
st.markdown("""
<style>
    * {
        box-sizing: border-box;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    }
    .stApp {
        background-color: #f0f2f5;
        max-width: 1200px;
        margin: 0 auto;
        padding: 10px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 600;
        transition: background-color 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .result-container {
        background: #fff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 15px;
    }
    .camera-section {
        background: #fff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Ensure results directory exists
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Global model and class variables
models = {}
class_names = []

# Function to load class names
def load_classes(classes_file="classes.txt"):
    try:
        if not os.path.exists(classes_file):
            default_classes = [
                "apple", "banana", "orange", "water bottle", "soda", "milk", "bread", 
                "cereal", "chips", "cookies", "pasta", "rice", "vegetables", "fruits",
                "meat", "cheese", "yogurt", "eggs", "juice", "candy", "snacks"
            ]
            with open(classes_file, 'w') as f:
                for cls in default_classes:
                    f.write(f"{cls}\n")
            logger.info(f"Created default classes file with {len(default_classes)} classes")
            return default_classes
        
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(classes)} classes from {classes_file}")
        return classes
    except Exception as e:
        logger.error(f"Error loading classes file {classes_file}: {str(e)}")
        st.error(f"Failed to load classes: {str(e)}")
        return ["product", "food", "beverage", "container", "package"]

# Function to initialize models
def init_models(model_paths=["280.pt", "maggie.pt"], conf_thres=0.35):
    global models, class_names
    try:
        for model_path in model_paths:
            if not os.path.exists(model_path):
                st.error(f"Model file {model_path} not found. Please ensure the model file exists in the app directory.")
                logger.error(f"Model file {model_path} not found")
                return False
                
        torch.serialization.add_safe_globals([DetectionModel])
        
        # Initialize both YOLO models
        for model_path in model_paths:
            model = YOLO(model_path)
            model.conf = conf_thres
            models[model_path] = model
            logger.info(f"Model {model_path} initialized with confidence threshold {conf_thres}")
        
        class_names = load_classes()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        st.error(f"Model initialization failed: {str(e)}")
        models = {}
        return False

# Draw bounding boxes with improved visibility
def draw_boxes(frame, boxes, confidences, class_ids, model_name):
    global class_names
    detected_objects = []
    
    img_height, img_width = frame.shape[:2]
    min_dimension = min(img_height, img_width)
    box_thickness = max(2, int(min_dimension / 300))
    font_scale = max(0.5, min_dimension / 1000)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = float(confidences[i])
        label_idx = int(class_ids[i])
        
        if label_idx >= len(class_names):
            logger.warning(f"Invalid class index {label_idx}, using default 'unknown'")
            label = "unknown"
        else:
            label = class_names[label_idx]
        
        # Different colors for different models
        color = (0, 255, 0) if model_name == "280.pt" else (255, 0, 0)  # Green for 280.pt, Blue for maggie.pt
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        text = f"{label}: {conf:.2f} ({model_name})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 0, 0), 1)
        
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        detected_objects.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2],
            "model": model_name
        })
    
    logger.info(f"Drew {len(detected_objects)} bounding boxes for {model_name}")
    return frame, detected_objects

# Process a single image with both models
def process_image(image_data, conf_thres=0.35, iou_thres=0.45):
    global models
    
    if not models:
        logger.info("Models not initialized, attempting to initialize")
        if not init_models():
            raise ValueError("Model initialization failed. Please check the model files.")
    
    try:
        if isinstance(image_data, np.ndarray):
            img = image_data
        else:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image data")
            raise ValueError("Invalid image data")
        
        logger.info(f"Processing image of size {img.shape}")
        original_size = img.shape[:2]
        
        max_dim = 1280
        h, w = original_size
        
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
            
        process_size = (new_h, new_w)
        img_resized = cv2.resize(img, (process_size[1], process_size[0]))
        
        processed_img = img.copy()
        all_detected_objects = []
        
        # Process with both models
        for model_name, model in models.items():
            results = model(
                img_resized,
                conf=conf_thres,
                iou=iou_thres,
                augment=False,
                verbose=False,
                max_det=50
            )
            
            scale_x = original_size[1] / process_size[1]
            scale_y = original_size[0] / process_size[0]
            
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            if boxes.shape[0] > 0:
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
            
            processed_img, detected_objects = draw_boxes(processed_img, boxes, confidences, class_ids, model_name)
            all_detected_objects.extend(detected_objects)
        
        logger.info(f"Image processed successfully, detected {len(all_detected_objects)} objects across both models")
        return processed_img, all_detected_objects
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Image processing error: {str(e)}")

# Streamlit app
def main():
    st.title("Grocery Product Detection")
    st.markdown("Detect multiple grocery products using two YOLO models simultaneously")
    
    # Initialize models at startup
    if 'models_initialized' not in st.session_state:
        with st.spinner("Loading models... This might take a moment"):
            st.session_state.models_initialized = init_models()

    # Create tabs for different detection methods
    tab1, tab2 = st.tabs(["Camera", "Upload Image"])
    
    with tab1:
        st.subheader("Camera Detection")
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="camera_conf"
        )
        
        st.markdown("""
        ### How to use the camera:
        1. Click the "Take picture" button below
        2. Allow camera permissions when prompted
        3. Take a clear photo of the grocery items
        4. Both models will automatically detect products in your photo
        """)
        
        camera_input = st.camera_input("Take picture", key="camera")
        
        if camera_input:
            try:
                with st.spinner("Processing image with both models..."):
                    img_bytes = camera_input.getvalue()
                    processed_img, detected_objects = process_image(img_bytes, conf_thres=conf_thres)
                
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                
                st.image(processed_img_rgb, caption="Detected Objects (Green: 280.pt, Blue: maggie.pt)", use_column_width=True)
                
                if detected_objects:
                    st.success(f"Detected {len(detected_objects)} product{'s' if len(detected_objects) > 1 else ''} across both models")
                    
                    df = pd.DataFrame([
                        {"Product": obj["label"], "Confidence": f"{obj['confidence']*100:.1f}%", "Model": obj["model"]}
                        for obj in detected_objects
                    ])
                    st.dataframe(df, use_container_width=True)
                    
                    filename = f"{uuid.uuid4()}.jpg"
                    result_path = RESULTS_DIR / filename
                    cv2.imwrite(str(result_path), processed_img)
                else:
                    st.warning("No products detected by either model. Try adjusting the confidence threshold or taking a clearer photo.")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Camera processing error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("Camera not working?"):
            st.markdown("""
            ### Troubleshooting Tips:
            
            1. **Permissions**: Make sure you've allowed camera permissions in your browser
            2. **Browser**: Try using Chrome or Safari on mobile devices
            3. **Lighting**: Ensure good lighting for better detection
            4. **Alternative**: If camera still doesn't work, use the "Upload Image" tab
            5. **Refresh**: Try refreshing the page if camera doesn't initialize
            """)
    
    with tab2:
        st.subheader("Image Upload Detection")
        
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="upload_conf"
        )
        
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG)", 
            type=["jpg", "jpeg", "png"],
            key="uploader"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.error("File size exceeds 10MB limit")
                    logger.error("File size exceeds 10MB")
                else:
                    img_data = uploaded_file.read()
                    
                    with st.spinner("Processing image with both models..."):
                        processed_img, detected_objects = process_image(img_data, conf_thres=conf_thres)
                    
                    filename = f"{uuid.uuid4()}.jpg"
                    result_path = RESULTS_DIR / filename
                    cv2.imwrite(str(result_path), processed_img)
                    
                    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.image(processed_img_rgb, caption="Detected Objects (Green: 280.pt, Blue: maggie.pt)", use_column_width=True)
                    
                    if detected_objects:
                        st.success(f"Detected {len(detected_objects)} product{'s' if len(detected_objects) > 1 else ''} across both models")
                        
                        df = pd.DataFrame([
                            {
                                "Product": obj["label"], 
                                "Confidence": f"{obj['confidence']*100:.1f}%",
                                "Model": obj["model"]
                            }
                            for obj in detected_objects
                        ])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No products detected by either model. Try adjusting the confidence threshold or using a clearer image.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    logger.info(f"Image detection completed with both models, saved as {filename}")
                
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Unexpected error in image upload: {str(e)}")

if __name__ == "__main__":
    main()