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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
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
    .stSelectbox, .stSlider, .stFileUploader {
        background: #fff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .result-image {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-message {
        color: #e74c3c;
        background: rgba(231, 76, 60, 0.1);
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .stDataFrame {
        overflow-x: auto;
    }
    @media (max-width: 600px) {
        h1 {
            font-size: 1.5rem;
        }
        .stButton>button {
            padding: 8px 15px;
        }
        .stSelectbox, .stSlider, .stFileUploader {
            padding: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Ensure results directory exists
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Global model and class variables
model = None
class_names = []

# Function to load class names
def load_classes(classes_file="classes.txt"):
    try:
        if not os.path.exists(classes_file):
            raise FileNotFoundError(f"Classes file {classes_file} not found")
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(classes)} classes from {classes_file}")
        return classes
    except Exception as e:
        logger.error(f"Error loading classes file {classes_file}: {str(e)}")
        st.error(f"Failed to load classes: {str(e)}")
        return []

# Function to initialize model
def init_model(model_path="280.pt", conf_thres=0.35):
    global model, class_names
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        # Allowlist the DetectionModel class for safe loading
        torch.serialization.add_safe_globals([DetectionModel])
        
        # Initialize the YOLO model
        model = YOLO(model_path)
        model.conf = conf_thres
        class_names = load_classes()
        logger.info(f"Model initialized with confidence threshold {conf_thres}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        st.error(f"Model initialization failed: {str(e)}")
        model = None  # Explicitly set model to None on failure

# Draw bounding boxes with improved visibility
def draw_boxes(frame, boxes, confidences, class_ids):
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
            logger.warning(f"Invalid class index {label_idx}, skipping")
            continue
            
        label = class_names[label_idx]
        
        color_hash = hash(label) % 0xFFFFFF
        color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        text = f"{label}: {conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 0, 0), 1)
        
        cv2.putText(frame, text, (x1 + 2, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        detected_objects.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })
    
    logger.info(f"Drew {len(detected_objects)} bounding boxes")
    return frame, detected_objects

# Process a single image
def process_image(image_data, conf_thres=0.35, iou_thres=0.45):
    global model
    
    if model is None:
        logger.error("Model is not initialized")
        try:
            logger.info("Attempting to initialize model")
            init_model(conf_thres=conf_thres)
            if model is None:
                raise ValueError("Model initialization failed")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise ValueError(f"Model initialization failed: {str(e)}")
    
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
        process_size = (1280, 1280)
        img_resized = cv2.resize(img, (process_size[1], process_size[0]))
        
        results = model(
            img_resized,
            conf=conf_thres,
            iou=iou_thres,
            augment=True,
            verbose=False
        )
        
        scale_x = original_size[1] / process_size[1]
        scale_y = original_size[0] / process_size[0]
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        if boxes.shape[0] > 0:
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
        
        processed_img, detected_objects = draw_boxes(img.copy(), boxes, confidences, class_ids)
        
        logger.info(f"Image processed successfully, detected {len(detected_objects)} objects")
        return processed_img, detected_objects
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Image processing error: {str(e)}")

# WebRTC video processor for real-time detection
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_thres = 0.35
    
    def set_conf_thres(self, conf_thres):
        self.conf_thres = conf_thres
    
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            processed_img, detected_objects = process_image(img, conf_thres=self.conf_thres)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return frame

# Streamlit app
def main():
    st.title("Grocery Product Detection")
    st.markdown("Detect grocery products using webcam or uploaded images")
    st.markdown("**Note**: Webcam Detection may not work on Streamlit Cloud due to WebRTC limitations. Use Image Upload for reliable results.")

    # Mode selector
    mode = st.selectbox("Select Mode", ["Webcam Detection", "Image Upload"], key="mode")

    if mode == "Webcam Detection":
        st.subheader("Real-time Webcam Detection")
        
        # Confidence threshold slider
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="webcam_conf"
        )
        
        # Initialize WebRTC streamer
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
            async_processing=True
        )
        
        if ctx.video_processor:
            ctx.video_processor.set_conf_thres(conf_thres)
        
        st.markdown("**Note**: Click 'Start' above to begin webcam feed. Ensure camera permissions are granted. This feature may not work on Streamlit Cloud.")
        
    else:
        st.subheader("Image Upload Detection")
        
        # Confidence threshold slider
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="upload_conf"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG)", 
            type=["jpg", "jpeg", "png"],
            key="uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Validate file size
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.error("File size exceeds 10MB limit")
                    logger.error("File size exceeds 10MB")
                    return
                
                # Read and process image
                img_data = uploaded_file.read()
                processed_img, detected_objects = process_image(img_data, conf_thres=conf_thres)
                
                # Save result
                filename = f"{uuid.uuid4()}.jpg"
                result_path = RESULTS_DIR / filename
                cv2.imwrite(str(result_path), processed_img)
                
                # Convert for display
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(processed_img_rgb)
                
                # Display results
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.image(pil_img, caption="Detected Objects", use_column_width=True)
                
                if detected_objects:
                    st.write(f"Detected {len(detected_objects)} object{'s' if len(detected_objects) != 1 else ''}")
                    df = pd.DataFrame([
                        {"Product": obj["label"], "Confidence": f"{obj['confidence']*100:.1f}%"}
                        for obj in detected_objects
                    ])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No objects detected")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                logger.info(f"Image detection completed, saved as {filename}")
                
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Unexpected error in image upload: {str(e)}")

if __name__ == "__main__":
    main()