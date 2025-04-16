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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import torch
import os
import time
from io import BytesIO
import base64

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
    .camera-fallback {
        background: #fff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 15px;
        text-align: center;
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
            # Create a fallback classes file with common grocery items if not found
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
        # Return some default classes as fallback
        return ["product", "food", "beverage", "container", "package"]

# Function to initialize model
def init_model(model_path="280.pt", conf_thres=0.35):
    global model, class_names
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file {model_path} not found. Please ensure the model file exists in the app directory.")
            logger.error(f"Model file {model_path} not found")
            return False
            
        # Allowlist the DetectionModel class for safe loading
        torch.serialization.add_safe_globals([DetectionModel])
        
        # Initialize the YOLO model
        model = YOLO(model_path)
        model.conf = conf_thres
        class_names = load_classes()
        logger.info(f"Model initialized with confidence threshold {conf_thres}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        st.error(f"Model initialization failed: {str(e)}")
        model = None  # Explicitly set model to None on failure
        return False

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
            logger.warning(f"Invalid class index {label_idx}, using default 'unknown'")
            label = "unknown"
        else:
            label = class_names[label_idx]
        
        color_hash = hash(label) % 0xFFFFFF
        color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        text = f"{label}: {conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        # Add a background to the text for better visibility
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 0, 0), 1)
        
        # Add text with outline for better visibility on various backgrounds
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
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
        logger.info("Model is not initialized, attempting to initialize")
        if not init_model(conf_thres=conf_thres):
            raise ValueError("Model initialization failed. Please check the model file.")
    
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
        
        # Use dynamic resizing based on image aspect ratio
        max_dim = 1280
        h, w = original_size
        
        # Calculate new dimensions while preserving aspect ratio
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
            
        process_size = (new_h, new_w)
        img_resized = cv2.resize(img, (process_size[1], process_size[0]))
        
        # Run inference with optimized parameters for multi-object detection
        results = model(
            img_resized,
            conf=conf_thres,
            iou=iou_thres,
            augment=False,  # Set to False for faster processing
            verbose=False,
            max_det=50  # Allow detecting up to 50 objects in a single frame
        )
        
        # Calculate scaling factors to map bounding boxes back to original image
        scale_x = original_size[1] / process_size[1]
        scale_y = original_size[0] / process_size[0]
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        if boxes.shape[0] > 0:
            # Scale bounding boxes back to original image dimensions
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
    def __init__(self, conf_threshold=0.35):
        self.conf_thres = conf_threshold
        self.frame_count = 0
        self.last_process_time = time.time()
        self.process_every_n_frames = 3  # Process every 3 frames for better mobile performance
    
    def set_conf_thres(self, conf_thres):
        self.conf_thres = conf_thres
    
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process every nth frame to improve performance on mobile
            current_time = time.time()
            if self.frame_count % self.process_every_n_frames == 0 or (current_time - self.last_process_time) >= 0.5:
                processed_img, _ = process_image(img, conf_thres=self.conf_thres)
                self.last_process_time = current_time
            else:
                processed_img = img
                
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            # Return original frame if processing fails
            return frame

# Function for mobile camera capture as fallback
def capture_from_camera():
    st.markdown('<div class="camera-fallback">', unsafe_allow_html=True)
    st.write("Take a photo using your device camera:")
    
    camera_component = st.camera_input("Take a picture")
    
    if camera_component is not None:
        try:
            conf_thres = st.session_state.get("upload_conf", 0.35)
            img_data = camera_component.getvalue()
            processed_img, detected_objects = process_image(img_data, conf_thres=conf_thres)
            
            # Save result
            filename = f"{uuid.uuid4()}.jpg"
            result_path = RESULTS_DIR / filename
            cv2.imwrite(str(result_path), processed_img)
            
            # Convert for display
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(processed_img_rgb)
            
            # Display results
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
                
            logger.info(f"Camera image detection completed, saved as {filename}")
            
        except Exception as e:
            st.error(f"Error processing camera image: {str(e)}")
            logger.error(f"Error in camera capture: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to get device type
def is_mobile():
    # Simple user agent check to determine if the user is on a mobile device
    try:
        user_agent = st.experimental_get_query_params().get('user_agent', [''])[0].lower()
        return any(device in user_agent for device in ['mobile', 'android', 'iphone', 'ipad', 'ipod'])
    except:
        return False

# Streamlit app
def main():
    st.title("Grocery Product Detection")
    st.markdown("Detect multiple grocery products in a single image or video frame")
    
    # Initialize model at startup
    if 'model_initialized' not in st.session_state:
        with st.spinner("Loading model... This might take a moment"):
            st.session_state.model_initialized = init_model()
    
    # Check if running on mobile
    mobile_device = is_mobile()
    
    # Mode selector with appropriate options for device type
    if mobile_device:
        mode_options = ["Camera Capture", "Image Upload", "Webcam Detection (Desktop Only)"]
        default_mode = "Camera Capture"
    else:
        mode_options = ["Webcam Detection", "Image Upload"]
        default_mode = "Webcam Detection"
    
    mode = st.selectbox("Select Mode", mode_options, index=mode_options.index(default_mode), key="mode")

    if mode == "Webcam Detection" or mode == "Webcam Detection (Desktop Only)":
        st.subheader("Real-time Webcam Detection")
        
        if mobile_device and mode == "Webcam Detection (Desktop Only)":
            st.warning("WebRTC webcam detection works best on desktop. Please use Camera Capture on mobile devices.")
        
        # Confidence threshold slider
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="webcam_conf"
        )
        
        # Additional WebRTC configuration for better mobile compatibility
        rtc_config = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]}
        )
        
        # Initialize WebRTC streamer with specific configuration for mobile
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=lambda: VideoProcessor(conf_threshold=conf_thres),
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},  # Lower resolution for mobile
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15},  # Lower framerate for better performance
                    "facingMode": {"ideal": "environment"}  # Use back camera on mobile
                },
                "audio": False
            },
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%", "height": "auto", "max-height": "80vh"},
                "controls": True,
                "autoPlay": True,
                "playsInline": True,  # Important for iOS
                "muted": True,  # Important for autoplay
            },
            mode=WebRtcMode.SENDRECV  # Ensure two-way communication for permissions
        )
        
        if ctx.video_processor:
            ctx.video_processor.set_conf_thres(conf_thres)
        
        st.markdown("""
        **WebRTC Instructions:**
        1. Click 'Start' above to begin webcam feed
        2. If prompted, grant camera permissions
        3. For mobile devices, ensure you're using a modern browser (Chrome/Safari)
        4. If camera doesn't start, try refreshing or using Image Upload instead
        """)
        
    elif mode == "Camera Capture":
        st.subheader("Mobile Camera Capture")
        
        # Confidence threshold slider
        st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="upload_conf"
        )
        
        # Use Streamlit's camera_input for direct camera access
        capture_from_camera()
        
    else:  # Image Upload
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
                
                # Process with status indicator
                with st.spinner("Processing image..."):
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
                    
                    # Create a more detailed dataframe for multiple objects
                    df = pd.DataFrame([
                        {
                            "Product": obj["label"], 
                            "Confidence": f"{obj['confidence']*100:.1f}%",
                            "Position": f"({obj['bbox'][0]},{obj['bbox'][1]})"
                        }
                        for obj in detected_objects
                    ])
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button for processed image
                    buf = BytesIO()
                    pil_img.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Processed Image",
                        data=byte_im,
                        file_name=f"detected_{filename}",
                        mime="image/jpeg"
                    )
                else:
                    st.write("No objects detected. Try adjusting the confidence threshold or using a clearer image.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                logger.info(f"Image detection completed, saved as {filename}")
                
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Unexpected error in image upload: {str(e)}")

if __name__ == "__main__":
    main()