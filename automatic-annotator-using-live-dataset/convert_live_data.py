import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Set page config
st.set_page_config(page_title="YOLOv8 Detection", layout="wide")

# Load model - UPDATE THIS PATH to your actual model path
@st.cache_resource
def load_model():
    # Change this to your actual model path
    model_path = "models/best.pt"  # or "yolov8n.pt" or whatever your model file is named
    try:
        model = YOLO(model_path)
        st.success(f"✅ Model loaded: {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

def main():
    st.title("🚀 YOLOv8 Object Detection")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Video stream URL - using your URL
    video_url = "http://10.2.3.140:8080/video"
    
    # Start/stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🎥 START DETECTION", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)
    
    # Placeholder for video
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Detection state
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if start_btn:
        st.session_state.running = True
    
    if stop_btn:
        st.session_state.running = False
    
    # Main detection loop
    if st.session_state.running:
        status_placeholder.info("🟢 Starting video stream...")
        
        # Open video stream
        cap = cv2.VideoCapture(video_url)
        
        if not cap.isOpened():
            status_placeholder.error("❌ Failed to open video stream. Check URL and connection.")
            st.session_state.running = False
        else:
            status_placeholder.success("🟢 Detection running...")
            
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.running:
                ret, frame = cap.read()
                
                if not ret:
                    status_placeholder.warning("⚠ Failed to read frame. Retrying...")
                    time.sleep(1)
                    continue
                
                # Run YOLOv8 detection
                results = model(frame, verbose=False)
                
                # Draw detections on frame
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Calculate FPS
                frame_count += 1
                fps = frame_count / (time.time() - start_time)
                
                # Display frame
                video_placeholder.image(rgb_frame, 
                                      caption=f"Live Detection - FPS: {fps:.1f}", 
                                      use_column_width=True)
            
            # Clean up
            cap.release()
            status_placeholder.info("🟡 Detection stopped")
            video_placeholder.info("Click START to begin detection")
    
    else:
        video_placeholder.info("👆 Click START DETECTION to begin")
        status_placeholder.warning("🔴 Ready to start")

if _name_ == "_main_":
    main()