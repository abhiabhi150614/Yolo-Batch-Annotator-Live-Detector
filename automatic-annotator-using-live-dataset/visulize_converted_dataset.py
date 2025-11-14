import streamlit as st
import cv2
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(page_title="YOLO Visualizer", layout="wide")

# Configuration
IMAGES_DIR = "annotated/annotated_images"  # or "images" if you want original images
LABELS_DIR = "annotated/labels"

# Get all images
image_paths = list(Path(IMAGES_DIR).glob("*"))
image_names = [img.name for img in image_paths]

if not image_names:
    st.error("No images found!")
    st.stop()

# Image selector
selected_image = st.selectbox("Select Image:", image_names)

if selected_image:
    # Load image
    img_path = Path(IMAGES_DIR) / selected_image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display image
    st.image(image, use_column_width=True, caption=selected_image)
    
    # Show label info
    label_path = Path(LABELS_DIR) / f"{Path(selected_image).stem}.txt"
    if label_path.exists():
        with open(label_path, 'r') as f:
            labels = f.readlines()
        st.write(f"*Detections:* {len(labels)}")
        
        for label in labels:
            class_id, x_center, y_center, width, height = label.split()
            st.write(f"Class {class_id} - Center: ({x_center}, {y_center}) Size: ({width}, {height})")