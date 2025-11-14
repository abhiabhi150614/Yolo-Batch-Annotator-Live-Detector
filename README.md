+# Automatic Annotator Using Live Dataset

A YOLO-based object detection system that can automatically annotate images and perform live detection from video streams.

## Features

- **Batch Image Annotation**: Automatically annotate multiple images using a trained YOLO model
- **Live Video Detection**: Real-time object detection from video streams
- **Dataset Visualization**: View annotated images and their detection labels

## Files

- `convert_images.py` - Batch annotate images and generate YOLO format labels
- `convert_live_data.py` - Streamlit app for live video detection
- `visulize_converted_dataset.py` - Streamlit app to visualize annotated dataset

## Requirements

```bash
pip install ultralytics opencv-python streamlit pathlib
```

## Setup

1. Place your trained YOLO model in `models/best.pt`
2. Put images to annotate in `images/` folder
3. Update paths in the scripts if needed

## Usage

### Batch Image Annotation
```bash
python convert_images.py
```
- Processes all images in `images/` folder
- Saves annotated images to `annotated/annotated_images/`
- Saves YOLO labels to `annotated/labels/`

### Live Video Detection
```bash
streamlit run convert_live_data.py
```
- Opens web interface for live detection
- Default video URL: `http://10.2.3.140:8080/video`
- Click START to begin detection

### Visualize Dataset
```bash
streamlit run visulize_converted_dataset.py
```
- Browse through annotated images
- View detection labels and coordinates

## Configuration

Update these variables in the scripts:
- `MODEL_PATH`: Path to your YOLO model
- `IMAGES_DIR`: Input images folder
- `OUTPUT_DIR`: Output folder for results
- `CONFIDENCE`: Detection confidence threshold (0.5 default)
