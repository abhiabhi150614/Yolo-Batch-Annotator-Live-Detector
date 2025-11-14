import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def auto_annotate_images():
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "models/best.pt"  # Path to your YOLO model
    IMAGES_DIR = "images"          # Folder containing images to annotate
    OUTPUT_DIR = "annotated"       # Folder to save results
    CONFIDENCE = 0.5              # Detection confidence threshold
    
    # Load model
    print("🚀 Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    # Create output directories
    annotated_dir = Path(OUTPUT_DIR) / "annotated_images"
    labels_dir = Path(OUTPUT_DIR) / "labels"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(IMAGES_DIR).glob(ext))
        image_paths.extend(Path(IMAGES_DIR).glob(ext.upper()))
    
    if not image_paths:
        print(f"❌ No images found in {IMAGES_DIR}")
        return
    
    print(f"📸 Found {len(image_paths)} images")
    print("⏳ Starting annotation...")
    
    total_detections = 0
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"Processing {i}/{len(image_paths)}: {img_path.name}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ❌ Failed to read {img_path.name}")
            continue
        
        # Run detection
        results = model(image, conf=CONFIDENCE, verbose=False)
        
        # Save annotated image
        annotated_image = results[0].plot()
        output_img_path = annotated_dir / img_path.name
        cv2.imwrite(str(output_img_path), annotated_image)
        
        # Save YOLO format labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                        total_detections += 1
        
        print(f"  ✅ Saved: {output_img_path.name} and {label_path.name}")
    
    print(f"\n🎉 Annotation completed!")
    print(f"📊 Total images processed: {len(image_paths)}")
    print(f"🎯 Total objects detected: {total_detections}")
    print(f"📁 Annotated images: {annotated_dir}")
    print(f"📄 Label files: {labels_dir}")

if _name_ == "_main_":
    auto_annotate_images()