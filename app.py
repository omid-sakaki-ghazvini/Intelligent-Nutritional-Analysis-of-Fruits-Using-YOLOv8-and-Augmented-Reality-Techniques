!pip install ultralytics opencv-python numpy matplotlib tqdm --upgrade --quiet

import cv2
import numpy as np
import os
from tqdm import tqdm
from google.colab import files
from datetime import datetime
from ultralytics import YOLO

# Environment setup
OUTPUT_DIR = "/content"

def upload_model():
    """Function to handle model upload"""
    print("\nPlease upload your trained model file (best.pt):")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded! Using default YOLOv8n model.")
        return None
    
    for filename in uploaded.keys():
        if filename.endswith('.pt'):
            model_path = os.path.join(OUTPUT_DIR, filename)
            with open(model_path, 'wb') as f:
                f.write(uploaded[filename])
            print(f"Model uploaded successfully: {filename}")
            return model_path
    
    print("No .pt file found in upload. Using default YOLOv8n model.")
    return None

# Complete nutrition database including kiwi
def create_nutrition_db():
    return {
        "apple": {"name": "Apple", "energy": 52, "protein": 0.3, "fat": 0.2, "carbs": 14, "color": (50, 205, 50)},
        "banana": {"name": "Banana", "energy": 89, "protein": 1.1, "fat": 0.3, "carbs": 23, "color": (0, 255, 255)},
        "grape": {"name": "Grape", "energy": 69, "protein": 0.7, "fat": 0.2, "carbs": 18, "color": (128, 0, 128)},
        "orange": {"name": "Orange", "energy": 47, "protein": 0.9, "fat": 0.1, "carbs": 12, "color": (0, 165, 255)},
        "pineapple": {"name": "Pineapple", "energy": 50, "protein": 0.5, "fat": 0.1, "carbs": 13, "color": (0, 215, 255)},
        "watermelon": {"name": "Watermelon", "energy": 30, "protein": 0.6, "fat": 0.2, "carbs": 8, "color": (0, 0, 255)},
        "kiwi": {"name": "Kiwi", "energy": 61, "protein": 1.1, "fat": 0.5, "carbs": 15, "color": (142, 229, 63)}
    }

# Extended label mapping including all fruit variations
LABEL_MAPPING = {
    "grapes": "grape", "grape_fresh": "grape",
    "apple": "apple", "apple_fresh": "apple",
    "banana": "banana", "banana_fresh": "banana",
    "orange": "orange", "orange_fresh": "orange",
    "kiwi": "kiwi", "kiwi_fruit": "kiwi", "kiwi_fresh": "kiwi",
    "pineapple": "pineapple", "pineapple_fresh": "pineapple",
    "watermelon": "watermelon", "watermelon_fresh": "watermelon"
}

def draw_nutrition_table(frame, nutrition, x, y):
    """Draws an enlarged nutrition information table"""
    table_width = 320  # Wider table
    table_height = 220 # Taller table
    header_height = 45
    row_height = 45    # More space between rows
    font_scale = 0.9   # Larger font size
    thickness = 1
    
    # Create table background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+table_width, y+table_height), (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Add colored header
    cv2.rectangle(frame, (x, y), (x+table_width, y+header_height), nutrition['color'], -1)
    
    # Add fruit name in header
    cv2.putText(frame, nutrition['name'], 
               (x+10, y+header_height-10), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness+1)
    
    # Nutrition facts data
    nutrients = [
        ("Energy", f"{nutrition['energy']} kcal"),
        ("Protein", f"{nutrition['protein']} g"),
        ("Fat", f"{nutrition['fat']} g"),
        ("Carbs", f"{nutrition['carbs']} g")
    ]
    
    # Draw each nutrition fact
    for i, (label, value) in enumerate(nutrients):
        y_pos = y + header_height + (i+1)*row_height
        cv2.putText(frame, label, (x+15, y_pos-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 255), thickness)
        cv2.putText(frame, value, (x+200, y_pos-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness+1)
        cv2.line(frame, (x+10, y_pos), (x+table_width-10, y_pos), (100, 100, 100), 1)
    
    return frame

def process_video(video_path, model, nutrition_db):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f'nutrition_analysis_{timestamp}.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects with lower confidence threshold
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), imgsz=640, conf=0.25)
        
        for result in results:
            # Debug: Print all detected objects
            print("Detected objects:", [result.names[int(cls)] for cls in result.boxes.cls])
            
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                original_label = result.names[int(cls)].lower()
                mapped_label = LABEL_MAPPING.get(original_label, original_label)
                
                print(f"Processing: {original_label} -> {mapped_label}")
                
                if mapped_label in nutrition_db:
                    nutrition = nutrition_db[mapped_label]
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), nutrition['color'], 3)
                    
                    # Draw nutrition table (automatically choose best position)
                    if x2 + 340 < frame_width:  # Right side
                        frame = draw_nutrition_table(frame, nutrition, x2+20, y1)
                    elif x1 - 340 > 0:  # Left side
                        frame = draw_nutrition_table(frame, nutrition, x1-340, y1)
                    else:  # Below if no space on sides
                        frame = draw_nutrition_table(frame, nutrition, x1, min(y2+20, frame_height-220))
        
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path

def main():
    print("==== Advanced Fruit Nutrition Analyzer ====")
    
    # Step 1: Model Loading
    print("\n[1/3] Model Loading Phase")
    model_path = upload_model()
    
    if model_path:
        try:
            model = YOLO(model_path)
            print("Model classes:", model.names)  # Show what fruits the model knows
            print("Custom model loaded successfully!")
        except Exception as e:
            print(f"Error loading custom model: {str(e)}")
            print("Falling back to YOLOv8n model")
            model = YOLO("yolov8n.pt")
    else:
        print("Using default YOLOv8n model")
        model = YOLO("yolov8n.pt")
    
    nutrition_db = create_nutrition_db()
    
    # Step 2: Video Upload
    print("\n[2/3] Please upload your fruit video (MP4 format, max 100MB):")
    uploaded = files.upload()
    if not uploaded:
        print("No video uploaded! Exiting...")
        return
    
    video_name = next(iter(uploaded))
    input_path = os.path.join(OUTPUT_DIR, video_name)
    os.rename(video_name, input_path)
    
    # Step 3: Processing
    print("\n[3/3] Processing video...")
    try:
        output_path = process_video(input_path, model, nutrition_db)
        print("\nProcessing complete! Downloading your video...")
        files.download(output_path)
        print("\nDone! The processed video has been downloaded.")
    except Exception as e:
        print(f"\nError processing video: {str(e)}")

if __name__ == "__main__":
    main()
