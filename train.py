!pip install ultralytics kagglehub pyyaml matplotlib seaborn --quiet

import os
import yaml
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from ultralytics import YOLO
from google.colab import files
import torch
import shutil

# Set random seed and visuals
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
sns.set_theme(style="darkgrid", font_scale=1.5)
plt.rcParams["figure.figsize"] = [12, 8]

# Dataset parameters
CLASSES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]
OUTPUT_DIR = "/content"

# Download and prepare dataset
def setup_dataset():
    print("📥 Downloading Fruits Detection dataset...")
    try:
        # Download dataset using kagglehub
        kaggle_path = kagglehub.dataset_download("lakshaytyagi01/fruit-detection")
        print(f"Dataset downloaded to: {kaggle_path}")
        
        # The actual dataset is inside 'Fruits-detection' subfolder
        data_dir = os.path.join(kaggle_path, "Fruits-detection")
        
        # Verify dataset structure
        required_dirs = {
            'train': os.path.join(data_dir, "train/images"),
            'val': os.path.join(data_dir, "valid/images"),
            'test': os.path.join(data_dir, "test/images")
        }
        
        for name, dir_path in required_dirs.items():
            if not os.path.exists(dir_path):
                print(f"Directory not found: {dir_path}")
                print(f"Contents of {data_dir}: {os.listdir(data_dir)}")
                raise FileNotFoundError("❌ Invalid dataset structure")
        
        print("Dataset structure verified successfully")
        
        # Create data.yaml
        data_yaml = {
            "path": data_dir,
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(CLASSES),
            "names": CLASSES
        }
        
        yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)
        
        print(f"📁 data.yaml created at: {yaml_path}")
        return data_dir
        
    except Exception as e:
        print(f"Failed to setup dataset: {str(e)}")
        raise

# Train model
def train_model(data_dir):
    print("🚀 Starting YOLOv8 training...")
    os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases
    
    # Load model
    model = YOLO("yolov8n.pt")  # Load pretrained nano model
    
    try:
        # Train the model
        results = model.train(
            data=os.path.join(OUTPUT_DIR, "data.yaml"),
            epochs=50,
            imgsz=640,
            batch=8,
            device="0" if torch.cuda.is_available() else "cpu",
            seed=42,
            patience=5,
            project=os.path.join(OUTPUT_DIR, "runs/detect"),
            name="fruit_detection"
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    try:
        metrics = model.val()
        print(f"mAP@0.5: {metrics.box.map:.3f}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
    
    # Test on sample images
    print("\n🖼️ Visualizing sample predictions:")
    val_images = glob.glob(os.path.join(data_dir, "valid/images", "*.jpg"))
    sample_imgs = random.sample(val_images, min(3, len(val_images)))
    
    for img in sample_imgs:
        results = model.predict(img, conf=0.3)
        for r in results:
            plt.figure(dpi=100)
            plt.imshow(r.plot()[:, :, ::-1])
            plt.title(f"{len(r.boxes)} fruits detected" if len(r.boxes) > 0 else "No fruits detected")
            plt.axis("off")
            plt.show()
    
    # Save and download model
    model_path = os.path.join(OUTPUT_DIR, "runs/detect/fruit_detection/weights/best.pt")
    if os.path.exists(model_path):
        print(f"✅ Model saved at: {model_path}")
        files.download(model_path)
        return model_path
    else:
        print("❌ Model file not found.")
        return None

# Main execution
def main():
    print("==== Fruit Detection Model Training ====")
    try:
        data_dir = setup_dataset()
        model_path = train_model(data_dir)
        if model_path:
            print(f"\n🎉 Training completed successfully! Model saved at: {model_path}")
        else:
            print("\n❌ Training failed. Please check the error messages above.")
    except Exception as e:
        print(f"\n🔥 Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
