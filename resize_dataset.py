import os
import glob
import cv2
import numpy as np
from tqdm import tqdm # Optional: for progress bar (pip install tqdm)

# --- CONFIGURATION ---
ORIGINAL_DIR = "./dataset_original"
PROCESSED_DIR = "./dataset" # New folder
TARGET_SIZE = (224, 224) # (Width, Height)

def preprocess_dataset():
    # 1. Create the new directory structure
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    classes = [d for d in os.listdir(ORIGINAL_DIR) if os.path.isdir(os.path.join(ORIGINAL_DIR, d))]
    
    for cls in classes:
        original_class_dir = os.path.join(ORIGINAL_DIR, cls)
        new_class_dir = os.path.join(PROCESSED_DIR, cls)
        
        if not os.path.exists(new_class_dir):
            os.makedirs(new_class_dir)
            
        video_files = glob.glob(os.path.join(original_class_dir, "*.avi"))
        print(f"Processing class '{cls}' ({len(video_files)} videos)...")
        
        for video_path in video_files:
            filename = os.path.basename(video_path)
            save_path = os.path.join(new_class_dir, filename)
            
            # Skip if already exists
            if os.path.exists(save_path):
                continue
                
            # Process Video
            resize_and_save_video(video_path, save_path)

def resize_and_save_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # Get original properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for .avi
    
    # Create Writer
    out = cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # RESIZE HERE (The expensive part)
        resized_frame = cv2.resize(frame, TARGET_SIZE)
        out.write(resized_frame)
        
    cap.release()
    out.release()

if __name__ == "__main__":
    preprocess_dataset()
    print("Done! Use './dataset_300x300' in your VideoLoader now.")