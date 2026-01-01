import os
import shutil
import random
from pathlib import Path

def merge_datasets():
    # Define sources
    sources = [
        # This one seems to be deep nested, let's target the leaf folders
        r"e:/SIH2/data/Pomegranate Diseases Dataset", 
        r"e:/SIH2/data/Pomegranate Fruit Diseases Dataset for Deep Learning Models/Pomegranate Diseases Dataset",
        # original one, if it has images. E:/SIH2/data/pomegranate might be raw zip or empty, 
        # let's assume dataset_processed was the reliable source.
        r"e:/SIH2/dataset_processed/train",
        r"e:/SIH2/dataset_processed/val",
        r"e:/SIH2/dataset_processed/test"
    ]
    
    # Target
    target_dir = r"e:/SIH2/dataset_merged"
    splits = ['train', 'val', 'test']
    classes = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']
    
    # Ratios
    train_ratio = 0.8
    val_ratio = 0.1
    # test is remainder
    
    # Create target structure
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(target_dir, s, c), exist_ok=True)
            
    print("Collecting images...")
    
    # Dictionary to hold all images per class
    # class_name -> list of file_paths
    image_pool = {c: [] for c in classes}
    
    for src in sources:
        if not os.path.exists(src):
            print(f"Source not found (skipping): {src}")
            continue
            
        print(f"Scanning {src}...")
        # Check if source has class folders directly or is one level up
        # We try to find class folders inside src
        for c in classes:
            # Check direct match or prefixed (e.g. pomegranate_Healthy)
            # dataset_processed has 'pomegranate_Healthy'
            
            # 1. Try exact name
            class_path = os.path.join(src, c)
            if not os.path.exists(class_path):
                # 2. Try prefix 'pomegranate_' which was in dataset_processed
                class_path = os.path.join(src, f"pomegranate_{c}")
                
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                for f in files:
                    image_pool[c].append(os.path.join(class_path, f))
                    
    # Distribute
    print("Distributing images...")
    total_imgs = 0
    for c, files in image_pool.items():
        unique_files = list(set(files)) # simple dedup by path string, might not be enough if copies exist
        # Better: dedup by filename? 
        # Many datasets result in "IMG_..." duplicates if merged.
        # Let's rename them uniquely to be safe.
        
        random.shuffle(unique_files)
        count = len(unique_files)
        total_imgs += count
        print(f"Class {c}: {count} images")
        
        train_end = int(count * train_ratio)
        val_end = train_end + int(count * val_ratio)
        
        train_files = unique_files[:train_end]
        val_files = unique_files[train_end:val_end]
        test_files = unique_files[val_end:]
        
        for idx, f in enumerate(train_files):
            shutil.copy(f, os.path.join(target_dir, 'train', c, f"merged_{idx}_{os.path.basename(f)}"))
            
        for idx, f in enumerate(val_files):
            shutil.copy(f, os.path.join(target_dir, 'val', c, f"merged_{idx}_{os.path.basename(f)}"))
            
        for idx, f in enumerate(test_files):
            shutil.copy(f, os.path.join(target_dir, 'test', c, f"merged_{idx}_{os.path.basename(f)}"))
            
    print(f"Done. Total images: {total_imgs}")
    print(f"Data ready at {target_dir}")

if __name__ == "__main__":
    merge_datasets()
