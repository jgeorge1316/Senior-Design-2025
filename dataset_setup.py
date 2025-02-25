import os
import shutil
import random

# Define paths
source_dir = r"/home/landon/Senior-Design/Training"
target_dir = r"/home/landon/Senior-Design/dataset2"

# Define classes
classes = ["narrowleaf_cattail", "none", "phragmites", "purple_loosestrife"]
splits = {"train": 70, "val": 15, "test": 15}  # Percentage split

# Create target directory structure
for split in splits:
    for class_name in classes:
        os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

# Process each class
for class_name in classes:
    source_path = os.path.join(source_dir, class_name)
    images = [f for f in os.listdir(source_path)]
    
    # Randomly sample up to 1500 images
    selected_images = random.sample(images, min(1500, len(images)))
    
    # Distribute images across train, val, test
    split_counts = {k: int(v * len(selected_images) / 100) for k, v in splits.items()}
    start_idx = 0

    for split, count in split_counts.items():
        split_path = os.path.join(target_dir, split, class_name)
        for i in range(count):
            img_name = f"{start_idx}_{class_name}.jpg"
            shutil.copy(os.path.join(source_path, selected_images[start_idx]), os.path.join(split_path, img_name))
            start_idx += 1