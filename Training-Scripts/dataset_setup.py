import os
import shutil
import random

# Define paths
source_dir = r"/home/landon/Senior-Design/Training"
target_dir = r"/home/landon/Senior-Design/dataset6"

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
    remaining_images = list(set(images) - set(selected_images))  # Get unselected images
    
    # Distribute images across train, val, test
    split_counts = {k: int(v * len(selected_images) / 100) for k, v in splits.items()}
    start_idx = 0

    for split, count in split_counts.items():
        split_path = os.path.join(target_dir, split, class_name)
        for i in range(count):
            img_name = f"{start_idx}_{class_name}.jpg"
            shutil.copy(os.path.join(source_path, selected_images[start_idx]), os.path.join(split_path, img_name))
            start_idx += 1

    print(f"Adding {len(remaining_images)} additional images to test set for class {class_name}")
    test_path = os.path.join(target_dir, "test", class_name)
    #gpt is dumb so making my own thing
    for i in range(len(remaining_images)):
        img_name = f"extra_{remaining_images[i]}.jpg"
        shutil.copy(os.path.join(source_path, remaining_images[i]), os.path.join(test_path, img_name))
    
    '''# Add remaining selected images to test set
    test_path = os.path.join(target_dir, "test", class_name)
    for i in range(start_idx, len(selected_images)):
        img_name = f"{i}_{class_name}.jpg"
        shutil.copy(os.path.join(source_path, selected_images[i]), os.path.join(test_path, img_name))
    
    # Add all unselected images to the test set
    print(f"Adding {len(remaining_images)} additional images to test set for class {class_name}")
    for img in remaining_images:
        img_name = f"extra_{img}"  # Keep original filename but prefix with 'extra_'
        shutil.copy(os.path.join(source_path, img), os.path.join(test_path, img_name))
    '''
    