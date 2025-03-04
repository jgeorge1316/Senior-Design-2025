import os
import shutil
import hashlib
from tqdm import tqdm

# Define paths
source_dir = r"/home/landon/Senior-Design/Training"  # Large dataset with all images
target_dir = r"/home/landon/Senior-Design/TestSet_ForDataset3"  # Destination for filtered test set
train_val_dirs = [
    r"/home/landon/Senior-Design/dataset3/train",
    r"/home/landon/Senior-Design/dataset3/val"
]

# Function to compute hash of an image
def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# Collect hashes of training and validation images
used_hashes = set()
for base_dir in train_val_dirs:
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        for img in tqdm(os.listdir(class_path), desc=f"Processing {class_name} (train/val)"):
            img_path = os.path.join(class_path, img)
            used_hashes.add(compute_hash(img_path))

# Process each class in the original dataset
for class_name in os.listdir(source_dir):
    source_class_path = os.path.join(source_dir, class_name)
    target_class_path = os.path.join(target_dir, class_name)
    os.makedirs(target_class_path, exist_ok=True)
    
    images = os.listdir(source_class_path)
    for img in tqdm(images, desc=f"Filtering {class_name} (test set)"):
        img_path = os.path.join(source_class_path, img)
        if compute_hash(img_path) not in used_hashes:
            shutil.copy(img_path, os.path.join(target_class_path, img))
