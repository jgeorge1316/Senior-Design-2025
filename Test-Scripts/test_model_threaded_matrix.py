from ultralytics import YOLO
import os
import time
import multiprocessing
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model once globally
model = YOLO("./models/single_model0.4.1.pt")

# Root test folder (contains subfolders for each class)
test_root = "/home/landon/Senior-Design/dataset6/test"

# Get class names and folder mapping
class_names = sorted(os.listdir(test_root))
class_to_index = {name: idx for idx, name in enumerate(class_names)}

def process_image(args):
    image_path, true_class_index = args
    with torch.no_grad():
        result = model.predict(image_path, stream=False, verbose=False)[0]
        pred_index = result.probs.top1
        return (true_class_index, pred_index)

if __name__ == "__main__":
    start_time = time.perf_counter()

    image_info = []

    # Collect all images and their true class index
    for class_name in class_names:
        folder_path = os.path.join(test_root, class_name)
        if not os.path.isdir(folder_path):
            continue
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_info.extend([(img_path, class_to_index[class_name]) for img_path in image_files])

    num_workers = min(multiprocessing.cpu_count(), 30)

    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_image, image_info), total=len(image_info), desc="Processing Images"))

    # Separate true and predicted labels
    y_true = [true for true, pred in results]
    y_pred = [pred for true, pred in results]

    # Compute accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    end_time = time.perf_counter()

    print(f"Total process time: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
