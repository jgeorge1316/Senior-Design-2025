from ultralytics import YOLO
import os
import time
import multiprocessing
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import torch.nn.functional as F

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

    # Generate and plot confusion matrix using torchmetrics
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    cm = MulticlassConfusionMatrix(num_classes=len(class_names))
    confusion_matrix = cm(y_pred_tensor, y_true_tensor)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix.numpy(), cmap='Blues')

    # Labeling the axes
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Display values in the matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig("confusion_matrix_ultralytics_style.png")
    plt.show()
