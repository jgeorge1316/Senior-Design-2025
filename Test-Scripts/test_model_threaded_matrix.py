from ultralytics import YOLO
import os
import time
import multiprocessing
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import csv

# Load model globally
model = YOLO("./models/single_model0.3.1.pt")

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
        return (image_path, true_class_index, pred_index)

def save_csv(results, class_names, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_class", "predicted_class", "true_index", "predicted_index"])
        for image_path, true_idx, pred_idx in results:
            writer.writerow([image_path, class_names[true_idx], class_names[pred_idx], true_idx, pred_idx])
    print(f"Saved CSV results to {output_file}")

def plot_confusion_matrix(conf_matrix, class_names, normalized=False, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = conf_matrix.numpy()

    if normalized:
        matrix = matrix.astype(float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums.clip(min=1e-6)  # avoid division by zero
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = format(matrix[i, j], fmt)
            ax.text(j, i, text, ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved confusion matrix to {filename}")

if __name__ == "__main__":
    start_time = time.perf_counter()

    image_info = []

    for class_name in class_names:
        folder_path = os.path.join(test_root, class_name)
        if not os.path.isdir(folder_path):
            continue
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_info.extend([(img_path, class_to_index[class_name]) for img_path in image_files])

    num_workers = min(multiprocessing.cpu_count(), 30)

    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_image, image_info), total=len(image_info), desc="Processing Images"))

    # Save to CSV
    csv_path = "classification_results.csv"
    save_csv(results, class_names, csv_path)

    # Extract true and predicted indices
    y_true = [true for _, true, _ in results]
    y_pred = [pred for _, _, pred in results]

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    end_time = time.perf_counter()
    print(f"Total process time: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Compute confusion matrix
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    cm_metric = MulticlassConfusionMatrix(num_classes=len(class_names))
    cm = cm_metric(y_pred_tensor, y_true_tensor)

    plot_confusion_matrix(cm, class_names, normalized=False, filename="confusion_matrix_raw.png")
    plot_confusion_matrix(cm, class_names, normalized=True, filename="confusion_matrix_normalized.png")
