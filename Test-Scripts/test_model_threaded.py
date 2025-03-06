from ultralytics import YOLO
import os
import time
import multiprocessing
import torch
from tqdm import tqdm

# Load the model once outside of the multiprocessing context
def load_model():
    return YOLO("./models/single_model0.3.1.pt")

# Function to process a single image with the model
def process_image(image_path):
    with torch.no_grad():  # Disable gradient computation to save memory and speed up inference
        result = model.predict(image_path, stream=False, verbose=False)[0]  # Get the first result
        #{0: 'narrowleaf_cattail', 1: 'none', 2: 'phragmites', 3: 'purple_loosestrife'}
        return (result.probs.top1 == 3), image_path  # Return if correct + image path

if __name__ == "__main__":
    start_time = time.perf_counter()

    # Path to folder containing images
    folder_path = "/home/landon/Senior-Design/dataset6/test/purple_loosestrife"
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    num_workers = min(multiprocessing.cpu_count(), 30)  # Limit to x workers or system's CPU count
    count_correct = 0
    count_wrong = 0

    # Load model once outside of the process pool
    model = load_model()

    # Use multiprocessing to process images concurrently with a progress bar
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_image, image_files), total=len(image_files), desc="Processing Images"))

    # Aggregate results
    for correct, _ in results:
        if correct:
            count_correct += 1
        else:
            count_wrong += 1

    count_total = count_correct + count_wrong
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Total process Time for the folder: {elapsed_time:.4f} seconds.")
    print(f"Correct: {count_correct} Wrong: {count_wrong} Total: {count_total}")
    print(f"Percentage Correct: {count_correct / count_total:.4f}" if count_total > 0 else "No images processed.")
