from ultralytics import YOLO
import time

if __name__ == "__main__":
    # Load a model
    model = YOLO("single_model0.1.1.pt")
    start_time=time.perf_counter()
    image_path = "/home/landon/Downloads/636029771937431354-IMG-8300-1-.webp"
    results1 = model.predict(image_path, visualize=True)
    
    for result in results1:
        result.show()
