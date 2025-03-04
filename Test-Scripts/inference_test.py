from ultralytics import YOLO
import time

if __name__ == "__main__":
    # Load a model
    model = YOLO("./models/single_model0.1.1.pt")
    start_time=time.perf_counter()
    image_path = "./narrowleaf_cattail-6-19-24-4746.JPG"
    results = model.predict(image_path)
    
    results[0].show()
