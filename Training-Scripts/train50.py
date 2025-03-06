from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("./yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="../dataset6", epochs=50, imgsz=640, batch=32, patience=20, device=1)