from ultralytics import YOLO

# Load your trained model
model = YOLO('./models/single_model0.3.1.pt')  

# Print the architecture summary
model.info(verbose=True)
print(model.model)