import cv2
import torch
import numpy as np
from torchvision import transforms
from yawn_detection_train import ResNet18BinaryClassifier  # Make sure this import matches your train file

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = ResNet18BinaryClassifier().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Define image transformations (must match train/test)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ["no_yawn", "yawn"]

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("[INFO] Starting real-time yawn detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Prepare frame for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds.item()]

    # Display the prediction on the frame
    color = (0, 255, 0) if label == "no_yawn" else (0, 0, 255)
    cv2.putText(frame, f"Prediction: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow('Yawn Detection (Press q to quit)', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
