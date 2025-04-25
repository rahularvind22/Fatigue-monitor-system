import os

import cv2
import torch
import numpy as np
from torchvision import transforms
from yawn_detection_train import CNNBinaryClassifier


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBinaryClassifier().to(device)
model.load_state_dict(torch.load("best_yawn_model.pth", map_location=device))
model.eval()

# Transform for preprocessing each frame
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Class labels
labels = ["No Yawn", "Yawn"]

# Start camera
cap = cv2.VideoCapture(0)  # 0 for default webcam

print("[INFO] Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Crop center of the frame or face detection
    img = cv2.resize(frame, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = labels[predicted.item()]

    # Display prediction on frame
    cv2.putText(frame, f"Prediction: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "No Yawn" else (0, 0, 255), 2)
    cv2.imshow("Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
