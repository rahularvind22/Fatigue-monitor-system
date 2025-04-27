import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

project_dir = os.getcwd()
eye_dir = os.path.join(project_dir, 'eye')
yawn_dir = os.path.join(project_dir, 'yawn')

eye_model_path = os.path.join(eye_dir, 'eye_model_best.pt')
yawn_model_path = os.path.join(yawn_dir, 'best_yawn_model.pth')

eye_model = torch.load(eye_model_path, map_location=torch.device('cpu'))
eye_model.eval()

yawn_model = torch.load(yawn_model_path, map_location=torch.device('cpu'))
yawn_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured.")
        break

    # Step 7: Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)

    # Step 8: Predict
    with torch.no_grad():
        eye_output = eye_model(input_tensor)
        yawn_output = yawn_model(input_tensor)

    eye_pred = torch.argmax(eye_output, dim=1).item()
    yawn_pred = torch.argmax(yawn_output, dim=1).item()

    # Step 9: Draw results
    label_eye = "Open" if eye_pred == 0 else "Closed"
    label_yawn = "No Yawn" if yawn_pred == 0 else "Yawning"

    cv2.putText(frame, f"Eye: {label_eye}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Yawn: {label_yawn}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow('Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()