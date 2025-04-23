import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ---------- Config ----------
IMAGE_SIZE = 100
MODEL_PATH = "eye_state_model.pth"  # Update path if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Closed', 'Open']

# ---------- Model Definition ----------
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- Load Trained Model ----------
model = EyeCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ---------- Haar Cascades ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Start Webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to access webcam.")
    exit()

print("🎥 Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = frame_rgb[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes[:2]:  # Limit to max 2 eyes
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_resized = cv2.resize(eye_img, (IMAGE_SIZE, IMAGE_SIZE))
            eye_rgb = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)
            eye_pil = Image.fromarray(eye_rgb)
            input_tensor = transform(eye_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                label = CLASSES[pred_class]

            color = (0, 255, 0) if label == 'Open' else (0, 0, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(roi_color, label, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Eye State Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
