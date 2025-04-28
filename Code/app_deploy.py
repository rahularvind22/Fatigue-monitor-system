import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from torchvision import transforms
from PIL import Image
import platform

# ----------------- Config -----------------
IMAGE_SIZE = 100
CLASSES = ['Closed', 'Open']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- Model -----------------
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.17),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, len(CLASSES))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_model(model_path):
    model = EyeCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

def preprocess_eye(eye_img):
    eye_pil = Image.fromarray(eye_img)
    eye_pil = eye_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    eye_tensor = transforms.ToTensor()(eye_pil)
    return eye_tensor, eye_pil

def predict_eye(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return CLASSES[preds.item()]

def beep_sound():
    """Play system beep depending on OS."""
    if platform.system() == "Darwin":  # macOS
        os.system('say "Alert!! wake up"')
    elif platform.system() == "Windows":  # Windows
        import winsound
        winsound.Beep(1000, 300)  # 1000 Hz for 300 ms
    else:  # Linux
        os.system('spd-say "wake up"')

# ----------------- Streamlit App -----------------
def main():
    st.title("🚗 Driver Drowsiness Detection")
    st.markdown("Detects if eyes are **Open** or **Closed** and plays a **continuous beep** when drowsiness is detected.")
    st.markdown("---")

    project_dir = os.getcwd()
    model_path = os.path.join(project_dir, 'eye', 'eye_model_best.pt')

    model = load_model(model_path)

    start_detection = st.button("Start Detection")

    frame_placeholder = st.empty()
    col1, col2 = st.columns(2)
    eye_placeholder_1 = col1.empty()
    eye_placeholder_2 = col2.empty()

    if start_detection:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Target 40 FPS
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        closed_start_time = None
        alert_triggered = False

        if not cap.isOpened():
            st.error("Could not open webcam.")
            return

        st.info("Press 'Stop' button on top right to end detection.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame. Exiting...")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)

            h_img, w_img, _ = frame_rgb.shape
            predictions = []
            eye_imgs = []

            for (x, y, w, h) in eyes:
                x1 = max(0, x - 20)
                y1 = max(0, y - 10)
                x2 = min(w_img, x + w + 20)
                y2 = min(h_img, y + h + 10)

                eye_img = frame_rgb[y1:y2, x1:x2]
                if eye_img.size == 0:
                    continue

                eye_tensor, eye_pil = preprocess_eye(eye_img)
                pred = predict_eye(model, eye_tensor)

                predictions.append(pred)
                eye_imgs.append(eye_pil)

                color = (0, 255, 0) if pred == 'Open' else (255, 0, 0)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_rgb, pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Drowsiness detection
            if len(predictions) >= 2 and predictions.count('Closed') >= 2:
                if closed_start_time is None:
                    closed_start_time = time.time()
                else:
                    elapsed = time.time() - closed_start_time
                    if elapsed >= 0.5:
                        alert_triggered = True
            else:
                closed_start_time = None
                alert_triggered = False

            if alert_triggered:
                beep_sound()
                cv2.putText(frame_rgb, "DROWSINESS ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            frame_placeholder.image(frame_rgb)

            if len(eye_imgs) >= 1:
                eye_placeholder_1.image(eye_imgs[0], caption="Eye 1")
            else:
                eye_placeholder_1.empty()

            if len(eye_imgs) >= 2:
                eye_placeholder_2.image(eye_imgs[1], caption="Eye 2")
            else:
                eye_placeholder_2.empty()

        cap.release()

if __name__ == "__main__":
    main()
