# ----------------- Imports -----------------
import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from torchvision import transforms, models
from PIL import Image
import platform
import mediapipe as mp

# ----------------- Config -----------------
IMAGE_SIZE = 100
CLASSES = ['Closed', 'Open']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- Model Definitions -----------------
class EyeCNN(nn.Module):
    def __init__(self, outputs):
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
        self.fc = nn.Linear(256, outputs)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def create_resnet18(outputs):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, outputs)
    return model

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_selected_model(model_name, model_path):
    if model_name == "EyeCNN":
        model = EyeCNN(outputs=2)
    elif model_name == "ResNet18":
        model = create_resnet18(outputs=2)
    else:
        raise ValueError("Invalid model name selected.")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

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
    if platform.system() == "Darwin":
        os.system('say "Alert! Wake up"')
    elif platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 300)
    else:
        os.system('spd-say "wake up"')


def crop_eye_landmarks(frame, landmarks, indices):
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*points)

    x1 = max(0, min(x_coords) - 10)
    y1 = max(0, min(y_coords) - 30)
    x2 = min(w, max(x_coords) + 10)
    y2 = min(h, max(y_coords) + 30)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")
    st.title("🚗 Driver Drowsiness Detection")
    st.markdown("Detects if eyes are **Open** or **Closed** and plays a **beep** when drowsiness is detected.")
    st.markdown("---")

    model_choice = st.selectbox("Select Model to Load", ("EyeCNN", "ResNet18"))

    project_dir = os.getcwd()
    model_paths = {
        "EyeCNN": os.path.join(project_dir, 'eye', 'eye_model_best.pt'),
        "ResNet18": os.path.join(project_dir, 'eye', 'resnet18_eye_model_best.pt')
    }

    selected_model_path = model_paths[model_choice]
    model = load_selected_model(model_choice, selected_model_path)

    start_detection = st.button("Start Detection")

    frame_placeholder = st.empty()
    col1, col2 = st.columns(2)
    eye_placeholder_1 = col1.empty()
    eye_placeholder_2 = col2.empty()

    if start_detection:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 40)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 155]
        RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]

        closed_start_time = None
        alert_triggered = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            predictions = []
            eye_imgs = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_crop, left_box = crop_eye_landmarks(frame, face_landmarks.landmark, LEFT_EYE)
                    right_eye_crop, right_box = crop_eye_landmarks(frame, face_landmarks.landmark, RIGHT_EYE)

                    for eye_crop, label, box in [(left_eye_crop, "Left", left_box), (right_eye_crop, "Right", right_box)]:
                        if eye_crop.size == 0:
                            continue
                        tensor, pil_eye = preprocess_eye(eye_crop)
                        pred = predict_eye(model, tensor)
                        predictions.append(pred)
                        eye_imgs.append(pil_eye)

                        color = (0, 255, 0) if pred == 'Open' else (255, 0, 0)
                        x1, y1, x2, y2 = box
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(rgb, f"{label}: {pred}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if len(predictions) >= 2 and predictions.count('Closed') >= 2:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time > 0.5:
                    alert_triggered = True
            else:
                closed_start_time = None
                alert_triggered = False

            if alert_triggered:
                beep_sound()
                cv2.putText(rgb, "DROWSINESS ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            frame_placeholder.image(rgb)

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
