# ✅ Enhanced Drowsiness Detection with Face Box, Scaled Eye Box, Input Display, and Separate Alerts

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
DEVICE = torch.device("cpu")
EYE_CLASSES = ['Closed', 'Open']
YAWN_CLASSES = ['no_yawn', 'yawn']

# ----------------- Eye Models -----------------
class EyeCNN(nn.Module):
    def __init__(self, outputs=2):
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

def create_resnet18_eye(outputs=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, outputs)
    return model

# ----------------- Yawn Models -----------------
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)

class ResNet18BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet18BinaryClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# ----------------- Loaders -----------------
@st.cache_resource
def load_model(model_type, task):
    if task == "eye":
        if model_type == "EyeCNN":
            model = EyeCNN(outputs=2)
            path = "eye/eye_model_best.pt"
            size = 100
            normalize = transforms.Normalize([0.5]*3, [0.5]*3)
        else:
            model = create_resnet18_eye(outputs=2)
            path = "eye/resnet18_eye_model_best.pt"
            size = 224
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        if model_type == "CNN":
            model = CNNBinaryClassifier()
            path = "yawn/best_yawn_model_baseline.pth"
            size = 100
            normalize = transforms.Normalize([0.5]*3, [0.5]*3)
        else:
            model = ResNet18BinaryClassifier()
            path = "yawn/best_model_main.pth"
            size = 224
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, size, normalize

# ----------------- Utilities -----------------
def preprocess_image(img, size, normalize):
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize
    ])
    pil_img = transforms.ToPILImage()(cv2.resize(img, (size, size)))
    return transform(img).unsqueeze(0).to(DEVICE), pil_img

def predict(model, tensor, class_names):
    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

def beep_alert(type="generic"):
    msg = "Alert! Wake Up" if type == "eye" else "Rest in Peace"
    if platform.system() == "Darwin":
        os.system(f'say "{msg}"')
    elif platform.system() == "Windows":
        import winsound
        winsound.Beep(1000 if type == "eye" else 1200, 400)
    else:
        os.system(f'spd-say "{msg}"')

# ----------------- App -----------------
def run_drowsiness_app():
    st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
    st.title("🚗 Driver Drowsiness Detection")

    eye_model_choice = st.selectbox("Select Eye Model", ["EyeCNN", "ResNet18"])
    yawn_model_choice = st.selectbox("Select Yawn Model", ["CNN", "ResNet18"])

    eye_model, eye_size, eye_norm = load_model(eye_model_choice, "eye")
    yawn_model, yawn_size, yawn_norm = load_model(yawn_model_choice, "yawn")

    stframe = st.empty()
    col_eye_left, col_eye_right, col_yawn = st.columns(3)
    eye_display_left = col_eye_left.empty()
    eye_display_right = col_eye_right.empty()
    yawn_display = col_yawn.empty()

    if st.button("Start Detection"):
        cap = cv2.VideoCapture(0)
        mp_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
        LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 155]
        RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]

        eye_closed_start = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            mesh_results = mp_mesh.process(rgb)
            face_results = mp_face.process(rgb)

            # --- Yawn Detection ---
            if face_results.detections:
                bboxC = face_results.detections[0].location_data.relative_bounding_box
                x1 = max(0, int(bboxC.xmin * w) - 50)
                y1 = max(0, int(bboxC.ymin * h) - 200)
                x2 = min(w, int((bboxC.xmin + bboxC.width) * w) + 50)
                y2 = min(h, int((bboxC.ymin + bboxC.height) * h) + 100)
                face_crop = rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    tensor, disp = preprocess_image(face_crop, yawn_size, yawn_norm)
                    label = predict(yawn_model, tensor, YAWN_CLASSES)
                    color = (0, 255, 0) if label == "no_yawn" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Yawn: {label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    yawn_display.image(disp, caption=f"Yawn: {label}", width=180)
                    if label == "yawn":
                        beep_alert("yawn")

            # --- Eye Detection ---
            if mesh_results.multi_face_landmarks:
                closed_count = 0
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                eye_results = []

                for eye_pts, eye_id in zip([LEFT_EYE, RIGHT_EYE], ["Left", "Right"]):
                    x_vals = [int(landmarks[i].x * w) for i in eye_pts]
                    y_vals = [int(landmarks[i].y * h) for i in eye_pts]
                    x1, y1 = max(0, min(x_vals) - 30), max(0, min(y_vals) - 50)
                    x2, y2 = min(w, max(x_vals) + 30), min(h, max(y_vals) + 50)
                    eye_crop = rgb[y1:y2, x1:x2]

                    if eye_crop.size > 0:
                        tensor, disp = preprocess_image(eye_crop, eye_size, eye_norm)
                        label = predict(eye_model, tensor, EYE_CLASSES)
                        if label == "Closed":
                            closed_count += 1
                        eye_results.append((disp, label))
                        color = (0, 255, 0) if label == "Open" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{eye_id} Eye: {label}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show both left and right eye inputs
                if len(eye_results) == 2:
                    eye_display_left.image(eye_results[0][0], caption=f"Left Eye: {eye_results[0][1]}", width=180)
                    eye_display_right.image(eye_results[1][0], caption=f"Right Eye: {eye_results[1][1]}", width=180)

                if closed_count == 2:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    elif time.time() - eye_closed_start > 0.5:
                        beep_alert("eye")
                        eye_closed_start = None
                else:
                    eye_closed_start = None

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("Detection stopped.")

run_drowsiness_app()
