# ✅ Final Streamlit App with Grad-CAM Overlays on Live Frame

import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import platform
import mediapipe as mp
import matplotlib.cm as cm

# ----------------- Config -----------------
DEVICE = torch.device("cpu")
EYE_CLASSES = ['Closed', 'Open']
YAWN_CLASSES = ['no_yawn', 'yawn']

# ----------------- Grad-CAM Overlay -----------------
def overlay_gradcam_on_frame(crop, tensor_input, model, class_idx, is_resnet=False):
    gradients, activations = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    h1 = last_conv.register_forward_hook(forward_hook)
    h2 = last_conv.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(tensor_input)
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=(0, 2, 3))
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap + 1e-6)
    heatmap = heatmap.detach().cpu().numpy()

    heatmap = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(crop, 0.6, heatmap_colored, 0.4, 0)

# ✅ Enhanced Drowsiness Detection with Display Panels

def run_drowsiness_app():
    st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
    st.title("Driver Drowsiness Detection")

    def predict(model, tensor, class_names):
        with torch.no_grad():
            outputs = model(tensor)
            _, preds = torch.max(outputs, 1)
        return class_names[preds.item()], preds.item()

    def preprocess_image(img, size, normalize):
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize
        ])
        processed = transform(img).unsqueeze(0).to(DEVICE)
        display_img = cv2.resize(img, (size, size))
        return processed, display_img

    def load_model(model_type, task):
        if task == "eye":
            if model_type == "CNN":
                model = EyeCNN(2)
                path = "eye/eye_model_best.pt"
                size = 100
                norm = transforms.Normalize([0.5]*3, [0.5]*3)
            else:
                model = create_resnet18_eye(2)
                path = "eye/resnet18_eye_model_best.pt"
                size = 224
                norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            if model_type == "CNN":
                model = CNNBinaryClassifier()
                path = "yawn/best_yawn_model_baseline.pth"
                size = 100
                norm = transforms.Normalize([0.5]*3, [0.5]*3)
            else:
                model = ResNet18BinaryClassifier()
                path = "yawn/best_model_main.pth"
                size = 224
                norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model.to(DEVICE).eval(), size, norm

    def beep_alert(msg):
        if platform.system() == "Darwin": os.system(f'say "{msg}"')
        elif platform.system() == "Windows": import winsound; winsound.Beep(1000, 400)
        else: os.system(f'spd-say "{msg}"')

    # Sidebar UI
    with st.sidebar:
        st.header("⚙️ Model Selection")
        st.markdown("### Recommended")
        st.markdown("**Best Eye Model**: `CNN`")
        st.markdown("**Best Yawn Model**: `ResNet18`")
        eye_model_choice = st.selectbox("Select Eye Model", ["CNN", "ResNet18"])
        yawn_model_choice = st.selectbox("Select Yawn Model", ["CNN", "ResNet18"])
        start_detection = st.button("Start Detection")

    eye_model, eye_size, eye_norm = load_model(eye_model_choice, "eye")
    yawn_model, yawn_size, yawn_norm = load_model(yawn_model_choice, "yawn")

    # Main Display
    stframe = st.empty()
    st.markdown("#### Eye and Yawn Detection (Raw & Grad-CAM)")

    col1, col2, col3 = st.columns(3)
    left_eye_disp = col1.empty()
    right_eye_disp = col2.empty()
    yawn_disp = col3.empty()

    if start_detection:
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
            face_results = mp_face.process(rgb)
            mesh_results = mp_mesh.process(rgb)

            # ---- Yawn Detection ----
            if face_results.detections:
                detection = face_results.detections[0]
                box = detection.location_data.relative_bounding_box
                x1, y1 = max(0, int(box.xmin * w) - 50), max(0, int(box.ymin * h) - 200)
                x2, y2 = min(w, int((box.xmin + box.width) * w) + 50), min(h, int((box.ymin + box.height) * h) + 80)
                face_crop = rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    tensor, raw = preprocess_image(face_crop, yawn_size, yawn_norm)
                    label, idx = predict(yawn_model, tensor, YAWN_CLASSES)
                    cam = overlay_gradcam_on_frame(raw, tensor, yawn_model, idx, yawn_model_choice == "ResNet18")
                    yawn_disp.image([raw, cam], caption=["Yawn Input", "Yawn Grad-CAM"], width=220)
                    if label == "yawn": beep_alert("Yawn detected")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Yawn: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # ---- Eye Detection ----
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                closed_count = 0
                for i, eye_pts in enumerate([LEFT_EYE, RIGHT_EYE]):
                    x = [int(landmarks[i].x * w) for i in eye_pts]
                    y = [int(landmarks[i].y * h) for i in eye_pts]
                    x1, y1 = max(min(x) - 30, 0), max(min(y) - 50, 0)
                    x2, y2 = min(max(x) + 30, w), min(max(y) + 40, h)
                    eye_crop = rgb[y1:y2, x1:x2]
                    if eye_crop.size > 0:
                        tensor, raw = preprocess_image(eye_crop, eye_size, eye_norm)
                        label, idx = predict(eye_model, tensor, EYE_CLASSES)
                        cam = overlay_gradcam_on_frame(raw, tensor, eye_model, idx, eye_model_choice == "ResNet18")
                        if i == 0:
                            left_eye_disp.image([raw, cam], caption=["Left Eye", "Grad-CAM"], width=220)
                        else:
                            right_eye_disp.image([raw, cam], caption=["Right Eye", "Grad-CAM"], width=220)
                        if label == "Closed": closed_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{['Left','Right'][i]}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                if closed_count == 2:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    elif time.time() - eye_closed_start > 0.5:
                        beep_alert("Wake up")
                        eye_closed_start = None
                else:
                    eye_closed_start = None

            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("Detection stopped.")

# Model Definitions

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

if __name__ == "__main__":
    run_drowsiness_app()