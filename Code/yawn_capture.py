import os
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
CLASSES = ['no_yawn', 'yawn']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- CNN Model -----------------
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

# ----------------- ResNet18 Model -----------------
class ResNet18BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet18BinaryClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_model(model_type):
    if model_type == "ResNet18":
        model = ResNet18BinaryClassifier()
        path = "yawn/best_model_main.pth"
        img_size = 224
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    else:
        model = CNNBinaryClassifier()
        path = "yawn/best_yawn_model_baseline.pth"
        img_size = 100
        normalize = transforms.Normalize([0.5]*3, [0.5]*3)

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, img_size, normalize

def preprocess_face(face_img, img_size, normalize):
    # Apply contrast reduction (alpha < 1) and brightness increase (beta > 0)
    face_img = cv2.convertScaleAbs(face_img, alpha=1.0, beta=50)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    tensor = transform(face_img)

    # Recreate display image from unnormalized tensor for sidebar preview
    display_img = transforms.ToPILImage()(tensor.cpu().detach().clone().clamp(0, 1))
    return tensor.unsqueeze(0).to(DEVICE), display_img


def predict_face(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
    return CLASSES[preds.item()]

def beep_sound():
    if platform.system() == "Darwin":
        os.system('say "Yawn detected!"')
    elif platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('spd-say "yawn detected"')

# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="Yawn Detection", page_icon="😴")
    st.title("🧠 Real-Time Yawn Detection with MediaPipe")
    st.markdown("Detects **yawn** vs **no_yawn** from webcam feed using CNN or ResNet18 model with brightness, contrast, and padding adjustments.")

    model_type = st.radio("Choose Model", ["CNN", "ResNet18"])
    model, img_size, normalize = load_model(model_type)

    start_detection = st.button("Start Detection")

    frame_placeholder = st.empty()
    sidebar_placeholder = st.sidebar.empty()
    st.sidebar.caption("Preprocessed face input to the model")

    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    if start_detection:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 60)

        st.info("Press the top-right 'Stop' button to end detection.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    x1 = max(0, x - 60)
                    y1 = max(0, y - 200)
                    x2 = min(iw, x + w + 60)
                    y2 = min(ih, y + h + 80)

                    face_crop = rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        tensor, display_img = preprocess_face(face_crop, img_size, normalize)
                        label = predict_face(model, tensor)

                        color = (0, 255, 0) if label == "no_yawn" else (0, 0, 255)
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(rgb, label.upper(), (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        if label == "yawn":
                            beep_sound()

                        sidebar_placeholder.image(display_img, caption=f"{label} ({img_size}x{img_size})", width=img_size)

                    break  # only the most confident detection

            # Stream video
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frame_placeholder.image(buffer.tobytes(), channels="BGR", use_column_width=True)

        cap.release()
        st.success("Detection stopped.")

if __name__ == "__main__":
    main()
