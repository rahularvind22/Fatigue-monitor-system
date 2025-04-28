import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# ------------------ Config ------------------
IMAGE_SIZE = 100
CLASSES = ['no_yawn', 'yawn']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Model ------------------
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

def load_model(model_path):
    model = CNNBinaryClassifier()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def predict_mouth(model, transform, img):
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    return CLASSES[preds.item()]

def main():
    # Load model
    project_dir = os.getcwd()
    model_path = os.path.join(project_dir, 'yawn', 'best_yawn_model.pth')
    model = load_model(model_path)
    transform = get_transform()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Modified mouth region selection
            mouth_y_start = y + h // 2 + 20  # move 20px down
            mouth_y_end = y + h + 40  # add 40px extra at bottom

            x1 = max(0, x - 30)  # expand left by 30px
            y1 = max(0, mouth_y_start)
            x2 = min(frame.shape[1], x + w + 30)  # expand right by 30px
            y2 = min(frame.shape[0], mouth_y_end)

            mouth_img = frame_rgb[y1:y2, x1:x2]
            if mouth_img.size == 0:
                continue

            pred = predict_mouth(model, transform, mouth_img)

            color = (0, 255, 0) if pred == 'no_yawn' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            break  # Only process first face

        cv2.imshow('Yawn Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
