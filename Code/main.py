import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from torchvision import transforms

IMAGE_SIZE = 100
CLASSES = ['Closed', 'Open']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def predict_eye(model, transform, img):
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    return CLASSES[preds.item()]

def draw_predictions(frame, eyes, model, transform):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = frame_rgb.shape
    predictions = []
    for (x, y, w, h) in eyes:
        x1 = max(0, x - 20)
        y1 = max(0, y - 10)
        x2 = min(w_img, x + w + 20)
        y2 = min(h_img, y + h + 10)

        eye_img = frame_rgb[y1:y2, x1:x2]
        if eye_img.size == 0:
            continue

        pred = predict_eye(model, transform, eye_img)
        predictions.append(pred)

        color = (0, 255, 0) if pred == 'Open' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return predictions

def check_drowsiness_alert(predictions, frame, closed_start_time, alert_triggered):
    if len(predictions) >= 2 and predictions.count('Closed') >= 2:
        if closed_start_time is None:
            closed_start_time = time.time()
        else:
            elapsed = time.time() - closed_start_time
            if elapsed >= 1.1:
                alert_triggered = True

    else:
        # Eyes are open, reset everything
        closed_start_time = None
        alert_triggered = False

    # Always display alert if triggered
    if alert_triggered:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return closed_start_time, alert_triggered


def main():
    project_dir = os.getcwd()
    model_path = os.path.join(project_dir, 'eye', 'eye_model_best.pt')

    model = load_model(model_path)
    transform = get_transform()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    closed_start_time = None
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)

        predictions = draw_predictions(frame, eyes, model, transform)  # <--- save predictions
        closed_start_time, alert_triggered = check_drowsiness_alert(predictions, frame, closed_start_time,
                                                                    alert_triggered)

        cv2.imshow('Eye State Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
