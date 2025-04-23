import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

# ------------------ Config ------------------ #
IMAGE_SIZE = 100
keep_classes = ['Closed', 'Open']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained = False
weights_path = "eye_model.pt"

# ------------------ Dummy Dataset for Class Names ------------------ #
class FilteredImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name in keep_classes]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

dummy_dataset = FilteredImageFolder("/home/ubuntu/deeplearning_project/data/train", transform=transforms.ToTensor())
labels = dummy_dataset.classes

# ------------------ Model Definition ------------------ #
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
        self.fc = nn.Linear(128, len(labels))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------ Load Model ------------------ #
if pretrained:
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    IMAGE_SIZE = 224
else:
    model = EyeCNN()

model = model.to(device)
model.eval()

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")

# ------------------ Transform ------------------ #
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ------------------ Webcam Inference with Annotation and Metrics ------------------ #
cap = cv2.VideoCapture(0)

predictions = []
true_labels = []

print("Press 'c' for Closed, 'o' for Open, 'q' to quit and evaluate.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = labels[pred]

    # Display prediction
    color = (0, 255, 0) if label == 'Open' else (0, 0, 255)
    cv2.putText(frame, f"Prediction: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)
    cv2.imshow("Eye State Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        predictions.append(pred)
        true_labels.append(labels.index('Closed'))
    elif key == ord('o'):
        predictions.append(pred)
        true_labels.append(labels.index('Open'))

cap.release()
cv2.destroyAllWindows()

# ------------------ Evaluation ------------------ #
if true_labels:
    print("\nEvaluation Report:")
    print(classification_report(true_labels, predictions, target_names=labels))
else:
    print("No labels captured for evaluation.")

# Save model
torch.save(model.state_dict(), "eye_model.pt")
print("✅ Model saved as eye_model.pt")

