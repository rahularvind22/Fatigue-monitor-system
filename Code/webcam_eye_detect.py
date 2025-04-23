import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ----- Config ----- #
IMAGE_SIZE = 100
labels = ['Closed', 'Open']  # Update if different
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'eye_model.pt'

# ----- Define Model ----- #
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.model = nn.Sequential(
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
        self.fc = nn.Linear(128, len(labels))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----- Load Model ----- #
model = EyeCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded model from {model_path}")

# ----- Preprocessing ----- #
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ----- Start Webcam ----- #
cap = cv2.VideoCapture(0)

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

    # Annotate and display
    color = (0, 255, 0) if label == 'Open' else (0, 0, 255)
    cv2.putText(frame, f"Eye: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)
    cv2.imshow("Live Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
