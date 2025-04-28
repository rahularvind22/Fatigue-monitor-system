import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

IMAGE_SIZE = 100
CLASSES = ['Closed', 'Open']

project_dir = os.getcwd()
eye_dir = os.path.join(project_dir, 'eye')
eye_model_path = os.path.join(eye_dir, 'eye_model_best.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = EyeCNN()
model.load_state_dict(torch.load(eye_model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])