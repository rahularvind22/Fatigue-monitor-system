import os
import cv2
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from tqdm import tqdm

# ------------------ Config ------------------ #
IMAGE_SIZE = 100  # Use 224 if using pretrained=True
BATCH_SIZE = 30
EPOCHS = 10
pretrained = False  # ✅ Change this to True for ResNet18
keep_classes = ['Closed', 'Open']

# ------------------ Transform ------------------ #
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
# ------------------ Custom ImageFolder to Filter Classes ------------------ #
class FilteredImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name in keep_classes]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# ------------------ Data Loaders ------------------ #
train_dir = "/home/ubuntu/deeplearning_project/data/train"
test_dir = "/home/ubuntu/deeplearning_project/data/test"

train_dataset = FilteredImageFolder(root=train_dir, transform=transform)
test_dataset = FilteredImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

OUTPUTS_a = len(train_dataset.classes)
print("Classes used:", train_dataset.classes)
# ------------------ Model Definition ------------------ #
class EyeCNN(nn.Module):
    def _init_(self):
        super(EyeCNN, self)._init_()
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
        self.fc = nn.Linear(128, OUTPUTS_a)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------ Instantiate Model ------------------ #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if pretrained:
    print("Using pretrained ResNet18...")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    model = model.to(device)
    IMAGE_SIZE = 224  # Update size for ResNet
else:
    print("Using custom EyeCNN...")
    model = EyeCNN().to(device)


