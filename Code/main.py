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
        self.fc = nn.Linear(128, OUTPUTS_a)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------ Instantiate Model ------------------ #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if pretrained:
    print("Using pretrained ResNet18...")
    IMAGE_SIZE = 224  # Update size for ResNet
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
else:
    print("Using custom EyeCNN...")
    model = EyeCNN().to(device)

# ------------------ Training Setup ------------------ #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

# ------------------ Training Loop ------------------ #
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "eye_model.pt")
print("✅ Model saved as eye_model.pt")


# ------------------ Evaluation ------------------ #
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTest Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
