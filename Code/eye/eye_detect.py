# ------------------ Imports ------------------ #
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from tqdm import tqdm

# ------------------ Config ------------------ #
OR_PATH = os.getcwd()
PATH = os.path.abspath(os.path.join(OR_PATH, "..", ".."))  # Go two levels up
DATA_DIR = os.path.join(PATH, 'data')
IMAGE_SIZE = 100
BATCH_SIZE = 30
EPOCHS = 30
LR = 0.0005
keep_classes = ['Closed', 'Open']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ Data Transforms ------------------ #
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ------------------ Custom Dataset ------------------ #
class FilteredImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name in keep_classes]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def load_data():
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')

    train_dataset = FilteredImageFolder(root=train_dir, transform=train_transform)
    test_dataset = FilteredImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    OUTPUTS_a = len(train_dataset.classes)
    print("Classes used:", train_dataset.classes)

    return train_loader, test_loader, OUTPUTS_a, train_dataset.classes

# ------------------ Model Definition ------------------ #
class EyeCNN(nn.Module):
    def __init__(self, outputs):
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

def create_model(OUTPUTS_a):
    model = EyeCNN(outputs=OUTPUTS_a).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    class_weights = torch.tensor([1.2, 0.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, optimizer, scheduler, criterion

# ------------------ Train and Test ------------------ #
def train_and_test(train_loader, test_loader, model, optimizer, scheduler, criterion, classes):
    best_f1 = 0.0
    save_path = "eye_model_best.pt"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

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

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}: Test F1 Macro = {f1_macro:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at epoch {epoch+1} with F1: {f1_macro:.4f}")

def final_evaluation(model, test_loader, classes):
    print("\nFinal Test Classification Report (Best Model Loaded):")
    model.load_state_dict(torch.load("eye_model_best.pt"))
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

    print(classification_report(all_labels, all_preds, target_names=classes))

# ------------------ Main ------------------ #
if __name__ == '__main__':
    train_loader, test_loader, OUTPUTS_a, classes = load_data()
    model, optimizer, scheduler, criterion = create_model(OUTPUTS_a)
    train_and_test(train_loader, test_loader, model, optimizer, scheduler, criterion, classes)
    final_evaluation(model, test_loader, classes)
