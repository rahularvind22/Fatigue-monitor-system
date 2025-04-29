import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import seaborn as sns

# ---------------------------- Paths ----------------------------------
OR_PATH = os.getcwd()
PATH = os.path.abspath(os.path.join(OR_PATH, "..", ".."))  # Go two levels up
DATA_DIR = os.path.join(PATH, 'data')


# ---------------------------- Seed & Device ----------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- Dataset Class ----------------------------------
class YawnDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_names = ["no_yawn", "yawn"]

        for label, class_name in enumerate(self.class_names):
            folder = os.path.join(root_dir, class_name)
            for img_name in os.listdir(folder):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(folder, img_name)
                    self.data.append(img_path)
                    self.labels.append(label)

        print(f"[INFO] Loaded {len(self.data)} images from {root_dir}")
        print(f"        no_yawn: {self.labels.count(0)}")
        print(f"        yawn   : {self.labels.count(1)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

# ---------------------------- CNN Model ----------------------------------
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

# ---------------------------- Loss with Label Smoothing -------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll + self.smoothing * smooth_loss
        return loss.mean()

# ---------------------------- Train & Eval Functions ------------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return running_loss / len(dataloader), acc, f1

def evaluate(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return val_loss / len(dataloader), acc, f1, all_preds, all_labels

# ---------------------------- Confusion Matrix -------------------------------
def plot_confusion_matrix(y_true, y_pred, labels=["no_yawn", "yawn"]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ---------------------------- DataLoader Builder ----------------------------
def load_data(transform):
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    train_ds = YawnDataset(train_dir, transform)
    test_ds = YawnDataset(test_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader

# ---------------------------- Training Script -------------------------------
def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_loader, test_loader = load_data(transform)

    model = CNNBinaryClassifier().to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_f1 = 0.0
    for epoch in range(30):
        print(f"\nEpoch {epoch + 1}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, test_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_yawn_model_baseline.pth")
            print("   Best model saved!")

    print("\n[INFO] Final Confusion Matrix:")
    plot_confusion_matrix(val_labels, val_preds)

# ---------------------------- Testing Script -------------------------------
def test_model():
    model_path = "best_yawn_model_baseline.pth"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    _, test_loader = load_data(transform)

    model = CNNBinaryClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report (Yawn vs No-Yawn):")
    print(classification_report(all_labels, all_preds, target_names=["no_yawn", "yawn"]))

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    print(f"Final Test F1 Score: {f1:.4f}")
    plot_confusion_matrix(all_labels, all_preds)

# ---------------------------- Run All --------------------------------------
if __name__ == "__main__":
    main()
    test_model()
