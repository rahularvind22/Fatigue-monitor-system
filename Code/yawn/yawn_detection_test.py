import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from yawn_detection_train import CNNBinaryClassifier  # make sure this is in same dir or import correctly
from yawn_detection_train import ResNet18BinaryClassifier

class YawnTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_names = ["no_yawn", "yawn"]

        for label, class_name in enumerate(self.class_names):
            folder = os.path.join(root_dir, class_name)
            for img_name in os.listdir(folder):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.data.append(os.path.join(folder, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names=["no_yawn", "yawn"]):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# test the best model for the training
def main():
    test_dir = "/home/ubuntu/Final-Project-Group1/data/test"
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    test_dataset = YawnTestDataset(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32)

    #model = CNNBinaryClassifier().to(device)
    model= ResNet18BinaryClassifier().to(device)
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


    print("\n Classification Report: of the yeawn and non yawn ")
    print(classification_report(all_labels, all_preds, target_names=["no_yawn", "yawn"]))

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print("\n  Accuracy Report : ")
    print(f"\n  Final Test Accuracy: {acc:.4f}")
    print(f" Final Test F1 Score: {f1:.4f}")
    plot_confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    main()
