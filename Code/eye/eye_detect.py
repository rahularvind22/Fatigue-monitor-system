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

