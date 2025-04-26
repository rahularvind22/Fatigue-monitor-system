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
