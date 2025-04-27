import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

project_dir = os.getcwd()
eye_dir = os.path.join(project_dir, 'Code', 'eye')
yawn_dir = os.path.join(project_dir, 'Code', 'yawn')

eye_model_path = os.path.join(eye_dir, 'eye_model_best.pt')
yawn_model_path = os.path.join(yawn_dir, 'best_yawn_model.pth')

eye_model = torch.load(eye_model_path, map_location=torch.device('cpu'))
eye_model.eval()

yawn_model = torch.load(yawn_model_path, map_location=torch.device('cpu'))
yawn_model.eval()