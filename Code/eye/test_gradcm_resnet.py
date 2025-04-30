import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score

# ------------------ Config ------------------ #
CLASSES = ['Closed', 'Open']
IMAGE_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ Grad-CAM Class ------------------ #
class GradCAM:
    def __init__(self, model, target_layer="layer4"):
        self.model = model
        self.gradients = []
        self.activations = []
        self._register_hook(target_layer)

    def _register_hook(self, layer_name):
        layer = dict([*self.model.named_modules()])[layer_name]
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations.append(output)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def generate(self, input_tensor, class_idx=None):
        self.gradients.clear()
        self.activations.clear()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        score = output[:, class_idx]
        self.model.zero_grad()
        score.backward()

        grads = self.gradients[0].squeeze().detach().cpu().numpy()
        acts = self.activations[0].squeeze().detach().cpu().numpy()

        weights = np.mean(grads, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * acts, axis=0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam -= cam.min()
        cam /= cam.max()
        return cam, class_idx.item()

# ------------------ Utility Functions ------------------ #
def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    return img, tensor

def save_heatmap(original_img, cam, image_path, pred_class_idx):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE)), 0.5, heatmap, 0.5, 0)

    output_dir = "gradcam_resnet_outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{CLASSES[pred_class_idx]}_{os.path.basename(image_path)}"
    save_path = os.path.join(output_dir, filename)
    success = cv2.imwrite(save_path, overlay)
    if success:
        print(f"Saved Grad-CAM to: {save_path}")
    else:
        print(f"Failed to save Grad-CAM to: {save_path}")

def evaluate_model(model, test_dir):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

# ------------------ Main ------------------ #
if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    TEST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "test"))
    MODEL_PATH = os.path.join(BASE_DIR, "resnet18_eye_model_best.pt")

    IMAGE_PATHS = [
        os.path.join(TEST_DIR, "Closed", "_72.jpg"),
        os.path.join(TEST_DIR, "Open", "_31.jpg")
    ]

    model = load_model(MODEL_PATH, num_classes=len(CLASSES))
    gradcam = GradCAM(model, target_layer="layer4")

    for image_path in IMAGE_PATHS:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        original_img, input_tensor = preprocess_image(image_path)
        cam, pred_class_idx = gradcam.generate(input_tensor)
        save_heatmap(original_img, cam, image_path, pred_class_idx)

    evaluate_model(model, TEST_DIR)
