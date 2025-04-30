import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ------------------ Config ------------------ #
IMAGE_SIZE = 100
CLASSES = ['Closed', 'Open']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ Model Definition ------------------ #
class EyeCNN(torch.nn.Module):
    def __init__(self, outputs):
        super(EyeCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.1),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.17),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(256, outputs)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------ Grad-CAM Class ------------------ #
class GradCAMVisualizer:
    def __init__(self, model_path, test_dir):
        self.test_dir = test_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        self.model = EyeCNN(outputs=len(CLASSES)).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.features = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.model[12]  # Last conv layer
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.features.append(output)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        return img, input_tensor

    def generate_gradcam(self, image_path):
        self.features.clear()
        self.gradients.clear()

        orig_img, input_tensor = self.preprocess_image(image_path)
        output = self.model(input_tensor)
        pred_class = torch.argmax(output).item()

        self.model.zero_grad()
        output[0, pred_class].backward()

        grads = self.gradients[0].squeeze().detach().cpu().numpy()
        feats = self.features[0].squeeze().detach().cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(feats.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feats[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam -= cam.min()
        cam /= cam.max()
        cam = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.resize(orig_img, (IMAGE_SIZE, IMAGE_SIZE)), 0.5, heatmap, 0.5, 0)

        save_path = os.path.join("gradcam_outputs", os.path.basename(image_path))
        os.makedirs("gradcam_outputs", exist_ok=True)
        cv2.imwrite(save_path, overlay)
        print(f"Saved Grad-CAM to: {save_path}")

    def evaluate_model(self):
        test_dataset = ImageFolder(self.test_dir, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_preds, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(imgs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

# ------------------ Main Call ------------------ #
if __name__ == "__main__":
    BASE_DIR = os.getcwd()  # You are in /home/ubuntu/Final-Project-Group1/Code/eye
    TEST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "test"))
    MODEL_PATH = os.path.join(BASE_DIR, "eye_model_best.pt")

    IMAGE_PATHS = [
        os.path.join(TEST_DIR, "Closed", "_72.jpg"),
        os.path.join(TEST_DIR, "Open", "_32.jpg"),  # <-- update this based on actual `ls`
    ]

    visualizer = GradCAMVisualizer(model_path=MODEL_PATH, test_dir=TEST_DIR)

    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        print(f"Generating Grad-CAM for: {img_path}")
        visualizer.generate_gradcam(img_path)

    visualizer.evaluate_model()


