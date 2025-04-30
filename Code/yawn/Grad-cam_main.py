import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import uuid
import torch.nn as nn

# ------------------ Paths ------------------
ROOT = os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "data"))
MODEL_PATH = "best_model_main.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ ResNet Model ------------------
class ResNet18BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet18BinaryClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# ------------------ Dataset ------------------
class YawnDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
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
        return image, torch.tensor(label)

# ------------------ Transform ------------------
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ Grad-CAM Logic ------------------
def visualize_gradcam(model, image_tensor, class_idx=None):
    model.eval()

    target_layers = [model.model.layer4[-1]]  # Last ResNet block
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = image_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # unnormalize
    rgb_img = np.clip(rgb_img, 0, 1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Show
    plt.imshow(visualization)
    plt.title(f"Grad-CAM for class: {class_idx}")
    plt.axis('off')
    plt.show()

    # Save
    os.makedirs("cam_outputs_main", exist_ok=True)
    out_path = f"cam_outputs_main/class_{class_idx}_{uuid.uuid4().hex[:8]}.jpg"

    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"[✅] Saved Grad-CAM image to: {out_path}")

# ------------------ Main ------------------
if __name__ == "__main__":
    test_dir = os.path.join(DATA_DIR, "test")
    dataset = YawnDataset(test_dir, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = ResNet18BinaryClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    for i, (img, label) in enumerate(dataloader):
        visualize_gradcam(model, img[0], class_idx=label[0].item())
        if i == 5:  # limit to first 6 samples
            break
