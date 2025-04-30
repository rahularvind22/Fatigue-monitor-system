import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn

# ----------------- Paths --------------------
ROOT = os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "data"))
MODEL_PATH = "best_yawn_model_baseline.pth"

# ----------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Custom CNN Model (Simple version) ---------------------
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

# ----------------- Dataset ---------------------
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

# ----------------- Transform ---------------------
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----------------- Grad-CAM Logic ---------------------
def visualize_gradcam(model, image_tensor, class_idx=None):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()

    target_layers = [model.net[-3]]
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = image_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img * 0.5) + 0.5
    rgb_img = np.clip(rgb_img, 0, 1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Show
    plt.imshow(visualization)
    plt.title(f"Grad-CAM for class: {class_idx}")
    plt.axis('off')
    plt.show()

    # Save
    os.makedirs("cam_outputs", exist_ok=True)
    import uuid
    unique_id = str(uuid.uuid4())[:8]  # short random ID
    out_path = f"cam_outputs/class_{class_idx}_{unique_id}.jpg"

    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"[✅] Saved Grad-CAM image to: {out_path}")




# ----------------- Main ---------------------
if __name__ == "__main__":
    test_dir = os.path.join(DATA_DIR, "test")
    dataset = YawnDataset(test_dir, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CNNBinaryClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # model finally matches!
    model.eval()

    for i, (img, label) in enumerate(dataloader):
        visualize_gradcam(model, img[0], class_idx=label[0].item())
        if i == 5:
            break
