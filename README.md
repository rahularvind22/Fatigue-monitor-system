# Deep-Learning-Project


#  How to Set Up and Run This Project

This section explains how to **clone the repository**, **download the dataset**, **unzip it**, and **set up everything correctly** to train and test the model.

---

## 1️ Clone the GitHub Repository

👉 **Command to clone the repo:**

```bash
git clone https://github.com/your-username/yawn-detection.git

## Then move into the repo folder:
cd yawn-detection

## 2️⃣ Download the Dataset
 Dataset link:
https://drive.google.com/file/d/1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA/view?usp=drive_link

 First install gdown (to download from Google Drive):
pip install gdown
Then download the dataset zip file:
gdown https://drive.google.com/uc?id=1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA -O yawn_dataset.zip

Unzip the Dataset:
unzip yawn_dataset.zip


data/
├── eye/
│   ├── train/
│   │   ├── open/
│   │   └── closed/
│   └── test/
│       ├── open/
│       └── closed/
└── yawn/
    ├── train/
    │   ├── yawn/
    │   └── no_yawn/
    └── test/
        ├── yawn/
        └── no_yawn/








