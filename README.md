# Deep-Learning-Project

# 💤 Driver Drowsiness Detection (Eye + Yawn Detection)

This project detects **driver drowsiness** using two modules:

- 👁️ **Eye Detection:** Classify eyes as open/closed  
- 😮 **Yawn Detection:** Classify faces as yawn/no_yawn

We use **custom CNN models** for both tasks with clean training & testing pipelines.

---

## 📂 Project Structure

```bash
Final-Project-Group1/
├── data/
│   ├── eye/
│   │   ├── train/
│   │   │   ├── open/
│   │   │   └── closed/
│   │   └── test/
│   │       ├── open/
│   │       └── closed/
│   └── yawn/
│       ├── train/
│       │   ├── yawn/
│       │   └── no_yawn/
│       └── test/
│           ├── yawn/
│           └── no_yawn/
├── eye_detection.py
├── yawn_detection.py
├── best_eye_model.pth
├── best_yawn_model.pth
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the GitHub Repository

```bash
git clone https://github.com/AswinBalajiTR/Final-Project-Group1
cd Final-Project-Group1
```

---

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
torchvision
opencv-python
numpy
tqdm
scikit-learn
```

---

### 3️⃣ Download the Datasets

#### 👁️ Eye Dataset

**Google Drive Link:**

https://drive.google.com/file/d/1XXXXXXX_EYE_DATASET_ID/view?usp=drive_link

Download using:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1XXXXXXX_EYE_DATASET_ID -O eye_dataset.zip
```

#### 😮 Yawn Dataset

**Google Drive Link:**

https://drive.google.com/file/d/1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA/view?usp=drive_link

Download using:

```bash
gdown https://drive.google.com/uc?id=1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA -O yawn_dataset.zip
```

---

### 4️⃣ Unzip the Datasets

```bash
unzip eye_dataset.zip
unzip yawn_dataset.zip
```

---

### 5️⃣ Set Up Data Folders

#### 👁️ Eye Dataset

Create folders:

```bash
mkdir -p data/eye/train/open data/eye/train/closed data/eye/test/open data/eye/test/closed
```

Move files:

```bash
mv eye_dataset/train/open/* data/eye/train/open/
mv eye_dataset/train/closed/* data/eye/train/closed/
mv eye_dataset/test/open/* data/eye/test/open/
mv eye_dataset/test/closed/* data/eye/test/closed/
```

#### 😮 Yawn Dataset

Create folders:

```bash
mkdir -p data/yawn/train/yawn data/yawn/train/no_yawn data/yawn/test/yawn data/yawn/test/no_yawn
```

Move files:

```bash
mv yawn_dataset/train/yawn/* data/yawn/train/yawn/
mv yawn_dataset/train/no_yawn/* data/yawn/train/no_yawn/
mv yawn_dataset/test/yawn/* data/yawn/test/yawn/
mv yawn_dataset/test/no_yawn/* data/yawn/test/no_yawn/
```

✅ Now your `data/` folder should look like:

```bash
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
```

---

# 💻 Running the Code

## 👁️ Eye Detection

**File:** `eye_detection.py`

### ✅ Train Eye Detection Model

```bash
python eye_detection.py --mode train --data_dir ./data/eye --save_model best_eye_model.pth
```

### ✅ Test Eye Detection Model

```bash
python eye_detection.py --mode test --data_dir ./data/eye --load_model best_eye_model.pth
```

---

## 😮 Yawn Detection

**File:** `yawn_detection.py`

### ✅ Train Yawn Detection Model

```bash
python yawn_detection.py --mode train --data_dir ./data/yawn --save_model best_yawn_model.pth
```

### ✅ Test Yawn Detection Model

```bash
python yawn_detection.py --mode test --data_dir ./data/yawn --load_model best_yawn_model.pth
```

---

# 🖼️ Expected Output

✅ **Training Logs Example:**

```bash
Epoch 1/10 - Train Loss: 0.42 - Train Acc: 83%
```

✅ **Test Output Example:**

```bash
Test Accuracy: 82%
Classification Report:
              precision    recall  f1-score   support
      open       0.84      0.87      0.85       100
    closed       0.80      0.75      0.77       100
```

---

# 🛠️ Notes & Troubleshooting

- ✅ Ensure your data is in the correct folder structure under `/data/eye/` and `/data/yawn/`
- ✅ Models will be saved as `best_eye_model.pth` and `best_yawn_model.pth`
- ✅ The models **overwrite each time you train**
- ✅ CUDA GPU is used automatically if available
- ✅ `opencv-python` is required for image loading

---

# 📄 What Each File Does

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `eye_detection.py`    | CNN model for eye state detection (train & test modes)                      |
| `yawn_detection.py`   | CNN model for yawn detection (train & test modes)                            |
| `requirements.txt`    | Lists all Python dependencies                                               |
| `README.md`           | This full guide                                                             |
| `data/`               | Holds both eye & yawn datasets (train & test splits)                         |
| `best_eye_model.pth`  | Trained model file for eye detection                                        |
| `best_yawn_model.pth` | Trained model file for yawn detection                                       |

---

# ✅ You're All Set 🚀

Clone ➔ Download ➔ Unzip ➔ Move files ➔ Train ➔ Test ➔ DONE ✔️

For any issues, please open a GitHub issue or contact the repo maintainer.
