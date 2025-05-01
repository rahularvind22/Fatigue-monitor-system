# Deep-Learning-Project

# 💤 Driver Drowsiness Detection (Eye + Yawn Detection)

This project implements driver drowsiness detection using **two modules:**

- 👁️ **Eye Detection**
- 😮 **Yawn Detection**

You can train and test **custom CNN models** and also use a **pretrained ResNet18 model** for eye detection.

---

## 📥 Clone the GitHub Repository

Run the following commands to clone the repository and move into the project folder:

```bash
git clone https://github.com/AswinBalajiTR/Final-Project-Group1
cd Final-Project-Group1
```

---

## ⬇️ Download the Dataset

Download the dataset zip file from the following link:

https://drive.google.com/file/d/1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA/view

---

## 📦 Unzip the Dataset

After downloading, unzip the data file using this command:

```bash
unzip data.zip
```

---

## 🚀 Run the Python Files

We have **different files for Eye Detection and Yawn Detection.**

---

### 👁️ Eye Detection

#### ✅ Run the Custom CNN Model

```bash
python3 eye_detection.py
```

#### ✅ Run the Pretrained ResNet18 Model

```bash
python3 eye_pretrained.py
```

---

### 😮 Yawn Detection

#### ✅ Run the Main Yawn Detection Model

```bash
python3 yawn_detection_main.py
```

#### ✅ Run the Baseline Yawn Detection Model

```bash
python3 yawn_detection_baseline.py
```

---

## 🖼️ View the Model Metrics

✅ After running any of the above files, you will be able to see **the performance metrics** of each model in your terminal (e.g., Accuracy, Precision, Recall, F1-Score).

---

## 🎬 Demo the Project (Streamlit App)

To run the **demo app** using Streamlit, use the following command:

```bash
streamlit run Main.py
```

---

# ✅ You’re All Set 🚀

This completes the setup:

- ✅ Clone the repo  
- ✅ Download & unzip the dataset  
- ✅ Run the eye/yawn detection models  
- ✅ Launch the Streamlit demo if needed

For any questions, open a GitHub issue or contact the repo maintainer.
