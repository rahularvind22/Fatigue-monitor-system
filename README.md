# Deep-Learning-Project

# Driver Drowsiness Detection (Eye + Yawn Detection)


This repository is a **fork** of the original Driver Drowsiness Detection project.  
While the main project covered both **eye state detection** and **yawn detection**,  
my focus and contribution was entirely on building a **robust Yawn Detection module** using Deep Learning.  

---

## 📌 My Contribution  

### 🔹 Problem  
Yawning is an early and critical indicator of driver drowsiness. Detecting it in real-time is challenging due to variations in lighting, head pose, and subtle facial movements.  

### 🔹 Action  
- Developed a **custom CNN** in PyTorch for binary classification (*yawn vs. non-yawn*).  
- Fine-tuned a **ResNet18** pretrained on ImageNet, improving robustness and performance.  
- Applied **data augmentation** (flips, rotations, color jitter) for generalization under diverse conditions.  
- Integrated **Grad-CAM explainability** to visualize attention maps, confirming focus on the **mouth region**.  

### 🔹 Results  
- **Custom CNN:** 88% accuracy (Macro F1: 0.87)  
- **ResNet18:** 100% precision, recall, and F1-score  
- Successfully integrated into the **real-time Driver Drowsiness Detection system** with live alerts for yawns.  

### 🔹 Future Improvements  
- Expand dataset with more diverse yawning samples.  
- Deploy as a **real-time pipeline** using Streamlit + OpenCV.  
- Explore **temporal modeling (CNN + LSTM)** to capture yawning sequences instead of static frames.  


This project implements driver drowsiness detection using **two modules:**

- **Eye Detection**
- **Yawn Detection**

You can train and test **custom CNN models** and also use a **pretrained ResNet18 model** for eye detection.

---

## Clone the GitHub Repository

Run the following commands to clone the repository and move into the project folder:

```bash
git clone https://github.com/AswinBalajiTR/Final-Project-Group1
cd Final-Project-Group1
```

---

## Download the Dataset

Download the dataset zip file from the following link:

https://drive.google.com/file/d/1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA/view

---

## Unzip the Dataset

After downloading, unzip the data file using this command:

```bash
unzip data.zip
```

---

## Run the Python Files

We have **different files for Eye Detection and Yawn Detection.**

---

### Eye Detection

#### Run the Custom CNN Model

```bash
python3 eye_detection.py
```

#### Run the Pretrained ResNet18 Model

```bash
python3 eye_pretrained.py
```

---

### Yawn Detection

#### Run the Main Yawn Detection Model

```bash
python3 yawn_detection_main.py
```

#### Run the Baseline Yawn Detection Model

```bash
python3 yawn_detection_baseline.py
```

---

## View the Model Metrics

After running any of the above files, you will be able to see **the performance metrics** of each model in your terminal (e.g., Accuracy, Precision, Recall, F1-Score).

---

## Demo the Project (Streamlit App)

To run the **demo app** using Streamlit, use the following command:

```bash
streamlit run Main.py
```

---

# You’re All Set 

This completes the setup:

- Clone the repo  
- Download & unzip the dataset  
- Run the eye/yawn detection models  
- Launch the Streamlit demo if needed

For any questions, open a GitHub issue or contact the repo maintainer.
