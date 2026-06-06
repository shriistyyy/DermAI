# 🔬 Skin Cancer Detection — Minor Project

A deep learning system to classify skin lesions into 7 categories using CNN (MobileNetV2 transfer learning), trained on the HAM10000 dataset.

---

## 📁 Project Structure

```
skin_cancer_detection/
│
├── train_model.py          ← Model training script
├── app.py                  ← Flask web application
├── requirements.txt        ← Python dependencies
│
├── templates/
│   └── index.html          ← Frontend UI
│
├── static/                 ← Generated plots saved here
│   ├── class_distribution.png
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── models/                 ← Saved model weights
│   └── skin_cancer_model.h5
│
├── notebooks/
│   └── EDA_and_Training.ipynb  ← Jupyter notebook
│
└── dataset/                ← Download HAM10000 here
    ├── HAM10000_metadata.csv
    └── HAM10000_images/
        └── *.jpg
```

---

## 🎯 Classes Detected (7 types)

| Code | Full Name | Risk Level |
|------|-----------|------------|
| nv | Melanocytic Nevi | Low |
| mel | Melanoma | **High** |
| bkl | Benign Keratosis | Low |
| bcc | Basal Cell Carcinoma | Medium |
| akiec | Actinic Keratoses | Medium |
| vasc | Vascular Lesions | Low |
| df | Dermatofibroma | Low |

---

## ⚙️ Setup Instructions

### 1. Clone and install dependencies

```bash
cd skin_cancer_detection
pip install -r requirements.txt
```

### 2. Download the Dataset

Download **HAM10000** from Kaggle (free):
👉 https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

Place files:
```
dataset/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   (merge all images)
└── HAM10000_images_part_2/
```

Merge all images into one folder:
```bash
mkdir -p dataset/HAM10000_images
cp dataset/HAM10000_images_part_1/*.jpg dataset/HAM10000_images/
cp dataset/HAM10000_images_part_2/*.jpg dataset/HAM10000_images/
```

### 3. Train the Model

```bash
python train_model.py
```

Training runs in **2 phases**:
- Phase 1: Frozen MobileNetV2 base → train classifier head (10 epochs)
- Phase 2: Fine-tune top 30 layers of base (30 epochs)

Expected accuracy: **~83-87%** on validation set

### 4. Launch the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(7, Softmax)
```

**Why MobileNetV2?**
- Lightweight and fast
- Pre-trained on ImageNet → strong feature extractor
- Good performance with small medical datasets
- Easy fine-tuning

---

## 📊 Data Augmentation

To handle class imbalance and improve generalization:
- Random rotation (±20°)
- Horizontal & vertical flips
- Width/height shift (10%)
- Zoom (10%)
- Shear transformation

---

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~83–87% |
| Melanoma AUC | ~0.89 |
| Training Time | ~2–4 hours (GPU) |

---

## 🚀 Running with Jupyter Notebook

```bash
jupyter notebook notebooks/EDA_and_Training.ipynb
```

Includes:
- Dataset exploration & visualization
- Class distribution analysis
- Sample images per class
- Training & evaluation
- Single image prediction

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | TensorFlow / Keras |
| Base CNN | MobileNetV2 (Transfer Learning) |
| Backend | Flask (Python) |
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Dataset | HAM10000 (10,015 images) |
| Visualization | Matplotlib, Seaborn |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **NOT** a medical device and should never replace professional medical diagnosis.
Always consult a qualified dermatologist for skin concerns.

---

## 👨‍💻 Minor Project Info

- **Subject**: Machine Learning / Deep Learning
- **Algorithm**: Convolutional Neural Network (Transfer Learning)
- **Dataset**: HAM10000 (Harvard Dataverse)
- **Framework**: TensorFlow 2.x

---

## 📚 References

1. Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).
2. Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR (2018).


© 2026 Shristy Goyal. All rights reserved.
