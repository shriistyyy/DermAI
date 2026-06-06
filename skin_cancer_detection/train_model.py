"""
Skin Cancer Detection - Model Training Script
==============================================
Uses CNN (MobileNetV2 transfer learning) to classify skin lesions
Dataset: HAM10000 or ISIC dataset (7 classes)

Classes:
  0 - Melanocytic nevi (nv)
  1 - Melanoma (mel)
  2 - Benign keratosis (bkl)
  3 - Basal cell carcinoma (bcc)
  4 - Actinic keratoses (akiec)
  5 - Vascular lesions (vasc)
  6 - Dermatofibroma (df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = 7
DATASET_DIR = "dataset/HAM10000_images"
METADATA    = "dataset/HAM10000_metadata.csv"
MODEL_PATH  = "models/skin_cancer_model.h5"

CLASS_NAMES = {
    'nv':    'Melanocytic Nevi',
    'mel':   'Melanoma',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc':  'Vascular Lesions',
    'df':    'Dermatofibroma'
}

# ─── DATA LOADING ───────────────────────────────────────────────────────────────
def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['path'] = df['image_id'].apply(
        lambda x: os.path.join(DATASET_DIR, x + '.jpg')
    )
    df = df[df['path'].apply(os.path.exists)]
    print(f"✅ Loaded {len(df)} images")
    print(f"📊 Class distribution:\n{df['dx'].value_counts()}\n")
    return df


def plot_class_distribution(df, save_path="static/class_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df['dx'].value_counts()
    labels = [CLASS_NAMES.get(c, c) for c in counts.index]
    colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c']
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title("Class Distribution in Dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Skin Lesion Type")
    ax.set_ylabel("Number of Samples")
    plt.xticks(rotation=30, ha='right')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(val), ha='center', fontsize=9)
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Saved class distribution plot → {save_path}")


# ─── DATA GENERATORS ────────────────────────────────────────────────────────────
def build_generators(df):
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_dataframe(
        train_df, x_col='path', y_col='dx',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_flow = val_gen.flow_from_dataframe(
        val_df, x_col='path', y_col='dx',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    return train_flow, val_flow, val_df


# ─── MODEL ARCHITECTURE ─────────────────────────────────────────────────────────
def build_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base initially
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print(model.summary())
    return model, base


# ─── CALLBACKS ──────────────────────────────────────────────────────────────────
def get_callbacks():
    os.makedirs("models", exist_ok=True)
    return [
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_auc', mode='max'),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7, monitor='val_loss'),
        TensorBoard(log_dir='logs', histogram_freq=1)
    ]


# ─── TRAINING ───────────────────────────────────────────────────────────────────
def train(model, base, train_flow, val_flow):
    print("\n🔵 Phase 1: Training classifier head (frozen base)\n")
    history1 = model.fit(
        train_flow, validation_data=val_flow,
        epochs=10, callbacks=get_callbacks()
    )

    # Fine-tune top layers of base
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("\n🟠 Phase 2: Fine-tuning top layers\n")
    history2 = model.fit(
        train_flow, validation_data=val_flow,
        epochs=EPOCHS, callbacks=get_callbacks()
    )

    return history1, history2


# ─── EVALUATION ─────────────────────────────────────────────────────────────────
def evaluate(model, val_flow, val_df):
    preds = model.predict(val_flow)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = val_flow.classes
    label_map = {v: k for k, v in val_flow.class_indices.items()}

    print("\n📊 Classification Report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=[CLASS_NAMES[label_map[i]] for i in range(NUM_CLASSES)]
    ))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CLASS_NAMES[label_map[i]] for i in range(NUM_CLASSES)],
                yticklabels=[CLASS_NAMES[label_map[i]] for i in range(NUM_CLASSES)])
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png", dpi=150)
    plt.close()
    print("📈 Saved confusion matrix → static/confusion_matrix.png")


def plot_training_history(h1, h2):
    acc  = h1.history['accuracy']  + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss']      + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc, label='Train', color='#3498db')
    axes[0].plot(val_acc, label='Val', color='#e74c3c')
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(loss, label='Train', color='#3498db')
    axes[1].plot(val_loss, label='Val', color='#e74c3c')
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("static/training_history.png", dpi=150)
    plt.close()
    print("📈 Saved training history → static/training_history.png")


# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔬 Skin Cancer Detection — Model Training")
    print("=" * 50)

    df = load_metadata(METADATA)
    plot_class_distribution(df)

    train_flow, val_flow, val_df = build_generators(df)
    model, base = build_model()

    h1, h2 = train(model, base, train_flow, val_flow)
    plot_training_history(h1, h2)
    evaluate(model, val_flow, val_df)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    print("🚀 Run `python app.py` to launch the web interface")
