"""
Training Script for TD_CNN-LSTM Mobile Snatching Detection Model
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from model import build_td_cnn_lstm, compile_model


def load_dataset(data_dir, sequence_length=10):
    """
    Load preprocessed .npy video frame sequences.

    Args:
        data_dir (str): Directory with 'snatching/' and 'normal/' subdirs
        sequence_length (int): Number of frames per sample. Default: 10

    Returns:
        X (np.ndarray): shape (N, sequence_length, 240, 240, 3)
        y (np.ndarray): shape (N,)  — 1=snatching, 0=normal
    """
    X, y = [], []
    classes = {'snatching': 1, 'normal': 0}

    for cls, label in classes.items():
        cls_dir = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        print(f"  Loading {len(files)} samples from class '{cls}'...")

        for f in files:
            frames = np.load(os.path.join(cls_dir, f))
            # Sample `sequence_length` frames uniformly
            indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
            sample = frames[indices]  # (sequence_length, H, W, 3)
            X.append(sample)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def plot_history(history, save_dir='results/'):
    """Plot and save training/validation accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Total Accuracy vs Total Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='red')
    axes[1].set_title('Total Loss vs Total Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.show()
    print(f"✅ Training curves saved to {save_dir}training_curves.png")


def train_model(
    data_dir='data/processed/',
    epochs=50,
    batch_size=16,
    learning_rate=0.0001,
    sequence_length=10,
    save_path='models/td_cnn_lstm_best.h5'
):
    """
    Full training pipeline for the TD_CNN-LSTM model.

    Args:
        data_dir (str): Path to preprocessed dataset
        epochs (int): Training epochs. Default: 50
        batch_size (int): Batch size. Default: 16
        learning_rate (float): Adam learning rate. Default: 0.0001
        sequence_length (int): Frames per sequence. Default: 10
        save_path (str): Path to save best model weights
    """
    print("📦 Loading dataset...")
    X, y = load_dataset(data_dir, sequence_length)
    print(f"  Dataset shape: {X.shape}, Labels: {y.shape}")
    print(f"  Snatching samples: {int(y.sum())}, Normal samples: {int((1 - y).sum())}")

    # Split: 70% train, 20% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp
    )

    print(f"\n📊 Split sizes:")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Save test set for evaluation
    os.makedirs('data/test_split/', exist_ok=True)
    np.save('data/test_split/X_test.npy', X_test)
    np.save('data/test_split/y_test.npy', y_test)

    print("\n🏗️ Building model...")
    model = build_td_cnn_lstm(
        num_frames=sequence_length,
        frame_height=240,
        frame_width=240,
        channels=3
    )
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=1, min_lr=1e-7)
    ]

    print(f"\n🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history)
    print(f"\n✅ Best model saved to: {save_path}")
    return model, history


if __name__ == '__main__':
    train_model()
