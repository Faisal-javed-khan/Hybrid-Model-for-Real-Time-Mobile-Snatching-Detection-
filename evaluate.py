"""
Evaluation Script — Accuracy, Precision, Recall, Confusion Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import tensorflow as tf
import os


def evaluate_model(model_path, test_dir=None, X_test=None, y_test=None):
    """
    Evaluate the trained TD_CNN-LSTM model.

    Provide either test_dir OR (X_test, y_test) directly.

    Args:
        model_path (str): Path to saved .h5 model
        test_dir (str): Directory with test split .npy files
        X_test (np.ndarray): Test features (optional)
        y_test (np.ndarray): Test labels (optional)

    Returns:
        dict: accuracy, precision, recall
    """
    print("📥 Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    if X_test is None or y_test is None:
        print("📦 Loading test data...")
        X_test = np.load(os.path.join(test_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(test_dir, 'y_test.npy'))

    print(f"  Test samples: {len(X_test)}")

    # Predict
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 45)
    print("         📊 EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print("=" * 45)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Snatching']))

    # Plot confusion matrix
    plot_confusion_matrix(cm)

    return {'accuracy': acc, 'precision': prec, 'recall': rec}


def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Snatching'],
                yticklabels=['Normal', 'Snatching'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix — TD-CNN-LSTM\nMobile Snatching Dataset')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved to: {save_path}")


if __name__ == '__main__':
    evaluate_model(
        model_path='models/td_cnn_lstm_best.h5',
        test_dir='data/test_split/'
    )
