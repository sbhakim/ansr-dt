# utils/evaluation.py

import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import os

def plot_metrics(history, figures_dir: str):
    """
    Plots and saves training and validation accuracy and loss.

    Parameters:
    - history: Keras History object.
    - figures_dir (str): Directory to save the plots.
    """
    logger = logging.getLogger(__name__)  # Initialize module-specific logger
    try:
        plt.figure(figsize=(12, 4))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Save plots
        plot_path = os.path.join(figures_dir, 'training_plots.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training plots saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to plot metrics: {e}")
        raise

def compute_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes classification report and confusion matrix.

    Parameters:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.

    Returns:
    - report (str): Classification report as a string.
    - cm (np.ndarray): Confusion matrix.
    """
    logger = logging.getLogger(__name__)  # Initialize module-specific logger
    try:
        report = classification_report(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        logger.info("Classification report and confusion matrix computed.")
        return report, cm
    except Exception as e:
        logger.error(f"Failed to compute additional metrics: {e}")
        raise
