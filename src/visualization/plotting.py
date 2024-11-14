# src/visualization/plotting.py

import os

import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def load_plot_config(config_path: str) -> dict:
    """
    Loads the plot configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - config (dict): Plot configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.getLogger(__name__).info(f"Plot configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load plot configuration: {e}")
        raise


def apply_plot_config(config: dict):
    """
    Applies the plot configuration to Matplotlib's rcParams.

    Parameters:
    - config (dict): Plot configuration parameters.
    """
    try:
        # Font settings
        matplotlib.rcParams['font.family'] = config['font']['family']
        matplotlib.rcParams['font.size'] = config['font']['size']

        # Figure settings
        matplotlib.rcParams['figure.figsize'] = config['figure']['figsize']
        matplotlib.rcParams['savefig.dpi'] = config['figure']['dpi']
        matplotlib.rcParams['figure.constrained_layout.use'] = config['figure']['tight_layout']

        # Lines settings
        matplotlib.rcParams['lines.linewidth'] = config['lines']['linewidth']
        matplotlib.rcParams['lines.markersize'] = config['lines']['markersize']

        # Ticks settings
        matplotlib.rcParams['xtick.major.size'] = config['ticks']['major_size']
        matplotlib.rcParams['xtick.minor.size'] = config['ticks']['minor_size']
        matplotlib.rcParams['ytick.major.size'] = config['ticks']['major_size']
        matplotlib.rcParams['ytick.minor.size'] = config['ticks']['minor_size']
        matplotlib.rcParams['xtick.direction'] = config['ticks']['direction']
        matplotlib.rcParams['ytick.direction'] = config['ticks']['direction']

        # Legend settings
        matplotlib.rcParams['legend.fontsize'] = config['legend']['fontsize']
        matplotlib.rcParams['legend.frameon'] = config['legend']['frameon']

        # Axes settings
        matplotlib.rcParams['axes.grid'] = config['axes']['grid']
        matplotlib.rcParams['axes.labelsize'] = config['axes']['labelsize']

        # Savefig settings
        matplotlib.rcParams['savefig.format'] = config['savefig']['format']
        matplotlib.rcParams['savefig.bbox'] = config['savefig']['bbox']

        logging.getLogger(__name__).info("Matplotlib rcParams updated with plot configuration.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to apply plot configuration: {e}")
        raise


def plot_metrics(history, figures_dir: str, config: dict):
    """
    Plots and saves training and validation accuracy and loss.

    Parameters:
    - history: Keras History object.
    - figures_dir (str): Directory to save the plots.
    - config (dict): Plot configuration parameters.
    """
    logger = logging.getLogger(__name__)
    try:
        plt.figure()

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history.get('accuracy', history.history.get('acc')), label='Train')
        plt.plot(history.history.get('val_accuracy', history.history.get('val_acc')), label='Validation')
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Save plots
        plot_path = os.path.join(figures_dir, 'training_plots.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training plots saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to plot metrics: {e}")
        raise


def plot_confusion_matrix(cm, classes, title: str, save_path: str, config: dict):
    """
    Plots and saves the confusion matrix.

    Parameters:
    - cm (np.ndarray): Confusion matrix.
    - classes (list): List of class names.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot.
    - config (dict): Plot configuration parameters.
    """
    logger = logging.getLogger(__name__)
    try:
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Normalize the confusion matrix.
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, f"{cm[i, j]} ({cm_normalized[i, j]:.2f})",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        raise


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str, config: dict):
    """
    Plots and saves the ROC curve.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_scores (np.ndarray): Predicted scores or probabilities.
    - save_path (str): Path to save the plot.
    - config (dict): Plot configuration parameters.
    """
    logger = logging.getLogger(__name__)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=1.5, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"ROC curve saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to plot ROC curve: {e}")
        raise


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str, config: dict):
    """
    Plots and saves the Precision-Recall curve.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_scores (np.ndarray): Predicted scores or probabilities.
    - save_path (str): Path to save the plot.
    - config (dict): Plot configuration parameters.
    """
    logger = logging.getLogger(__name__)
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        plt.figure()
        plt.plot(recall, precision, color='blue',
                 lw=1.5, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Precision-Recall curve saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to plot Precision-Recall curve: {e}")
        raise
