# src/visualization/plotting.py

import os
import numpy as np  # Ensure this import exists
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
        plt.figure(figsize=(12, 5))  # Adjusted figure size for better layout

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        accuracy = history.history.get('accuracy') or history.history.get('acc')
        val_accuracy = history.history.get('val_accuracy') or history.history.get('val_acc')
        plt.plot(accuracy, label='Train')
        plt.plot(val_accuracy, label='Validation')
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plot_path = os.path.join(figures_dir, 'training_plots.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training plots saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to plot metrics: {e}")
        raise


def plot_confusion_matrix(cm, classes, title: str, save_path: str, config: dict):
    """
    Plots and saves the confusion matrix with improved layout handling.
    """
    logger = logging.getLogger(__name__)
    try:
        # Create figure and axis with constrained layout
        fig, ax = plt.subplots(figsize=config['figure']['figsize'],
                               constrained_layout=True)

        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # Add colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        # Configure ticks
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            percentage = cm[i, j] / np.sum(cm[i, :]) * 100 if np.sum(cm[i, :]) != 0 else 0
            ax.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        # Save figure
        plt.savefig(save_path, dpi=config['figure']['dpi'],
                    bbox_inches='tight', format=config['savefig']['format'])
        plt.close(fig)

        logger.info(f"Confusion matrix plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        raise


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str, config: dict):
    """
    Plots and saves the ROC curve with improved layout handling.
    """
    logger = logging.getLogger(__name__)
    try:
        fig, ax = plt.subplots(figsize=config['figure']['figsize'],
                               constrained_layout=True)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        ax.plot(fpr, tpr, color='darkorange',
                lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")

        plt.savefig(save_path, dpi=config['figure']['dpi'],
                    bbox_inches='tight', format=config['savefig']['format'])
        plt.close(fig)

        logger.info(f"ROC curve saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot ROC curve: {e}")
        raise


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str, config: dict):
    """
    Plots and saves the Precision-Recall curve with improved layout handling.
    """
    logger = logging.getLogger(__name__)
    try:
        fig, ax = plt.subplots(figsize=config['figure']['figsize'],
                               constrained_layout=True)

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        ax.plot(recall, precision, color='blue',
                lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="upper right")

        plt.savefig(save_path, dpi=config['figure']['dpi'],
                    bbox_inches='tight', format=config['savefig']['format'])
        plt.close(fig)

        logger.info(f"Precision-Recall curve saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot Precision-Recall curve: {e}")
        raise
