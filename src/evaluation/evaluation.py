# src/evaluation/evaluation.py

import logging
import os
from src.visualization.plotting import (
    load_plot_config,
    apply_plot_config,
    plot_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(y_true, y_pred, y_scores, figures_dir: str, plot_config_path: str):
    """
    Evaluates the model and generates evaluation plots.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - y_scores (np.ndarray): Predicted probabilities or scores.
    - figures_dir (str): Directory to save the plots.
    - plot_config_path (str): Path to the plot configuration YAML file.

    Returns:
    - None
    """
    logger = logging.getLogger(__name__)
    try:
        # Load and apply plot configuration
        plot_config = load_plot_config(plot_config_path)
        apply_plot_config(plot_config)

        # Compute classification metrics
        report = classification_report(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        logger.info("Classification Report:\n" + report)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Save classification report to a text file
        report_path = os.path.join(figures_dir, 'test_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {report_path}")

        # Generate and save confusion matrix plot
        cm_plot_path = os.path.join(figures_dir, 'test_confusion_matrix.png')
        plot_confusion_matrix(
            cm=cm,
            classes=['Normal', 'Anomaly'],
            title='Confusion Matrix',
            save_path=cm_plot_path,
            config=plot_config
        )

        # Generate and save ROC curve
        roc_plot_path = os.path.join(figures_dir, 'roc_curve.png')
        plot_roc_curve(
            y_true=y_true,
            y_scores=y_scores,
            save_path=roc_plot_path,
            config=plot_config
        )

        # Generate and save Precision-Recall curve
        pr_plot_path = os.path.join(figures_dir, 'precision_recall_curve.png')
        plot_precision_recall_curve(
            y_true=y_true,
            y_scores=y_scores,
            save_path=pr_plot_path,
            config=plot_config
        )

    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise
