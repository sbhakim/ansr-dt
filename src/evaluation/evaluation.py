# src/evaluation/evaluation.py

# src/evaluation/evaluation.py

import logging
import os
from src.visualization.plotting import (
    load_plot_config,
    apply_plot_config,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import numpy as np
from src.reasoning.reasoning import SymbolicReasoner
import json


def _get_rules_path(config_path: str) -> str:
    """
    Extracts the symbolic reasoning rules path from the configuration.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - rules_path (str): Path to the Prolog rules file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        rules_path = config.get('paths', {}).get('reasoning_rules_path', '')
        if not rules_path:
            raise ValueError("Reasoning rules path not found in configuration.")
        return rules_path
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to extract rules path from config: {e}")
        raise


def evaluate_model(y_true, y_pred, y_scores, figures_dir: str, plot_config_path: str, config_path: str,
                   sensor_data: np.ndarray):
    """
    Evaluates the model and generates evaluation plots along with symbolic reasoning insights.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - y_scores (np.ndarray): Predicted probabilities or scores.
    - figures_dir (str): Directory to save the plots.
    - plot_config_path (str): Path to the plot configuration YAML file.
    - config_path (str): Path to the main configuration YAML file.
    - sensor_data (np.ndarray): Sensor data corresponding to each prediction.
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

        # Apply symbolic reasoning if enabled
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        symbolic_reasoning_enabled = config.get('symbolic_reasoning', {}).get('enabled', False)

        if symbolic_reasoning_enabled:
            # Get the rules path from config
            rules_path = _get_rules_path(config_path)

            # Initialize the SymbolicReasoner
            symbolic_reasoner = SymbolicReasoner(rules_path=rules_path)
            insights = []

            for i in range(sensor_data.shape[0]):
                sensor_dict = {
                    'temperature': sensor_data[i][0],
                    'vibration': sensor_data[i][1],
                    'pressure': sensor_data[i][2],
                    'operational_hours': sensor_data[i][3],
                    'efficiency_index': sensor_data[i][4]
                }
                insight = symbolic_reasoner.reason(sensor_dict)
                insights.append(insight)

            # Save insights to a JSON file
            insights_path = os.path.join(figures_dir, 'symbolic_reasoning_insights.json')
            with open(insights_path, 'w') as f:
                json.dump(insights, f, indent=2)
            logger.info(f"Symbolic reasoning insights saved to {insights_path}")

    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise

def load_symbolic_rules(config_path: str) -> list:
    """
    Loads symbolic reasoning rules from the configuration file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - rules (list): List of symbolic reasoning rules.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        rules = config.get('symbolic_reasoning', {}).get('rules', [])
        logging.getLogger(__name__).info(f"Loaded {len(rules)} symbolic reasoning rules.")
        return rules
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load symbolic reasoning rules: {e}")
        raise

