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
from src.visualization.model_visualization import ModelVisualizer
import yaml
import numpy as np
from src.reasoning.reasoning import SymbolicReasoner
import json
from typing import Optional, Dict, List, Any


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


def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_scores: np.ndarray,
                   figures_dir: str,
                   plot_config_path: str,
                   config_path: str,
                   sensor_data: np.ndarray,
                   model: Optional[Any] = None) -> Dict[str, Any]:
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
    - model: Optional trained model for visualization features.

    Returns:
    - Dict[str, Any]: Evaluation results and metrics
    """
    logger = logging.getLogger(__name__)

    # Initialize paths and results
    cnn_path = None
    importance_path = None
    insights = []
    evaluation_results = {}

    try:
        # Load and apply plot configuration
        plot_config = load_plot_config(plot_config_path)
        apply_plot_config(plot_config)

        # Create figures directory if it doesn't exist
        os.makedirs(figures_dir, exist_ok=True)

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

        # Model visualization if model is provided
        importance_scores = None
        if model is not None:
            viz_dir = os.path.join(figures_dir, 'model_visualization')
            os.makedirs(viz_dir, exist_ok=True)

            try:
                visualizer = ModelVisualizer(model, logger)

                # Prepare sample data for visualization
                sample_window = sensor_data[:10]  # Take first window
                sample_data = sample_window.reshape(1, *sample_window.shape)

                # Visualize CNN features
                cnn_path = os.path.join(viz_dir, 'cnn_features.png')
                feature_maps, cnn_success = visualizer.visualize_cnn_features(
                    input_data=sample_data,
                    layer_name='conv1d',
                    save_path=cnn_path
                )
                if not cnn_success:
                    cnn_path = None

                # Calculate and visualize feature importance
                importance_path = os.path.join(viz_dir, 'feature_importance.png')
                importance, importance_success = visualizer.get_feature_importance(
                    input_data=sample_data,
                    save_path=importance_path
                )

                if importance_success and importance is not None:
                    importance_scores = {
                        'temperature': float(importance[0]),
                        'vibration': float(importance[1]),
                        'pressure': float(importance[2]),
                        'operational_hours': float(importance[3]),
                        'efficiency_index': float(importance[4]),
                        'system_state': float(importance[5]),
                        'performance_score': float(importance[6])
                    }

                    with open(os.path.join(viz_dir, 'feature_importance.json'), 'w') as f:
                        json.dump(importance_scores, f, indent=2)
                else:
                    importance_path = None

            except Exception as viz_error:
                logger.warning(f"Model visualization failed: {viz_error}. Continuing with evaluation...")
                cnn_path = None
                importance_path = None

        # Apply symbolic reasoning if enabled
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        symbolic_reasoning_enabled = config.get('symbolic_reasoning', {}).get('enabled', False)

        if symbolic_reasoning_enabled:
            try:
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
                        'efficiency_index': sensor_data[i][4],
                        'system_state': sensor_data[i][5],
                        'performance_score': sensor_data[i][6]
                    }
                    insight = symbolic_reasoner.reason(sensor_dict)
                    insights.append(insight)

                # Save insights to a JSON file
                insights_path = os.path.join(figures_dir, 'symbolic_reasoning_insights.json')
                with open(insights_path, 'w') as f:
                    json.dump(insights, f, indent=2)
                logger.info(f"Symbolic reasoning insights saved to {insights_path}")

            except Exception as reasoning_error:
                logger.warning(f"Symbolic reasoning failed: {reasoning_error}. Continuing with evaluation...")
                insights = []

        # Create an evaluation summary dictionary
        evaluation_summary = {
            'classification_metrics': {
                'report': report,
                'confusion_matrix': cm.tolist()
            },
            'symbolic_insights_count': len(insights),
            'visualization_paths': {
                'confusion_matrix': cm_plot_path,
                'roc_curve': roc_plot_path,
                'precision_recall': pr_plot_path,
                'cnn_features': cnn_path,
                'feature_importance': importance_path
            },
            'feature_importance_scores': importance_scores,
            'timestamp': str(np.datetime64('now'))
        }

        # Save evaluation summary
        summary_path = os.path.join(figures_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        logger.info(f"Evaluation summary saved to {summary_path}")

        evaluation_results = {
            'success': True,
            'classification_report': report,
            'confusion_matrix': cm,
            'insights': insights,
            'importance_scores': importance_scores,
            'paths': {
                'summary': summary_path,
                'confusion_matrix': cm_plot_path,
                'roc_curve': roc_plot_path,
                'precision_recall': pr_plot_path,
                'cnn_features': cnn_path,
                'feature_importance': importance_path
            }
        }

        return evaluation_results

    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        evaluation_results = {
            'success': False,
            'error': str(e),
            'paths': {
                'confusion_matrix': cm_plot_path if 'cm_plot_path' in locals() else None,
                'roc_curve': roc_plot_path if 'roc_plot_path' in locals() else None,
                'precision_recall': pr_plot_path if 'pr_plot_path' in locals() else None,
                'cnn_features': cnn_path,
                'feature_importance': importance_path
            }
        }
        return evaluation_results


def load_symbolic_rules(config_path: str) -> List[str]:
    """
    Loads symbolic reasoning rules from the configuration file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - List[str]: List of symbolic reasoning rules.
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