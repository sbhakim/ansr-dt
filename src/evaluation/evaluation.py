# src/evaluation/evaluation.py

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc
)
from src.visualization.plotting import (
    load_plot_config,
    apply_plot_config,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from src.visualization.model_visualization import ModelVisualizer
from src.reasoning.reasoning import SymbolicReasoner


def get_project_root(config_path: str) -> str:
    """Get project root directory from config path."""
    return os.path.dirname(os.path.dirname(config_path))


def setup_evaluation_dirs(results_dir: str) -> Dict[str, str]:
    """
    Create and return evaluation directory structure.

    Args:
        results_dir: Base results directory

    Returns:
        Dictionary of directory paths
    """
    dirs = {
        'base': results_dir,
        'figures': os.path.join(results_dir, 'visualization'),
        'model_viz': os.path.join(results_dir, 'visualization', 'model_visualization'),
        'metrics': os.path.join(results_dir, 'metrics')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities

    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        'classification_report': classification_report(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'roc_auc': float(roc_auc_score(y_true, y_scores)),
        'avg_precision': float(average_precision_score(y_true, y_scores))
    }

    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    metrics['pr_auc'] = float(auc(recall, precision))

    return metrics


def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        figures_dir: str,
        plot_config_path: str,
        config_path: str,
        sensor_data: np.ndarray,
        model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with visualization and symbolic reasoning.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores
        figures_dir: Directory to save figures
        plot_config_path: Path to plotting configuration
        config_path: Path to main configuration
        sensor_data: Sensor data for analysis
        model: Optional trained model for visualization

    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.yaml')
        logger.info(f"Using config file: {config_path}")

    symbolic_insights = []

    try:
        # Check if config_path is a file
        if not os.path.isfile(config_path):
            logger.error(f"Config path is not a file: {config_path}")
            raise ValueError(f"Config path is not a file: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Determine project root
        project_root = get_project_root(config_path)

        # Initialize symbolic reasoner with correct path
        rules_path = config['paths'].get('reasoning_rules_path')
        if not rules_path:
            logger.error("reasoning_rules_path not found in configuration.")
            raise KeyError("reasoning_rules_path not found in configuration.")

        if not os.path.isabs(rules_path):
            rules_path = os.path.join(project_root, rules_path)

        # Check if rules_path is a file
        if not os.path.isfile(rules_path):
            logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")

        # Initialize reasoner with proper parameters
        input_shape = (config['model']['window_size'], len(config['model']['feature_names']))
        reasoner = SymbolicReasoner(
            rules_path=rules_path,
            input_shape=input_shape,
            model=model,
            logger=logger
        )

        # Setup evaluation directories
        eval_dirs = setup_evaluation_dirs(figures_dir)

        # Load and apply plot configuration
        plot_config = load_plot_config(plot_config_path)
        apply_plot_config(plot_config)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        logger.info(f"Classification Report:\n{metrics['classification_report']}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        # Save classification report
        report_path = os.path.join(eval_dirs['metrics'], 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(metrics['classification_report'])

        # Generate and save plots
        plot_paths = {}

        # Confusion Matrix
        cm_path = os.path.join(eval_dirs['figures'], 'confusion_matrix.png')
        plot_confusion_matrix(
            cm=np.array(metrics['confusion_matrix']),
            classes=['Normal', 'Anomaly'],
            title='Confusion Matrix',
            save_path=cm_path,
            config=plot_config
        )
        plot_paths['confusion_matrix'] = cm_path

        # ROC Curve
        roc_path = os.path.join(eval_dirs['figures'], 'roc_curve.png')
        plot_roc_curve(
            y_true=y_true,
            y_scores=y_scores,
            save_path=roc_path,
            config=plot_config
        )
        plot_paths['roc_curve'] = roc_path

        # Precision-Recall Curve
        pr_path = os.path.join(eval_dirs['figures'], 'precision_recall_curve.png')
        plot_precision_recall_curve(
            y_true=y_true,
            y_scores=y_scores,
            save_path=pr_path,
            config=plot_config
        )
        plot_paths['precision_recall'] = pr_path

        # Model visualization if model is provided
        feature_importance_scores = None
        if model is not None:
            try:
                visualizer = ModelVisualizer(model, logger)

                if hasattr(model, 'input_shape'):
                    # Prepare sample data for visualization
                    sample_data = sensor_data[:10].reshape(1, 10, -1)

                    # Ensure the model is built
                    if not model.built:
                        logger.info("Building the model with sample data for visualization.")
                        model.predict(sample_data)

                    # Generate feature importance visualization
                    importance_path = os.path.join(eval_dirs['model_viz'], 'feature_importance.png')
                    importance, success = visualizer.get_feature_importance(
                        input_data=sample_data,
                        save_path=importance_path
                    )

                    if success and importance is not None:
                        feature_importance_scores = dict(zip(
                            ['temperature', 'vibration', 'pressure', 'operational_hours',
                             'efficiency_index', 'system_state', 'performance_score'],
                            importance.tolist()
                        ))

                        # Save feature importance scores
                        scores_path = os.path.join(eval_dirs['model_viz'], 'feature_importance.json')
                        with open(scores_path, 'w') as f:
                            json.dump(feature_importance_scores, f, indent=2)
                        plot_paths['feature_importance'] = importance_path

                # Save model architecture
                arch_path = os.path.join(eval_dirs['model_viz'], 'model_architecture.txt')
                visualizer.visualize_model_architecture(arch_path)

            except Exception as viz_error:
                logger.warning(f"Model visualization warning: {viz_error}")

        # Symbolic reasoning
        try:
            # Process each timestep
            for i in range(len(sensor_data)):
                sensor_dict = {
                    'temperature': float(sensor_data[i][0]),
                    'vibration': float(sensor_data[i][1]),
                    'pressure': float(sensor_data[i][2]),
                    'operational_hours': float(sensor_data[i][3]),
                    'efficiency_index': float(sensor_data[i][4]),
                    'system_state': float(sensor_data[i][5]),
                    'performance_score': float(sensor_data[i][6])
                }
                insight = reasoner.reason(sensor_dict)
                if insight:
                    symbolic_insights.append({
                        'timestep': i,
                        'insights': insight,
                        'readings': sensor_dict
                    })

            # Save insights
            insights_path = os.path.join(eval_dirs['metrics'], 'symbolic_insights.json')
            with open(insights_path, 'w') as f:
                json.dump(symbolic_insights, f, indent=2)

        except Exception as reasoning_error:
            logger.warning(f"Symbolic reasoning warning: {reasoning_error}")

        # Compile final evaluation results
        evaluation_results = {
            'metrics': metrics,
            'plot_paths': plot_paths,
            'feature_importance': feature_importance_scores,
            'symbolic_insights': {
                'total_insights': len(symbolic_insights),
                'insights_path': insights_path if symbolic_insights else None
            },
            'timestamp': str(np.datetime64('now')),
            'success': True
        }

        # Save evaluation summary
        summary_path = os.path.join(eval_dirs['base'], 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info("Evaluation completed successfully")
        return evaluation_results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'error': str(e),
            'success': False
        }


def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """
    Load saved evaluation results.

    Args:
        results_path: Path to evaluation results file

    Returns:
        Dictionary containing loaded results
    """
    try:
        if not os.path.isfile(results_path):
            logging.getLogger(__name__).error(f"Results path is not a file: {results_path}")
            raise ValueError(f"Results path is not a file: {results_path}")

        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load evaluation results: {e}")
        raise
