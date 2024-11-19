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
from .pattern_metrics import PatternEvaluator


def get_project_root(config_path: str) -> str:
    """Get project root directory from config path."""
    return os.path.dirname(os.path.dirname(config_path))

def setup_evaluation_dirs(results_dir: str) -> Dict[str, str]:
    """Create and return evaluation directory structure."""
    dirs = {
        'base': results_dir,
        'figures': os.path.join(results_dir, 'visualization'),
        'model_viz': os.path.join(results_dir, 'visualization', 'model_visualization'),
        'metrics': os.path.join(results_dir, 'metrics'),
        'patterns': os.path.join(results_dir, 'pattern_analysis')  # New directory
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'classification_report': classification_report(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'roc_auc': float(roc_auc_score(y_true, y_scores)),
        'avg_precision': float(average_precision_score(y_true, y_scores))
    }

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
    Enhanced model evaluation with visualization, symbolic reasoning, and pattern analysis.
    """
    logger = logging.getLogger(__name__)
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.yaml')
        logger.info(f"Using config file: {config_path}")

    symbolic_insights = []
    pattern_insights = []

    try:
        # Config validation
        if not os.path.isfile(config_path):
            logger.error(f"Config path is not a file: {config_path}")
            raise ValueError(f"Config path is not a file: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        project_root = get_project_root(config_path)

        # Initialize components
        rules_path = os.path.join(project_root, config['paths'].get('reasoning_rules_path', ''))
        if not os.path.isfile(rules_path):
            logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")

        input_shape = (config['model']['window_size'], len(config['model']['feature_names']))
        reasoner = SymbolicReasoner(
            rules_path=rules_path,
            input_shape=input_shape,
            model=model,
            logger=logger
        )

        # Initialize pattern evaluator
        pattern_evaluator = PatternEvaluator()

        # Setup directories and configuration
        eval_dirs = setup_evaluation_dirs(figures_dir)
        plot_config = load_plot_config(plot_config_path)
        apply_plot_config(plot_config)

        # Calculate base metrics
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        logger.info(f"Classification Report:\n{metrics['classification_report']}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        # Save report
        report_path = os.path.join(eval_dirs['metrics'], 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(metrics['classification_report'])

        # Generate plots
        plot_paths = {}

        # Standard plots
        plot_paths['confusion_matrix'] = os.path.join(eval_dirs['figures'], 'confusion_matrix.png')
        plot_confusion_matrix(
            cm=np.array(metrics['confusion_matrix']),
            classes=['Normal', 'Anomaly'],
            title='Confusion Matrix',
            save_path=plot_paths['confusion_matrix'],
            config=plot_config
        )

        plot_paths['roc_curve'] = os.path.join(eval_dirs['figures'], 'roc_curve.png')
        plot_roc_curve(
            y_true=y_true,
            y_scores=y_scores,
            save_path=plot_paths['roc_curve'],
            config=plot_config
        )

        plot_paths['precision_recall'] = os.path.join(eval_dirs['figures'], 'precision_recall_curve.png')
        plot_precision_recall_curve(
            y_true=y_true,
            y_scores=y_scores,
            save_path=plot_paths['precision_recall'],
            config=plot_config
        )

        # Model visualization and feature importance
        feature_importance_scores = None
        if model is not None:
            try:
                visualizer = ModelVisualizer(model, logger)
                if hasattr(model, 'input_shape'):
                    sample_data = sensor_data[:10].reshape(1, 10, -1)
                    if not model.built:
                        model.predict(sample_data)

                    importance_path = os.path.join(eval_dirs['model_viz'], 'feature_importance.png')
                    importance, success = visualizer.get_feature_importance(
                        input_data=sample_data,
                        save_path=importance_path
                    )

                    if success and importance is not None:
                        feature_importance_scores = dict(zip(
                            config['model']['feature_names'],
                            importance.tolist()
                        ))

                        scores_path = os.path.join(eval_dirs['model_viz'], 'feature_importance.json')
                        with open(scores_path, 'w') as f:
                            json.dump(feature_importance_scores, f, indent=2)
                        plot_paths['feature_importance'] = importance_path

                arch_path = os.path.join(eval_dirs['model_viz'], 'model_architecture.txt')
                visualizer.visualize_model_architecture(arch_path)

            except Exception as viz_error:
                logger.warning(f"Model visualization warning: {viz_error}")

        # Enhanced symbolic reasoning with pattern analysis
        try:
            # Process sequences
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

                # Get symbolic insights
                insight = reasoner.reason(sensor_dict)
                if insight:
                    symbolic_insights.append({
                        'timestep': i,
                        'insights': insight,
                        'readings': sensor_dict
                    })

            # Evaluate patterns
            pattern_metrics = pattern_evaluator.evaluate_rules(
                predictions=y_pred,
                actual_anomalies=y_true,
                rule_activations=reasoner.get_rule_activations() if hasattr(reasoner, 'get_rule_activations') else []
            )

            # Save insights and patterns
            insights_path = os.path.join(eval_dirs['metrics'], 'symbolic_insights.json')
            with open(insights_path, 'w') as f:
                json.dump(symbolic_insights, f, indent=2)

            patterns_path = os.path.join(eval_dirs['patterns'], 'pattern_analysis.json')
            with open(patterns_path, 'w') as f:
                json.dump(pattern_metrics, f, indent=2)

        except Exception as reasoning_error:
            logger.warning(f"Reasoning and pattern analysis warning: {reasoning_error}")

        # Compile results
        evaluation_results = {
            'metrics': metrics,
            'plot_paths': plot_paths,
            'feature_importance': feature_importance_scores,
            'symbolic_insights': {
                'total_insights': len(symbolic_insights),
                'insights_path': insights_path if symbolic_insights else None
            },
            'pattern_analysis': {
                'metrics': pattern_metrics,
                'analysis_path': patterns_path
            },
            'timestamp': str(np.datetime64('now')),
            'success': True
        }

        # Save summary
        summary_path = os.path.join(eval_dirs['base'], 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info("Enhanced evaluation completed successfully")
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
