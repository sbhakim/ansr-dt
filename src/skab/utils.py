import csv
import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import DEFAULT_LOG_FORMAT


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_run_dir(base_dir: str, run_name: str = None) -> str:
    ensure_directory(base_dir)
    run_name = run_name or __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_directory(os.path.dirname(path))
    with open(path, 'w') as handle:
        json.dump(to_serializable(payload), handle, indent=2)


def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_directory(os.path.dirname(path))
    if not rows:
        with open(path, 'w') as handle:
            handle.write('')
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_serializable(v) for k, v in row.items()})


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'positive_rate': float(np.mean(y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        metrics['roc_auc'] = None

    try:
        metrics['avg_precision'] = float(average_precision_score(y_true, y_scores))
    except ValueError:
        metrics['avg_precision'] = None

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = float(auc(recall_curve, precision_curve))
    except ValueError:
        metrics['pr_auc'] = None

    return metrics


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def select_threshold(y_true: np.ndarray, y_scores: np.ndarray, metric_name: str = 'f1') -> Tuple[float, Dict[str, float]]:
    rounded_scores = np.unique(np.round(np.asarray(y_scores, dtype=float), 4))
    if rounded_scores.size == 0:
        return 0.5, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'positive_rate': 0.0}

    best_threshold = 0.5
    best_metrics = None
    best_value = -1.0

    y_true = np.asarray(y_true).astype(int)
    for threshold in rounded_scores:
        y_pred = (y_scores >= threshold).astype(int)
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0
        candidate = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_rate': float(np.mean(y_pred)),
        }
        if candidate[metric_name] > best_value:
            best_value = candidate[metric_name]
            best_threshold = float(threshold)
            best_metrics = candidate

    return best_threshold, best_metrics or {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'positive_rate': 0.0}
