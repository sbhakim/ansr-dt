# src/evaluation/pattern_metrics.py

import numpy as np
from typing import Dict, List
import logging


class PatternEvaluator:
    def __init__(self):
        """Initialize pattern evaluator."""
        self.logger = logging.getLogger(__name__)

    def evaluate_rules(self,
                       predictions: np.ndarray,
                       actual_anomalies: np.ndarray,
                       rule_activations: List[Dict]) -> Dict[str, float]:
        """
        Evaluate effectiveness of learned rules.
        """
        try:
            total_rules = len(rule_activations)
            correct_predictions = 0

            for activation in rule_activations:
                idx = activation['index']
                if idx < len(predictions) and predictions[idx] == actual_anomalies[idx]:
                    correct_predictions += 1

            accuracy = correct_predictions / total_rules if total_rules > 0 else 0

            return {
                'rule_accuracy': accuracy,
                'total_rules': total_rules,
                'correct_predictions': correct_predictions
            }

        except Exception as e:
            self.logger.error(f"Error evaluating rules: {e}")
            return {}