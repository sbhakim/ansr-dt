import numpy as np
from typing import Dict, List
import logging


class PatternEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_rules(self, rules: List[Dict], predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """Evaluate effectiveness of learned rules."""
        try:
            total_rules = len(rules)
            correct_predictions = 0
            rule_coverage = 0

            # Evaluate each rule's effectiveness
            for rule in rules:
                condition_met = rule['condition_met']
                idx = rule['timestep']

                if idx < len(predictions):
                    if condition_met and predictions[idx] == actual[idx]:
                        correct_predictions += 1
                    rule_coverage += 1

            metrics = {
                'accuracy': correct_predictions / total_rules if total_rules > 0 else 0,
                'coverage': rule_coverage / len(predictions),
                'total_rules': total_rules,
                'valid_predictions': correct_predictions
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error in rule evaluation: {e}")
            return {}