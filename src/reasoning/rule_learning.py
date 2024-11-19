# src/reasoning/rule_learning.py


import numpy as np
from typing import List, Dict, Tuple
import logging


class RuleLearner:
    def __init__(self, threshold: float = 0.7):
        """Initialize with confidence threshold."""
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self.learned_patterns = []

    def analyze_temporal_patterns(self,
                                  sequences: np.ndarray,
                                  labels: np.ndarray) -> List[Dict[str, any]]:
        """
        Analyze sequences to identify temporal patterns leading to anomalies.
        """
        try:
            patterns = []
            for i in range(len(sequences) - 1):
                if labels[i + 1] == 1:  # Anomaly detected
                    current_seq = sequences[i]
                    next_seq = sequences[i + 1]

                    # Check for significant changes
                    temp_change = next_seq[0] - current_seq[0]  # Temperature
                    vib_change = next_seq[1] - current_seq[1]  # Vibration
                    press_change = next_seq[2] - current_seq[2]  # Pressure

                    if abs(temp_change) > 5 or abs(vib_change) > 3 or abs(press_change) > 2:
                        pattern = {
                            'sequence_idx': i,
                            'changes': {
                                'temperature': temp_change,
                                'vibration': vib_change,
                                'pressure': press_change
                            },
                            'confidence': float(labels[i + 1])
                        }
                        patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error in temporal pattern analysis: {e}")
            return []

    def extract_rules(self, patterns: List[Dict[str, any]]) -> List[str]:
        """
        Convert patterns to Prolog rules.
        """
        try:
            rules = []
            for i, pattern in enumerate(patterns):
                if pattern['confidence'] >= self.threshold:
                    changes = pattern['changes']
                    rule = (f"temporal_pattern_{i} :- "
                            f"temp_change({changes['temperature']:.1f}), "
                            f"vib_change({changes['vibration']:.1f}), "
                            f"press_change({changes['pressure']:.1f}).")
                    rules.append(rule)

            return rules

        except Exception as e:
            self.logger.error(f"Error in rule extraction: {e}")
            return []