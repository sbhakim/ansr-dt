# src/reasoning/rule_learning.py


import numpy as np
from typing import List, Dict, Tuple, Any
import logging


class RuleLearner:
    def __init__(self, base_threshold=0.7, window_size=10):
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        self.learned_patterns = []

        # Add thresholds based on data statistics
        self.thresholds = {
            'temperature': {
                'high': 80.0,
                'low': 40.0,
                'gradient': 10.0
            },
            'vibration': {
                'high': 55.0,
                'low': 20.0,
                'gradient': 5.0
            },
            'pressure': {
                'high': 40.0,
                'low': 20.0,
                'gradient': 2.0
            },
            'efficiency_index': {
                'high': 0.9,
                'low': 0.6,
                'gradient': 0.1
            }
        }

    def analyze_temporal_patterns(self, sequences: np.ndarray,
                                  labels: np.ndarray) -> List[Dict[str, Any]]:
        """Enhanced temporal pattern analysis."""
        try:
            patterns = []
            for i in range(len(sequences) - self.window_size):
                window = sequences[i:i + self.window_size]
                if labels[i + self.window_size - 1] == 1:  # Anomaly detected

                    # Calculate multi-feature gradients
                    gradients = {
                        'temperature': np.gradient(window[:, 0]),
                        'vibration': np.gradient(window[:, 1]),
                        'pressure': np.gradient(window[:, 2]),
                        'efficiency': np.gradient(window[:, 4])
                    }

                    # Extract significant patterns
                    significant_changes = self._detect_significant_changes(
                        window, gradients)

                    if significant_changes:
                        pattern = {
                            'sequence_idx': i,
                            'window_size': self.window_size,
                            'changes': significant_changes,
                            'confidence': float(labels[i + self.window_size - 1]),
                            'state_transition': self._detect_state_transition(window),
                            'sensor_correlations': self._analyze_correlations(window),
                            'performance_impact': self._analyze_performance(window)
                        }
                        patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error in temporal pattern analysis: {e}")
            return []

    def _detect_significant_changes(self, window: np.ndarray,
                                    gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect significant changes in sensor readings."""
        changes = {}

        # Temperature patterns
        temp_grad = np.max(np.abs(gradients['temperature']))
        if temp_grad > self.thresholds['temperature']['gradient']:
            changes['temperature'] = {
                'gradient': float(temp_grad),
                'pattern': 'rapid_change',
                'confidence': self._calculate_confidence(temp_grad,
                                                         self.thresholds['temperature']['gradient'])
            }

        # Vibration patterns
        vib_grad = np.max(np.abs(gradients['vibration']))
        if vib_grad > self.thresholds['vibration']['gradient']:
            changes['vibration'] = {
                'gradient': float(vib_grad),
                'pattern': 'rapid_change',
                'confidence': self._calculate_confidence(vib_grad,
                                                         self.thresholds['vibration']['gradient'])
            }

        # Check for combined patterns
        if ('temperature' in changes and 'vibration' in changes):
            changes['combined'] = {
                'pattern': 'temp_vib_correlation',
                'confidence': min(changes['temperature']['confidence'],
                                  changes['vibration']['confidence'])
            }

        return changes

    def _analyze_correlations(self, window: np.ndarray) -> Dict[str, float]:
        """Analyze sensor correlations within window."""
        correlations = {}

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(window.T)

        # Extract relevant correlations
        correlations['temp_vib'] = float(corr_matrix[0, 1])
        correlations['temp_press'] = float(corr_matrix[0, 2])
        correlations['vib_press'] = float(corr_matrix[1, 2])

        return correlations

    def _analyze_performance(self, window: np.ndarray) -> Dict[str, float]:
        """Analyze performance metrics."""
        return {
            'efficiency_change': float(window[-1, 4] - window[0, 4]),
            'performance_change': float(window[-1, 6] - window[0, 6]),
            'state_severity': float(window[-1, 5])
        }

    def _detect_state_transition(self, window: np.ndarray) -> Dict[str, Any]:
        """Detect state transitions in the window."""
        states = window[:, 5].astype(int)
        transitions = []

        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                transitions.append({
                    'from': int(states[i - 1]),
                    'to': int(states[i]),
                    'timestep': i
                })

        return {
            'transitions': transitions,
            'initial_state': int(states[0]),
            'final_state': int(states[-1]),
            'transition_count': len(transitions)
        }

    def _calculate_confidence(self, value: float, threshold: float) -> float:
        """Calculate confidence score based on threshold."""
        return min(1.0, value / threshold)

    def extract_rules(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Convert patterns to Prolog rules with confidence scores."""
        rules = []
        for i, pattern in enumerate(patterns):
            if pattern['confidence'] >= self.base_threshold:
                # Extract pattern components
                changes = pattern['changes']
                correlations = pattern.get('sensor_correlations', {})
                performance = pattern.get('performance_impact', {})

                # Build rule conditions
                conditions = []

                # Add temporal conditions
                for sensor, change in changes.items():
                    if sensor != 'combined':
                        conditions.append(
                            f"{sensor}_gradient({change['gradient']:.1f})"
                        )

                # Add correlation conditions
                if correlations.get('temp_vib', 0) > 0.8:
                    conditions.append("temp_vib_correlation")

                # Add performance conditions
                if performance.get('efficiency_change', 0) < -0.1:
                    conditions.append(f"efficiency_drop({abs(performance['efficiency_change']):.1f})")

                # Create rule
                rule_name = f"temporal_pattern_{i}"
                rule_body = ", ".join(conditions)
                rule = f"{rule_name} :- {rule_body}."

                rules.append(rule)

        return rules