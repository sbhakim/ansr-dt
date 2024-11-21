# src/reasoning/rule_learning.py

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
import logging
from datetime import datetime
from pyswip import Prolog
import os


@dataclass
class Rule:
    id: str
    conditions: List[str]
    confidence: float
    support: int
    timestamp: datetime
    last_activation: Optional[datetime] = None
    activation_count: int = 0
    source: str = 'learned'  # 'learned' or 'predefined'


class RuleLearner:
    def __init__(self,
                 base_threshold: float = 0.7,
                 window_size: int = 10,
                 rules_path: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize RuleLearner with integrated Prolog management.

        Args:
            base_threshold (float): Confidence threshold for rule acceptance
            window_size (int): Window size for temporal pattern analysis
            rules_path (str): Path to Prolog rules file
            logger (logging.Logger): Optional logger instance
        """
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.rules_path = rules_path
        self.logger = logger or logging.getLogger(__name__)
        self.learned_rules: Dict[str, Rule] = {}
        self.prolog = Prolog()

        # Track rule relationships
        self.rule_dependencies: Dict[str, Set[str]] = {}
        self.conflicting_rules: Dict[str, Set[str]] = {}

        # Initialize thresholds based on data statistics
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

        # Load existing rules if path provided
        if rules_path and os.path.exists(rules_path):
            self._load_existing_rules()

    def _load_existing_rules(self) -> None:
        """Load existing Prolog rules into the system."""
        try:
            self.prolog.consult(self.rules_path)
            # Extract existing rules and convert to Rule objects
            for rule in self.prolog.query("clause(Head, Body)"):
                rule_str = f"{rule['Head']} :- {rule['Body']}"
                self._add_predefined_rule(rule_str)
            self.logger.info(f"Loaded existing rules from {self.rules_path}")
        except Exception as e:
            self.logger.error(f"Failed to load existing rules: {e}")

    def analyze_temporal_patterns(self,
                                  sequences: np.ndarray,
                                  labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze temporal patterns in sequences for rule extraction.

        Args:
            sequences (np.ndarray): Input sequences
            labels (np.ndarray): Sequence labels

        Returns:
            List[Dict[str, Any]]: Detected patterns with metadata
        """
        try:
            patterns = []
            for i in range(len(sequences) - self.window_size):
                window = sequences[i:i + self.window_size]
                if labels[i + self.window_size - 1] == 1:  # Anomaly detected
                    # Calculate feature gradients
                    gradients = {
                        'temperature': np.gradient(window[:, 0]),
                        'vibration': np.gradient(window[:, 1]),
                        'pressure': np.gradient(window[:, 2]),
                        'efficiency': np.gradient(window[:, 4])
                    }

                    # Detect significant changes
                    significant_changes = self._detect_significant_changes(window, gradients)
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

    def _detect_significant_changes(self,
                                    window: np.ndarray,
                                    gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect significant changes in sensor readings."""
        changes = {}

        # Temperature patterns
        temp_grad = np.max(np.abs(gradients['temperature']))
        if temp_grad > self.thresholds['temperature']['gradient']:
            changes['temperature'] = {
                'gradient': float(temp_grad),
                'pattern': 'rapid_change',
                'confidence': self._calculate_confidence(
                    temp_grad,
                    self.thresholds['temperature']['gradient']
                )
            }

        # Vibration patterns
        vib_grad = np.max(np.abs(gradients['vibration']))
        if vib_grad > self.thresholds['vibration']['gradient']:
            changes['vibration'] = {
                'gradient': float(vib_grad),
                'pattern': 'rapid_change',
                'confidence': self._calculate_confidence(
                    vib_grad,
                    self.thresholds['vibration']['gradient']
                )
            }

        # Combined patterns
        if ('temperature' in changes and 'vibration' in changes):
            changes['combined'] = {
                'pattern': 'temp_vib_correlation',
                'confidence': min(
                    changes['temperature']['confidence'],
                    changes['vibration']['confidence']
                )
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
        """Analyze performance metrics within window."""
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
        """
        Convert patterns to ProbLog-compliant rules.
        """
        try:
            rules = []
            for i, pattern in enumerate(patterns):
                if pattern['confidence'] >= self.base_threshold:
                    conditions = []

                    # Add temperature conditions
                    if pattern.get('high_temp', False):
                        conditions.append("high_temp")

                    # Add vibration conditions
                    if pattern.get('high_vib', False):
                        conditions.append("high_vib")

                    # Add pressure conditions
                    if pattern.get('low_press', False):
                        conditions.append("low_press")

                    if conditions:
                        # Format as ProbLog rule with probability
                        rule_name = f"pattern_rule_{i}"
                        confidence = pattern['confidence']
                        rule = f"{confidence}::{rule_name} :- {', '.join(conditions)}."
                        rules.append(rule)

            return rules

        except Exception as e:
            self.logger.error(f"Error extracting ProbLog rules: {e}")
            return []

    def update_rule_base(self, new_rules: List[Rule], min_confidence: float = 0.7) -> None:
        """Update rule base with new rules."""
        try:
            for rule in new_rules:
                if rule.confidence >= min_confidence:
                    # Check for similar rules
                    similar_rules = self._find_similar_rules(rule)
                    if similar_rules:
                        self._consolidate_rules(rule, similar_rules)
                    else:
                        self.learned_rules[rule.id] = rule
                        self._update_rule_relationships(rule)

            # Save to Prolog file if path provided
            if self.rules_path:
                self._save_rules_to_file()

            self.logger.info(f"Added {len(new_rules)} new rules to the rule base")
        except Exception as e:
            self.logger.error(f"Failed to update rule base: {e}")

    def _find_similar_rules(self, new_rule: Rule) -> List[str]:
        """Find rules with similar conditions."""
        similar_rules = []
        new_conditions = set(new_rule.conditions)

        for rule_id, existing_rule in self.learned_rules.items():
            existing_conditions = set(existing_rule.conditions)
            similarity = len(new_conditions & existing_conditions) / len(new_conditions | existing_conditions)

            if similarity > 0.8:  # High similarity threshold
                similar_rules.append(rule_id)

        return similar_rules

    def _consolidate_rules(self, new_rule: Rule, similar_rules: List[str]) -> None:
        """Consolidate similar rules."""
        best_rule = max(
            [self.learned_rules[rule_id] for rule_id in similar_rules],
            key=lambda r: r.confidence * r.support
        )

        if new_rule.confidence * new_rule.support > best_rule.confidence * best_rule.support:
            # Replace existing rule
            for rule_id in similar_rules:
                self._remove_rule(rule_id)
            self.learned_rules[new_rule.id] = new_rule
            self._update_rule_relationships(new_rule)

    def _update_rule_relationships(self, rule: Rule) -> None:
        """Update rule dependencies and conflicts."""
        self.rule_dependencies[rule.id] = set()
        self.conflicting_rules[rule.id] = set()

        for existing_id, existing_rule in self.learned_rules.items():
            if existing_id == rule.id:
                continue

            if self._check_dependency(rule, existing_rule):
                self.rule_dependencies[rule.id].add(existing_id)

            if self._check_conflict(rule, existing_rule):
                self.conflicting_rules[rule.id].add(existing_id)

    def _check_dependency(self, rule1: Rule, rule2: Rule) -> bool:
        """Check if rule1 depends on rule2."""
        return any(cond in rule2.conditions for cond in rule1.conditions)

    def _check_conflict(self, rule1: Rule, rule2: Rule) -> bool:
        """Check if rules have conflicting conditions."""
        return any(
            self._are_conditions_conflicting(cond1, cond2)
            for cond1 in rule1.conditions
            for cond2 in rule2.conditions
        )

    def _are_conditions_conflicting(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions conflict."""
        # Basic implementation - could be expanded
        if cond1 == cond2:
            return False
        if cond1.split('(')[0] == cond2.split('(')[0]:
            return True
        return False

    def _remove_rule(self, rule_id: str) -> None:
        """Remove a rule and its relationships."""
        if rule_id in self.learned_rules:
            del self.learned_rules[rule_id]
        if rule_id in self.rule_dependencies:
            del self.rule_dependencies[rule_id]
        if rule_id in self.conflicting_rules:
            del self.conflicting_rules[rule_id]

    def _save_rules_to_file(self) -> None:
        """Save current rule base to Prolog file."""
        if not self.rules_path:
            return

        try:
            with open(self.rules_path, 'w') as f:
                # Write header
                f.write("%% NEXUS-DT Rule Base\n")
                f.write("%% Generated: " + datetime.now().isoformat() + "\n\n")

                # Write rules
                for rule in self.learned_rules.values():
                    formatted_rule = (
                        f"{rule.id} :- "
                        f"{', '.join(rule.conditions)}. "
                        f"% Confidence: {rule.confidence:.2f}\n"
                    )
                    f.write(formatted_rule)

            self.logger.info(f"Rules saved to {self.rules_path}")
        except Exception as e:
            self.logger.error(f"Failed to save rules to file: {e}")

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current rule base."""
        return {
            'total_rules': len(self.learned_rules),
            'high_confidence_rules': sum(
                1 for r in self.learned_rules.values()
                if r.confidence >= self.base_threshold
            ),
            'average_confidence': np.mean([
                r.confidence for r in self.learned_rules.values()
            ]) if self.learned_rules else 0.0,
            'rule_sources': {
                'learned': sum(1 for r in self.learned_rules.values() if r.source == 'learned'),
                'predefined': sum(1 for r in self.learned_rules.values() if r.source == 'predefined')
            },
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger("RuleLearner")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        # Initialize RuleLearner
        learner = RuleLearner(
            base_threshold=0.7,
            window_size=10,
            rules_path="rules/learned_rules.pl",
            logger=logger
        )

        # Example data (dummy sequences and labels)
        sequences = np.random.rand(100, 10, 7)  # 100 sequences, 10 timesteps, 7 features
        labels = np.random.randint(0, 2, size=100)  # Binary labels

        # Analyze patterns
        patterns = learner.analyze_temporal_patterns(sequences, labels)
        logger.info(f"Found {len(patterns)} patterns")

        # Extract rules from patterns
        new_rules = learner.extract_rules(patterns)
        logger.info(f"Extracted {len(new_rules)} rules")

        # Update rule base
        learner.update_rule_base(new_rules)

        # Get and print statistics
        stats = learner.get_rule_statistics()
        logger.info("Rule Base Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"  {subkey}: {subvalue}")
            else:
                logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Error in example execution: {e}")
        raise