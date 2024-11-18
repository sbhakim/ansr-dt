# src/reasoning/reasoning.py

import logging
from pyswip import Prolog
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple


class SymbolicReasoner:
    def __init__(self, rules_path: str):
        self.logger = logging.getLogger(__name__)
        self.prolog = Prolog()
        self.rules_path = rules_path
        self.learned_rules = []
        self.rule_confidence = {}

        if not os.path.exists(rules_path):
            self.logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")

        self.prolog.consult(rules_path)
        self.logger.info(f"Prolog rules loaded from {rules_path}")

    def extract_rules_from_neural_model(
            self,
            model: tf.keras.Model,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7
    ) -> List[str]:
        """
        Extract rules from CNN-LSTM model using gradient-based feature importance.

        Args:
            model: Trained CNN-LSTM model
            input_data: Input data that triggered anomaly detection
            feature_names: Names of input features
            threshold: Importance threshold for rule extraction

        Returns:
            List of extracted rules in Prolog format
        """
        try:
            # Get neural network's prediction and gradients
            with tf.GradientTape() as tape:
                input_tensor = tf.convert_to_tensor(input_data)
                tape.watch(input_tensor)
                predictions = model(input_tensor)

            # Calculate gradients for feature importance
            gradients = tape.gradient(predictions, input_tensor)
            feature_importance = np.abs(gradients.numpy()).mean(axis=1)[0]  # Shape: [n_features]

            # Extract rules based on important features
            new_rules = []

            # For high anomaly predictions, extract rules
            if predictions[0][0] > threshold:
                important_features = []

                # Identify important features
                for idx, importance in enumerate(feature_importance):
                    if importance > threshold:
                        feature_name = feature_names[idx]
                        feature_value = float(input_data[0, -1, idx])  # Get last timestep value
                        important_features.append((feature_name, feature_value, importance))

                if important_features:
                    # Create rule conditions based on important features
                    conditions = []
                    for name, value, importance in important_features:
                        if name == 'temperature':
                            conditions.append(f"temperature > {value:.1f}")
                        elif name == 'vibration':
                            conditions.append(f"vibration > {value:.1f}")
                        elif name == 'pressure':
                            conditions.append(f"pressure < {value:.1f}")
                        elif name == 'efficiency_index':
                            conditions.append(f"efficiency_index < {value:.2f}")

                    if conditions:
                        # Create Prolog rule
                        rule_head = f"neural_anomaly_{len(self.learned_rules) + 1}"
                        rule_body = ", ".join(conditions)
                        rule = f"{rule_head} :- {rule_body}."

                        # Add confidence based on prediction strength
                        confidence = float(predictions[0][0])

                        new_rules.append((rule, confidence))
                        self.logger.info(f"Extracted rule from neural model: {rule} (confidence: {confidence:.2f})")

            return new_rules

        except Exception as e:
            self.logger.error(f"Error extracting rules from neural model: {e}")
            return []

    def analyze_neural_patterns(
            self,
            model: tf.keras.Model,
            anomalous_sequences: np.ndarray,
            normal_sequences: np.ndarray,
            feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Analyze patterns in neural network predictions to generate rules.

        Args:
            model: Trained CNN-LSTM model
            anomalous_sequences: Sequences labeled as anomalous
            normal_sequences: Sequences labeled as normal
            feature_names: Names of input features

        Returns:
            List of (rule, confidence) tuples
        """
        try:
            rules = []

            # Build model first with sample data
            if not model.built:
                dummy_input = tf.convert_to_tensor(anomalous_sequences[0:1])
                _ = model(dummy_input)  # This builds the model

            # Now create feature extractor
            lstm_layer = None
            for layer in model.layers:
                if 'lstm' in layer.name:
                    lstm_layer = layer
                    break

            if lstm_layer is None:
                raise ValueError("No LSTM layer found in model")

            # Get model's internal representations
            feature_extractor = tf.keras.Model(
                inputs=model.input,
                outputs=model.get_layer('lstm_1').output  # Assuming this layer exists
            )

            # Analyze anomalous patterns
            anomalous_features = feature_extractor.predict(anomalous_sequences)
            normal_features = feature_extractor.predict(normal_sequences)

            # Calculate mean patterns
            anomaly_pattern = np.mean(anomalous_features, axis=0)
            normal_pattern = np.mean(normal_features, axis=0)

            # Find significant differences
            pattern_diff = np.abs(anomaly_pattern - normal_pattern)
            significant_dims = np.where(pattern_diff > np.mean(pattern_diff) + np.std(pattern_diff))[0]

            # Generate rules from significant patterns
            for sequence in anomalous_sequences:
                # Get last timestep values
                current_values = sequence[-1]

                conditions = []
                for dim in significant_dims:
                    feature_name = feature_names[dim % len(feature_names)]
                    value = float(current_values[dim % len(feature_names)])

                    if feature_name == 'temperature':
                        conditions.append(f"temperature > {value:.1f}")
                    elif feature_name == 'vibration':
                        conditions.append(f"vibration > {value:.1f}")
                    elif feature_name == 'pressure':
                        conditions.append(f"pressure < {value:.1f}")
                    elif feature_name == 'efficiency_index':
                        conditions.append(f"efficiency_index < {value:.2f}")

                if conditions:
                    rule_head = f"neural_pattern_{len(self.learned_rules) + 1}"
                    rule_body = ", ".join(conditions)
                    rule = f"{rule_head} :- {rule_body}."

                    # Calculate confidence based on model prediction
                    confidence = float(model.predict(sequence[np.newaxis, ...])[0])

                    rules.append((rule, confidence))

            return rules

        except Exception as e:
            self.logger.error(f"Error analyzing neural patterns: {e}")
            return []

    def update_rules(self, new_rules: List[Tuple[str, float]], min_confidence: float = 0.7):
        """Update rules with proper formatting."""
        try:
            # Read existing content
            with open(self.rules_path, 'r') as f:
                existing_content = f.read()

            # Add new rules with proper formatting
            with open(self.rules_path, 'a') as f:
                if not existing_content.endswith('\n\n'):
                    f.write('\n\n')
                f.write("% New Neural-Extracted Rules\n")
                for rule, confidence in new_rules:
                    if confidence >= min_confidence and rule not in self.learned_rules:
                        formatted_rule = (
                            f"{rule}  "
                            f"% Confidence: {confidence:.2f}, "
                            f"Extracted: {str(np.datetime64('now'))}\n"
                        )
                        f.write(formatted_rule)
                        self.learned_rules.append(rule)
                        self.rule_confidence[rule] = confidence

                f.write("\n")  # Add spacing after new rules

            # Reload rules
            self.prolog.consult(self.rules_path)
            self.logger.info(
                f"Added {len(new_rules)} new rules with "
                f"avg confidence: {np.mean([c for r, c in new_rules]):.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error updating rules: {e}")
            raise

    def reason(self, sensor_data: dict) -> list:
        """Apply both base and learned rules from neural model."""
        insights = []
        try:
            # Extract sensor values
            temperature = float(sensor_data.get('temperature', 0))
            vibration = float(sensor_data.get('vibration', 0))
            pressure = float(sensor_data.get('pressure', 0))
            efficiency_index = float(sensor_data.get('efficiency_index', 0))

            # Apply base rules
            try:
                for _ in self.prolog.query(f"degraded_state({temperature}, {vibration})."):
                    insights.append("Degraded State (Base Rule)")
                    break
            except Exception as e:
                self.logger.warning(f"Error in base rule query: {e}")

            # Apply neural-extracted rules
            for rule in self.learned_rules:
                try:
                    rule_name = rule.split(":-")[0].strip()
                    if list(self.prolog.query(rule_name)):
                        confidence = self.rule_confidence.get(rule, 0.0)
                        insights.append(f"Neural Rule {rule_name} (Confidence: {confidence:.2f})")
                except Exception as e:
                    self.logger.warning(f"Error applying neural rule {rule}: {e}")

            return insights

        except Exception as e:
            self.logger.error(f"Error during reasoning: {e}")
            raise

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rule base."""
        return {
            'total_rules': len(self.learned_rules) + 4,  # 4 base rules
            'neural_derived_rules': len(self.learned_rules),
            'high_confidence_rules': sum(1 for conf in self.rule_confidence.values() if conf >= 0.7),
            'average_confidence': np.mean(list(self.rule_confidence.values())) if self.rule_confidence else 0.0,
            'rules_confidence': self.rule_confidence.copy()
        }