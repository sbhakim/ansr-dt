# src/reasoning/reasoning.py

import logging
from pyswip import Prolog
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple, Optional
from src.utils.model_utils import load_model_with_initialization


class SymbolicReasoner:
    def __init__(
            self,
            rules_path: str,
            input_shape: tuple,
            model: Optional[tf.keras.Model] = None,
            model_path: Optional[str] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Symbolic Reasoner.

        Args:
            rules_path (str): Path to the Prolog rules file.
            input_shape (tuple): Shape of the input data (window_size, n_features).
            model (tf.keras.Model, optional): Trained Keras model. If provided, it will be used directly.
            model_path (str, optional): Path to the trained Keras model file. Used only if `model` is not provided.
            logger (logging.Logger, optional): Logger instance for logging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.prolog = Prolog()
        self.rules_path = rules_path
        self.learned_rules = []
        self.rule_confidence = {}
        self.model = model  # Use the provided model if available
        self.input_shape = input_shape

        # Validate rules file existence
        if not os.path.exists(rules_path):
            self.logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")

        # Load Prolog rules
        self.prolog.consult(rules_path)
        self.logger.info(f"Prolog rules loaded from {rules_path}")

        # Load and initialize the model if not provided
        if self.model is None and model_path is not None:
            if os.path.exists(model_path):
                self.model = load_model_with_initialization(
                    path=model_path,
                    logger=self.logger,
                    input_shape=input_shape
                )
                self.logger.info("Model loaded and initialized successfully.")
            else:
                self.logger.warning(f"Model path {model_path} does not exist. Model not loaded.")
        elif self.model is not None:
            self.logger.info("Model provided directly to SymbolicReasoner.")
        else:
            self.logger.warning("No model provided to SymbolicReasoner.")

    def extract_rules_from_neural_model(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7,
            model: Optional[tf.keras.Model] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract rules from neural model using gradient analysis and predefined thresholds.

        Parameters:
            input_data (np.ndarray): Input data used for rule extraction.
            feature_names (List[str]): Names of the input features.
            threshold (float): Threshold to determine feature importance.
            model (Optional[tf.keras.Model]): Model to use for rule extraction. If None, uses self.model.

        Returns:
            List[Tuple[str, float]]: List of extracted rules with their confidence scores.
        """
        try:
            # Use provided model if available, otherwise use self.model
            model_to_use = model if model is not None else self.model

            if model_to_use is None:
                self.logger.error("No model available for rule extraction.")
                return []

            # Format input data
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)
            elif len(input_data.shape) != 3:
                self.logger.error(f"Invalid input shape: {input_data.shape}. Expected (batch, timesteps, features)")
                return []

            self.logger.debug(f"Input data shape: {input_data.shape}")
            input_data = input_data.astype(np.float32)

            # Get predictions
            predictions = model_to_use.predict(input_data, verbose=0).flatten()
            self.logger.debug(f"Model predictions shape: {predictions.shape}")

            # Find anomalous sequences
            anomaly_indices = np.where(predictions > threshold)[0]
            self.logger.info(f"Found {len(anomaly_indices)} anomalous sequences based on the threshold of {threshold}.")

            if len(anomaly_indices) == 0:
                self.logger.info("No anomalous sequences found. No rules extracted.")
                return []

            new_rules = []

            # Process each anomalous sequence
            for idx in anomaly_indices:
                sequence = input_data[idx]
                last_timestep = sequence[-1]

                # Extract relevant conditions
                important_features = []

                # Add conditions based on feature values and predefined thresholds
                for feat_idx, feat_name in enumerate(feature_names):
                    feat_value = float(last_timestep[feat_idx])

                    if feat_name == 'temperature' and feat_value > 80:
                        important_features.append((feat_name, feat_value, 'high_temp'))
                    elif feat_name == 'vibration' and feat_value > 55:
                        important_features.append((feat_name, feat_value, 'high_vib'))
                    elif feat_name == 'pressure' and feat_value < 20:
                        important_features.append((feat_name, feat_value, 'low_pressure'))
                    elif feat_name == 'efficiency_index' and feat_value < 0.6:
                        important_features.append((feat_name, feat_value, 'low_efficiency'))

                # Generate rule if important features found
                if important_features:
                    # Format conditions based on feature type
                    conditions = []
                    for feat_name, feat_value, condition_type in important_features:
                        if feat_name == 'efficiency_index':
                            conditions.append(f"{feat_name}({feat_value:.2f})")
                        else:
                            conditions.append(f"{feat_name}({int(feat_value)})")

                    # Create rule
                    rule_name = f"neural_rule_{len(self.learned_rules) + 1}"
                    rule_body = ", ".join(conditions) + "."
                    rule = f"{rule_name} :- {rule_body}"

                    # Calculate rule confidence from model prediction
                    confidence = float(predictions[idx])

                    # Add rule and confidence
                    new_rules.append((rule, confidence))
                    self.rule_confidence[rule] = confidence
                    self.logger.info(f"Extracted rule: {rule} with confidence: {confidence:.2f}")

            self.logger.info(f"Total new rules extracted: {len(new_rules)}")
            return new_rules

        except Exception as e:
            self.logger.error(f"Error extracting rules from neural model: {e}")
            raise

    def analyze_neural_patterns(
            self,
            anomalous_sequences: np.ndarray,
            normal_sequences: np.ndarray,
            feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Analyze patterns in neural model activations to extract additional rules.

        Parameters:
            anomalous_sequences (np.ndarray): Sequences identified as anomalous.
            normal_sequences (np.ndarray): Sequences identified as normal.
            feature_names (List[str]): Names of the input features.

        Returns:
            List[Tuple[str, float]]: List of generated pattern rules with their confidence scores.
        """
        try:
            rules = []
            if anomalous_sequences.shape[0] == 0 or normal_sequences.shape[0] == 0:
                self.logger.warning("Insufficient sequences for pattern analysis.")
                return rules

            # Ensure model is built by making a prediction
            try:
                if not self.model.built:
                    # Build model with first sequence
                    dummy_input = np.expand_dims(anomalous_sequences[0], axis=0)
                    _ = self.model.predict(dummy_input, verbose=0)
                    self.logger.debug("Model built with dummy prediction")

                # Build again with a single prediction to ensure internal states
                dummy_pred = self.model.predict(dummy_input, verbose=0)
                self.logger.debug(f"Model built successfully. Dummy prediction shape: {dummy_pred.shape}")
            except Exception as build_error:
                self.logger.error(f"Failed to build model: {build_error}")
                return []

            # Find LSTM layer
            lstm_layer = None
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, tf.keras.layers.LSTM):
                    lstm_layer = layer
                    self.logger.debug(f"Found LSTM layer at index {i}")
                    break

            if lstm_layer is None:
                self.logger.error("No LSTM layer found in the model.")
                return []

            # Create feature extractor with built model
            try:
                # Get input layer explicitly
                input_layer = self.model.get_layer(index=0)
                self.logger.debug(f"Using input layer: {input_layer.name}")

                # Create feature extractor with explicit inputs
                feature_extractor = tf.keras.Model(
                    inputs=input_layer.input,
                    outputs=lstm_layer.output,
                    name='feature_extractor'
                )

                # Build feature extractor
                feature_extractor.build(input_shape=(None,) + self.input_shape)
                self.logger.info("Feature extractor created and built successfully")
            except Exception as fe:
                self.logger.error(f"Failed to create feature extractor: {fe}")
                return []

            # Extract features
            try:
                # Ensure proper input shapes
                anomalous_sequences = anomalous_sequences.astype(np.float32)
                normal_sequences = normal_sequences.astype(np.float32)

                # Extract features
                anomalous_features = feature_extractor.predict(anomalous_sequences, verbose=0)
                normal_features = feature_extractor.predict(normal_sequences, verbose=0)

                self.logger.debug(f"Anomalous features shape: {anomalous_features.shape}")
                self.logger.debug(f"Normal features shape: {normal_features.shape}")
            except Exception as pe:
                self.logger.error(f"Failed to extract features: {pe}")
                return []

            # Calculate mean patterns
            anomaly_pattern = np.mean(anomalous_features, axis=0)
            normal_pattern = np.mean(normal_features, axis=0)

            # Determine significant differences
            pattern_diff = np.abs(anomaly_pattern - normal_pattern)
            threshold_diff = np.mean(pattern_diff) + np.std(pattern_diff)
            significant_dims = np.where(pattern_diff > threshold_diff)[0]
            self.logger.info(f"Found {len(significant_dims)} significant dimensions")

            # Generate rules
            for sequence in anomalous_sequences:
                # Get last timestep values
                current_values = sequence[-1]
                important_features = []

                # Check significant features
                for dim in significant_dims:
                    feature_idx = dim % len(feature_names)
                    feature_name = feature_names[feature_idx]
                    feat_value = float(current_values[feature_idx])

                    # Domain-specific thresholds
                    if feature_name == 'temperature' and feat_value > 80:
                        important_features.append((feature_name, feat_value, 'high'))
                    elif feature_name == 'vibration' and feat_value > 55:
                        important_features.append((feature_name, feat_value, 'high'))
                    elif feature_name == 'pressure' and feat_value < 20:
                        important_features.append((feature_name, feat_value, 'low'))
                    elif feature_name == 'efficiency_index' and feat_value < 0.6:
                        important_features.append((feature_name, feat_value, 'low'))

                # Create rules for important features
                if important_features:
                    # Format conditions
                    conditions = []
                    for feat_name, feat_value, condition_type in important_features:
                        if feat_name == 'efficiency_index':
                            conditions.append(f"{feat_name}({feat_value:.2f})")
                        else:
                            conditions.append(f"{feat_name}({int(feat_value)})")

                    # Create rule
                    rule_name = f"pattern_rule_{len(self.learned_rules) + 1}"
                    rule_body = ", ".join(conditions) + "."
                    rule = f"{rule_name} :- {rule_body}"

                    # Calculate confidence
                    try:
                        sequence_reshaped = np.expand_dims(sequence, axis=0)
                        confidence = float(self.model.predict(sequence_reshaped, verbose=0)[0])

                        rules.append((rule, confidence))
                        self.rule_confidence[rule] = confidence
                        self.logger.info(f"Generated rule: {rule} with confidence: {confidence:.2f}")
                    except Exception as pred_error:
                        self.logger.warning(f"Failed to calculate confidence for rule {rule}: {pred_error}")
                        continue

            self.logger.info(f"Generated {len(rules)} pattern-based rules")
            return rules

        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            raise

    def update_rules(self, new_rules: List[Tuple[str, float]], min_confidence: float = 0.7):
        """
        Update the Prolog rules file with newly extracted rules that meet the confidence threshold.

        Parameters:
            new_rules (List[Tuple[str, float]]): List of new rules with their confidence scores.
            min_confidence (float): Minimum confidence required to add a rule.
        """
        try:
            # Read existing Prolog rules
            with open(self.rules_path, 'r') as f:
                existing_content = f.read()

            # Prepare to append new rules
            with open(self.rules_path, 'a') as f:
                if not existing_content.endswith('\n\n'):
                    f.write('\n\n')
                f.write("% New Neural-Extracted Rules\n")
                for rule, confidence in new_rules:
                    if confidence >= min_confidence and rule not in self.learned_rules:
                        formatted_rule = (
                            f"{rule}  "
                            f"% Confidence: {confidence:.2f}, "
                            f"Extracted: {np.datetime64('now')}\n"
                        )
                        f.write(formatted_rule)
                        self.learned_rules.append(rule)
                        self.rule_confidence[rule] = confidence

                f.write("\n")  # Add spacing after new rules

            # Reload the updated Prolog rules
            self.prolog.consult(self.rules_path)
            avg_confidence = np.mean([c for _, c in new_rules]) if new_rules else 'N/A'
            self.logger.info(
                f"Added {len(new_rules)} new rules with avg confidence: {avg_confidence}"
            )

        except Exception as e:
            self.logger.error(f"Error updating Prolog rules: {e}")
            raise

    def reason(self, sensor_data: Dict[str, Any]) -> List[str]:
        """
        Apply both base and learned rules to the provided sensor data to generate insights.

        Parameters:
            sensor_data (Dict[str, Any]): Dictionary containing sensor readings.

        Returns:
            List[str]: List of insights generated from applying the rules.
        """
        insights = []
        try:
            # Extract and cast sensor values
            temperature = float(sensor_data.get('temperature', 0))
            vibration = float(sensor_data.get('vibration', 0))
            pressure = float(sensor_data.get('pressure', 0))
            efficiency_index = float(sensor_data.get('efficiency_index', 0))
            operational_hours = int(sensor_data.get('operational_hours', 0))

            # Log extracted sensor data
            self.logger.debug(f"Sensor Data - Temperature: {temperature}, Vibration: {vibration}, "
                              f"Pressure: {pressure}, Efficiency Index: {efficiency_index}, "
                              f"Operational Hours: {operational_hours}")

            # Apply base rules
            base_queries = {
                "Degraded State (Base Rule)": f"degraded_state({temperature}, {vibration}).",
                "System Stress (Base Rule)": f"system_stress({pressure}).",
                "Critical State (Base Rule)": f"critical_state({efficiency_index}).",
                "Maintenance Needed (Base Rule)": f"maintenance_needed({operational_hours})."
            }

            for insight, query in base_queries.items():
                try:
                    self.logger.debug(f"Executing Prolog query: {query}")
                    if list(self.prolog.query(query)):
                        insights.append(insight)
                        self.logger.info(f"Prolog Query Success: {insight}")
                    else:
                        self.logger.debug(f"Prolog Query Failed: {insight}")
                except Exception as e:
                    self.logger.warning(f"Error executing Prolog query for {insight}: {e}")

            # Apply neural-extracted rules
            for rule in self.learned_rules:
                try:
                    rule_name = rule.split(":-")[0].strip()
                    rule_query = f"{rule_name}."
                    self.logger.debug(f"Executing Prolog query: {rule_query}")
                    if list(self.prolog.query(rule_query)):
                        confidence = self.rule_confidence.get(rule, 0.0)
                        insight = f"Neural Rule {rule_name} (Confidence: {confidence:.2f})"
                        insights.append(insight)
                        self.logger.info(f"Prolog Query Success: {insight}")
                    else:
                        self.logger.debug(f"Prolog Query Failed: Neural Rule {rule_name}")
                except Exception as e:
                    self.logger.warning(f"Error executing Prolog query for neural rule '{rule}': {e}")

            if not insights:
                self.logger.info("No insights generated from reasoning.")

            self.logger.debug(f"Generated Insights: {insights}")
            return insights

        except Exception as e:
            self.logger.error(f"Error during reasoning process: {e}")
            raise

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Retrieve statistics about the current rule base.

        Returns:
            Dict[str, Any]: Dictionary containing rule statistics.
        """
        stats = {
            'total_rules': len(self.learned_rules) + 4,  # 4 base rules
            'neural_derived_rules': len(self.learned_rules),
            'high_confidence_rules': sum(1 for conf in self.rule_confidence.values() if conf >= 0.7),
            'average_confidence': np.mean(list(self.rule_confidence.values())) if self.rule_confidence else 0.0,
            'rules_confidence': self.rule_confidence.copy()
        }
        self.logger.debug(f"Rule Statistics: {stats}")
        return stats
