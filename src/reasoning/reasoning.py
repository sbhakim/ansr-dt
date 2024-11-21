# src/reasoning/reasoning.py

import logging
from pyswip import Prolog
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple, Optional
from src.utils.model_utils import load_model_with_initialization
from .rule_learning import RuleLearner
from .state_tracker import StateTracker



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
        self.rules_dir = os.path.dirname(rules_path)  # Store rules directory
        self.learned_rules = []
        self.rule_confidence = {}
        self.model = model  # Use the provided model if available
        self.input_shape = input_shape
        self.rule_activations = []
        self.rule_confidence = {}
        self.state_history = []

        self.rule_learner = RuleLearner()
        self.state_tracker = StateTracker()

        # Validate rules file existence
        if not os.path.exists(rules_path):
            self.logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")

        # Load Prolog files in correct order
        self._load_prolog_files([
            'load_config.pl',    # Load configuration first
            'prob_rules.pl',     # Load probabilistic rules
            'integrate_prob_log.pl',  # Load integration
            'rules.pl'            # Load main rules last
        ])

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

    def _load_prolog_files(self, files: List[str]):
        """Load Prolog files in specified order."""
        for file in files:
            path = os.path.join(self.rules_dir, file)
            if os.path.exists(path):
                try:
                    self.prolog.consult(path)
                    self.logger.info(f"Loaded Prolog file: {file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {file}: {e}")

    def extract_rules_from_neural_model(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7,
            model: Optional[tf.keras.Model] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract probabilistic rules from neural model using ProbLog compliant format.
        """
        try:
            model_to_use = model if model is not None else self.model
            if model_to_use is None:
                self.logger.error("No model available for rule extraction")
                return []

            # Format input data
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)

            predictions = model_to_use.predict(input_data, verbose=0).flatten()
            anomaly_indices = np.where(predictions > threshold)[0]

            new_rules = []
            for idx in anomaly_indices:
                sequence = input_data[idx]
                current_values = sequence[-1]

                # Generate ProbLog compliant rules
                confidence = float(predictions[idx])
                conditions = []

                # Add temperature conditions
                if current_values[0] > 80:
                    conditions.append("high_temp")

                # Add vibration conditions
                if current_values[1] > 55:
                    conditions.append("high_vib")

                # Add pressure conditions
                if current_values[2] < 20:
                    conditions.append("low_press")

                # Create ProbLog rule
                if conditions:
                    rule_name = f"neural_rule_{len(self.learned_rules) + 1}"
                    rule_body = ", ".join(conditions)
                    rule = f"{confidence}::{rule_name} :- {rule_body}."

                    new_rules.append((rule, confidence))

            return new_rules

        except Exception as e:
            self.logger.error(f"Error extracting rules: {e}")
            return []

    def update_problog_rules(self, new_rules: List[Tuple[str, float]], min_confidence: float = 0.7):
        """
        Update ProbLog rules file with new learned rules.
        """
        try:
            with open(self.rules_path, 'r') as f:
                existing_content = f.read()

            with open(self.rules_path, 'a') as f:
                if not existing_content.endswith('\n\n'):
                    f.write('\n\n')

                f.write("% Neural-derived probabilistic rules\n")
                for rule, confidence in new_rules:
                    if confidence >= min_confidence:
                        f.write(f"% Confidence: {confidence:.2f}\n")
                        f.write(f"{rule}\n")

                f.write("\n")

            # Reload ProbLog rules
            self.prolog.consult(self.rules_path)

        except Exception as e:
            self.logger.error(f"Error updating ProbLog rules: {e}")

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

    def get_rule_activations(self) -> List[Dict]:
        """
        Get list of rule activations with confidence scores.

        Returns:
            List[Dict]: List of rule activations with metadata
        """
        try:
            activations = []
            for rule, confidence in self.rule_confidence.items():
                activation = {
                    'rule': rule,
                    'confidence': confidence,
                    'timestep': len(self.rule_activations),
                    'type': 'learned' if rule.startswith('neural_rule') else 'base'
                }
                activations.append(activation)
            return activations
        except Exception as e:
            self.logger.error(f"Error getting rule activations: {e}")
            return []

    def reason(self, sensor_dict: Dict[str, float]) -> List[str]:
        """
        Apply symbolic reasoning rules to sensor data and generate insights.

        Args:
            sensor_dict (Dict[str, float]): Dictionary containing sensor readings with keys:
                - temperature
                - vibration
                - pressure
                - operational_hours
                - efficiency_index
                - system_state
                - performance_score

        Returns:
            List[str]: List of insights generated from rule application
        """
        try:
            insights = []

            # Input validation
            required_keys = [
                'temperature', 'vibration', 'pressure',
                'operational_hours', 'efficiency_index',
                'system_state', 'performance_score'
            ]

            # Verify all required keys exist
            missing_keys = [key for key in required_keys if key not in sensor_dict]
            if missing_keys:
                self.logger.error(f"Missing required sensor values: {missing_keys}")
                return insights

            # Extract and validate sensor values
            try:
                temperature = float(sensor_dict['temperature'])
                vibration = float(sensor_dict['vibration'])
                pressure = float(sensor_dict['pressure'])
                efficiency_index = float(sensor_dict['efficiency_index'])
                operational_hours = int(float(sensor_dict['operational_hours']))
                system_state = int(sensor_dict['system_state'])
                performance_score = float(sensor_dict['performance_score'])
            except (ValueError, TypeError) as e:
                self.logger.error(f"Invalid sensor value format: {e}")
                return insights

            # Calculate thermal gradients if history exists
            thermal_gradient = 0.0
            if len(self.state_history) > 0:
                prev_temp = float(self.state_history[-1]['sensor_readings']['temperature'])
                thermal_gradient = abs(temperature - prev_temp)

            # Apply base rules with enhanced queries
            base_queries = {
                "Degraded State (Base Rule)": f"degraded_state({temperature}, {vibration}).",
                "System Stress (Base Rule)": f"system_stress({pressure}).",
                "Critical State (Base Rule)": f"critical_state({efficiency_index}).",
                "Maintenance Needed (Base Rule)": f"maintenance_needed({operational_hours}).",
                "Thermal Stress (Base Rule)": f"thermal_stress({temperature}, {thermal_gradient}).",
                "Sensor Correlation (Base Rule)": f"sensor_correlation({temperature}, {vibration}, {pressure})."
            }

            # Execute base rule queries
            for insight, query in base_queries.items():
                try:
                    if list(self.prolog.query(query)):
                        insights.append(insight)
                        self.logger.info(f"Prolog Query Success: {insight}")
                except Exception as e:
                    self.logger.warning(f"Query failed for {insight}: {e}")
                    continue

            # Apply learned rules if available
            if hasattr(self, 'learned_rules'):
                for rule in self.learned_rules:
                    try:
                        rule_name = rule.split(":-")[0].strip()
                        # Create context dict for rule evaluation
                        context = {
                            'temperature': temperature,
                            'vibration': vibration,
                            'pressure': pressure,
                            'efficiency_index': efficiency_index,
                            'operational_hours': operational_hours,
                            'thermal_gradient': thermal_gradient,
                            'system_state': system_state
                        }

                        # Evaluate rule with context
                        if list(self.prolog.query(f"{rule_name}.", context=context)):
                            confidence = self.rule_confidence.get(rule, 0.0)
                            insight = f"Learned Rule {rule_name} (Confidence: {confidence:.2f})"
                            insights.append(insight)
                            self.logger.info(f"Learned Rule Activated: {insight}")
                    except Exception as e:
                        self.logger.warning(f"Failed to apply learned rule {rule}: {e}")
                        continue

            # Record rule activation with enhanced metadata
            activation_record = {
                'timestep': len(self.rule_activations) if hasattr(self, 'rule_activations') else 0,
                'insights': insights,
                'sensor_values': sensor_dict,
                'system_state': system_state,
                'performance': performance_score,
                'thermal_gradient': thermal_gradient,
                'active_rules': len(insights),
                'timestamp': str(np.datetime64('now'))
            }

            # Update state history with retention
            if hasattr(self, 'rule_activations'):
                self.rule_activations.append(activation_record)
                if len(self.rule_activations) > 1000:
                    self.rule_activations = self.rule_activations[-1000:]

            # Update state transition matrix
            if hasattr(self, 'state_tracker'):
                self.state_tracker.update({
                    'system_state': system_state,
                    'sensor_readings': sensor_dict,
                    'insights': insights,
                    'thermal_gradient': thermal_gradient
                })

            return insights

        except Exception as e:
            self.logger.error(f"Error during reasoning process: {str(e)}")
            return []

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