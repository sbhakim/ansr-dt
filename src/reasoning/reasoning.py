# src/reasoning/reasoning.py

import logging
# --- CHANGE 1: Modified Import ---
from pyswip import Prolog # Removed PrologError import
# --- END CHANGE 1 ---
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple, Optional
from src.utils.model_utils import load_model_with_initialization
from .rule_learning import RuleLearner
from .state_tracker import StateTracker
from datetime import datetime

# Define a marker for separating base rules from dynamic ones
DYNAMIC_RULES_MARKER = "%% START DYNAMIC RULES %%"

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
        self.rules_dir = os.path.dirname(rules_path)
        self.learned_rules = {}
        self.model = model
        self.input_shape = input_shape
        self.rule_activations = []
        self.state_history = []

        self.rule_learner = RuleLearner()
        self.state_tracker = StateTracker()

        if not os.path.exists(rules_path):
            self.logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")

        self._load_prolog_files([
            'load_config.pl',
            'integrate_prob_log.pl',
            'rules.pl'
        ])

        self._load_dynamic_rules_from_file()

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
            if not self.model.built and self.input_shape:
                 try:
                     dummy_input = np.zeros((1,) + self.input_shape, dtype=np.float32)
                     self.model.predict(dummy_input, verbose=0)
                     self.logger.info("Externally provided model built successfully.")
                 except Exception as e:
                     self.logger.error(f"Failed to build externally provided model: {e}")
        else:
            self.logger.warning("No model provided to SymbolicReasoner.")

    def _load_prolog_files(self, files: List[str]):
        """Load Prolog files in specified order."""
        for file in files:
            path = os.path.join(self.rules_dir, file)
            if os.path.exists(path):
                try:
                    list(self.prolog.query(f"consult('{path}')"))
                    self.logger.info(f"Loaded Prolog file: {file}")
                # --- CHANGE 2: Catch general Exception ---
                except Exception as e:
                    # Check if it looks like a Prolog error from pyswip
                    if "PrologError" in str(type(e)) or "pyswip" in str(e).lower():
                         self.logger.error(f"PrologError consulting file {file}: {e}")
                    else:
                         self.logger.warning(f"Failed to load {file}: {type(e).__name__} - {e}")
                # --- END CHANGE 2 ---
            else:
                self.logger.warning(f"Prolog file not found, skipping: {path}")

    def _load_dynamic_rules_from_file(self):
        """Loads existing dynamic rules from the rules file at initialization."""
        try:
            with open(self.rules_path, 'r') as f:
                content = f.readlines()

            in_dynamic_section = False
            self.learned_rules = {} # Ensure it's clear before loading

            for line in content:
                line = line.strip()
                if line == DYNAMIC_RULES_MARKER:
                    in_dynamic_section = True # Start dynamic section
                    continue
                if line == "%% END DYNAMIC RULES %%": # Optional end marker
                    in_dynamic_section = False
                    continue

                if in_dynamic_section and line and not line.startswith('%'):
                    confidence = 0.8 # Default confidence
                    timestamp_str = datetime.now().isoformat() # Default timestamp
                    activations = 0 # Default activations

                    if '%' in line:
                        rule_part, comment_part = line.split('%', 1)
                        rule_part = rule_part.strip()
                        comment_part = comment_part.strip()
                        try:
                            parts = comment_part.split(',')
                            for part in parts:
                                part = part.strip()
                                if part.lower().startswith('confidence:'):
                                    confidence = float(part.split(':')[1].strip())
                                elif part.lower().startswith('extracted:'):
                                    timestamp_str = part.split(':', 1)[1].strip()
                                elif part.lower().startswith('activations:'):
                                     activations = int(part.split(':')[1].strip()) # Load activations
                        except Exception:
                            self.logger.warning(f"Could not parse metadata from comment: {comment_part}")
                    else:
                        rule_part = line

                    if rule_part.endswith('.'):
                         rule_key = rule_part
                         if rule_key not in self.learned_rules:
                             try:
                                 timestamp_dt = datetime.fromisoformat(timestamp_str)
                             except ValueError:
                                 self.logger.warning(f"Invalid timestamp format '{timestamp_str}', using current time.")
                                 timestamp_dt = datetime.now()

                             self.learned_rules[rule_key] = {
                                 'confidence': confidence,
                                 'timestamp': timestamp_dt,
                                 'activations': activations
                             }
                         else:
                             self.logger.debug(f"Rule '{rule_key}' already loaded, skipping duplicate.")

            self.logger.info(f"Loaded {len(self.learned_rules)} dynamic rules from {self.rules_path}")

        except FileNotFoundError:
            self.logger.warning(f"Rules file not found at {self.rules_path}, no dynamic rules loaded.")
        except Exception as e:
            self.logger.error(f"Error loading dynamic rules from file: {e}")


    def _rewrite_rules_file(self):
        """Rewrites the rules file with base rules and current dynamic rules."""
        try:
            base_rules_content = []
            # Use a flag to ensure we only capture lines before the dynamic marker
            before_marker = True
            with open(self.rules_path, 'r') as f:
                for line in f:
                    if line.strip() == DYNAMIC_RULES_MARKER:
                        before_marker = False
                        # Add the marker itself to the base content
                        base_rules_content.append(line)
                        break # Stop reading after marker
                    if before_marker:
                         base_rules_content.append(line)

            # Ensure the marker is present if file was empty or marker missing
            if DYNAMIC_RULES_MARKER not in "".join(base_rules_content):
                 base_rules_content.append(f"\n{DYNAMIC_RULES_MARKER}\n")


            with open(self.rules_path, 'w') as f:
                f.writelines(base_rules_content)
                f.write("%% Automatically managed section - Do not edit manually below this line %%\n")

                sorted_rule_items = sorted(self.learned_rules.items())

                for rule, metadata in sorted_rule_items:
                     formatted_rule = (
                         f"{rule}  "
                         f"% Confidence: {metadata['confidence']:.3f}, "
                         f"Extracted: {metadata['timestamp'].isoformat()}, "
                         f"Activations: {metadata['activations']}\n"
                     )
                     f.write(formatted_rule)

                f.write(f"\n%% END DYNAMIC RULES %%\n")

            self.logger.info(f"Rewrote rules file {self.rules_path} with {len(self.learned_rules)} dynamic rules.")

            # Re-consult the entire file
            try:
                 list(self.prolog.query(f"consult('{self.rules_path}')"))
                 self.logger.info(f"Re-consulted Prolog file: {self.rules_path}")
            # --- CHANGE 2: Catch general Exception ---
            except Exception as e:
                # Check if it looks like a Prolog error from pyswip
                if "PrologError" in str(type(e)) or "pyswip" in str(e).lower():
                     self.logger.error(f"PrologError re-consulting file after rewrite: {e}")
                else:
                     self.logger.error(f"Failed to re-consult {self.rules_path} after rewrite: {type(e).__name__} - {e}")
            # --- END CHANGE 2 ---

        except Exception as e:
            self.logger.error(f"Error rewriting Prolog rules file: {e}")


    def extract_rules_from_neural_model(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7,
            model: Optional[tf.keras.Model] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract rules from neural model using gradient analysis and predefined thresholds.
        This method focuses on generating potential rule candidates based on instance predictions.

        Parameters:
            input_data (np.ndarray): Input data used for rule extraction (batch, timesteps, features).
            feature_names (List[str]): Names of the input features.
            threshold (float): Anomaly score threshold to consider sequences for rule extraction.
            model (Optional[tf.keras.Model]): Model to use for rule extraction. If None, uses self.model.

        Returns:
            List[Tuple[str, float]]: List of extracted potential rule candidates (rule_string, confidence_score).
                                      Confidence here is based on the model's prediction for that instance.
        """
        try:
            model_to_use = model if model is not None else self.model
            if model_to_use is None:
                self.logger.error("No model available for rule extraction.")
                return []

            if not model_to_use.built:
                 self.logger.warning("Model is not built, attempting to build for rule extraction.")
                 if self.input_shape:
                     try:
                         dummy_input = np.zeros((1,) + self.input_shape, dtype=np.float32)
                         model_to_use.predict(dummy_input, verbose=0)
                     except Exception as e:
                         self.logger.error(f"Failed to build model during rule extraction: {e}")
                         return []
                 else:
                    self.logger.error("Cannot build model: input_shape not available.")
                    return []


            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)
            elif len(input_data.shape) != 3:
                self.logger.error(f"Invalid input shape: {input_data.shape}. Expected (batch, timesteps, features)")
                return []

            self.logger.debug(f"Rule extraction input data shape: {input_data.shape}")
            input_data = input_data.astype(np.float32)

            predictions = model_to_use.predict(input_data, verbose=0).flatten()
            self.logger.debug(f"Model predictions shape for rule extraction: {predictions.shape}")

            anomaly_indices = np.where(predictions > threshold)[0]
            self.logger.info(f"Found {len(anomaly_indices)} potential rule sequences based on threshold {threshold}.")

            if len(anomaly_indices) == 0:
                return []

            potential_new_rules = []

            for idx in anomaly_indices:
                sequence = input_data[idx]
                if sequence.shape[0] < 2:
                    continue

                current_values = sequence[-1, :]
                previous_values = sequence[-2, :]

                feature_conditions = []

                for feat_idx, feat_name in enumerate(feature_names):
                    feat_value = float(current_values[feat_idx])
                    prev_value = float(previous_values[feat_idx])

                    if feat_name == 'temperature':
                        if feat_value > 80: feature_conditions.append(f"temperature({int(feat_value)})")
                        if abs(feat_value - prev_value) > 10: feature_conditions.append(f"temperature_change({int(abs(feat_value - prev_value))})")
                    elif feat_name == 'vibration':
                         if feat_value > 55: feature_conditions.append(f"vibration({int(feat_value)})")
                         if abs(feat_value - prev_value) > 5: feature_conditions.append(f"vibration_change({int(abs(feat_value - prev_value))})")
                    elif feat_name == 'pressure':
                         if feat_value < 20: feature_conditions.append(f"pressure({int(feat_value)})")
                    elif feat_name == 'efficiency_index':
                         if feat_value < 0.6: feature_conditions.append(f"efficiency_index({feat_value:.2f})")
                    elif feat_name == 'operational_hours':
                         if feat_value % 1000 < 10: feature_conditions.append(f"maintenance_needed({int(feat_value)})")
                    elif feat_name == 'system_state':
                         if feat_value != prev_value: feature_conditions.append(f"state_transition({int(prev_value)}, {int(feat_value)})")

                temp_val = float(current_values[feature_names.index('temperature')])
                vib_val = float(current_values[feature_names.index('vibration')])
                if temp_val > 75 and vib_val > 50:
                    feature_conditions.append(f"combined_high_temp_vib")

                press_val = float(current_values[feature_names.index('pressure')])
                eff_val = float(current_values[feature_names.index('efficiency_index')])
                if press_val < 25 and eff_val < 0.7:
                    feature_conditions.append(f"combined_low_press_eff")

                if feature_conditions:
                    rule_body = ", ".join(sorted(list(set(feature_conditions)))) + "."
                    # Note: Simple naming might lead to many similar rules. Consider hashing body for uniqueness later.
                    rule_name = f"neural_rule_{len(self.learned_rules) + len(potential_new_rules) + 1}"
                    rule = f"{rule_name} :- {rule_body}"
                    confidence = float(predictions[idx])
                    potential_new_rules.append((rule, confidence))

            self.logger.info(f"Potential new rules extracted: {len(potential_new_rules)}")
            return potential_new_rules

        except Exception as e:
            self.logger.error(f"Error extracting rules from neural model: {e}", exc_info=True)
            return []


    def analyze_neural_patterns(
            self,
            anomalous_sequences: np.ndarray,
            normal_sequences: np.ndarray,
            feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Analyze patterns in neural model activations (e.g., LSTM layer outputs)
        to extract potentially more complex or abstract rules.
        (NOTE: This is more experimental and requires careful validation).

        Parameters:
            anomalous_sequences (np.ndarray): Sequences identified as anomalous (batch, timesteps, features).
            normal_sequences (np.ndarray): Sequences identified as normal (batch, timesteps, features).
            feature_names (List[str]): Names of the input features.

        Returns:
            List[Tuple[str, float]]: List of generated pattern rules with confidence scores.
        """
        try:
            pattern_rules = []
            if self.model is None:
                self.logger.error("Cannot analyze patterns: Model not available.")
                return []
            if anomalous_sequences.shape[0] == 0 or normal_sequences.shape[0] == 0:
                self.logger.warning("Insufficient sequences for pattern analysis.")
                return pattern_rules

            if not self.model.built:
                 self.logger.warning("Model is not built before calling analyze_neural_patterns. Attempting to build.")
                 try:
                     build_input = np.expand_dims(anomalous_sequences[0], axis=0) if anomalous_sequences.shape[0] > 0 else (np.expand_dims(normal_sequences[0], axis=0) if normal_sequences.shape[0] > 0 else None)
                     if build_input is not None:
                         _ = self.model.predict(build_input.astype(np.float32), verbose=0)
                         self.logger.info("Model built successfully within analyze_neural_patterns.")
                     else:
                         self.logger.error("Cannot build model: No sequences available.")
                         return []
                 except Exception as build_error:
                     self.logger.error(f"Failed to build model within analyze_neural_patterns: {build_error}")
                     return []

            feature_layer = None
            for layer in reversed(self.model.layers):
                 if isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.Dense)) and layer != self.model.layers[-1]:
                    feature_layer = layer
                    break

            if feature_layer is None:
                self.logger.error("Could not find a suitable layer for pattern analysis.")
                return []
            self.logger.info(f"Using layer '{feature_layer.name}' for pattern analysis feature extraction.")

            try:
                feature_extractor = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=feature_layer.output,
                    name='pattern_feature_extractor'
                )
            except Exception as fe:
                self.logger.error(f"Failed to create feature extractor for patterns: {fe}")
                return []

            try:
                anomalous_features = feature_extractor.predict(anomalous_sequences.astype(np.float32), verbose=0)
                normal_features = feature_extractor.predict(normal_sequences.astype(np.float32), verbose=0)
                if len(anomalous_features.shape) == 3: anomalous_features = anomalous_features[:, -1, :]
                if len(normal_features.shape) == 3: normal_features = normal_features[:, -1, :]
            except Exception as pe:
                self.logger.error(f"Failed to extract features for patterns: {pe}")
                return []

            if anomalous_features.shape[0] > 0 and normal_features.shape[0] > 0:
                anomaly_pattern_mean = np.mean(anomalous_features, axis=0)
                normal_pattern_mean = np.mean(normal_features, axis=0)
                pattern_diff = np.abs(anomaly_pattern_mean - normal_pattern_mean)
                threshold_diff = np.percentile(pattern_diff, 75)

                significant_dims = np.where(pattern_diff > threshold_diff)[0]
                if len(significant_dims) > 0:
                    self.logger.info(f"Found {len(significant_dims)} significant feature dimensions in layer '{feature_layer.name}'.")
                    pattern_rule_name = f"abstract_pattern_{len(self.learned_rules) + len(pattern_rules) + 1}"
                    pattern_rule_body = f"internal_pattern({feature_layer.name}, {significant_dims.tolist()})."
                    pattern_rule = f"{pattern_rule_name} :- {pattern_rule_body}"
                    avg_confidence = np.mean(self.model.predict(anomalous_sequences.astype(np.float32), verbose=0))
                    pattern_rules.append((pattern_rule, float(avg_confidence)))

            self.logger.info(f"Generated {len(pattern_rules)} abstract pattern-based rules.")
            return pattern_rules

        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}", exc_info=True)
            return []


    def update_rules(self,
                     potential_new_rules: List[Tuple[str, float]],
                     min_confidence: float = 0.7,
                     max_learned_rules: int = 100,
                     pruning_strategy: str = 'confidence'
                    ):
        """
        Adds new rules meeting confidence, potentially prunes old rules,
        and rewrites the rules file.

        Parameters:
            potential_new_rules (List[Tuple[str, float]]): List of candidate rules with instance confidence.
            min_confidence (float): Minimum confidence required to add/keep a rule.
            max_learned_rules (int): Maximum number of dynamic rules to keep.
            pruning_strategy (str): Method for pruning ('confidence' or 'lru').
        """
        try:
            added_count = 0
            updated_count = 0
            now = datetime.now()
            needs_rewrite = False # Flag to check if rewrite is needed

            for rule, confidence in potential_new_rules:
                if confidence >= min_confidence:
                    if rule in self.learned_rules:
                        # Update timestamp and activation count (reset on update?)
                        self.learned_rules[rule]['timestamp'] = now
                        # Optionally average confidence here if desired
                        updated_count +=1
                        needs_rewrite = True # Timestamp updated
                    else:
                        # Add new rule
                        self.learned_rules[rule] = {
                            'confidence': confidence,
                            'timestamp': now,
                            'activations': 0
                        }
                        added_count += 1
                        needs_rewrite = True # New rule added

            self.logger.info(f"Considered {len(potential_new_rules)} potential rules. Added: {added_count}, Updated: {updated_count}.")

            # Pruning logic
            if len(self.learned_rules) > max_learned_rules:
                num_to_remove = len(self.learned_rules) - max_learned_rules
                rules_to_remove = []
                needs_rewrite = True # Pruning will happen

                if pruning_strategy == 'confidence':
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['confidence'])
                    rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                    self.logger.info(f"Pruning {num_to_remove} rules based on lowest confidence.")
                elif pruning_strategy == 'lru':
                     sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['timestamp'])
                     rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                     self.logger.info(f"Pruning {num_to_remove} rules based on LRU timestamp.")
                # Add 'lra' pruning if activation tracking is implemented
                # elif pruning_strategy == 'lra': ...

                else: # Default to confidence
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['confidence'])
                    rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                    self.logger.info(f"Pruning {num_to_remove} rules based on lowest confidence (default).")

                for rule in rules_to_remove:
                    if rule in self.learned_rules:
                        del self.learned_rules[rule]

            # Rewrite the rules file and re-consult only if necessary
            if needs_rewrite:
                 self._rewrite_rules_file()

        except Exception as e:
            self.logger.error(f"Error updating Prolog rules: {e}", exc_info=True)


    def get_rule_activations(self) -> List[Dict]:
        """
        Get history of rule activations during reasoning steps.

        Returns:
            List[Dict]: List of rule activation records.
        """
        return self.rule_activations


    def reason(self, sensor_dict: Dict[str, float]) -> List[str]:
        """
        Apply symbolic reasoning rules to sensor data and generate insights.

        Args:
            sensor_dict (Dict[str, float]): Dictionary containing current sensor readings.

        Returns:
            List[str]: List of insights (activated rule names/descriptions).
        """
        try:
            insights = []
            activated_rule_names_this_step = []

            required_keys = [
                'temperature', 'vibration', 'pressure',
                'operational_hours', 'efficiency_index',
                'system_state', 'performance_score'
            ]
            missing_keys = [key for key in required_keys if key not in sensor_dict]
            if missing_keys:
                self.logger.error(f"Missing required sensor values in reason(): {missing_keys}")
                return insights

            try:
                temperature = float(sensor_dict['temperature'])
                vibration = float(sensor_dict['vibration'])
                pressure = float(sensor_dict['pressure'])
                efficiency_index = float(sensor_dict['efficiency_index'])
                operational_hours = int(float(sensor_dict['operational_hours']))
                system_state = int(float(sensor_dict['system_state']))
                performance_score = float(sensor_dict['performance_score'])
            except (ValueError, TypeError) as e:
                self.logger.error(f"Invalid sensor value format in reason(): {e}")
                return insights

            self.state_history.append(sensor_dict)
            if len(self.state_history) > 2:
                 self.state_history.pop(0)

            thermal_gradient = 0.0
            if len(self.state_history) == 2:
                try:
                    prev_temp = float(self.state_history[0].get('temperature', temperature))
                    thermal_gradient = abs(temperature - prev_temp)
                except (ValueError, TypeError):
                     self.logger.warning("Could not calculate thermal gradient due to invalid previous data.")
                     thermal_gradient = 0.0

            # Define base queries matching rules.pl
            base_queries = {
                "Degraded State (Base Rule)": f"degraded_state({temperature}, {vibration}).",
                "System Stress (Base Rule)": f"system_stress({pressure}).",
                "Critical State (Base Rule)": f"critical_state({efficiency_index}).",
                "Maintenance Needed (Base Rule)": f"maintenance_needed({operational_hours}).",
                "Thermal Stress (Base Rule)": f"thermal_stress({temperature}, {thermal_gradient}).",
                # --- CHANGE #2: Use correct predicate name ---
                "Sensor Correlation Alert (Base Rule)": f"sensor_correlation_alert({temperature}, {vibration}, {pressure})."
                # --- END CHANGE #2 ---
            }

            # Execute base rule queries
            for insight_desc, query_string in base_queries.items():
                try:
                    solutions = list(self.prolog.query(query_string))
                    if solutions:
                        insights.append(insight_desc)
                        rule_name = query_string.split('(')[0]
                        activated_rule_names_this_step.append(rule_name + "_base")
                        self.logger.debug(f"Base Rule Success: {insight_desc}")
                # --- CHANGE 2: Catch general Exception ---
                except Exception as e:
                    # Check if it looks like a Prolog error from pyswip
                    if "PrologError" in str(type(e)) or "pyswip" in str(e).lower():
                        self.logger.warning(f"Prolog query failed for '{insight_desc}' with query '{query_string}': {e}")
                    else:
                        self.logger.warning(f"Generic query failed for {insight_desc}: {type(e).__name__} - {e}")
                    continue
                # --- END CHANGE 2 ---

            # Apply learned rules
            for rule_string, metadata in self.learned_rules.items():
                 try:
                     rule_head = rule_string.split(":-")[0].strip()
                     if not rule_head: continue

                     query_string = f"{rule_head}."
                     solutions = list(self.prolog.query(query_string))
                     if solutions:
                         confidence = metadata['confidence']
                         insight = f"Learned Rule {rule_head} (Confidence: {confidence:.2f})"
                         insights.append(insight)
                         activated_rule_names_this_step.append(rule_head)
                         # Increment activation count for the rule
                         self.learned_rules[rule_string]['activations'] += 1
                         self.logger.debug(f"Learned Rule Activated: {insight}")
                 # --- CHANGE 2: Catch general Exception ---
                 except Exception as e:
                     # Check if it looks like a Prolog error from pyswip
                     if "PrologError" in str(type(e)) or "pyswip" in str(e).lower():
                         self.logger.warning(f"Prolog query failed for learned rule '{rule_head}': {e}")
                     else:
                          self.logger.warning(f"Failed to apply learned rule {rule_string}: {type(e).__name__} - {e}")
                     continue
                 # --- END CHANGE 2 ---

            # Record rule activations for this step
            activation_record = {
                'timestep': len(self.rule_activations),
                'activated_rules': activated_rule_names_this_step,
                'insights_generated': insights,
                'sensor_values': sensor_dict,
                'timestamp': datetime.now().isoformat()
            }
            self.rule_activations.append(activation_record)
            if len(self.rule_activations) > 1000:
                self.rule_activations.pop(0)

            self.state_tracker.update(sensor_dict)

            return insights

        except Exception as e:
            self.logger.error(f"Critical error during reasoning process: {str(e)}", exc_info=True)
            return []


    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Retrieve statistics about the current rule base (base + learned).
        """
        num_base_rules = 6 # Approximate count
        num_learned_rules = len(self.learned_rules)
        high_conf_learned = sum(1 for meta in self.learned_rules.values() if meta['confidence'] >= 0.7)
        avg_conf_learned = np.mean([meta['confidence'] for meta in self.learned_rules.values()]) if self.learned_rules else 0.0

        stats = {
            'total_rules': num_base_rules + num_learned_rules,
            'base_rules': num_base_rules,
            'learned_rules_count': num_learned_rules,
            'learned_high_confidence': high_conf_learned,
            'learned_average_confidence': float(avg_conf_learned),
        }
        self.logger.debug(f"Rule Statistics: {stats}")
        return stats