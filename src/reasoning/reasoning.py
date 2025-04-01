# src/reasoning/reasoning.py

import logging
from pyswip import Prolog  # Import PrologError for specific catching
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple, Optional
from src.utils.model_utils import load_model_with_initialization
from .rule_learning import RuleLearner
from .state_tracker import StateTracker
from datetime import datetime
import math  # For isnan checks

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
        """Initializes the Symbolic Reasoner."""
        self.logger = logger or logging.getLogger(__name__)
        try:
            self.prolog = Prolog()
            # Optionally, increase default stack sizes if needed.
        except Exception as e:
             self.logger.critical(f"Failed to initialize PySWIP Prolog instance: {e}", exc_info=True)
             raise RuntimeError(f"Prolog initialization failed: {e}") from e

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

        # Load Prolog files
        self._load_prolog_files([
            'load_config.pl',
            'integrate_prob_log.pl',
            'rules.pl'
        ])

        # Load dynamic rules stored in the main rules file
        self._load_dynamic_rules_from_file()

        # --- Model Loading/Building ---
        if self.model is None and model_path is not None:
            if os.path.exists(model_path):
                self.model = load_model_with_initialization(
                    path=model_path,
                    logger=self.logger,
                    input_shape=input_shape  # Pass shape for build check
                )
                self.logger.info("Model loaded and initialized successfully.")
            else:
                self.logger.warning(f"Model path {model_path} does not exist. Model not loaded.")
        elif self.model is not None:
            self.logger.info("Model provided directly to SymbolicReasoner.")
            if not self.model.built and self.input_shape:
                 try:
                     dummy_input = np.zeros((1,) + self.input_shape, dtype=np.float32)
                     _ = self.model.predict(dummy_input, verbose=0)
                     self.logger.info("Externally provided model built successfully.")
                 except Exception as e:
                     self.logger.error(f"Failed to build externally provided model: {e}")
        else:
            self.logger.warning("No neural model provided or loaded for SymbolicReasoner.")
        # --- End Model Loading/Building ---

    def _load_prolog_files(self, files: List[str]):
        """Load Prolog files, handling potential errors during consult."""
        for file in files:
            path = os.path.join(self.rules_dir, file)
            if os.path.exists(path):
                try:
                    prolog_path = path.replace("\\", "/")  # Ensure prolog-friendly path
                    list(self.prolog.query(f"consult('{prolog_path}')"))
                    self.logger.info(f"Loaded Prolog file: {file}")
                except Exception as e:
                     self.logger.warning(f"Potential issue loading {file} at {path}: {type(e).__name__} - {e}")
            else:
                self.logger.warning(f"Prolog file not found, skipping: {path}")

    def _load_dynamic_rules_from_file(self):
        """Loads existing dynamic rules from the rules file during initialization."""
        try:
            with open(self.rules_path, 'r') as f:
                content = f.readlines()

            in_dynamic_section = False
            self.learned_rules = {}  # Reset before loading

            for line_num, line in enumerate(content):
                stripped_line = line.strip()
                if stripped_line == DYNAMIC_RULES_MARKER:
                    in_dynamic_section = True
                    continue
                if stripped_line == "%% END DYNAMIC RULES %%":
                    in_dynamic_section = False
                    continue

                if in_dynamic_section and stripped_line and not stripped_line.startswith('%'):
                    confidence = 0.7
                    timestamp_str = datetime.now().isoformat()
                    activations = 0
                    rule_part = stripped_line
                    comment_part = ""

                    if '%' in stripped_line:
                        try:
                            rule_part, comment_part = stripped_line.split('%', 1)
                            rule_part = rule_part.strip()
                            comment_part = comment_part.strip()
                            meta_parts = comment_part.split(',')
                            for part in meta_parts:
                                part = part.strip()
                                if ':' in part:
                                    key, value = part.split(':', 1)
                                    key = key.strip().lower()
                                    value = value.strip()
                                    if key == 'confidence': confidence = float(value)
                                    elif key == 'extracted': timestamp_str = value
                                    elif key == 'activations': activations = int(value)
                        except Exception as parse_error:
                            self.logger.warning(f"L{line_num+1}: Could not parse metadata '{comment_part}': {parse_error}")
                            rule_part = stripped_line  # Fallback to using the whole line
                            comment_part = ""

                    if rule_part.endswith('.'):
                         rule_key = rule_part
                         if rule_key not in self.learned_rules:
                             try:
                                 timestamp_dt = datetime.fromisoformat(timestamp_str)
                             except ValueError:
                                 self.logger.warning(f"L{line_num+1}: Invalid ISO timestamp '{timestamp_str}' for rule '{rule_key}', using current time.")
                                 timestamp_dt = datetime.now()
                             self.learned_rules[rule_key] = {
                                 'confidence': confidence, 'timestamp': timestamp_dt, 'activations': activations
                             }
                         else:
                             self.logger.debug(f"L{line_num+1}: Rule '{rule_key}' already loaded, skipping duplicate.")
                    elif rule_part:
                         self.logger.warning(f"L{line_num+1}: Skipping potential rule due to missing period: '{rule_part}'")

            self.logger.info(f"Loaded {len(self.learned_rules)} dynamic rules from {self.rules_path}")

        except FileNotFoundError:
            self.logger.warning(f"Rules file {self.rules_path} not found, no dynamic rules loaded.")
        except Exception as e:
            self.logger.error(f"Error loading dynamic rules from file: {e}", exc_info=True)

    def _rewrite_rules_file(self):
        """Rewrites the rules file safely."""
        temp_rules_path = self.rules_path + ".tmp"
        try:
            base_rules_content = []
            found_marker = False
            try:
                with open(self.rules_path, 'r') as f_read:
                    for line in f_read:
                        if line.strip() == DYNAMIC_RULES_MARKER:
                            found_marker = True
                            base_rules_content.append(line)  # Include marker
                            break
                        base_rules_content.append(line)
                    # If marker wasn't found, add it
                    if not found_marker:
                        self.logger.warning(f"'{DYNAMIC_RULES_MARKER}' not found in {self.rules_path}. Appending it.")
                        if base_rules_content and not base_rules_content[-1].endswith('\n'):
                             base_rules_content.append('\n')
                        base_rules_content.append(f"\n{DYNAMIC_RULES_MARKER}\n")
            except FileNotFoundError:
                 self.logger.warning(f"Rules file {self.rules_path} not found for rewrite. Creating fresh file content.")
                 base_rules_content = [f"{DYNAMIC_RULES_MARKER}\n"]

            # Write base + dynamic rules to temp file
            with open(temp_rules_path, 'w') as f_temp:
                f_temp.writelines(base_rules_content)
                f_temp.write("%% Automatically managed section - Do not edit manually below this line %%\n")
                sorted_rule_items = sorted(self.learned_rules.items())
                for rule, metadata in sorted_rule_items:
                    formatted_rule = (
                        f"{rule}  "
                        f"% Confidence: {metadata.get('confidence', 0.0):.3f}, "
                        f"Extracted: {metadata.get('timestamp', datetime.now()).isoformat()}, "
                        f"Activations: {metadata.get('activations', 0)}\n"
                    )
                    f_temp.write(formatted_rule)
                f_temp.write(f"\n%% END DYNAMIC RULES %%\n")

            # Atomic replace
            os.replace(temp_rules_path, self.rules_path)
            self.logger.info(f"Safely rewrote rules file {self.rules_path} with {len(self.learned_rules)} dynamic rules.")

            # Re-consult Prolog
            try:
                prolog_path = self.rules_path.replace("\\", "/")
                list(self.prolog.query(f"consult('{prolog_path}')"))
                self.logger.info(f"Re-consulted Prolog file: {prolog_path}")
            except Exception as e:
                 self.logger.error(f"Failed to re-consult {prolog_path} after rewrite: {type(e).__name__} - {e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error rewriting Prolog rules file: {e}", exc_info=True)
            if os.path.exists(temp_rules_path):
                try:
                    os.remove(temp_rules_path)
                except OSError as rm_error:
                    self.logger.error(f"Could not remove temporary rules file {temp_rules_path}: {rm_error}")

    def extract_rules_from_neural_model(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7,
            model: Optional[tf.keras.Model] = None
    ) -> List[Tuple[str, float]]:
        """Extracts potential rule candidates from neural model predictions with improved diversity."""
        try:
            model_to_use = model if model is not None else self.model
            if model_to_use is None:
                self.logger.error("No model available for rule extraction.")
                return []

            # Ensure model is built
            if not model_to_use.built:
                 self.logger.warning("Model not built, attempting build for rule extraction.")
                 if self.input_shape and input_data.shape[1:] == self.input_shape:
                     try:
                         _ = model_to_use.predict(input_data[:1].astype(np.float32), verbose=0)
                         self.logger.info("Model built successfully during rule extraction.")
                     except Exception as e:
                         self.logger.error(f"Failed to build model during rule extraction with data shape {input_data.shape[1:]} vs expected {self.input_shape}: {e}")
                         return []
                 else:
                    self.logger.error(f"Cannot build model: input_shape {self.input_shape} unavailable or mismatch with data {input_data.shape[1:]}.")
                    return []

            # Ensure input is 3D
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)
            elif len(input_data.shape) != 3:
                self.logger.error(f"Invalid input shape: {input_data.shape}. Expected 3D.")
                return []

            # Feature dimension check
            if input_data.shape[2] != len(feature_names):
                 self.logger.error(f"Input data feature dim ({input_data.shape[2]}) != feature_names len ({len(feature_names)}).")
                 return []

            self.logger.debug(f"Rule extraction input data shape: {input_data.shape}")
            input_data = input_data.astype(np.float32)

            predictions = model_to_use.predict(input_data, verbose=0).flatten()
            anomaly_indices = np.where(predictions > threshold)[0]
            self.logger.info(f"Found {len(anomaly_indices)} potential rule sequences (prediction > {threshold}).")

            if len(anomaly_indices) == 0:
                return []

            potential_new_rules = []
            current_learned_rule_count = len(self.learned_rules)
            unique_rule_bodies = set()  # Track unique rule bodies to prevent duplicates

            for i, idx in enumerate(anomaly_indices):
                sequence = input_data[idx]
                if sequence.shape[0] < 2:
                    continue

                current_values = sequence[-1, :]
                previous_values = sequence[-2, :]
                feature_conditions = []
                condition_confidences = []

                # --- Generate conditions using predicate names defined in rules.pl ---
                for feat_idx, feat_name in enumerate(feature_names):
                    try:
                        feat_value = float(current_values[feat_idx])
                        prev_value = float(previous_values[feat_idx])
                        change = abs(feat_value - prev_value)
                    except (IndexError, ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing feature '{feat_name}' (idx {feat_idx}) for rule gen at sequence {idx}: {e}")
                        continue

                    # Example: Check if temperature is 'high'
                    if feat_name == 'temperature' and list(self.prolog.query(f"feature_threshold(temperature, {feat_value}, high).")):
                        feature_conditions.append("feature_threshold(temperature, _, high)")
                        condition_confidences.append(min(1.0, feat_value / 80.0))

                    # Example: Check if vibration gradient is 'high'
                    if feat_name == 'vibration' and list(self.prolog.query(f"feature_gradient(vibration, {change}, high).")):
                        feature_conditions.append("feature_gradient(vibration, _, high)")
                        condition_confidences.append(min(1.0, change / 2.0))

                    # Example: Check if pressure is 'low'
                    if feat_name == 'pressure' and list(self.prolog.query(f"feature_threshold(pressure, {feat_value}, low).")):
                        feature_conditions.append("feature_threshold(pressure, _, low)")
                        condition_confidences.append(min(1.0, (100 - feat_value) / 70.0))

                    # Example: Check if efficiency is 'low'
                    if feat_name == 'efficiency_index' and list(self.prolog.query(f"feature_threshold(efficiency_index, {feat_value}, low).")):
                        feature_conditions.append("feature_threshold(efficiency_index, _, low)")
                        condition_confidences.append(min(1.0, (1.0 - feat_value) / 0.4))

                    # Example: Check maintenance_needed condition
                    op_hours_int = int(round(feat_value))
                    if feat_name == 'operational_hours' and list(self.prolog.query(f"maintenance_needed({op_hours_int}).")):
                        feature_conditions.append("maintenance_needed(_)")
                        condition_confidences.append(1.0)

                    # Example: Check state_transition condition
                    if feat_name == 'system_state':
                        prev_state_int = int(round(prev_value))
                        curr_state_int = int(round(feat_value))
                        if curr_state_int != prev_state_int and list(self.prolog.query(f"state_transition({prev_state_int}, {curr_state_int}).")):
                            feature_conditions.append(f"state_transition({prev_state_int}, {curr_state_int})")
                            condition_confidences.append(1.0)

                    # --- Temporal Trend Detection ---
                    # Calculate trend over a short window (using last two points here)
                    if change > 0:
                        # For example, for temperature:
                        if feat_name == 'temperature':
                            if change > 3.0:
                                feature_conditions.append("trend(temperature, increasing)")
                                condition_confidences.append(min(1.0, change / 3.0))
                            elif change > 1.0:
                                feature_conditions.append("trend(temperature, medium)")
                                condition_confidences.append(min(1.0, change / 1.0))
                        # Similarly for other features (vibration, pressure, etc.)
                # --- End condition generation ---

                # --- Add combined conditions (for example, feature correlation) ---
                if len(feature_conditions) >= 2:
                    for i in range(len(feature_names)):
                        for j in range(i+1, len(feature_names)):
                            f1, f2 = feature_names[i], feature_names[j]
                            v1, v2 = current_values[i], current_values[j]
                            if abs(v1 - v2) < 10.0:  # Arbitrary threshold for correlation
                                feature_conditions.append(f"correlated({f1}, {f2})")
                                condition_confidences.append(1.0 - abs(v1 - v2) / 10.0)

                if feature_conditions:
                    # Create rule string with unique conditions sorted
                    rule_body = ", ".join(sorted(list(set(feature_conditions)))) + "."
                    # Skip if we've already seen this exact rule body
                    if rule_body in unique_rule_bodies:
                        continue
                    unique_rule_bodies.add(rule_body)
                    rule_name = f"neural_rule_{current_learned_rule_count + len(potential_new_rules) + 1}"
                    rule_string = f"{rule_name} :- {rule_body}"
                    instance_confidence = float(predictions[idx])
                    # Optionally, you can combine average condition confidence and prediction confidence.
                    avg_confidence = sum(condition_confidences) / len(condition_confidences) if condition_confidences else 0.0
                    final_confidence = max(avg_confidence, instance_confidence)
                    potential_new_rules.append((rule_string, final_confidence))

            self.logger.info(f"Generated {len(potential_new_rules)} potential new rule candidates.")
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
        Placeholder for analyzing patterns in neural model activations.
        (Current implementation is experimental and might need refinement).
        """
        try:
            pattern_rules = []
            if self.model is None:
                self.logger.error("Cannot analyze patterns: Model not available.")
                return []
            if anomalous_sequences.size == 0 or normal_sequences.size == 0:
                self.logger.warning("Insufficient sequences provided for pattern analysis.")
                return pattern_rules

            if not self.model.built:
                 self.logger.warning("Model is not built before calling analyze_neural_patterns. Attempting build.")
                 try:
                     build_input = np.expand_dims(anomalous_sequences[0], axis=0) if anomalous_sequences.size > 0 else (
                                   np.expand_dims(normal_sequences[0], axis=0) if normal_sequences.size > 0 else None)
                     if build_input is not None:
                         _ = self.model.predict(build_input.astype(np.float32), verbose=0)
                         self.logger.info("Model built successfully within analyze_neural_patterns.")
                     else:
                         self.logger.error("Cannot build model: No valid sequences available for build input.")
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
                self.logger.error("Could not find a suitable intermediate layer for pattern analysis.")
                return []
            self.logger.info(f"Using layer '{feature_layer.name}' for pattern analysis feature extraction.")

            try:
                feature_extractor = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=feature_layer.output,
                    name='pattern_feature_extractor'
                )
            except Exception as fe:
                self.logger.error(f"Failed to create feature extractor for patterns using layer '{feature_layer.name}': {fe}")
                return []

            try:
                anomalous_features = feature_extractor.predict(anomalous_sequences.astype(np.float32), verbose=0)
                normal_features = feature_extractor.predict(normal_sequences.astype(np.float32), verbose=0)
                if len(anomalous_features.shape) == 3:
                    anomalous_features = anomalous_features[:, -1, :]
                if len(normal_features.shape) == 3:
                    normal_features = normal_features[:, -1, :]
            except Exception as pe:
                self.logger.error(f"Failed to extract features for patterns using layer '{feature_layer.name}': {pe}")
                return []

            if anomalous_features.shape[0] > 0 and normal_features.shape[0] > 0:
                anomaly_pattern_mean = np.mean(anomalous_features, axis=0)
                normal_pattern_mean = np.mean(normal_features, axis=0)
                pattern_diff = np.abs(anomaly_pattern_mean - normal_pattern_mean)
                threshold_diff = np.percentile(pattern_diff, 75) if pattern_diff.size > 0 else 0

                significant_dims = np.where(pattern_diff > threshold_diff)[0]
                if len(significant_dims) > 0:
                    self.logger.info(f"Found {len(significant_dims)} significant feature dimensions in layer '{feature_layer.name}'.")
                    pattern_rule_name = f"abstract_pattern_{len(self.learned_rules) + len(pattern_rules) + 1}"
                    dims_str = str(significant_dims.tolist()).replace(' ', '')
                    pattern_rule_body = f"internal_pattern('{feature_layer.name}', {dims_str})."
                    pattern_rule = f"{pattern_rule_name} :- {pattern_rule_body}"
                    avg_confidence = np.mean(self.model.predict(anomalous_sequences.astype(np.float32), verbose=0))
                    pattern_rules.append((pattern_rule, float(avg_confidence)))
            self.logger.info(f"Generated {len(pattern_rules)} abstract pattern-based rules.")
            return pattern_rules

        except Exception as e:
            self.logger.error(f"Error in analyze_neural_patterns: {e}", exc_info=True)
            return []

    def update_rules(self,
                     potential_new_rules: List[Tuple[str, float]],
                     min_confidence: float = 0.7,
                     max_learned_rules: int = 100,
                     pruning_strategy: str = 'confidence'
                    ):
        """Adds new rules meeting confidence, potentially prunes old rules."""
        try:
            added_count = 0
            updated_count = 0
            now = datetime.now()
            needs_rewrite = False

            for rule_string, instance_confidence in potential_new_rules:
                if instance_confidence >= min_confidence:
                    if rule_string in self.learned_rules:
                        self.learned_rules[rule_string]['timestamp'] = now
                        updated_count += 1
                        needs_rewrite = True
                    else:
                        self.learned_rules[rule_string] = {
                            'confidence': instance_confidence, 'timestamp': now, 'activations': 0
                        }
                        added_count += 1
                        needs_rewrite = True

            self.logger.info(f"Considered {len(potential_new_rules)} potential rules. Added: {added_count}, Updated: {updated_count}.")

            if len(self.learned_rules) > max_learned_rules:
                num_to_remove = len(self.learned_rules) - max_learned_rules
                rules_to_remove = []
                needs_rewrite = True

                if pruning_strategy == 'confidence':
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['confidence'])
                    rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                    self.logger.info(f"Pruning {num_to_remove} rules based on lowest confidence.")
                elif pruning_strategy == 'lru':
                     sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['timestamp'])
                     rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                     self.logger.info(f"Pruning {num_to_remove} rules based on LRU timestamp.")
                elif pruning_strategy == 'lra':
                     sorted_rules = sorted(self.learned_rules.items(), key=lambda item: (item[1]['activations'], item[1]['timestamp']))
                     rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                     self.logger.info(f"Pruning {num_to_remove} rules based on LRA.")
                else:
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['confidence'])
                    rules_to_remove = [item[0] for item in sorted_rules[:num_to_remove]]
                    self.logger.info(f"Pruning {num_to_remove} rules based on lowest confidence (default).")

                for rule_key in rules_to_remove:
                    if rule_key in self.learned_rules:
                        del self.learned_rules[rule_key]

            if needs_rewrite:
                 self._rewrite_rules_file()

        except Exception as e:
            self.logger.error(f"Error updating Prolog rules: {e}", exc_info=True)

    def get_rule_activations(self) -> List[Dict]:
        """Get history of detailed rule activations."""
        return self.rule_activations

    def reason(self, sensor_dict: Dict[str, float]) -> List[str]:
        """Apply symbolic reasoning rules after asserting current facts."""
        insights = []
        activated_rules_details = []

        # --- 1. Validate Input ---
        required_keys = [
            'temperature', 'vibration', 'pressure', 'operational_hours',
            'efficiency_index', 'system_state', 'performance_score'
        ]
        missing_keys = [key for key in required_keys if key not in sensor_dict or sensor_dict[key] is None]
        if missing_keys:
            self.logger.error(f"Missing or None values for required keys in reason(): {missing_keys}")
            return insights

        # --- 2. Safely Convert Input Values ---
        try:
            current_values = {key: float(sensor_dict[key]) for key in required_keys}
            current_values['operational_hours'] = int(current_values['operational_hours'])
            current_values['system_state'] = int(current_values['system_state'])
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid sensor value format in reason(): {e}")
            return insights

        # --- 3. Manage State History & Calculate Changes ---
        previous_values = self.state_history[-1] if self.state_history else None
        self.state_history.append(current_values)  # Store processed dict
        if len(self.state_history) > 2:
             self.state_history.pop(0)

        changes = {}
        if previous_values:
            for key in ['temperature', 'vibration', 'pressure', 'efficiency_index']:
                 changes[key] = abs(current_values.get(key, 0.0) - previous_values.get(key, 0.0))

        # --- 4. Assert Current Facts into Prolog KB (with error handling) ---
        try:
            # Retract old facts
            retract_queries = [
                "retractall(current_sensor_value(_, _))", "retractall(sensor_change(_, _))",
                "retractall(current_state(_))", "retractall(previous_state(_))"
            ]
            for q in retract_queries:
                list(self.prolog.query(q))

            # Assert new facts
            for key in ['temperature', 'vibration', 'pressure', 'efficiency_index', 'operational_hours']:
                list(self.prolog.query(f"assertz(current_sensor_value({key}, {current_values[key]}))"))
            for key, change_val in changes.items():
                list(self.prolog.query(f"assertz(sensor_change({key}, {change_val}))"))
            list(self.prolog.query(f"assertz(current_state({current_values['system_state']}))"))
            if previous_values:
                 prev_state_val = int(previous_values.get('system_state', 0))
                 list(self.prolog.query(f"assertz(previous_state({prev_state_val}))"))

        except Exception as e:
            self.logger.error(f"Unexpected error managing Prolog facts: {e}", exc_info=True)
            return []

        # --- 5. Define and Execute Base Rule Queries ---
        base_queries = {
            "Degraded State (Base Rule)": f"degraded_state({current_values['temperature']}, {current_values['vibration']}).",
            "System Stress (Base Rule)": f"system_stress({current_values['pressure']}).",
            "Critical State (Base Rule)": f"critical_state({current_values['efficiency_index']}).",
            "Maintenance Needed (Base Rule)": f"base_maintenance_needed({current_values['operational_hours']}).",
            "Thermal Stress (Base Rule)": f"thermal_stress({current_values['temperature']}, {changes.get('temperature', 0.0)}).",
            "Sensor Correlation Alert (Base Rule)": f"sensor_correlation_alert({current_values['temperature']}, {current_values['vibration']}, {current_values['pressure']})."
        }
        for insight_desc, query_string in base_queries.items():
            try:
                if list(self.prolog.query(query_string)):
                    insights.append(insight_desc)
                    rule_name = query_string.split('(')[0]
                    activated_rules_details.append({'rule': rule_name + "_base", 'confidence': 1.0, 'type': 'base'})
                    self.logger.debug(f"Base Rule Activated: {insight_desc}")
            except Exception as e:
                self.logger.warning(f"Error querying base rule '{insight_desc}': {type(e).__name__} - {e}")

        # --- 6. Execute Learned Rule Queries ---
        for rule_string, metadata in self.learned_rules.items():
             try:
                 rule_head = rule_string.split(":-")[0].strip()
                 if not rule_head:
                     continue

                 query_string = f"{rule_head}."
                 if list(self.prolog.query(query_string)):
                     confidence = metadata['confidence']
                     insight = f"Learned Rule Activated: {rule_head} (Conf: {confidence:.2f})"
                     insights.append(insight)
                     activated_rules_details.append({'rule': rule_head, 'confidence': confidence, 'type': 'learned'})
                     self.learned_rules[rule_string]['activations'] = metadata.get('activations', 0) + 1
                     self.logger.debug(f"Learned Rule Activated: {insight}")
             except Exception as e:
                  self.logger.warning(f"Failed applying learned rule {rule_string}: {type(e).__name__} - {e}")

        # --- 7. Record Activations for this Timestep ---
        activation_record = {
            'timestep': len(self.rule_activations),
            'activated_rules_detailed': activated_rules_details,
            'activated_rule_names': [r['rule'] for r in activated_rules_details],
            'insights_generated': insights,
            'sensor_values': current_values,
            'timestamp': datetime.now().isoformat()
        }
        self.rule_activations.append(activation_record)
        if len(self.rule_activations) > 1000:
            self.rule_activations.pop(0)

        return insights

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Retrieve statistics about the current rule base."""
        num_base_rules = 6  # Estimate based on base_queries
        num_learned_rules = len(self.learned_rules)

        if num_learned_rules > 0:
            confidences = [meta.get('confidence', 0.0) for meta in self.learned_rules.values()]
            high_conf_learned = sum(1 for conf in confidences if conf >= 0.7)
            avg_conf_learned = np.mean(confidences)
            total_activations = sum(meta.get('activations', 0) for meta in self.learned_rules.values())
        else:
            high_conf_learned = 0
            avg_conf_learned = 0.0
            total_activations = 0

        stats = {
            'total_rules': num_base_rules + num_learned_rules,
            'base_rules_approx': num_base_rules,
            'learned_rules_count': num_learned_rules,
            'learned_high_confidence': high_conf_learned,
            'learned_average_confidence': float(avg_conf_learned),
            'learned_total_activations': total_activations
        }
        self.logger.debug(f"Rule Statistics: {stats}")
        return stats
