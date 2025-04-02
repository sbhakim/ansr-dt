# src/reasoning/reasoning.py

import logging
from pyswip import Prolog  # Import PrologError for specific catching
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple, Optional
from src.utils.model_utils import load_model_with_initialization
# Assuming rule_learning.py provides methods for more advanced pattern analysis if needed
from .rule_learning import RuleLearner
from .state_tracker import StateTracker
from datetime import datetime
import math  # For isnan checks
import re  # For parsing metadata and rule strings

# <<< SUGGESTION 3: No longer need DYNAMIC_RULES_MARKER for this approach >>>

class SymbolicReasoner:
    def __init__(
            self,
            rules_path: str,
            input_shape: tuple,
            model: Optional[tf.keras.Model] = None,
            model_path: Optional[str] = None,  # Can be removed if model always loaded externally
            logger: Optional[logging.Logger] = None
    ):
        """
        Initializes the Symbolic Reasoner.
        Manages interaction with Prolog engine, loads base rules,
        and dynamically manages learned rules in a separate file.
        """
        self.logger = logger or logging.getLogger(__name__)
        try:
            self.prolog = Prolog()
            # Optionally, increase default stack sizes if needed, e.g.,
            # list(self.prolog.query("set_prolog_flag(stack_limit, 4*10**9)"))
            self.logger.info("PySWIP Prolog instance initialized.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize PySWIP Prolog instance: {e}", exc_info=True)
            raise RuntimeError(f"Prolog initialization failed: {e}") from e

        self.rules_path = os.path.abspath(rules_path)  # Store absolute path to base rules
        self.rules_dir = os.path.dirname(self.rules_path)
        # Define path for learned rules file relative to base rules file
        self.learned_rules_path = os.path.splitext(self.rules_path)[0] + "_learned.pl"

        self.learned_rules: Dict[str, Dict[str, Any]] = {}  # Store learned rules in memory {rule_string: metadata}
        self.model = model  # Expect model instance to be passed
        self.input_shape = input_shape
        self.rule_activations: List[Dict[str, Any]] = []  # History of activations per step
        # History of sensor dicts used for reasoning (limit size for memory)
        self.state_history_limit = 10  # Keep short history for change calculation
        self.state_history: List[Dict[str, Any]] = []

        self.rule_learner = RuleLearner()  # Consider if this is still needed/used
        self.state_tracker = StateTracker()  # Tracks state transitions

        if not os.path.exists(self.rules_path):
            self.logger.error(f"Base Prolog rules file not found at: {self.rules_path}")
            raise FileNotFoundError(f"Base Prolog rules file not found at: {self.rules_path}")

        # Load base and learned rules files
        self._load_prolog_files([
            # Files relative to self.rules_dir
            os.path.basename(self.rules_path),
            os.path.basename(self.learned_rules_path)  # Ensure this is loaded/created
        ])

        # Load learned rules from the separate file into memory
        self._load_dynamic_rules_from_file()  # Reads from self.learned_rules_path

        # Model Validation/Building
        if self.model is not None:
            self.logger.info("Model provided directly to SymbolicReasoner.")
            # Add check if model is built if required by extraction logic later
            if not self.model.built:
                self.logger.warning("Provided model is not built. Rule extraction might attempt to build it.")
        else:
            self.logger.error("No neural model provided to SymbolicReasoner. Rule extraction will fail.")
            # Raise error if model is strictly required immediately
            # raise ValueError("SymbolicReasoner requires a valid Keras model instance.")

    # Updated Prolog file loading
    def _load_prolog_files(self, files: List[str]):
        """
        Load Prolog files, handling potential errors during consult.
        Ensures the learned rules file exists.
        """
        for file in files:
            # Construct path relative to the directory of the main rules file
            path = os.path.join(self.rules_dir, file)

            # Create learned rules file if it's the one and doesn't exist
            if file == os.path.basename(self.learned_rules_path) and not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        f.write(f"% Dynamically learned rules for ANSR-DT - Automatically managed\n")
                        # Add necessary directives if rules need them
                        f.write(":- discontiguous(neural_rule/0).\n")
                        f.write(":- discontiguous(gradient_rule/0).\n")
                        f.write(":- discontiguous(pattern_rule/0).\n")
                        f.write(":- discontiguous(abstract_pattern/0).\n\n")  # Add others if used
                    self.logger.info(f"Created empty learned rules file: {path}")
                except IOError as e:
                    self.logger.error(f"Failed to create learned rules file {path}: {e}")
                    # Consider raising an error if learned rules are critical
                    # raise RuntimeError(f"Failed to create learned rules file: {path}") from e

            if os.path.exists(path):
                try:
                    prolog_path = path.replace("\\", "/")  # Ensure prolog-friendly path
                    # Use consult/1 for loading files
                    query = f"consult('{prolog_path}')"
                    # Execute query and check for errors explicitly if possible
                    result = list(self.prolog.query(query))
                    # Basic check: if consult fails critically, PrologError is often raised
                    self.logger.info(f"Consulted Prolog file: {path}")
                except Exception as e:
                    self.logger.error(f"Error consulting {path}: {type(e).__name__} - {e}", exc_info=True)
                    raise RuntimeError(f"Failed to consult Prolog file: {path}") from e
            else:
                # Log warning only if it's not the learned rules file
                if file != os.path.basename(self.learned_rules_path):
                    self.logger.warning(f"Prolog file not found, skipping: {path}")

    # Updated dynamic rule loading from separate file
    def _load_dynamic_rules_from_file(self):
        """Loads existing dynamic rules from the separate learned rules file."""
        self.learned_rules = {}  # Reset before loading
        try:
            # Use self.learned_rules_path
            if not os.path.exists(self.learned_rules_path):
                self.logger.info(f"Learned rules file {self.learned_rules_path} not found, starting fresh.")
                return  # Nothing to load

            with open(self.learned_rules_path, 'r') as f:
                content = f.readlines()

            # Parsing logic for rule and metadata comment
            for line_num, line in enumerate(content):
                stripped_line = line.strip()
                # Skip empty lines and comments that are not metadata or directives
                if not stripped_line or (stripped_line.startswith('%') and 'Confidence:' not in stripped_line) or stripped_line.startswith(':- discontiguous'):
                    continue

                confidence = 0.7  # Default confidence
                timestamp_str = datetime.now().isoformat()  # Default timestamp
                activations = 0  # Default activations
                rule_part = stripped_line
                comment_part = ""

                if '%' in stripped_line:
                    try:
                        rule_part, comment_part = stripped_line.split('%', 1)
                        rule_part = rule_part.strip()
                        comment_part = comment_part.strip()
                        # Use regex for more robust parsing of metadata
                        conf_match = re.search(r'Confidence:\s*([0-9.]+)', comment_part, re.IGNORECASE)
                        ts_match = re.search(r'Extracted:\s*([\d\-T:.+]+)', comment_part, re.IGNORECASE)  # Handle timezone
                        act_match = re.search(r'Activations:\s*(\d+)', comment_part, re.IGNORECASE)

                        if conf_match:
                            confidence = float(conf_match.group(1))
                        if ts_match:
                            timestamp_str = ts_match.group(1)
                        if act_match:
                            activations = int(act_match.group(1))

                    except Exception as parse_error:
                        self.logger.warning(f"L{line_num+1} in {self.learned_rules_path}: Could not parse metadata '{comment_part}': {parse_error}")
                        if '%' in stripped_line:
                            rule_part = stripped_line.split('%', 1)[0].strip()
                        else:
                            rule_part = stripped_line  # Use whole line if no comment found

                # Validate and store the rule (must end with '.')
                if rule_part.endswith('.'):
                    rule_key = rule_part  # Use the full rule string as the key
                    if rule_key not in self.learned_rules:
                        try:
                            # Attempt to parse ISO format, fallback to now()
                            timestamp_dt = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            self.logger.warning(f"L{line_num+1}: Invalid ISO timestamp '{timestamp_str}' for rule '{rule_key}', using current time.")
                            timestamp_dt = datetime.now()
                        self.learned_rules[rule_key] = {
                            'confidence': confidence, 'timestamp': timestamp_dt, 'activations': activations
                        }
                    # else: Log duplicate if needed: self.logger.debug(...)
                elif rule_part and not rule_part.startswith(':-'):
                    # Log if it looks like a rule but lacks the period and isn't a directive
                    self.logger.warning(f"L{line_num+1} in {self.learned_rules_path}: Skipping potential rule due to missing period: '{rule_part}'")

            self.logger.info(f"Loaded {len(self.learned_rules)} dynamic rules from {self.learned_rules_path}")

        except FileNotFoundError:
            self.logger.warning(f"Learned rules file {self.learned_rules_path} not found during load attempt.")
        except Exception as e:
            self.logger.error(f"Error loading dynamic rules from file {self.learned_rules_path}: {e}", exc_info=True)

    # Updated rewrite function for separate learned file
    def _rewrite_rules_file(self):
        """Rewrites ONLY the learned rules file safely."""
        temp_learned_rules_path = self.learned_rules_path + ".tmp"
        try:
            # Write only learned rules to temp file
            with open(temp_learned_rules_path, 'w') as f_temp:
                # Write header comments
                f_temp.write(f"% Dynamically learned rules for ANSR-DT - Automatically managed\n")
                f_temp.write(f"% Last updated: {datetime.now().isoformat()}\n\n")

                # Add necessary Prolog directives at the start of the file
                f_temp.write(":- discontiguous(neural_rule/0).\n")
                f_temp.write(":- discontiguous(gradient_rule/0).\n")
                f_temp.write(":- discontiguous(pattern_rule/0).\n")
                f_temp.write(":- discontiguous(abstract_pattern/0).\n\n")  # Add others if used

                # Write learned rules sorted by rule string (key) for consistency
                sorted_rule_items = sorted(self.learned_rules.items())
                for rule, metadata in sorted_rule_items:
                    # Ensure rule ends with a period before appending comment
                    rule_str = rule if rule.endswith('.') else rule + '.'
                    formatted_rule = (
                        f"{rule_str}  "
                        f"% Confidence: {metadata.get('confidence', 0.0):.3f}, "
                        f"Extracted: {metadata.get('timestamp', datetime.now()).isoformat()}, "
                        f"Activations: {metadata.get('activations', 0)}\n"
                    )
                    f_temp.write(formatted_rule)

            # Atomic replace for the learned rules file
            os.replace(temp_learned_rules_path, self.learned_rules_path)
            self.logger.info(f"Safely rewrote learned rules file {self.learned_rules_path} with {len(self.learned_rules)} rules.")

            # Re-consult ONLY the learned rules file
            try:
                prolog_path = self.learned_rules_path.replace("\\", "/")
                query = f"consult('{prolog_path}')"
                list(self.prolog.query(query))
                self.logger.info(f"Re-consulted Prolog learned rules file: {prolog_path}")
            except Exception as e:
                self.logger.error(f"Failed to re-consult {prolog_path} after rewrite: {type(e).__name__} - {e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error rewriting learned rules file {self.learned_rules_path}: {e}", exc_info=True)
            if os.path.exists(temp_learned_rules_path):
                try:
                    os.remove(temp_learned_rules_path)
                except OSError as rm_error:
                    self.logger.error(f"Could not remove temporary learned rules file {temp_learned_rules_path}: {rm_error}")

    # --- Start of provided extract_rules_from_neural_model ---
    def extract_rules_from_neural_model(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7,
            model: Optional[tf.keras.Model] = None
    ) -> List[Tuple[str, float]]:
        """
        Extracts potential rule candidates from neural model predictions.
        Focuses on translating high-confidence anomaly predictions into symbolic preconditions.
        """
        try:
            model_to_use = model if model is not None else self.model
            if model_to_use is None:
                self.logger.error("No model available for rule extraction.")
                return []

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

            if input_data.ndim == 2:
                if input_data.shape[0] == self.input_shape[0] and input_data.shape[1] == self.input_shape[1]:
                    input_data = np.expand_dims(input_data, axis=0)
                else:
                    self.logger.error(f"Invalid 2D input shape: {input_data.shape}. Expected ({self.input_shape[0]}, {self.input_shape[1]}).")
                    return []
            elif input_data.ndim != 3:
                self.logger.error(f"Invalid input shape: {input_data.shape}. Expected 3D (batch, timesteps, features).")
                return []

            if input_data.shape[2] != len(feature_names):
                self.logger.error(f"Input data feature dim ({input_data.shape[2]}) != feature_names len ({len(feature_names)}).")
                return []

            self.logger.debug(f"Rule extraction input data shape: {input_data.shape}")
            input_data = input_data.astype(np.float32)

            predictions = model_to_use.predict(input_data, verbose=0).flatten()
            anomaly_indices = np.where(predictions > threshold)[0]
            self.logger.info(f"Found {len(anomaly_indices)} sequences meeting rule extraction threshold (prediction > {threshold}).")

            if len(anomaly_indices) == 0:
                return []

            potential_new_rules = []
            current_learned_rule_count = len(self.learned_rules)
            unique_rule_bodies_this_batch = set()

            for i, sequence_idx in enumerate(anomaly_indices):
                sequence = input_data[sequence_idx]
                if sequence.shape[0] < 2:
                    self.logger.warning(f"Skipping sequence index {sequence_idx}: Not enough timesteps ({sequence.shape[0]})")
                    continue

                current_values = sequence[-1, :]
                previous_values = sequence[-2, :]
                feature_conditions = []
                condition_confidences = []

                for feat_idx, feat_name in enumerate(feature_names):
                    try:
                        feat_value = float(current_values[feat_idx])
                        prev_value = float(previous_values[feat_idx])
                        if math.isnan(feat_value) or math.isinf(feat_value) or \
                           math.isnan(prev_value) or math.isinf(prev_value):
                            self.logger.debug(f"Skipping feature '{feat_name}' due to NaN/Inf.")
                            continue
                        change = abs(feat_value - prev_value)
                    except (IndexError, ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing feature '{feat_name}' (idx {feat_idx}) for rule gen at sequence {sequence_idx}: {e}")
                        continue

                    try:
                        if list(self.prolog.query(f"feature_threshold({feat_name}, {feat_value}, high).")):
                            feature_conditions.append(f"feature_threshold({feat_name}, _, high)")
                            threshold_high = 80.0 if feat_name == 'temperature' else (55.0 if feat_name == 'vibration' else 40.0)
                            condition_confidences.append(min(1.0, max(0.0, (feat_value - threshold_high) / (10.0 if feat_name == 'temperature' else 5.0))))
                        elif list(self.prolog.query(f"feature_threshold({feat_name}, {feat_value}, low).")):
                            feature_conditions.append(f"feature_threshold({feat_name}, _, low)")
                            threshold_low = 40.0 if feat_name == 'temperature' else (20.0 if feat_name == 'vibration' else 20.0)
                            condition_confidences.append(min(1.0, max(0.0, (threshold_low - feat_value) / (10.0 if feat_name == 'temperature' else 5.0))))

                        if list(self.prolog.query(f"feature_gradient({feat_name}, {change}, high).")):
                            feature_conditions.append(f"feature_gradient({feat_name}, _, high)")
                            gradient_threshold = 2.0 if feat_name == 'temperature' else (1.5 if feat_name == 'vibration' else 1.0)
                            condition_confidences.append(min(1.0, max(0.0, (change - gradient_threshold) / (gradient_threshold))))

                        if feat_value > prev_value and list(self.prolog.query(f"trend({feat_name}, increasing).")):
                            feature_conditions.append(f"trend({feat_name}, increasing)")
                            condition_confidences.append(0.8)
                        elif feat_value < prev_value and list(self.prolog.query(f"trend({feat_name}, decreasing).")):
                            feature_conditions.append(f"trend({feat_name}, decreasing)")
                            condition_confidences.append(0.8)

                        if feat_name == 'operational_hours':
                            op_hours_int = int(round(feat_value))
                            if list(self.prolog.query(f"base_maintenance_needed({op_hours_int}).")):
                                feature_conditions.append(f"base_maintenance_needed({op_hours_int})")
                                condition_confidences.append(1.0)

                        if feat_name == 'system_state':
                            prev_state_int = int(round(prev_value))
                            curr_state_int = int(round(feat_value))
                            if curr_state_int != prev_state_int and list(self.prolog.query(f"state_transition({prev_state_int}, {curr_state_int}).")):
                                feature_conditions.append(f"state_transition({prev_state_int}, {curr_state_int})")
                                condition_confidences.append(1.0)

                    except Exception as qe:
                        self.logger.warning(f"Unexpected error querying Prolog for feature '{feat_name}': {qe}")

                try:
                    if list(self.prolog.query(f"correlated(temperature, vibration).")):
                        feature_conditions.append("correlated(temperature, vibration)")
                        condition_confidences.append(0.85)
                    if list(self.prolog.query(f"sequence_pattern(efficiency_drop_with_temp_rise).")):
                        feature_conditions.append("sequence_pattern(efficiency_drop_with_temp_rise)")
                        condition_confidences.append(0.9)
                except Exception as corr_e:
                    self.logger.warning(f"Error querying correlation/sequence patterns: {corr_e}")

                if feature_conditions:
                    unique_sorted_conditions = sorted(list(set(feature_conditions)))
                    rule_body = ", ".join(unique_sorted_conditions)
                    if not rule_body:
                        continue

                    if rule_body in unique_rule_bodies_this_batch:
                        continue
                    unique_rule_bodies_this_batch.add(rule_body)

                    rule_name = f"neural_rule_{current_learned_rule_count + len(potential_new_rules) + 1}"
                    rule_string = f"{rule_name} :- {rule_body}."
                    neural_confidence = float(predictions[sequence_idx])
                    avg_condition_confidence = np.mean(condition_confidences) if condition_confidences else 0.0
                    final_confidence = (neural_confidence * 0.6) + (avg_condition_confidence * 0.4)
                    final_confidence = max(0.0, min(1.0, final_confidence))
                    potential_new_rules.append((rule_string, final_confidence))

            self.logger.info(f"Generated {len(potential_new_rules)} potential new rule candidates from {len(anomaly_indices)} sequences.")
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
        Analyzes patterns in neural model activations to derive abstract rules.
        (Experimental - Requires careful validation and refinement).
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

                if anomalous_features.ndim == 3:
                    anomalous_features = anomalous_features[:, -1, :]
                if normal_features.ndim == 3:
                    normal_features = normal_features[:, -1, :]

            except Exception as pe:
                self.logger.error(f"Failed to extract features for patterns using layer '{feature_layer.name}': {pe}")
                return []

            if anomalous_features.shape[0] > 0 and normal_features.shape[0] > 0:
                if anomalous_features.ndim != 2 or normal_features.ndim != 2:
                    self.logger.error(f"Unexpected feature dimensions after extraction: Anom {anomalous_features.shape}, Norm {normal_features.shape}")
                    return []

                anomaly_pattern_mean = np.mean(anomalous_features, axis=0)
                normal_pattern_mean = np.mean(normal_features, axis=0)

                if anomaly_pattern_mean.shape != normal_pattern_mean.shape:
                    self.logger.error(f"Mean feature shapes differ: Anom {anomaly_pattern_mean.shape}, Norm {normal_pattern_mean.shape}")
                    return []

                pattern_diff = np.abs(anomaly_pattern_mean - normal_pattern_mean)

                if pattern_diff.size > 0:
                    threshold_diff = np.percentile(pattern_diff, 75)
                else:
                    threshold_diff = 0

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
        """
        Adds new rules meeting confidence to the in-memory store,
        potentially prunes old rules, and triggers rewrite of the learned rules file.
        """
        try:
            added_count = 0
            updated_count = 0
            now = datetime.now()
            needs_rewrite = False

            for rule_string, instance_confidence in potential_new_rules:
                if not isinstance(rule_string, str) or not rule_string.endswith('.'):
                    self.logger.warning(f"Skipping invalid rule string format: {rule_string}")
                    continue

                if instance_confidence >= min_confidence:
                    if rule_string in self.learned_rules:
                        self.learned_rules[rule_string]['timestamp'] = now
                        updated_count += 1
                        needs_rewrite = True
                    else:
                        self.learned_rules[rule_string] = {
                            'confidence': instance_confidence,
                            'timestamp': now,
                            'activations': 0
                        }
                        added_count += 1
                        needs_rewrite = True

            if added_count > 0 or updated_count > 0:
                self.logger.info(f"Processed {len(potential_new_rules)} potential rules. Added: {added_count}, Updated existing: {updated_count}.")

            if len(self.learned_rules) > max_learned_rules:
                num_to_remove = len(self.learned_rules) - max_learned_rules
                rules_to_remove_keys = []
                needs_rewrite = True

                self.logger.info(f"Learned rules ({len(self.learned_rules)}) exceed limit ({max_learned_rules}). Pruning {num_to_remove} rules using '{pruning_strategy}' strategy.")

                if pruning_strategy == 'confidence':
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: (item[1]['confidence'], item[1]['timestamp']))
                elif pruning_strategy == 'lru':
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['timestamp'])
                elif pruning_strategy == 'lra':
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: (item[1]['activations'], item[1]['timestamp']))
                else:
                    self.logger.warning(f"Unknown pruning strategy '{pruning_strategy}'. Defaulting to 'confidence'.")
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: (item[1]['confidence'], item[1]['timestamp']))

                rules_to_remove_keys = [item[0] for item in sorted_rules[:num_to_remove]]

                for rule_key in rules_to_remove_keys:
                    if rule_key in self.learned_rules:
                        del self.learned_rules[rule_key]
                self.logger.info(f"Pruned {len(rules_to_remove_keys)} rules. New count: {len(self.learned_rules)}.")

            if needs_rewrite:
                self._rewrite_rules_file()

        except Exception as e:
            self.logger.error(f"Error updating Prolog rules: {e}", exc_info=True)

    # <<< SUGGESTION 3: Method to get full activation history >>>
    def get_rule_activations(self) -> List[Dict]:
        """Get the complete history of detailed rule activations recorded by the reasoner."""
        return self.rule_activations

    def reason(self, sensor_dict: Dict[str, Any]) -> List[str]:
        """
        Apply symbolic reasoning rules based on the current sensor state.
        Asserts current facts, queries base and learned rules, and records activations.

        Args:
            sensor_dict (Dict[str, Any]): Dictionary containing current sensor readings
                                          and potentially other metrics (e.g., efficiency).
                                          Keys must match Prolog expectations (e.g., 'temperature').

        Returns:
            List[str]: A list of human-readable insights derived from activated rules.
        """
        insights = []
        activated_rules_details = []  # Track rules activated in this specific call

        # --- 1. Validate Input ---
        required_keys = [
            'temperature', 'vibration', 'pressure', 'operational_hours',
            'efficiency_index', 'system_state', 'performance_score'
        ]
        if not all(key in sensor_dict and sensor_dict[key] is not None for key in required_keys):
            missing_or_none = [k for k in required_keys if k not in sensor_dict or sensor_dict[k] is None]
            self.logger.error(f"Missing or None values for required keys in reason(): {missing_or_none}")
            return []

        # --- 2. Safely Convert Input Values ---
        try:
            current_values = {}
            for key in required_keys:
                value = sensor_dict[key]
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    self.logger.warning(f"NaN/Inf found for key '{key}'. Replacing with 0.")
                    value = 0.0

                if key in ['system_state', 'operational_hours']:
                    current_values[key] = int(round(float(value)))
                else:
                    current_values[key] = float(value)
            self.logger.debug(f"Current values for reasoning: {current_values}")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid sensor value format provided to reason(): {e}")
            return []

        # --- 3. Manage State History & Calculate Changes ---
        previous_values = self.state_history[-1] if self.state_history else None
        self.state_history.append(current_values)
        if len(self.state_history) > self.state_history_limit:
            self.state_history.pop(0)

        changes = {}
        if previous_values:
            for key in ['temperature', 'vibration', 'pressure', 'efficiency_index']:
                if key in current_values and key in previous_values:
                    try:
                        current_val = current_values.get(key, 0.0)
                        previous_val = previous_values.get(key, 0.0)
                        if current_val is not None and previous_val is not None:
                            changes[key] = abs(float(current_val) - float(previous_val))
                        else:
                            changes[key] = 0.0
                    except (TypeError, ValueError):
                        changes[key] = 0.0
                else:
                    changes[key] = 0.0
        self.logger.debug(f"Calculated changes: {changes}")

        # --- 4. Assert Current Facts into Prolog KB ---
        try:
            retract_queries = [
                "retractall(current_sensor_value(_, _))",
                "retractall(feature_threshold(_, _, _))",
                "retractall(feature_gradient(_, _, _))",
                "retractall(sensor_change(_, _))",
                "retractall(current_state(_))",
                "retractall(previous_state(_))",
                "retractall(base_maintenance_needed(_))"
            ]
            for q in retract_queries:
                list(self.prolog.query(q))

            # 1. Assert basic sensor values
            for key, value in current_values.items():
                if isinstance(value, (int, float)):
                    assert_query = f"assertz(current_sensor_value({key}, {value}))"
                    list(self.prolog.query(assert_query))

            # 2. Assert threshold facts that match rule conditions
            temp = current_values.get('temperature', 0.0)
            vib = current_values.get('vibration', 0.0)
            press = current_values.get('pressure', 0.0)
            eff = current_values.get('efficiency_index', 0.0)
            op_hours = current_values.get('operational_hours', 0)

            if temp > 80:
                list(self.prolog.query(f"assertz(feature_threshold(temperature, {temp}, high))"))
            elif temp < 40:
                list(self.prolog.query(f"assertz(feature_threshold(temperature, {temp}, low))"))

            if vib > 55:
                list(self.prolog.query(f"assertz(feature_threshold(vibration, {vib}, high))"))
            elif vib < 20:
                list(self.prolog.query(f"assertz(feature_threshold(vibration, {vib}, low))"))

            if press > 40:
                list(self.prolog.query(f"assertz(feature_threshold(pressure, {press}, high))"))
            elif press < 20:
                list(self.prolog.query(f"assertz(feature_threshold(pressure, {press}, low))"))

            if eff < 0.6:
                list(self.prolog.query(f"assertz(feature_threshold(efficiency_index, {eff}, low))"))

            # 3. Assert gradient facts for significant changes
            for key, change in changes.items():
                threshold = 0.0
                if key == 'temperature':
                    threshold = 2.0
                elif key == 'vibration':
                    threshold = 1.5
                elif key == 'pressure':
                    threshold = 1.0
                elif key == 'efficiency_index':
                    threshold = 0.1

                if change > threshold:
                    list(self.prolog.query(f"assertz(feature_gradient({key}, {change}, high))"))
                    list(self.prolog.query(f"assertz(sensor_change({key}, {change}))"))

            # 4. Assert maintenance facts
            if op_hours % 1000 == 0 and op_hours > 0:
                list(self.prolog.query(f"assertz(base_maintenance_needed({op_hours}))"))

            # 5. Assert state transitions
            list(self.prolog.query(f"assertz(current_state({current_values['system_state']}))"))
            if previous_values:
                prev_state = int(previous_values.get('system_state', 0))
                list(self.prolog.query(f"assertz(previous_state({prev_state}))"))

            self.logger.debug("Asserted facts into Prolog KB that match rule patterns")
        except Exception as e:
            self.logger.error(f"Error asserting facts: {e}", exc_info=True)
            return []

        # --- 5. Execute Base Rule Queries ---
        base_queries = {
            "Degraded State Triggered": "degraded_state(_, _)",
            "System Stress Triggered": "system_stress(_)",
            "Critical State Triggered": "critical_state(_)",
            "Maintenance Needed Soon": f"base_maintenance_needed({current_values['operational_hours']})",
            "Thermal Stress Detected": "thermal_stress(_, _)",
            "Sensor Correlation Alert": "sensor_correlation_alert(_, _, _)"
        }
        for insight_desc, query_string in base_queries.items():
            try:
                solutions = list(self.prolog.query(query_string))
                if solutions:
                    insights.append(insight_desc)
                    rule_name = query_string.split('(')[0]
                    activated_rules_details.append({
                        'rule': rule_name + "_base",
                        'confidence': 1.0,
                        'type': 'base'
                    })
                    self.logger.debug(f"Base Rule Activated: {insight_desc}")
            except Exception as e:
                self.logger.warning(f"Error querying base rule '{insight_desc}' ({query_string}): {type(e).__name__} - {e}")

        # --- 6. Execute Learned Rule Queries ---
        updated_learned_rules = False
        for rule_string, metadata in list(self.learned_rules.items()):
            try:
                rule_head_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:-", rule_string)
                if not rule_head_match:
                    self.logger.warning(f"Could not parse head from learned rule: {rule_string}")
                    continue
                rule_head = rule_head_match.group(1)
                query_string = f"{rule_head}."
                solutions = list(self.prolog.query(query_string))
                if solutions:
                    confidence = metadata.get('confidence', 0.0)
                    insight = f"Learned Rule Activated: {rule_head} (Conf: {confidence:.2f})"
                    insights.append(insight)
                    activated_rules_details.append({
                        'rule': rule_head,
                        'confidence': confidence,
                        'type': 'learned'
                    })
                    current_activations = metadata.get('activations', 0)
                    self.learned_rules[rule_string]['activations'] = current_activations + 1
                    updated_learned_rules = True
                    self.logger.debug(f"Learned Rule Activated: {insight} (Total Activations: {current_activations + 1})")
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
        history_limit = 1000
        if len(self.rule_activations) > history_limit:
            self.rule_activations.pop(0)

        # --- 8. Return Insights ---
        self.logger.info(f"Reasoning complete. Generated {len(insights)} insights.")
        return insights

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Retrieve statistics about the current rule base (base + learned in memory)."""
        num_base_rules_queried = 6
        num_learned_rules = len(self.learned_rules)

        learned_stats = {
            'count': num_learned_rules,
            'high_confidence_count': 0,
            'average_confidence': 0.0,
            'total_activations': 0,
            'confidence_distribution': {}
        }

        if num_learned_rules > 0:
            confidences = [meta.get('confidence', 0.0) for meta in self.learned_rules.values()]
            activations = [meta.get('activations', 0) for meta in self.learned_rules.values()]

            learned_stats['high_confidence_count'] = sum(1 for conf in confidences if conf >= 0.7)
            learned_stats['average_confidence'] = float(np.mean(confidences))
            learned_stats['total_activations'] = sum(activations)

            bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
            hist, _ = np.histogram(confidences, bins=bins)
            learned_stats['confidence_distribution'] = {
                f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(count) for i, count in enumerate(hist)
            }

        stats = {
            'total_rules_managed': num_learned_rules,
            'base_rules_queried_approx': num_base_rules_queried,
            'learned_rules': learned_stats,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.debug(f"Rule Statistics: {stats}")
        return stats
