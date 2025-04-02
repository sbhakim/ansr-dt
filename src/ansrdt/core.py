# src/ansrdt/core.py

import numpy as np
import logging
from typing import Dict, Any, Optional
import tensorflow as tf
from stable_baselines3 import PPO
from src.config.config_manager import load_config
from src.utils.model_utils import load_model_with_initialization as load_model
from src.reasoning.reasoning import SymbolicReasoner
from src.integration.adaptive_controller import AdaptiveController
import json
import os
import math # <<< SUGGESTION 2: Added import math >>>

class ANSRDTCore:
    """
    Core class for the ANSR-DT system, handling inference, adaptation, and explanation.
    Integrates the CNN-LSTM anomaly detection model, PPO agent for control, and symbolic reasoning.
    """

    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None,
                 cnn_lstm_model: Optional[tf.keras.Model] = None,
                 ppo_agent: Optional[PPO] = None):
        """
        Initialize core ANSR-DT components.

        Parameters:
        - config_path (str): Path to the configuration file.
        - logger (Optional[logging.Logger]): Optional logger instance. If not provided, a default logger is created.
        - cnn_lstm_model (Optional[tf.keras.Model]): Pre-loaded CNN-LSTM model. If None, the model will be loaded from file.
        - ppo_agent (Optional[PPO]): Pre-loaded PPO agent. If None, the agent will be loaded from file.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = load_config(config_path)

        # Determine base directories based on config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.project_root = os.path.dirname(self.config_dir)

        # Set results directory from configuration
        # Ensure results_dir path exists in config before joining
        results_rel_path = self.config.get('paths', {}).get('results_dir', 'results')
        self.results_dir = os.path.join(self.project_root, results_rel_path)
        os.makedirs(self.results_dir, exist_ok=True) # Ensure it exists

        self.adaptive_controller = AdaptiveController(
            window_size=self.config['model']['window_size']
        )

        # Initialize control parameters
        self.control_params = {
            'temperature_adjustment': 0.0,
            'vibration_adjustment': 0.0,
            'pressure_adjustment': 0.0,
            'efficiency_target': 0.8
        }

        # Initialize models
        if cnn_lstm_model is not None:
            self.cnn_lstm = cnn_lstm_model
            self.logger.info("CNN-LSTM model provided externally.")
        else:
            self.cnn_lstm = self._load_cnn_lstm_model()

        if ppo_agent is not None:
            self.ppo_agent = ppo_agent
            self.logger.info("PPO agent provided externally.")
        else:
            self.ppo_agent = self._load_ppo_agent()

        # Extract and set window_size and feature_names from configuration
        try:
            self.window_size = self.config['model']['window_size']
            self.feature_names = self.config['model']['feature_names']
            if not self.feature_names or not isinstance(self.feature_names, list):
                raise ValueError("feature_names must be a non-empty list in configuration.")
            self.logger.debug(f"Window Size: {self.window_size}")
            self.logger.debug(f"Feature Names: {self.feature_names}")
        except KeyError as ke:
            self.logger.error(f"Missing key in model configuration: {ke}")
            raise
        except ValueError as ve:
             self.logger.error(f"Invalid configuration for feature_names: {ve}")
             raise

        # Initialize symbolic reasoner with correct path
        self.reasoner = self._initialize_reasoner()

        # Initialize state tracking
        self.current_state = None
        # Limit state history size for memory management
        self.state_history_limit = self.config.get('logging', {}).get('state_history_limit', 1000)
        self.state_history = []
        self.decision_history = [] # Also consider limiting this if needed

    def _load_cnn_lstm_model(self) -> tf.keras.Model:
        """
        Load the trained CNN-LSTM model from the specified path in results_dir.

        Returns:
            tf.keras.Model: Loaded CNN-LSTM model.
        """
        try:
            # Use the class attribute self.results_dir
            model_path = os.path.join(self.results_dir, 'best_model.keras')
            self.logger.info(f"Loading CNN-LSTM model from: {model_path}")
            if not os.path.exists(model_path):
                self.logger.error(f"CNN-LSTM model file not found at: {model_path}")
                raise FileNotFoundError(f"CNN-LSTM model file not found at: {model_path}")
            # Assume input shape is needed if model isn't built
            input_shape_tuple = (self.window_size, len(self.feature_names))
            model = load_model(model_path, self.logger, input_shape=input_shape_tuple)
            self.logger.info("CNN-LSTM model loaded successfully.")
            return model
        except FileNotFoundError as fe:
            self.logger.error(f"CNN-LSTM model file not found: {fe}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load CNN-LSTM model: {e}", exc_info=True)
            raise

    def _load_ppo_agent(self) -> PPO:
        """
        Load the trained PPO agent from the specified path in results_dir.

        Returns:
            PPO: Loaded PPO agent.
        """
        try:
            # Use the class attribute self.results_dir
            ppo_path = os.path.join(self.results_dir, 'ppo_ansr_dt.zip')
            self.logger.info(f"Loading PPO agent from: {ppo_path}")
            if not os.path.exists(ppo_path):
                self.logger.error(f"PPO model file not found at {ppo_path}")
                raise FileNotFoundError(f"PPO model not found at {ppo_path}")
            ppo_agent = PPO.load(ppo_path)
            self.logger.info("PPO agent loaded successfully.")
            return ppo_agent
        except FileNotFoundError as fe:
            self.logger.error(f"PPO agent file not found: {fe}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load PPO agent: {e}", exc_info=True)
            raise

    def _initialize_reasoner(self) -> Optional[SymbolicReasoner]:
        """
        Initialize symbolic reasoner with proper path resolution.

        Returns:
            Optional[SymbolicReasoner]: Initialized symbolic reasoner or None if disabled.
        """
        try:
            symbolic_reasoning_config = self.config.get('symbolic_reasoning', {})
            if symbolic_reasoning_config.get('enabled', False):
                # Resolve rules path using paths from config (already resolved in main)
                rules_path_key = 'reasoning_rules_path'
                if rules_path_key not in self.config.get('paths', {}):
                    self.logger.error(f"'{rules_path_key}' not found in configuration paths.")
                    raise KeyError(f"'{rules_path_key}' not found in configuration.")

                rules_path = self.config['paths'][rules_path_key]

                # Check path existence *after* potential resolution
                if not os.path.exists(rules_path):
                    self.logger.error(f"Rules file not found at resolved path: {rules_path}")
                    raise FileNotFoundError(f"Rules file not found at: {rules_path}")

                self.logger.info(f"Initializing Symbolic Reasoner with rules from: {rules_path}")

                # Define input_shape for reasoner
                input_shape = (self.window_size, len(self.feature_names))

                # Ensure the model is available before passing to reasoner
                if self.cnn_lstm is None:
                    self.logger.error("CNN-LSTM model is not loaded, cannot initialize SymbolicReasoner with model.")
                    raise ValueError("CNN-LSTM model required for reasoner initialization but not available.")

                reasoner = SymbolicReasoner(
                    rules_path=rules_path,
                    input_shape=input_shape,
                    model=self.cnn_lstm, # Pass the loaded model instance
                    logger=self.logger
                )
                self.logger.info("Symbolic Reasoner initialized successfully.")
                return reasoner
            else:
                self.logger.info("Symbolic Reasoning is disabled in the configuration.")
                return None
        except FileNotFoundError as fnf:
             self.logger.error(f"Initialization failed: {fnf}")
             raise
        except KeyError as ke:
             self.logger.error(f"Initialization failed due to missing config key: {ke}")
             raise
        except ValueError as ve:
             self.logger.error(f"Initialization failed due to value error: {ve}")
             raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Symbolic Reasoner: {e}", exc_info=True)
            raise

    def preprocess_data(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Preprocess sensor data for models with proper shape handling.

        Args:
            sensor_data (np.ndarray): Input data of shape (timesteps, features) or (batch, timesteps, features).

        Returns:
            np.ndarray: Preprocessed data with correct shape (batch, timesteps, features).
        """
        try:
            # Ensure input is a NumPy array
            if not isinstance(sensor_data, np.ndarray):
                sensor_data = np.array(sensor_data, dtype=np.float32)
            else:
                # Ensure correct dtype if already an ndarray
                sensor_data = sensor_data.astype(np.float32)

            # Reshape 2D input to 3D (add batch dimension)
            if sensor_data.ndim == 2:
                # Check if shape matches (timesteps, features)
                if sensor_data.shape[0] != self.window_size or sensor_data.shape[1] != len(self.feature_names):
                    raise ValueError(f"Expected 2D shape ({self.window_size}, {len(self.feature_names)}), got {sensor_data.shape}")
                sensor_data = np.expand_dims(sensor_data, axis=0) # Shape becomes (1, timesteps, features)
            elif sensor_data.ndim != 3:
                raise ValueError(f"Expected 2D or 3D input, got {sensor_data.ndim}D shape {sensor_data.shape}")

            # Validate the shape after potential reshaping
            expected_inner_shape = (self.window_size, len(self.feature_names))
            if sensor_data.shape[1:] != expected_inner_shape:
                raise ValueError(
                    f"Expected inner shape (timesteps, features) to be {expected_inner_shape}, but got {sensor_data.shape[1:]} in final shape {sensor_data.shape}"
                )

            # Check for NaN/Inf values after potential type conversion
            if np.isnan(sensor_data).any() or np.isinf(sensor_data).any():
                 self.logger.warning("NaN or Inf detected in input sensor data. Attempting to replace with zeros.")
                 sensor_data = np.nan_to_num(sensor_data, nan=0.0, posinf=0.0, neginf=0.0)


            return sensor_data

        except ValueError as ve:
             self.logger.error(f"Data shape/type validation failed during preprocessing: {ve}")
             raise
        except Exception as e:
            self.logger.error(f"Error in preprocessing data: {e}", exc_info=True)
            raise

    def _prepare_ppo_observation(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare observation for PPO agent with correct shape handling.
        Handles potential batch dimension.

        Args:
            data (np.ndarray): Input data of shape (batch, window_size, features).

        Returns:
            np.ndarray: Observation ready for PPO predict (batch, window_size*features) or (window_size, features)
                       depending on SB3 requirements (often expects features flattened or specific shape).
                       This implementation returns the standard (window_size, features) if batch=1,
                       or the full batch otherwise (assuming SB3 handles it or expects VecEnv).
                       *Check SB3 documentation for specific PPO policy input requirements.*
        """
        try:
            if data.ndim != 3:
                raise ValueError(f"Expected 3D input (batch, window_size, features), got shape {data.shape}")

            # If batch size is 1, return the single observation (window_size, features)
            # Some SB3 policies might expect this directly if not using VecEnv internally
            if data.shape[0] == 1:
                return data[0]
            # If batch size > 1, return the full batch (SB3 usually handles this with VecEnvs)
            else:
                return data

        except ValueError as ve:
             self.logger.error(f"Input shape error preparing PPO observation: {ve}")
             raise
        except Exception as e:
            self.logger.error(f"Error preparing PPO observation: {e}", exc_info=True)
            raise

    def update_state(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Update system state using all components: neural, symbolic, and adaptive control.

        Args:
            sensor_data (np.ndarray): Input sensor data of shape (timesteps, features)
                                      or (batch, timesteps, features). Expects unscaled data.

        Returns:
            Dict[str, Any]: Complete system state including predictions, insights,
                           control parameters, and history

        Raises:
            ValueError: If sensor data shape is invalid
            RuntimeError: If state update fails
        """
        try:
            # --- 1. Preprocessing ---
            # Note: Assumes sensor_data is raw and needs scaling if model was trained on scaled data.
            #       If the pipeline requires raw data for PPO/Symbolic but scaled for CNN-LSTM,
            #       scaling needs to happen just before CNN-LSTM prediction.
            #       Assuming here the models work on potentially unscaled data or scaling is handled internally/beforehand.
            #       Let's preprocess shape first.
            try:
                data = self.preprocess_data(sensor_data) # Ensures correct shape (batch, win, feat)
                self.logger.debug(f"Shape-Preprocessed data shape: {data.shape}")
                # TODO: Add scaling here if CNN-LSTM requires it and `data` is raw
                # scaler = load_scaler(...) # Load the scaler used during training
                # data_scaled_for_cnn = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
                # Use data_scaled_for_cnn for cnn_lstm.predict, use raw `data` for PPO/Symbolic state extraction
            except Exception as prep_error:
                raise RuntimeError(f"Data preprocessing failed: {str(prep_error)}") from prep_error

            # --- 2. Anomaly Prediction (Neural) ---
            try:
                # Use the appropriately scaled data if scaling was done above
                # anomaly_scores = self.cnn_lstm.predict(data_scaled_for_cnn, verbose=0)
                anomaly_scores = self.cnn_lstm.predict(data, verbose=0) # Assuming model handles raw or is trained on raw for now
                # Ensure prediction output is handled correctly (e.g., for batch > 1)
                prediction_confidence = float(anomaly_scores[0][0]) if anomaly_scores.ndim > 1 else float(anomaly_scores[0])
                anomaly_detected = prediction_confidence > 0.5 # Threshold from config?
                self.logger.debug(f"Anomaly prediction - Score: {prediction_confidence:.3f}, Detected: {anomaly_detected}")
            except Exception as pred_error:
                self.logger.error(f"Anomaly prediction failed: {pred_error}", exc_info=True)
                raise RuntimeError(f"Anomaly prediction failed: {str(pred_error)}") from pred_error

            # --- 3. Action Recommendation (RL) ---
            try:
                # Prepare observation for PPO agent
                obs = self._prepare_ppo_observation(data) # Get shape PPO expects
                # Use deterministic=True for consistent action during inference, False during exploration/training
                action, _ = self.ppo_agent.predict(obs, deterministic=True)
                # Ensure action is numpy array for clipping
                action = np.array(action)
                # Clip action to valid range defined by action space? Assumed [-5, 5] here.
                action_validated = np.clip(action, -5.0, 5.0)
                self.logger.debug(f"PPO action generated: {action_validated.tolist()}")
            except Exception as ppo_error:
                self.logger.error(f"PPO prediction failed: {ppo_error}", exc_info=True)
                # Define a default safe action on failure
                action_validated = np.zeros(self.ppo_agent.action_space.shape)
                self.logger.warning(f"PPO prediction failed. Using default action: {action_validated.tolist()}")
                # Optional: Decide if this should be a fatal error (raise RuntimeError)

            # --- 4. Extract Current State for Reasoning ---
            try:
                 # Extract from the *last* timestep of the *first* batch element
                 last_timestep_data = data[0, -1, :]
                 current_readings = self._extract_current_state(last_timestep_data)
                 # Optional: Add range checks as warnings
                 # if not (15 < current_readings['temperature'] < 90 ...):
                 #     self.logger.warning("Sensor readings outside expected operational ranges.")
            except IndexError:
                 self.logger.error(f"IndexError extracting current state from data shape {data.shape}. Need at least {self.window_size} timesteps.")
                 raise RuntimeError(f"State extraction failed due to insufficient data.")
            except Exception as extract_error:
                 self.logger.error(f"State extraction failed: {extract_error}", exc_info=True)
                 raise RuntimeError(f"State extraction failed: {str(extract_error)}") from extract_error

            # --- 5. Symbolic Reasoning ---
            insights = []
            rule_activations_current_step = [] # Store activations for this specific step
            state_info = {} # Placeholder for state tracker info

            if self.reasoner:
                try:
                    # Pass the dictionary of current sensor readings
                    insights = self.reasoner.reason(current_readings)
                    # Get rule activations specifically from the *last* call to reason()
                    # This relies on the reasoner storing the last step's activations separately
                    # or modifying get_rule_activations to return only the last step.
                    # Let's assume get_rule_activations returns the full history for now (fix suggested elsewhere)
                    full_activation_history = self.reasoner.get_rule_activations()
                    if full_activation_history:
                        rule_activations_current_step = full_activation_history[-1].get('activated_rules_detailed', [])

                    # Update state tracker if it's part of the reasoner
                    if hasattr(self.reasoner, 'state_tracker'):
                        state_info = self.reasoner.state_tracker.update(current_readings)

                    self.logger.debug(f"Symbolic insights generated: {len(insights)}")
                    self.logger.debug(f"Rules activated this step: {len(rule_activations_current_step)}")

                except Exception as reason_error:
                    self.logger.warning(f"Symbolic reasoning error: {str(reason_error)}", exc_info=True)
                    # Continue execution, but log the warning

            # --- 6. Adaptive Control (Parameter Adjustment) ---
            # This seems less about direct action and more about adjusting system parameters/targets
            try:
                control_params = self.adaptive_controller.adapt_control_parameters(
                    current_state=current_readings, # Use the extracted dictionary
                    predictions=np.array([prediction_confidence]), # Pass prediction score
                    rule_activations=rule_activations_current_step # Pass activations for this step
                )
                # Apply clipping/validation to control parameters
                control_params = {
                    'temperature_adjustment': np.clip(control_params.get('temperature_adjustment', 0.0), -5.0, 5.0),
                    'vibration_adjustment': np.clip(control_params.get('vibration_adjustment', 0.0), -5.0, 5.0),
                    'pressure_adjustment': np.clip(control_params.get('pressure_adjustment', 0.0), -5.0, 5.0),
                    'efficiency_target': np.clip(control_params.get('efficiency_target', 0.8), 0.0, 1.0)
                }
                self.logger.debug(f"Adaptive control parameters updated: {control_params}")
            except Exception as control_error:
                self.logger.warning(f"Adaptive control parameter update error: {str(control_error)}", exc_info=True)
                # Use default control params on error
                control_params = self.control_params.copy() # Use defaults

            # --- 7. Compile Final State Dictionary ---
            try:
                # Use the validated sensor readings from step 4
                sensor_readings_dict = current_readings

                self.current_state = {
                    'anomaly_score': prediction_confidence,
                    'anomaly_detected': anomaly_detected,
                    'recommended_action': action_validated.tolist(), # RL action
                    'control_parameters': control_params, # Adaptive controller params
                    'sensor_readings': sensor_readings_dict, # From step 4
                    'system_state': sensor_readings_dict.get('system_state', 0), # Extracted state
                    'system_health': { # Extracted metrics
                        'efficiency_index': sensor_readings_dict.get('efficiency_index', 0.0),
                        'performance_score': sensor_readings_dict.get('performance_score', 0.0)
                    },
                    'insights': insights, # From step 5
                    'rule_activations': rule_activations_current_step, # Activations *this step*
                    'state_transitions': state_info.get('transition_matrix', []), # From state tracker
                    'timestamp': str(np.datetime64('now')), # Current timestamp
                    'confidence': prediction_confidence, # Alias for anomaly_score
                    'history_length': len(self.state_history),
                    # Add key sensors directly for easier access (e.g., by KG)
                    'temperature': sensor_readings_dict.get('temperature', 70.0),
                    'vibration': sensor_readings_dict.get('vibration', 50.0),
                    'pressure': sensor_readings_dict.get('pressure', 30.0),
                    'operational_hours': sensor_readings_dict.get('operational_hours', 0.0),
                    'efficiency_index': sensor_readings_dict.get('efficiency_index', 0.8),
                    'performance_score': sensor_readings_dict.get('performance_score', 80.0)
                }
            except Exception as state_error:
                self.logger.error(f"State compilation failed: {state_error}", exc_info=True)
                raise RuntimeError(f"State compilation failed: {str(state_error)}") from state_error

            # --- 8. Update State History ---
            try:
                self.state_history.append(self.current_state)
                # Limit history size
                if len(self.state_history) > self.state_history_limit:
                    self.state_history.pop(0)
            except Exception as history_error:
                self.logger.warning(f"State history update failed: {str(history_error)}")

            # --- 9. Logging ---
            if anomaly_detected:
                self.logger.info(
                    f"Anomaly detected (conf: {prediction_confidence:.2f}). Action: {action_validated.tolist()}. "
                    f"Insights: {insights}. Control Params: {control_params}."
                )
            else:
                self.logger.info(f"Normal operation (conf: {prediction_confidence:.2f}). Action: {action_validated.tolist()}.")


            return self.current_state

        except (ValueError, RuntimeError) as vr_error:
             # Log errors already raised for flow control or validation
             self.logger.error(f"State update failed due to validation/runtime error: {vr_error}", exc_info=True)
             # Optionally create a minimal error state or re-raise
             raise # Re-raise critical errors
        except Exception as e:
            self.logger.error(f"Critical unexpected error in update_state: {e}", exc_info=True)
            # Define a minimal error state to return or raise
            error_state = {
                'error': f"Critical unexpected error: {str(e)}",
                'anomaly_detected': True, # Assume worst case on error
                'recommended_action': [0.0, 0.0, 0.0],
                'control_parameters': self.control_params.copy(),
                'timestamp': str(np.datetime64('now')),
                'status': 'error'
            }
            # Decide whether to return error state or raise exception
            # return error_state
            raise RuntimeError(f"Failed to update system state: {str(e)}") from e

    def adapt_and_explain(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Run state update, trigger rule learning, and generate decision dictionary.
        NOTE: This method now focuses on orchestrating the update and rule learning trigger.
              Explanation generation is moved to the ExplainableANSRDT subclass.

        Args:
            sensor_data (np.ndarray): Input sensor data (raw).

        Returns:
            Dict[str, Any]: The updated system state dictionary.
        """
        try:
            # --- 1. Update the system state ---
            state = self.update_state(sensor_data) # This performs prediction, action, reasoning

            # --- 2. Trigger Dynamic Rule Learning (if anomaly detected) ---
            # Use a threshold from config, e.g., config['symbolic_reasoning']['extraction_threshold']
            extraction_threshold = self.config.get('symbolic_reasoning', {}).get('extraction_threshold', 0.7)
            if state.get('anomaly_detected', False) and state.get('anomaly_score', 0.0) >= extraction_threshold:
                if self.reasoner:
                    self.logger.info(f"Anomaly score {state['anomaly_score']:.2f} >= {extraction_threshold}. Triggering rule extraction.")
                    # Pass the raw input data used for this state update
                    input_data_for_rules = self.preprocess_data(sensor_data) # Ensure correct shape
                    new_rules = self.reasoner.extract_rules_from_neural_model(
                        model=self.cnn_lstm,
                        input_data=input_data_for_rules,
                        feature_names=self.feature_names,
                        threshold=extraction_threshold # Use the configured threshold
                    )
                    if new_rules:
                        update_confidence = self.config.get('symbolic_reasoning', {}).get('min_confidence', 0.7)
                        self.reasoner.update_rules(new_rules, min_confidence=update_confidence)
                        self.logger.info(f"Attempted to update with {len(new_rules)} new rules (min_conf: {update_confidence}).")
                else:
                     self.logger.warning("Anomaly detected but Symbolic Reasoner is not enabled. Skipping rule extraction.")

            # --- 3. Return the comprehensive state dictionary ---
            # The explanation generation is handled by the calling class (ExplainableANSRDT)
            return state

        except (ValueError, RuntimeError) as vr_error:
             self.logger.error(f"Error in adapt_and_explain orchestration: {vr_error}", exc_info=True)
             # Return an error dictionary or re-raise
             return {'error': str(vr_error), 'status': 'error'}
        except Exception as e:
            self.logger.error(f"Unexpected error in adapt_and_explain: {e}", exc_info=True)
             # Return an error dictionary or re-raise
            return {'error': f"Unexpected error: {str(e)}", 'status': 'error'}


    def _generate_explanation(self, state: Dict[str, Any]) -> str:
        """
        [DEPRECATED in Core - Moved to ExplainableANSRDT]
        Generate detailed explanation of system state and actions.
        Kept here temporarily for reference, should be removed or marked deprecated.
        """
        self.logger.warning("_generate_explanation called in ANSRDTCore. This logic is now in ExplainableANSRDT.")
        explanation = []
        # Basic fallback explanation
        explanation.append(f"State Timestamp: {state.get('timestamp', 'N/A')}")
        explanation.append(f"Anomaly Detected: {state.get('anomaly_detected', 'Unknown')}")
        explanation.append(f"Confidence: {state.get('confidence', 0.0):.2%}")
        if state.get('action'):
            explanation.append(f"Action: {state['action']}")
        return " | ".join(explanation)


    def save_state(self, output_path: str):
        """
        Save current state and recent history to a JSON file.

        Args:
            output_path (str): Path to save the state file.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Use NumpyEncoder if needed for complex state objects
            # from src.pipeline.pipeline import NumpyEncoder # Import if needed
            state_data = {
                'current_state': self.current_state,
                'state_history_tail': self.state_history[-100:], # Save only tail of history
                'timestamp': str(np.datetime64('now'))
            }
            with open(output_path, 'w') as f:
                # Add cls=NumpyEncoder if state contains numpy types
                json.dump(state_data, f, indent=2)
            self.logger.info(f"Core state saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving core state: {e}", exc_info=True)
            # Do not raise here, saving state is not critical to core function

    def load_state(self, input_path: str):
        """
        Load saved state from a JSON file.

        Args:
            input_path (str): Path to the state file.
        """
        try:
            if not os.path.exists(input_path):
                 self.logger.error(f"State file not found for loading: {input_path}")
                 return # Or raise error?

            with open(input_path, 'r') as f:
                state_data = json.load(f)
            # Basic validation
            if 'current_state' not in state_data or 'state_history_tail' not in state_data:
                 self.logger.error(f"Invalid state file format in {input_path}")
                 return

            self.current_state = state_data['current_state']
            # Overwrite history with the loaded tail - full history is lost
            self.state_history = state_data['state_history_tail']
            self.logger.info(f"Core state loaded from {input_path}. History reset to loaded tail.")
        except Exception as e:
            self.logger.error(f"Error loading core state: {e}", exc_info=True)
            # Do not raise usually, allow system to continue with default state

    def integrated_inference(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        [DEPRECATED - Use update_state or adapt_and_explain]
        Run integrated inference combining all components: anomaly detection,
        adaptive control, and symbolic reasoning.
        """
        self.logger.warning("integrated_inference is deprecated. Use update_state or adapt_and_explain instead.")
        # Simply call update_state as it now performs the integrated steps
        return self.update_state(sensor_data)


    def _extract_current_state(self, sensor_values: np.ndarray) -> Dict[str, float]:
        """
        Extract current state dictionary from the last timestep's sensor values array.
        Uses self.feature_names for indexing.

        Args:
            sensor_values (np.ndarray): 1D array of sensor values for the last timestep.

        Returns:
            Dict[str, Any]: Dictionary of current sensor readings and metrics.
                           Values are cast to appropriate types (float, int).
        """
        state_dict = {}
        if len(sensor_values) != len(self.feature_names):
             self.logger.error(f"Mismatch between sensor_values length ({len(sensor_values)}) and feature_names count ({len(self.feature_names)}).")
             # Return empty or default dict?
             return {name: 0.0 for name in self.feature_names} # Basic fallback

        try:
             for i, name in enumerate(self.feature_names):
                  value = sensor_values[i]
                  # Handle potential NaN/Inf before casting
                  if np.isnan(value) or np.isinf(value):
                       self.logger.warning(f"NaN/Inf found for feature '{name}'. Replacing with 0.")
                       value = 0.0

                  # Cast to appropriate type based on feature name convention
                  if name == 'system_state':
                      state_dict[name] = int(round(float(value))) # Ensure it's an integer 0, 1, 2
                  elif name == 'operational_hours':
                       state_dict[name] = int(round(float(value)))
                  else:
                       state_dict[name] = float(value)
             return state_dict
        except Exception as e:
            self.logger.error(f"Error extracting state dictionary: {e}", exc_info=True)
            # Return default dictionary on error
            return {name: 0 for name in self.feature_names}