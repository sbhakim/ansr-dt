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
        self.results_dir = os.path.join(self.project_root, self.config['paths']['results_dir'])

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
            self.logger.debug(f"Window Size: {self.window_size}")
            self.logger.debug(f"Feature Names: {self.feature_names}")
        except KeyError as ke:
            self.logger.error(f"Missing key in configuration: {ke}")
            raise

        # Initialize symbolic reasoner with correct path
        self.reasoner = self._initialize_reasoner()

        # Initialize state tracking
        self.current_state = None
        self.state_history = []
        self.decision_history = []

    def _load_cnn_lstm_model(self) -> tf.keras.Model:
        """
        Load the trained CNN-LSTM model from the specified path.

        Returns:
            tf.keras.Model: Loaded CNN-LSTM model.
        """
        try:
            model_path = os.path.join(self.results_dir, 'best_model.keras')
            self.logger.info(f"Loading CNN-LSTM model from: {model_path}")
            model = load_model(model_path, self.logger)
            self.logger.info("CNN-LSTM model loaded successfully.")
            return model
        except FileNotFoundError as fe:
            self.logger.error(f"CNN-LSTM model file not found: {fe}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load CNN-LSTM model: {e}")
            raise

    def _load_ppo_agent(self) -> PPO:
        """
        Load the trained PPO agent from the specified path.

        Returns:
            PPO: Loaded PPO agent.
        """
        try:
            ppo_path = os.path.join(self.results_dir, 'ppo_ansr_dt.zip')
            self.logger.info(f"Loading PPO agent from: {ppo_path}")
            if not os.path.exists(ppo_path):
                raise FileNotFoundError(f"PPO model not found at {ppo_path}")
            ppo_agent = PPO.load(ppo_path)
            self.logger.info("PPO agent loaded successfully.")
            return ppo_agent
        except FileNotFoundError as fe:
            self.logger.error(f"PPO agent file not found: {fe}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load PPO agent: {e}")
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
                # Resolve rules path
                rules_path = self.config['paths'].get('reasoning_rules_path')
                if not rules_path:
                    self.logger.error("Reasoning rules path not specified in configuration.")
                    raise KeyError("reasoning_rules_path not found in configuration.")

                if not os.path.isabs(rules_path):
                    rules_path = os.path.join(self.project_root, rules_path)

                if not os.path.exists(rules_path):
                    self.logger.error(f"Rules file not found at: {rules_path}")
                    raise FileNotFoundError(f"Rules file not found at: {rules_path}")

                self.logger.info(f"Initializing Symbolic Reasoner with rules from: {rules_path}")

                # Define input_shape for reasoner
                input_shape = (self.window_size, len(self.feature_names))

                reasoner = SymbolicReasoner(
                    rules_path=rules_path,
                    input_shape=input_shape,
                    model=self.cnn_lstm,
                    logger=self.logger
                )
                self.logger.info("Symbolic Reasoner initialized successfully.")
                return reasoner
            else:
                self.logger.info("Symbolic Reasoning is disabled in the configuration.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to initialize Symbolic Reasoner: {e}")
            raise

    def preprocess_data(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Preprocess sensor data for models with proper shape handling.

        Args:
            sensor_data (np.ndarray): Input data of shape (timesteps, features) or (batch, timesteps, features).

        Returns:
            np.ndarray: Preprocessed data with correct shape.
        """
        try:
            sensor_data = np.array(sensor_data, dtype=np.float32)

            if len(sensor_data.shape) == 2:
                sensor_data = np.expand_dims(sensor_data, axis=0)
            elif len(sensor_data.shape) != 3:
                raise ValueError(f"Expected 2D or 3D input, got shape {sensor_data.shape}")

            expected_shape = (self.window_size, len(self.feature_names))
            if sensor_data.shape[1:] != expected_shape:
                raise ValueError(
                    f"Expected shape (batch, {self.window_size}, {len(self.feature_names)}), got {sensor_data.shape}"
                )

            return sensor_data

        except Exception as e:
            self.logger.error(f"Error in preprocessing data: {e}")
            raise

    def _prepare_ppo_observation(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare observation for PPO agent with correct shape handling.

        Args:
            data (np.ndarray): Input data of shape (batch, window_size, features).

        Returns:
            np.ndarray: Properly shaped observation for PPO.
        """
        try:
            if len(data.shape) != 3:
                raise ValueError(f"Expected 3D input, got shape {data.shape}")
            if data.shape[0] == 1:
                return data[0]
            return data

        except Exception as e:
            self.logger.error(f"Error preparing PPO observation: {e}")
            raise

    def update_state(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Update system state using all components: neural, symbolic, and adaptive control.

        Args:
            sensor_data (np.ndarray): Input sensor data of shape (timesteps, features)
                                      or (batch, timesteps, features)

        Returns:
            Dict[str, Any]: Complete system state including predictions, insights,
                           control parameters, and history

        Raises:
            ValueError: If sensor data shape is invalid
            RuntimeError: If state update fails
        """
        try:
            if not isinstance(sensor_data, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(sensor_data)}")

            if len(sensor_data.shape) not in [2, 3]:
                raise ValueError(f"Expected 2D or 3D array, got shape {sensor_data.shape}")

            try:
                data = self.preprocess_data(sensor_data)
                self.logger.debug(f"Preprocessed data shape: {data.shape}")
            except Exception as prep_error:
                raise RuntimeError(f"Data preprocessing failed: {str(prep_error)}")

            try:
                anomaly_scores = self.cnn_lstm.predict(data, verbose=0)
                anomaly_detected = float(anomaly_scores[0]) > 0.5
                prediction_confidence = float(anomaly_scores[0])
                self.logger.debug(f"Anomaly prediction - Score: {prediction_confidence:.3f}, Detected: {anomaly_detected}")
            except Exception as pred_error:
                raise RuntimeError(f"Anomaly prediction failed: {str(pred_error)}")

            try:
                obs = self._prepare_ppo_observation(data)
                action, _ = self.ppo_agent.predict(obs, deterministic=not anomaly_detected)
                action_validated = np.clip(action, -5.0, 5.0)
                self.logger.debug(f"PPO action generated: {action_validated.tolist()}")
            except Exception as ppo_error:
                raise RuntimeError(f"PPO prediction failed: {str(ppo_error)}")

            try:
                current_readings = self._extract_current_state(data[0, -1])
                if not (15 < current_readings['temperature'] < 90 and
                        10 < current_readings['vibration'] < 65 and
                        18 < current_readings['pressure'] < 50):
                    self.logger.warning("Sensor readings outside expected ranges")
            except Exception as extract_error:
                raise RuntimeError(f"State extraction failed: {str(extract_error)}")

            # Enhance state extraction: try to extract full sensor readings from last timestep
            try:
                if data.shape[1] >= self.window_size and data.shape[2] >= len(self.feature_names):
                    last_timestep = data[0, -1, :]
                    sensor_readings = {
                        'temperature': float(last_timestep[self.feature_names.index('temperature')] if 'temperature' in self.feature_names else 70.0),
                        'vibration': float(last_timestep[self.feature_names.index('vibration')] if 'vibration' in self.feature_names else 50.0),
                        'pressure': float(last_timestep[self.feature_names.index('pressure')] if 'pressure' in self.feature_names else 30.0),
                        'operational_hours': float(last_timestep[self.feature_names.index('operational_hours')] if 'operational_hours' in self.feature_names else 0.0),
                        'efficiency_index': float(last_timestep[self.feature_names.index('efficiency_index')] if 'efficiency_index' in self.feature_names else 0.8),
                        'system_state': int(float(last_timestep[self.feature_names.index('system_state')] if 'system_state' in self.feature_names else 0)),
                        'performance_score': float(last_timestep[self.feature_names.index('performance_score')] if 'performance_score' in self.feature_names else 80.0)
                    }
                    # Validate numeric values and assign defaults if necessary
                    for key, value in sensor_readings.items():
                        if isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
                            if key == 'system_state':
                                sensor_readings[key] = 0
                            elif key == 'temperature':
                                sensor_readings[key] = 70.0
                            elif key == 'vibration':
                                sensor_readings[key] = 50.0
                            elif key == 'pressure':
                                sensor_readings[key] = 30.0
                            elif key == 'efficiency_index':
                                sensor_readings[key] = 0.8
                            elif key == 'performance_score':
                                sensor_readings[key] = 80.0
                            else:
                                sensor_readings[key] = 0.0
                else:
                    sensor_readings = current_readings
            except Exception as state_err:
                self.logger.warning(f"Enhanced state extraction failed: {state_err}. Using current_readings.")
                sensor_readings = current_readings

            insights = []
            rule_activations = []
            state_info = {}

            if self.reasoner:
                try:
                    insights = self.reasoner.reason(sensor_readings)
                    rule_activations = self.reasoner.get_rule_activations() if hasattr(self.reasoner, 'get_rule_activations') else []
                    if hasattr(self.reasoner, 'state_tracker'):
                        state_info = self.reasoner.state_tracker.update(sensor_readings)
                    self.logger.debug(f"Symbolic insights generated: {len(insights)}")
                except Exception as reason_error:
                    self.logger.warning(f"Symbolic reasoning error: {str(reason_error)}")

            try:
                control_params = self.adaptive_controller.adapt_control_parameters(
                    current_state=sensor_readings,
                    predictions=anomaly_scores,
                    rule_activations=rule_activations
                )
            except Exception as control_error:
                self.logger.warning(f"Adaptive control error: {str(control_error)}")
                control_params = {
                    'temperature_adjustment': 0.0,
                    'vibration_adjustment': 0.0,
                    'pressure_adjustment': 0.0,
                    'efficiency_target': sensor_readings.get('efficiency_index', 0.8)
                }

            control_params = {
                'temperature_adjustment': np.clip(control_params.get('temperature_adjustment', 0.0), -5.0, 5.0),
                'vibration_adjustment': np.clip(control_params.get('vibration_adjustment', 0.0), -5.0, 5.0),
                'pressure_adjustment': np.clip(control_params.get('pressure_adjustment', 0.0), -5.0, 5.0),
                'efficiency_target': np.clip(control_params.get('efficiency_target', 0.8), 0.0, 1.0)
            }

            try:
                self.current_state = {
                    'anomaly_score': prediction_confidence,
                    'anomaly_detected': anomaly_detected,
                    'recommended_action': action_validated.tolist(),
                    'control_parameters': control_params,
                    'sensor_readings': sensor_readings,
                    'system_state': sensor_readings.get('system_state', 0),
                    'system_health': {
                        'efficiency_index': sensor_readings.get('efficiency_index', 0.0),
                        'performance_score': sensor_readings.get('performance_score', 0.0)
                    },
                    'insights': insights,
                    'rule_activations': rule_activations,
                    'state_transitions': state_info.get('transition_matrix', []),
                    'timestamp': str(np.datetime64('now')),
                    'confidence': prediction_confidence,
                    'history_length': len(self.state_history),
                    # Directly include key sensor fields for the knowledge graph
                    'temperature': sensor_readings.get('temperature', 70.0),
                    'vibration': sensor_readings.get('vibration', 50.0),
                    'pressure': sensor_readings.get('pressure', 30.0),
                    'operational_hours': sensor_readings.get('operational_hours', 0.0),
                    'efficiency_index': sensor_readings.get('efficiency_index', 0.8),
                    'performance_score': sensor_readings.get('performance_score', 80.0)
                }
            except Exception as state_error:
                raise RuntimeError(f"State compilation failed: {str(state_error)}")

            try:
                self.state_history.append(self.current_state)
                if len(self.state_history) > 1000:
                    self.state_history.pop(0)
            except Exception as history_error:
                self.logger.warning(f"State history update failed: {str(history_error)}")

            if anomaly_detected:
                self.logger.info(
                    f"Anomaly detected (confidence: {prediction_confidence:.2f}) - "
                    f"Control adjustments: Temperature ({control_params['temperature_adjustment']:.2f}), "
                    f"Vibration ({control_params['vibration_adjustment']:.2f}), "
                    f"Pressure ({control_params['pressure_adjustment']:.2f})"
                )

            if insights:
                self.logger.info(f"Symbolic insights generated: {insights}")

            return self.current_state

        except Exception as e:
            self.logger.error(f"Critical error in state update: {str(e)}")
            default_state = {
                'error': str(e),
                'anomaly_detected': True,
                'control_parameters': {
                    'temperature_adjustment': 0.0,
                    'vibration_adjustment': 0.0,
                    'pressure_adjustment': 0.0,
                    'efficiency_target': 0.8
                },
                'timestamp': str(np.datetime64('now')),
                'status': 'error'
            }
            raise RuntimeError(f"Failed to update system state: {str(e)}") from e

    def adapt_and_explain(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate actions and explanations based on sensor data.

        Args:
            sensor_data (np.ndarray): Input sensor data.

        Returns:
            Dict[str, Any]: Decision with action and explanation.
        """
        try:
            state = self.update_state(sensor_data)
            if state['anomaly_score'] > 0.5 and self.reasoner:
                new_rules = self.reasoner.extract_rules_from_neural_model(
                    model=self.cnn_lstm,
                    input_data=sensor_data,
                    feature_names=self.feature_names,
                    threshold=0.7
                )
                if new_rules:
                    self.reasoner.update_rules(new_rules, min_confidence=0.7)
                    self.logger.info(f"Extracted {len(new_rules)} new rules from current prediction")
            decision = {
                'timestamp': state['timestamp'],
                'action': None,
                'explanation': 'Normal operation',
                'confidence': state['anomaly_score']
            }
            if state['anomaly_score'] > 0.5:
                decision.update({
                    'action': state['recommended_action'],
                    'explanation': self._generate_explanation(state),
                    'insights': state['insights']
                })
            return decision

        except Exception as e:
            self.logger.error(f"Error in adapt_and_explain: {e}")
            raise

    def _generate_explanation(self, state: Dict[str, Any]) -> str:
        """
        Generate detailed explanation of system state and actions.

        Args:
            state (Dict[str, Any]): Current state information.

        Returns:
            str: Generated explanation string.
        """
        explanation = []
        explanation.append(f"Anomaly detected with {state['anomaly_score']:.2%} confidence.")
        if state['insights']:
            explanation.append("System insights: " + ", ".join(state['insights']))
        action = state['recommended_action']
        explanation.append(
            f"Recommended adjustments: Temperature ({action[0]:.2f}), Vibration ({action[1]:.2f}), Pressure ({action[2]:.2f})."
        )
        return " | ".join(explanation)

    def save_state(self, output_path: str):
        """
        Save current state and history to a JSON file.

        Args:
            output_path (str): Path to save the state file.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            state_data = {
                'current_state': self.current_state,
                'state_history': self.state_history[-100:],
                'timestamp': str(np.datetime64('now'))
            }
            with open(output_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            self.logger.info(f"State saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            raise

    def load_state(self, input_path: str):
        """
        Load saved state from a JSON file.

        Args:
            input_path (str): Path to the state file.
        """
        try:
            with open(input_path, 'r') as f:
                state_data = json.load(f)
            self.current_state = state_data['current_state']
            self.state_history = state_data['state_history']
            self.logger.info(f"State loaded from {input_path}")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            raise

    def integrated_inference(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Run integrated inference combining all components: anomaly detection,
        adaptive control, and symbolic reasoning.

        Args:
            sensor_data (np.ndarray): Input sensor data.

        Returns:
            Dict[str, Any]: Comprehensive inference results.
        """
        try:
            data = self.preprocess_data(sensor_data)
            anomaly_scores = self.cnn_lstm.predict(data, verbose=0)
            anomaly_detected = anomaly_scores[0] > 0.5
            current_state = self._extract_current_state(data[0, -1])
            if self.reasoner:
                symbolic_insights = self.reasoner.reason(current_state)
            else:
                symbolic_insights = []
            obs = self._prepare_ppo_observation(data)
            action, _ = self.ppo_agent.predict(obs, deterministic=not anomaly_detected)
            result = {
                'timestamp': str(np.datetime64('now')),
                'anomaly_score': float(anomaly_scores[0]),
                'anomaly_detected': bool(anomaly_detected),
                'symbolic_insights': symbolic_insights,
                'recommended_action': action.tolist(),
                'current_state': current_state
            }
            self.current_state = result
            self.state_history.append(result)
            if len(self.state_history) > 1000:
                self.state_history.pop(0)
            return result
        except Exception as e:
            self.logger.error(f"Error in integrated inference: {e}")
            raise

    def _extract_current_state(self, sensor_values: np.ndarray) -> Dict[str, float]:
        """
        Extract current state from sensor values.

        Args:
            sensor_values (np.ndarray): Array of sensor values.

        Returns:
            Dict[str, float]: Dictionary of current sensor readings.
        """
        return {
            'temperature': float(sensor_values[0]),
            'vibration': float(sensor_values[1]),
            'pressure': float(sensor_values[2]),
            'operational_hours': float(sensor_values[3]),
            'efficiency_index': float(sensor_values[4]),
            'system_state': float(sensor_values[5]),
            'performance_score': float(sensor_values[6])
        }
