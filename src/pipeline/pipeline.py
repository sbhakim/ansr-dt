# src/pipeline/pipeline.py

import json
import logging
import os
from typing import Tuple, Dict, Any, List

import numpy as np
import tensorflow as tf

# Import components
from src.models.cnn_lstm_model import create_cnn_lstm_model
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import preprocess_sequences
from src.data.data_processing import map_labels
from src.training.trainer import train_model
from src.evaluation.evaluation import evaluate_model
from src.utils.model_utils import save_model, save_scaler
from src.visualization.plotting import load_plot_config, plot_metrics
from src.reasoning.reasoning import SymbolicReasoner

# Custom JSON Encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts NumPy data types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def validate_config(config: dict, logger: logging.Logger, project_root: str, config_dir: str):
    """Validates configuration and paths."""
    try:
        required_keys = ['model', 'training', 'paths']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                raise KeyError(f"Missing required configuration key: {key}")

        # Define required paths with their base directories
        required_paths = {
            'data_file': {'relative_to': project_root, 'default': 'data/synthetic_sensor_data_with_anomalies.npz'},
            'results_dir': {'relative_to': project_root, 'default': 'results'},
            'plot_config_path': {'relative_to': config_dir, 'default': 'plot_config.yaml'},
            'reasoning_rules_path': {'relative_to': project_root, 'default': 'src/reasoning/rules.pl'}
        }

        for key, path_info in required_paths.items():
            path = config['paths'].get(key, path_info['default'])
            base_dir = path_info['relative_to']
            full_path = os.path.join(base_dir, path)

            if key.endswith('_dir'):
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Directory '{key}' ensured at: {full_path}")
            else:
                if not os.path.exists(full_path):
                    logger.error(f"Required file '{key}' not found at: {full_path}")
                    raise FileNotFoundError(f"Required file '{key}' not found at: {full_path}")
                logger.info(f"File '{key}' found at: {full_path}")

            config['paths'][key] = full_path

        logger.info("Configuration validation passed.")

    except Exception as e:
        logger.exception(f"Configuration validation failed: {e}")
        raise


class NEXUSDTPipeline:
    """NEXUS-DT Pipeline with Neurosymbolic Rule Learning."""

    def __init__(self, config: dict, config_path: str, logger: logging.Logger):
        """Initialize pipeline."""
        self.config = config
        self.logger = logger
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.project_root = os.path.dirname(self.config_dir)

        # Initialize components
        self.data_loader = DataLoader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size']
        )

        # Add model state tracking
        self.model = None
        self.model_built = False

        # Add rule extraction configuration
        self.rule_extraction_config = {
            'confidence_threshold': 0.7,
            'importance_threshold': 0.5,
            'min_support': 3  # Minimum sequences to form a rule
        }

        # Define feature names for rule extraction
        self.feature_names = [
            'temperature', 'vibration', 'pressure', 'operational_hours',
            'efficiency_index', 'system_state', 'performance_score'
        ]

        # Configure logging for rule extraction
        self.logger.info(f"Initialized with {len(self.feature_names)} features for rule extraction")
        self.logger.info(f"Rule extraction thresholds: {self.rule_extraction_config}")

        # Initialize SymbolicReasoner as None; will initialize after model is trained
        self.reasoner = None

        # Initialize rule activations list
        self.rule_activations: List[Dict[str, Any]] = []

    def extract_neural_rules(
            self,
            model: tf.keras.Model,
            input_data: np.ndarray,
            y_pred: np.ndarray,
            threshold: float = 0.8
    ) -> None:
        """
        Extract rules from neural model predictions with enhanced pattern detection.

        Args:
            model: Trained neural network model
            input_data: Input sequences
            y_pred: Model predictions
            threshold: Confidence threshold for rule extraction
        """
        try:
            # Find anomalous and normal sequences
            anomalous_idx = np.where(y_pred > threshold)[0]
            normal_idx = np.where(y_pred < 0.2)[0]

            self.logger.info(f"Found {len(anomalous_idx)} anomalous sequences based on threshold {threshold}")

            if len(anomalous_idx) == 0:
                self.logger.info("No anomalous sequences found. No rules extracted.")
                return

            # Get sequences
            anomalous_sequences = input_data[anomalous_idx]
            normal_sequences = input_data[normal_idx]

            # Calculate gradients for all features
            gradients = {
                'temperature': np.gradient(input_data[:, :, 0], axis=1),
                'vibration': np.gradient(input_data[:, :, 1], axis=1),
                'pressure': np.gradient(input_data[:, :, 2], axis=1),
                'efficiency_index': np.gradient(input_data[:, :, 4], axis=1)
            }

            gradient_rules = []
            pattern_rules = []

            # Process each anomalous sequence
            for idx in anomalous_idx:
                sequence = input_data[idx]
                current_values = sequence[-1]  # Last timestep
                previous_values = sequence[-2] if sequence.shape[0] > 1 else current_values  # Previous timestep

                # Update feature_values dictionary
                feature_values = {
                    'temperature': float(current_values[0]),
                    'vibration': float(current_values[1]),
                    'pressure': float(current_values[2]),
                    'operational_hours': float(current_values[3]),
                    'efficiency_index': float(current_values[4]),
                    'system_state': float(current_values[5]),
                    'performance_score': float(current_values[6])
                }

                # Check for gradient-based patterns
                feature_patterns = []
                for feature, gradient in gradients.items():
                    max_grad = np.max(np.abs(gradient[idx]))
                    if max_grad > 2.0:  # Significant change threshold
                        feature_patterns.append({
                            'feature': feature,
                            'gradient': float(max_grad),
                            'value': feature_values[feature]
                        })

                # Generate gradient rules
                if feature_patterns:
                    for pattern in feature_patterns:
                        rule_conditions = []

                        # Add gradient condition
                        rule_conditions.append(
                            f"{pattern['feature']}_gradient({pattern['gradient']:.2f})"
                        )

                        # Add value condition
                        rule_conditions.append(
                            f"{pattern['feature']}({int(pattern['value'])})"
                        )

                        # Add state transition if applicable
                        if abs(current_values[5] - previous_values[5]) > 0:
                            rule_conditions.append(
                                f"state_transition({int(previous_values[5])}->{int(current_values[5])})"
                            )

                        # Create rule
                        rule_name = f"gradient_rule_{len(gradient_rules) + 1}"
                        rule_body = ", ".join(rule_conditions) + "."
                        rule = f"{rule_name} :- {rule_body}"
                        confidence = float(y_pred[idx])

                        gradient_rules.append({
                            'rule': rule,
                            'confidence': confidence,
                            'patterns': feature_patterns,
                            'timestep': idx
                        })

                # Check for combined feature patterns
                if feature_values['temperature'] > 75 and feature_values['vibration'] > 50:
                    pattern_rules.append({
                        'rule': (f"pattern_rule_{len(pattern_rules) + 1} :- "
                                 f"temperature({int(feature_values['temperature'])}), "
                                 f"vibration({int(feature_values['vibration'])})."),
                        'confidence': float(y_pred[idx]),
                        'type': 'temp_vib_correlation',
                        'timestep': idx
                    })

                if feature_values['pressure'] < 25 and feature_values['efficiency_index'] < 0.7:
                    pattern_rules.append({
                        'rule': (f"pattern_rule_{len(pattern_rules) + 1} :- "
                                 f"pressure({int(feature_values['pressure'])}), "
                                 f"efficiency_index({feature_values['efficiency_index']:.2f})."),
                        'confidence': float(y_pred[idx]),
                        'type': 'press_eff_correlation',
                        'timestep': idx
                    })

            # Extract additional patterns using sequence analysis via SymbolicReasoner
            temporal_patterns = self.reasoner.analyze_neural_patterns(
                anomalous_sequences=anomalous_sequences,
                normal_sequences=normal_sequences,
                feature_names=self.feature_names
            )

            # Combine all rules
            all_rules = []

            # Add gradient rules
            for rule in gradient_rules:
                if rule['confidence'] >= threshold:
                    all_rules.append((rule['rule'], rule['confidence']))

            # Add pattern rules
            for rule in pattern_rules:
                if rule['confidence'] >= threshold:
                    all_rules.append((rule['rule'], rule['confidence']))

            # Add temporal patterns
            for pattern in temporal_patterns:
                if pattern.get('confidence', 0) >= threshold:
                    all_rules.append((pattern['rule'], pattern['confidence']))

            # Update symbolic knowledge base
            if all_rules:
                self.update_rules(all_rules)

                # Log statistics
                stats = self.get_rule_statistics()
                self.logger.info(f"Neural rule extraction stats: {stats}")

                # Save extracted rules summary
                rules_summary = {
                    'gradient_rules': gradient_rules,
                    'pattern_rules': pattern_rules,
                    'temporal_patterns': temporal_patterns,
                    'statistics': stats,
                    'total_rules': len(all_rules),
                    'timestamp': str(np.datetime64('now'))
                }

                # Save summary
                summary_path = os.path.join(
                    self.config['paths']['results_dir'],
                    'neurosymbolic_rules.json'
                )

                # Serialize rules_summary with custom encoder
                with open(summary_path, 'w') as f:
                    json.dump(rules_summary, f, indent=2, cls=NumpyEncoder)

                self.logger.info(f"Neurosymbolic rules saved to {summary_path}")

                # Track rule activations
                self.rule_activations.extend([{
                    'rule': rule[0],
                    'confidence': rule[1],
                    'timestep': len(self.rule_activations),
                    'type': 'neural_derived'
                } for rule in all_rules])

        except Exception as e:
            self.logger.error(f"Error in neural rule extraction: {e}")
            raise

    def run(self):
        """Execute complete pipeline with neurosymbolic learning."""
        try:
            self.logger.info("Starting pipeline execution.")

            # 1. Validate Configuration
            validate_config(self.config, self.logger, self.project_root, self.config_dir)

            # 2. Load and process data
            X, y = self.data_loader.load_data()
            self.logger.info(f"Data loaded with shapes - X: {X.shape}, y: {y.shape}")

            # 3. Create sequences
            X_seq, y_seq = self.data_loader.create_sequences(X, y)
            self.logger.info(f"Sequences created with shapes - X_seq: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            # 4. Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
                X_seq, y_seq,
                validation_split=self.config['training']['validation_split'],
                test_split=self.config['training']['test_split']
            )
            self.logger.info("Data split completed.")

            # 5. Map labels to binary
            y_train_binary = map_labels(y_train, self.logger)
            y_val_binary = map_labels(y_val, self.logger)
            y_test_binary = map_labels(y_test, self.logger)

            # 6. Preprocess sequences
            scaler, X_train_scaled = preprocess_sequences(X_train)
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            # Save scaler
            scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            save_scaler(scaler, scaler_path, self.logger)

            # 7. Create model
            if self.config['model'].get('architecture', 'cnn_lstm') == 'cnn_lstm':
                self.model = create_cnn_lstm_model(
                    input_shape=X_train_scaled.shape[1:],
                    learning_rate=self.config['training']['learning_rate']
                )
            else:
                raise ValueError(f"Unsupported architecture: {self.config['model'].get('architecture')}")

            self.logger.info("Model created.")

            # Build model with dummy data
            dummy_shape = (1, self.config['model']['window_size'], len(self.feature_names))
            dummy_input = np.zeros(dummy_shape)
            _ = self.model(dummy_input, training=False)

            # Train model
            history, trained_model = train_model(
                model=self.model,
                X_train=X_train_scaled,
                y_train=y_train_binary,
                X_val=X_val_scaled,
                y_val=y_val_binary,
                config=self.config,
                results_dir=self.config['paths']['results_dir'],
                logger=self.logger
            )
            self.logger.info("Model training completed.")

            # Initialize Symbolic Reasoner with the trained model
            rules_path = self.config['paths'].get('reasoning_rules_path')
            if not os.path.isabs(rules_path):
                rules_path = os.path.join(self.project_root, rules_path)

            input_shape = (
                self.config['model']['window_size'],
                len(self.feature_names)
            )

            self.reasoner = SymbolicReasoner(
                rules_path=rules_path,
                input_shape=input_shape,
                model=trained_model,  # Pass the trained model directly
                logger=self.logger
            )
            self.logger.info("Symbolic Reasoner initialized.")

            # 8. Plot metrics
            figures_dir = os.path.join(self.config['paths']['results_dir'], 'visualization')
            os.makedirs(figures_dir, exist_ok=True)

            plot_config = load_plot_config(self.config['paths']['plot_config_path'])
            plot_metrics(history, figures_dir, plot_config)

            # 9. Evaluate model and extract rules
            y_test_pred = trained_model.predict(X_test_scaled).ravel()
            y_test_pred_classes = (y_test_pred > 0.5).astype(int)

            # Prepare sensor data for evaluation
            sensor_data_test = np.column_stack([
                X_test_scaled[:, -1, 0:7]  # Last timestep of each sequence
            ])

            # Extract rules from neural model
            self.logger.info("Extracting rules from neural model...")
            self.extract_neural_rules(
                model=trained_model,
                input_data=X_test_scaled,
                y_pred=y_test_pred,
                threshold=0.8
            )

            # Evaluate with both neural and symbolic components
            evaluate_model(
                y_true=y_test_binary,
                y_pred=y_test_pred_classes,
                y_scores=y_test_pred,
                figures_dir=figures_dir,
                plot_config_path=self.config['paths']['plot_config_path'],
                config_path=self.config_dir,
                sensor_data=sensor_data_test,
                model=trained_model
            )

            # 10. Save final model
            best_model_path = os.path.join(self.config['paths']['results_dir'], 'best_model.keras')
            save_model(trained_model, best_model_path, self.logger)

            self.logger.info("Pipeline completed successfully.")

        except Exception as e:
            self.logger.exception(f"Pipeline error: {e}")
            raise

    def update_rules(self, rules: List[Tuple[str, float]]):
        """
        Update the Prolog rules file with new rules.

        Args:
            rules: List of tuples containing rule strings and their confidence scores.
        """
        try:
            rules_file_path = self.config['paths']['reasoning_rules_path']
            with open(rules_file_path, 'a') as f:
                for rule, confidence in rules:
                    # Optionally, you can append confidence as a comment or integrate it into the rule
                    f.write(f"% Confidence: {confidence:.2f}\n")
                    f.write(f"{rule}\n")
            self.logger.info(f"Added {len(rules)} new rules to {rules_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to update rules: {e}")
            raise

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the extracted rules.

        Returns:
            Dictionary containing rule statistics.
        """
        try:
            total_rules = len(self.rule_activations)
            high_confidence = len([r for r in self.rule_activations if r['confidence'] >= 0.8])
            low_confidence = total_rules - high_confidence

            stats = {
                'total_rules': total_rules,
                'high_confidence': high_confidence,
                'low_confidence': low_confidence
            }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get rule statistics: {e}")
            return {}
