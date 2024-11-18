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
        self.model = None  # Add this line

        # Initialize components
        self.data_loader = DataLoader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size']
        )

        # Initialize symbolic reasoner
        rules_path = os.path.join(self.project_root, self.config['paths']['reasoning_rules_path'])
        self.reasoner = SymbolicReasoner(rules_path)

        # Define feature names for rule extraction
        self.feature_names = [
            'temperature', 'vibration', 'pressure', 'operational_hours',
            'efficiency_index', 'system_state', 'performance_score'
        ]

    def extract_neural_rules(
            self,
            model: tf.keras.Model,
            X_test: np.ndarray,
            y_pred: np.ndarray,
            threshold: float = 0.8
    ) -> None:
        """Extract rules from neural model predictions."""
        try:
            # Find strong anomaly predictions
            anomalous_idx = np.where(y_pred > threshold)[0]
            normal_idx = np.where(y_pred < 0.2)[0]  # Clear normal cases

            if len(anomalous_idx) > 0:
                # Get sequences for analysis
                anomalous_sequences = X_test[anomalous_idx]
                normal_sequences = X_test[normal_idx]

                # Extract rules using gradient-based method
                gradient_rules = self.reasoner.extract_rules_from_neural_model(
                    model=model,
                    input_data=anomalous_sequences,
                    feature_names=self.feature_names,
                    threshold=0.7
                )

                # Extract rules from pattern analysis
                pattern_rules = self.reasoner.analyze_neural_patterns(
                    model=model,
                    anomalous_sequences=anomalous_sequences,
                    normal_sequences=normal_sequences,
                    feature_names=self.feature_names
                )

                # Update symbolic knowledge base
                all_rules = gradient_rules + pattern_rules
                self.reasoner.update_rules(all_rules)

                # Log statistics
                stats = self.reasoner.get_rule_statistics()
                self.logger.info(f"Neural rule extraction stats: {stats}")

                # Save extracted rules summary
                rules_summary = {
                    'gradient_rules': gradient_rules,
                    'pattern_rules': pattern_rules,
                    'statistics': stats,
                    'timestamp': str(np.datetime64('now'))
                }

                summary_path = os.path.join(
                    self.config['paths']['results_dir'],
                    'neurosymbolic_rules.json'
                )

                with open(summary_path, 'w') as f:
                    json.dump(rules_summary, f, indent=2)

                self.logger.info(f"Neurosymbolic rules saved to {summary_path}")

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

            # 7. Create model before using it
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
                X_test=X_test_scaled,
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

            # After evaluating model and before saving final model
            self.logger.info("Extracting rules from neural model predictions...")
            anomaly_indices = np.where(y_test_pred > 0.8)[0]  # Get strong anomaly predictions
            if len(anomaly_indices) > 0:
                anomaly_sequences = X_test_scaled[anomaly_indices]
                # Extract and update rules
                new_rules = self.reasoner.extract_rules_from_neural_model(
                    model=trained_model,
                    input_data=anomaly_sequences,
                    feature_names=self.feature_names,
                    threshold=0.7
                )
                pattern_rules = self.reasoner.analyze_neural_patterns(
                    model=trained_model,
                    anomalous_sequences=anomaly_sequences,
                    normal_sequences=X_test_scaled[y_test_pred < 0.2],
                    feature_names=self.feature_names
                )
                # Update rules with confidence
                self.reasoner.update_rules(new_rules + pattern_rules, min_confidence=0.7)
                self.logger.info(f"Extracted {len(new_rules)} direct rules and {len(pattern_rules)} pattern rules")

            # 10. Save final model
            best_model_path = os.path.join(self.config['paths']['results_dir'], 'best_model.keras')
            save_model(trained_model, best_model_path, self.logger)

            self.logger.info("Pipeline completed successfully.")

        except Exception as e:
            self.logger.exception(f"Pipeline error: {e}")
            raise