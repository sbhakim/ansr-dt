# src/pipeline/pipeline.py

import logging
import os
from typing import Tuple, Dict

import numpy as np
from tensorflow.keras.models import Model

# Import the CNN-LSTM model
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
    """
    Validates the presence of required configuration keys and the existence of critical files.

    Parameters:
    - config (dict): Configuration dictionary.
    - logger (logging.Logger): Logger for logging errors.
    - project_root (str): Path to the project root directory.
    - config_dir (str): Path to the configuration directory.
    """
    required_keys = ['model', 'training', 'paths']
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            raise KeyError(f"Missing required configuration key: {key}")

    # Validate plot_config_path
    plot_config_path = config['paths'].get('plot_config_path')
    if not plot_config_path:
        logger.error("Missing 'plot_config_path' in configuration.")
        raise KeyError("Missing 'plot_config_path' in configuration.")

    full_plot_config_path = os.path.join(config_dir, plot_config_path)
    if not os.path.exists(full_plot_config_path):
        logger.error(f"Plot configuration file not found at: {full_plot_config_path}")
        raise FileNotFoundError(f"Plot configuration file not found at: {full_plot_config_path}")

    # Validate reasoning_rules_path
    reasoning_rules_path = config['paths'].get('reasoning_rules_path')
    if not reasoning_rules_path:
        logger.error("Missing 'reasoning_rules_path' in configuration.")
        raise KeyError("Missing 'reasoning_rules_path' in configuration.")

    full_rules_path = os.path.join(project_root, reasoning_rules_path)
    if not os.path.exists(full_rules_path):
        logger.error(f"Prolog rules file not found at: {full_rules_path}")
        raise FileNotFoundError(f"Prolog rules file not found at: {full_rules_path}")

    # Add more validations as needed
    logger.info("Configuration validation passed.")


class NEXUSDTPipeline:
    """
    The NEXUSDTPipeline class orchestrates the entire workflow of the NEXUS-DT project,
    including data loading, preprocessing, model training, evaluation, and saving results.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        """
        Initializes the pipeline with configuration and logger.

        Parameters:
        - config (dict): Configuration parameters loaded from config.yaml.
        - logger (logging.Logger): Configured logger for logging messages.
        """
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size']
        )

        # Determine base directories
        self.config_dir = os.path.dirname(os.path.abspath('configs/config.yaml'))
        self.project_root = os.path.dirname(self.config_dir)

    def run(self):
        """
        Executes the pipeline:
        1. Validates configuration.
        2. Loads and processes data.
        3. Splits data into training, validation, and test sets.
        4. Maps labels to binary.
        5. Preprocesses sequences (scaling).
        6. Creates and trains the CNN-LSTM model.
        7. Plots training metrics.
        8. Evaluates the model on the test set.
        9. Applies symbolic reasoning to generate insights.
        10. Saves the trained model.
        """
        try:
            self.logger.info("Starting pipeline execution.")

            # 0. Validate Configuration
            validate_config(self.config, self.logger, self.project_root, self.config_dir)

            # 1. Load and process data
            X, y = self.data_loader.load_data()
            self.logger.info(f"Data loaded with shapes - X: {X.shape}, y: {y.shape}")

            # 2. Create sequences
            X_seq, y_seq = self.data_loader.create_sequences(X, y)
            self.logger.info(f"Sequences created with shapes - X_seq: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            # 3. Split into training, validation, and test sets
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
                X_seq, y_seq,
                validation_split=self.config['training']['validation_split'],
                test_split=self.config['training']['test_split']
            )
            self.logger.info("Data split into training, validation, and test sets.")

            # 4. Map labels to binary
            y_train_binary = map_labels(y_train, self.logger)
            y_val_binary = map_labels(y_val, self.logger)
            y_test_binary = map_labels(y_test, self.logger)

            # 5. Preprocess sequences (fit scaler on training data)
            scaler, X_train_scaled = preprocess_sequences(X_train)
            self.logger.info("Training data preprocessing completed.")

            # Apply scaler to validation and test data
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            self.logger.info("Validation and test data scaling completed.")

            # Save scaler for future use
            scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            save_scaler(scaler, scaler_path, self.logger)
            self.logger.info(f"Scaler saved to {scaler_path}")

            # 6. Create and train CNN-LSTM model based on configuration
            architecture = self.config['model'].get('architecture', 'cnn_lstm')
            if architecture == 'cnn_lstm':
                self.logger.info("Creating CNN-LSTM model.")
                model = create_cnn_lstm_model(
                    input_shape=X_train_scaled.shape[1:],  # (window_size, features)
                    learning_rate=self.config['training']['learning_rate']
                )
            else:
                self.logger.error(f"Unsupported model architecture: {architecture}")
                raise ValueError(f"Unsupported model architecture: {architecture}")

            self.logger.info("Model created.")

            # Train the model
            history, trained_model = train_model(
                model=model,
                X_train=X_train_scaled,
                y_train=y_train_binary,
                X_val=X_val_scaled,
                y_val=y_val_binary,
                config=self.config,
                results_dir=self.config['paths']['results_dir'],
                logger=self.logger
            )
            self.logger.info("Model training completed.")

            # 7. Plot training metrics
            figures_dir = os.path.join(self.config['paths']['results_dir'], 'visualization')
            os.makedirs(figures_dir, exist_ok=True)
            self.logger.info(f"Figures directory ensured at {figures_dir}")

            # Load plot configuration from plot_config_path
            plot_config_path = os.path.join(self.config_dir, self.config['paths']['plot_config_path'])
            plot_config = load_plot_config(plot_config_path)
            self.logger.info(f"Plot configuration loaded from {plot_config_path}")

            # Apply plot configuration and plot metrics
            plot_metrics(history, figures_dir, plot_config)
            self.logger.info("Training and validation metrics plotted.")

            # 8. Evaluate the model on the test set
            y_test_pred = trained_model.predict(X_test_scaled).ravel()
            y_test_pred_classes = (y_test_pred > 0.5).astype(int)

            # Pass sensor data corresponding to test set
            sensor_data_test = {
                'temperature': X_test_scaled[:, -1, 0],
                'vibration': X_test_scaled[:, -1, 1],
                'pressure': X_test_scaled[:, -1, 2],
                'operational_hours': X_test_scaled[:, -1, 3],
                'efficiency_index': X_test_scaled[:, -1, 4],
                'system_state': X_test_scaled[:, -1, 5],
                'performance_score': X_test_scaled[:, -1, 6]
            }

            evaluate_model(
                y_true=y_test_binary,
                y_pred=y_test_pred_classes,
                y_scores=y_test_pred,
                figures_dir=figures_dir,
                plot_config_path=plot_config_path,
                config_path='configs/config.yaml',
                sensor_data=np.column_stack([
                    sensor_data_test['temperature'],
                    sensor_data_test['vibration'],
                    sensor_data_test['pressure'],
                    sensor_data_test['operational_hours'],
                    sensor_data_test['efficiency_index'],
                    sensor_data_test['system_state'],
                    sensor_data_test['performance_score']
                ]),
                model=trained_model  # Pass the trained model for visualization
            )
            self.logger.info("Model evaluation on test set completed.")

            # 9. Save the final trained model
            # To avoid confusion, save only the best model from training
            best_model_path = os.path.join(self.config['paths']['results_dir'], 'best_model.keras')
            save_model(trained_model, best_model_path, self.logger)
            self.logger.info(f"Best trained model saved to {best_model_path}")

        except KeyError as ke:
            self.logger.exception(f"Missing configuration key: {ke}")
            raise
        except Exception as e:
            self.logger.exception(f"An error occurred in the pipeline: {e}")
            raise
