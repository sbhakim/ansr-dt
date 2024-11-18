# src/pipeline/pipeline.py

import logging
import os
from typing import Tuple, Dict, Any

import numpy as np

# Import the CNN-LSTM model
from src.models.cnn_lstm_model import create_cnn_lstm_model

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import preprocess_sequences
from src.data.data_processing import map_labels
from src.training.trainer import train_model
from src.evaluation.evaluation import evaluate_model
from src.utils.model_utils import save_model, save_scaler
from src.visualization.plotting import load_plot_config, plot_metrics


def validate_config(config: dict, logger: logging.Logger, project_root: str, config_dir: str):
    """
    Validates the presence of required configuration keys and the existence of critical files.

    Parameters:
    - config (dict): Configuration dictionary.
    - logger (logging.Logger): Logger for logging errors.
    - project_root (str): Path to the project root directory.
    - config_dir (str): Path to the configuration directory.
    """
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
                # Ensure directory exists
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Directory '{key}' ensured at: {full_path}")
            else:
                # Ensure file exists
                if not os.path.exists(full_path):
                    logger.error(f"Required file '{key}' not found at: {full_path}")
                    raise FileNotFoundError(f"Required file '{key}' not found at: {full_path}")
                logger.info(f"File '{key}' found at: {full_path}")

            # Update the config with absolute paths
            config['paths'][key] = full_path

        # Additional validations can be added here (e.g., checking numerical parameters)

        logger.info("Configuration validation passed.")

    except KeyError as ke:
        logger.exception(f"Configuration validation failed: {ke}")
        raise
    except FileNotFoundError as fe:
        logger.exception(f"Configuration validation failed: {fe}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during configuration validation: {e}")
        raise


class NEXUSDTPipeline:
    """
    The NEXUSDTPipeline class orchestrates the entire workflow of the NEXUS-DT project,
    including data loading, preprocessing, model training, evaluation, and saving results.
    """

    def __init__(self, config: dict, config_path: str, logger: logging.Logger):
        """
        Initializes the pipeline with configuration and logger.

        Parameters:
        - config (dict): Configuration parameters loaded from config.yaml.
        - config_path (str): Path to the configuration file.
        - logger (logging.Logger): Configured logger for logging messages.
        """
        self.config = config
        self.logger = logger

        # Determine base directories based on config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.project_root = os.path.dirname(self.config_dir)

        # Initialize DataLoader
        self.data_loader = DataLoader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size']
        )

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
        9. Saves the trained model.
        """
        try:
            self.logger.info("Starting pipeline execution.")

            # 1. Validate Configuration
            self.logger.debug("Validating configuration.")
            validate_config(self.config, self.logger, self.project_root, self.config_dir)

            # 2. Load and process data
            self.logger.debug("Loading data.")
            X, y = self.data_loader.load_data()
            self.logger.info(f"Data loaded with shapes - X: {X.shape}, y: {y.shape}")

            # 3. Create sequences
            self.logger.debug("Creating sequences.")
            X_seq, y_seq = self.data_loader.create_sequences(X, y)
            self.logger.info(f"Sequences created with shapes - X_seq: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            # 4. Split into training, validation, and test sets
            self.logger.debug("Splitting data into training, validation, and test sets.")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
                X_seq, y_seq,
                validation_split=self.config['training']['validation_split'],
                test_split=self.config['training']['test_split']
            )
            self.logger.info("Data split into training, validation, and test sets.")

            # 5. Map labels to binary
            self.logger.debug("Mapping labels to binary.")
            y_train_binary = map_labels(y_train, self.logger)
            y_val_binary = map_labels(y_val, self.logger)
            y_test_binary = map_labels(y_test, self.logger)

            # 6. Preprocess sequences (scaling)
            self.logger.debug("Preprocessing training data (scaling).")
            scaler, X_train_scaled = preprocess_sequences(X_train)
            self.logger.info("Training data preprocessing (scaling) completed.")

            self.logger.debug("Applying scaler to validation and test data.")
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            self.logger.info("Validation and test data scaling completed.")

            # Save scaler for future use
            scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            self.logger.debug(f"Saving scaler to {scaler_path}.")
            save_scaler(scaler, scaler_path, self.logger)
            self.logger.info(f"Scaler saved to {scaler_path}")

            # 7. Create and train CNN-LSTM model based on configuration
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
            self.logger.debug("Starting model training.")
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

            # 8. Plot training metrics
            self.logger.debug("Plotting training metrics.")
            figures_dir = os.path.join(self.config['paths']['results_dir'], 'visualization')
            os.makedirs(figures_dir, exist_ok=True)
            self.logger.info(f"Figures directory ensured at {figures_dir}")

            # Load plot configuration from plot_config_path
            plot_config_path = self.config['paths']['plot_config_path']
            self.logger.debug(f"Loading plot configuration from {plot_config_path}.")
            plot_config = load_plot_config(plot_config_path)
            self.logger.info(f"Plot configuration loaded from {plot_config_path}")

            # Plot metrics
            plot_metrics(history, figures_dir, plot_config)
            self.logger.info("Training and validation metrics plotted.")

            # 9. Evaluate the model on the test set
            self.logger.debug("Evaluating the model on the test set.")
            y_test_pred = trained_model.predict(X_test_scaled).ravel()
            y_test_pred_classes = (y_test_pred > 0.5).astype(int)
            self.logger.info("Model predictions on test set obtained.")

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
                config_path=self.config_dir,  # Use config_dir to locate rules
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

            # 10. Save the final trained model
            self.logger.debug("Saving the final trained model.")
            best_model_path = os.path.join(self.config['paths']['results_dir'], 'best_model.keras')
            save_model(trained_model, best_model_path, self.logger)
            self.logger.info(f"Best trained model saved to {best_model_path}")

            self.logger.info("Pipeline completed successfully.")

        except KeyError as ke:
            self.logger.exception(f"Missing configuration key: {ke}")
            raise
        except FileNotFoundError as fe:
            self.logger.exception(f"File not found during pipeline execution: {fe}")
            raise
        except ValueError as ve:
            self.logger.exception(f"Value error during pipeline execution: {ve}")
            raise
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred in the pipeline: {e}")
            raise
