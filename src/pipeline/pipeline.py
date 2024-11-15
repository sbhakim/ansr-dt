# src/pipeline/pipeline.py

import logging
import os
from typing import Tuple, Dict

import numpy as np
from tensorflow.keras.models import Model

from src.models.lstm_model import create_lstm_model
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import preprocess_sequences
from src.data.data_processing import map_labels, split_data
from src.training.trainer import train_model
from src.evaluation.evaluation import evaluate_model
from src.utils.model_utils import save_model, save_scaler, load_model
from src.visualization import plot_metrics
from src.visualization.plotting import load_plot_config
from src.reasoning.reasoning import SymbolicReasoner  # Import the SymbolicReasoner


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

    def run(self):
        """
        Executes the pipeline:
        1. Loads and processes data.
        2. Splits data into training, validation, and test sets.
        3. Maps labels to binary.
        4. Preprocesses sequences (scaling).
        5. Creates and trains the LSTM model.
        6. Plots training metrics.
        7. Evaluates the model on the test set.
        8. Applies symbolic reasoning to generate insights.
        9. Saves the trained model.
        """
        try:
            self.logger.info("Starting pipeline execution.")

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

            # 6. Create and train LSTM model
            model = create_lstm_model(
                input_shape=X_train_scaled.shape[1:],  # (window_size, features)
                learning_rate=self.config['training']['learning_rate']
            )
            self.logger.info("LSTM model created.")

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
            plot_config_path = self.config['paths']['plot_config_path']
            plot_config = load_plot_config(plot_config_path)
            self.logger.info(f"Plot configuration loaded from {plot_config_path}")

            # Apply plot configuration and plot metrics
            plot_metrics(history, figures_dir, plot_config)
            self.logger.info("Training and validation metrics plotted.")

            # 8. Evaluate the model on the test set
            y_test_pred = trained_model.predict(X_test_scaled).ravel()
            y_test_pred_classes = (y_test_pred > 0.5).astype(int)

            # Pass sensor data corresponding to test set
            # Assuming the sensor data is part of X_test_scaled
            sensor_data_test = {
                'temperature': X_test_scaled[:, -1, 0],  # Example indexing; adjust as per actual data
                'vibration': X_test_scaled[:, -1, 1],
                'pressure': X_test_scaled[:, -1, 2],
                'operational_hours': X_test_scaled[:, -1, 3],
                'efficiency_index': X_test_scaled[:, -1, 4]
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
                    sensor_data_test['efficiency_index']
                ])  # Passing as a 2D array
            )
            self.logger.info("Model evaluation on test set completed.")

            # 9. Save the final trained model
            model_path = os.path.join(self.config['paths']['results_dir'], 'model.keras')
            save_model(trained_model, model_path, self.logger)
            self.logger.info(f"Trained model saved to {model_path}")

            self.logger.info("Pipeline execution completed successfully.")

        except KeyError as ke:
            self.logger.exception(f"Missing configuration key: {ke}")
            raise
        except Exception as e:
            self.logger.exception(f"An error occurred in the pipeline: {e}")
            raise
