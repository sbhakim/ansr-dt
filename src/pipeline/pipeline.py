# src/pipeline/pipeline.py

import logging
import os
from typing import Tuple

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


class NEXUSDTPipeline:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size']
        )

    def run(self):
        try:
            # Load and process data
            X, y = self.data_loader.load_data()
            self.logger.info(f"Data loaded with shapes - X: {X.shape}, y: {y.shape}")

            X_seq, y_seq = self.data_loader.create_sequences(X, y)
            self.logger.info(f"Sequences created with shapes - X_seq: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            # Split into training, validation, and test sets
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
                X_seq, y_seq,
                validation_split=self.config['training']['validation_split'],
                test_split=self.config['training']['test_split']
            )
            self.logger.info(f"Data split into training, validation, and test sets.")

            # Map labels to binary
            y_train_binary = map_labels(y_train, self.logger)
            y_val_binary = map_labels(y_val, self.logger)
            y_test_binary = map_labels(y_test, self.logger)

            # Preprocess sequences (fit scaler on training data)
            scaler, X_train = preprocess_sequences(X_train)
            self.logger.info("Training data preprocessing completed.")

            # Apply scaler to validation and test data
            X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            self.logger.info("Validation and test data scaling completed.")

            # Save scaler
            scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            save_scaler(scaler, scaler_path, self.logger)

            # Create and train model
            model = create_lstm_model(
                input_shape=X_train.shape[1:],
                learning_rate=self.config['training']['learning_rate']
            )
            self.logger.info("LSTM model created.")

            history, trained_model = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train_binary,
                X_val=X_val,
                y_val=y_val_binary,
                config=self.config,
                results_dir=self.config['paths']['results_dir'],
                logger=self.logger
            )

            # Plot training metrics
            figures_dir = os.path.join(self.config['paths']['results_dir'], 'visualization')
            os.makedirs(figures_dir, exist_ok=True)
            plot_metrics(history, figures_dir, self.config['plot_config'])
            self.logger.info("Training and validation metrics plotted.")

            # Evaluate on test set
            y_test_pred = trained_model.predict(X_test).ravel()
            y_test_pred_classes = (y_test_pred > 0.5).astype(int)

            evaluate_model(
                y_true=y_test_binary,
                y_pred=y_test_pred_classes,
                y_scores=y_test_pred,
                figures_dir=figures_dir,
                plot_config_path=self.config['plot_config_path']
            )

            # Save final model
            model_path = os.path.join(self.config['paths']['results_dir'], 'model.keras')
            save_model(trained_model, model_path, self.logger)

        except Exception as e:
            self.logger.exception(f"An error occurred in the pipeline: {e}")
            raise
