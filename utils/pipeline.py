# utils/pipeline.py

import logging
import os
from typing import Tuple

import numpy as np
from tensorflow.keras.models import Model

from models.lstm_model import create_lstm_model
from utils.data_loader import DataLoader
from utils.preprocessing import preprocess_sequences
from utils.data_processing import map_labels, split_data
from utils.trainer import train_model
from utils.evaluation import plot_metrics, compute_additional_metrics
from utils.model_utils import save_model, save_scaler

class NEXUSDTPipeline:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(self.config['paths']['data_file'], window_size=self.config['model']['window_size'])

    def run(self):
        try:
            # Load and process data
            X, y = self.data_loader.load_data()
            self.logger.info(f"Data loaded with shapes - X: {X.shape}, y: {y.shape}")

            X_seq, y_seq = self.data_loader.create_sequences(X, y)
            self.logger.info(f"Sequences created with shapes - X_seq: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            y_seq_binary = map_labels(y_seq, self.logger)
            y_seq = y_seq_binary

            scaler, X_seq = preprocess_sequences(X_seq)
            self.logger.info("Data preprocessing completed.")

            scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            save_scaler(scaler, scaler_path, self.logger)

            X_train, X_val, y_train, y_val = split_data(X_seq, y_seq, self.config['training']['validation_split'], self.logger)

            # Create and train model
            model = create_lstm_model(
                input_shape=X_train.shape[1:],
                learning_rate=self.config['training']['learning_rate']
            )
            self.logger.info("LSTM model created.")

            history, trained_model = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config=self.config,
                results_dir=self.config['paths']['results_dir'],
                logger=self.logger
            )

            # Plot metrics
            figures_dir = os.path.join(self.config['paths']['results_dir'], 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            plot_metrics(history, figures_dir)
            self.logger.info("Training and validation metrics plotted.")

            # Evaluate model
            y_pred = trained_model.predict(X_val)
            y_pred_classes = (y_pred > 0.5).astype(int)
            report, cm = compute_additional_metrics(y_val, y_pred_classes)

            self.logger.info("Model evaluation:")
            self.logger.info(f"\nClassification Report:\n{report}")
            self.logger.info(f"\nConfusion Matrix:\n{cm}")

            # Save final model
            model_path = os.path.join(self.config['paths']['results_dir'], 'model.keras')
            save_model(trained_model, model_path, self.logger)

        except Exception as e:
            self.logger.exception(f"An error occurred in the pipeline: {e}")
            raise
