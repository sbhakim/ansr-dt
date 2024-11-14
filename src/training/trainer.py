# src/training/trainer.py

import logging
import os
from typing import Tuple, Dict

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import joblib


from src.models.lstm_model import create_lstm_model  # Corrected import
from src.utils.model_utils import save_model, save_scaler

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    results_dir: str,
    logger: logging.Logger
) -> Tuple:
    """
    Trains the provided model using the training data and validates on validation data.

    Parameters:
    - model: Compiled Keras model to train.
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training labels.
    - X_val (np.ndarray): Validation features.
    - y_val (np.ndarray): Validation labels.
    - config (Dict): Configuration parameters.
    - results_dir (str): Directory to save results like models and plots.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - history: Keras History object containing training history.
    - model: Trained Keras model.
    """
    try:
        logger.info("Handling class imbalance using class weights.")

        # Compute class weights
        class_weights_values = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train.flatten()
        )
        class_weights = {i: class_weights_values[i] for i in range(len(class_weights_values))}
        logger.info(f"Class weights: {class_weights}")

        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        )

        checkpoint_path = os.path.join(results_dir, 'best_model.keras')
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True
        )

        callbacks = [early_stopping, checkpoint]
        logger.info("Callbacks for training have been set up.")

        # Train the model
        logger.info("Starting model training.")
        history = model.fit(
            X_train, y_train,
            batch_size=config['training']['batch_size'],
            epochs=config['training']['epochs'],
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Model training completed.")

        return history, model

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise
