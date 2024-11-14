# src/utils/model_utils.py

import logging
import os
from typing import Any
import joblib
import tensorflow as tf  # Added import

def save_scaler(scaler: Any, path: str, logger: logging.Logger):
    """
    Saves the scaler object to the specified path.

    Parameters:
    - scaler: Scaler object to save.
    - path (str): File path to save the scaler.
    - logger (logging.Logger): Logger for logging messages.
    """
    try:
        joblib.dump(scaler, path)
        logger.info(f"Scaler saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save scaler: {e}")
        raise

def load_scaler(path: str, logger: logging.Logger) -> Any:
    """
    Loads the scaler object from the specified path.

    Parameters:
    - path (str): File path to load the scaler from.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - scaler: Loaded scaler object.
    """
    try:
        scaler = joblib.load(path)
        logger.info(f"Scaler loaded from {path}")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise

def save_model(model: Any, path: str, logger: logging.Logger):
    """
    Saves the Keras model to the specified path.

    Parameters:
    - model: Keras model to save.
    - path (str): File path to save the model.
    - logger (logging.Logger): Logger for logging messages.
    """
    try:
        model.save(path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model(path: str, logger: logging.Logger) -> Any:
    """
    Loads the Keras model from the specified path.

    Parameters:
    - path (str): File path to load the model from.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - model: Loaded Keras model.
    """
    try:
        model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
