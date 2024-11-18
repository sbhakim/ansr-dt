# src/utils/model_utils.py

import logging
import os
from typing import Any, Optional
import joblib
import numpy as np
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
        logger.info(f"Attempting to save model to {path}")
        model.save(path)
        logger.info(f"Model successfully saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def load_model_with_initialization(
    path: str, logger: logging.Logger, input_shape: Optional[tuple] = None
) -> Any:
    """
    Loads the Keras model from the specified path and ensures it's built by performing a dummy prediction.

    Parameters:
    - path (str): File path to load the model from.
    - logger (logging.Logger): Logger for logging messages.
    - input_shape (tuple, optional): Shape of the input data (excluding batch size).
                                     Required if the model wasn't built with an input shape.

    Returns:
    - model: Loaded and built Keras model.
    """
    try:
        model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")

        # Check if the model is built
        if not model.built:
            if input_shape is None:
                # Attempt to infer input shape from the first layer
                if hasattr(model.layers[0], 'input_shape'):
                    inferred_shape = model.layers[0].input_shape[1:]  # Exclude batch size
                    if None in inferred_shape:
                        raise ValueError(
                            "Cannot infer input shape. Please provide a valid input_shape."
                        )
                    input_shape = inferred_shape
                else:
                    raise ValueError(
                        "Model does not have layers with input_shape attribute. Please provide input_shape."
                    )
            else:
                # Validate provided input_shape against the model's input
                expected_shape = model.input_shape[1:]
                if expected_shape != input_shape:
                    logger.warning(
                        f"Provided input_shape {input_shape} does not match model's expected input_shape {expected_shape}."
                    )

            # Create a dummy input with batch size 1
            dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
            model.predict(dummy_input)
            logger.info("Model built successfully with dummy input.")

        else:
            logger.info("Model was already built.")

        return model
    except Exception as e:
        logger.error(f"Failed to load and build model: {e}")
        raise
