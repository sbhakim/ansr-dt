# utils/data_processing.py

import logging
from typing import Tuple

import numpy as np


def map_labels(y: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Maps multi-class labels to binary labels.
    Label '0' remains '0', others are mapped to '1'.

    Parameters:
    - y (np.ndarray): Original labels.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - y_binary (np.ndarray): Binary labels.
    """
    try:
        y_binary = np.where(y == 0, 0, 1)
        logger.info(f"Binary labels created. Unique labels: {np.unique(y_binary)}")
        return y_binary
    except Exception as e:
        logger.error(f"Failed to map labels: {e}")
        raise


def split_data(
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float,
        logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training and validation sets.

    Parameters:
    - X (np.ndarray): Features.
    - y (np.ndarray): Labels.
    - validation_split (float): Fraction of data to use for validation.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - X_train (np.ndarray): Training features.
    - X_val (np.ndarray): Validation features.
    - y_train (np.ndarray): Training labels.
    - y_val (np.ndarray): Validation labels.
    """
    try:
        split_index = int((1 - validation_split) * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        logger.info(f"Data split into training and validation sets with validation_split={validation_split}.")
        logger.info(f"Training set - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Validation set - X_val: {X_val.shape}, y_val: {y_val.shape}")

        return X_train, X_val, y_train, y_val
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise
