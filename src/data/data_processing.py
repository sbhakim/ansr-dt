# src/data/data_processing.py

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
        test_split: float,
        logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training, validation, and test sets.

    Parameters:
    - X (np.ndarray): Features.
    - y (np.ndarray): Labels.
    - validation_split (float): Fraction of data to use for validation.
    - test_split (float): Fraction of data to use for testing.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    try:
        total = len(X)
        test_size = int(total * test_split)
        validation_size = int(total * validation_split)

        X_test = X[:test_size]
        y_test = y[:test_size]

        X_val = X[test_size:test_size + validation_size]
        y_val = y[test_size:test_size + validation_size]

        X_train = X[test_size + validation_size:]
        y_train = y[test_size + validation_size:]

        logger.info(
            f"Data split into training, validation, and test sets with validation_split={validation_split} and test_split={test_split}."
        )
        logger.info(f"Training set - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Validation set - X_val: {X_val.shape}, y_val: {y_val.shape}")
        logger.info(f"Test set - X_test: {X_test.shape}, y_test: {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise
