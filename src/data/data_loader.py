# src/data/data_loader.py

import numpy as np
import logging
import os
from typing import Tuple

class DataLoader:
    def __init__(self, data_file, window_size=10):
        self.data_file = data_file
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            data = np.load(self.data_file)
            self.logger.info(f"Keys in the .npz file: {data.files}")

            # List of features to include
            feature_keys = [
                'temperature',
                'vibration',
                'pressure',
                'operational_hours',
                'efficiency_index',
                'system_state',
                'performance_score'
            ]

            # Verify all feature keys are present
            for key in feature_keys:
                if key not in data.files:
                    self.logger.error(f"Feature '{key}' not found in the data file.")
                    raise KeyError(f"Feature '{key}' not found in the data file.")

            # Stack features into a 2D array (samples, features)
            X = np.stack([data[key] for key in feature_keys], axis=1)  # Shape: (5000, 7)
            y = data['anomaly']  # Shape: (5000,)

            self.logger.info(f"Data loaded successfully from {self.data_file}")
            return X, y

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def create_sequences(self, X, y):
        """
        Create sequences of data for LSTM input.

        Parameters:
        - X: 2D array of shape (samples, features)
        - y: 1D array of shape (samples,)

        Returns:
        - X_seq: 3D array of shape (samples - window_size +1, window_size, features)
        - y_seq: 1D array of shape (samples - window_size +1,)
        """
        try:
            X_seq = []
            y_seq = []
            for i in range(len(X) - self.window_size + 1):
                X_seq.append(X[i:i + self.window_size])
                y_seq.append(y[i + self.window_size - 1])  # Assign label of the last timestep

            X_seq = np.array(X_seq)  # Shape: (samples - window_size +1, window_size, features)
            y_seq = np.array(y_seq)  # Shape: (samples - window_size +1,)

            self.logger.info(f"Created sequences with window size {self.window_size}")
            self.logger.info(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            return X_seq, y_seq

        except Exception as e:
            self.logger.error(f"Failed to create sequences: {e}")
            raise

    def split_data(self, X, y, validation_split: float, test_split: float) -> Tuple:
        """
        Splits the data into training, validation, and test sets.

        Parameters:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        - validation_split (float): Fraction of data to use for validation.
        - test_split (float): Fraction of data to use for testing.

        Returns:
        - X_train, X_val, X_test, y_train, y_val, y_test
        """
        try:
            total = len(X)
            test_size = int(total * test_split)
            val_size = int(total * validation_split)

            X_test = X[:test_size]
            y_test = y[:test_size]

            X_val = X[test_size:test_size + val_size]
            y_val = y[test_size:test_size + val_size]

            X_train = X[test_size + val_size:]
            y_train = y[test_size + val_size:]

            self.logger.info(
                f"Data split into training, validation, and test sets with validation_split={validation_split} and test_split={test_split}.")
            self.logger.info(f"Training set - X_train: {X_train.shape}, y_train: {y_train.shape}")
            self.logger.info(f"Validation set - X_val: {X_val.shape}, y_val: {y_val.shape}")
            self.logger.info(f"Test set - X_test: {X_test.shape}, y_test: {y_test.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            self.logger.error(f"Failed to split data: {e}")
            raise
