# utils/data_loader.py

import numpy as np
import logging
import os

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

if __name__ == "__main__":
    # Test the DataLoader
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Dynamically construct the path to the data file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, '..', 'data', 'synthetic_sensor_data_with_anomalies.npz')

    # Initialize DataLoader with the correct path
    data_loader = DataLoader(data_file, window_size=10)
    X, y = data_loader.load_data()
    X_seq, y_seq = data_loader.create_sequences(X, y)
    print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
