# utils/preprocessing.py

import logging
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_sequences(X_seq: np.ndarray):
    """
    Preprocesses the sequenced data by scaling.

    Parameters:
    - X_seq (np.ndarray): Sequenced feature data.

    Returns:
    - scaler (StandardScaler): Fitted scaler object.
    - X_seq_scaled (np.ndarray): Scaled sequenced data.
    """
    logger = logging.getLogger(__name__)  # Initialize module-specific logger
    try:
        scaler = StandardScaler()
        X_shape = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])  # Flatten for scaler
        X_seq_scaled = scaler.fit_transform(X_seq_reshaped)
        X_seq_scaled = X_seq_scaled.reshape(X_shape)  # Reshape back to original
        logger.info("Data scaling completed.")
        return scaler, X_seq_scaled
    except Exception as e:
        logger.error(f"Failed to preprocess sequences: {e}")
        raise
