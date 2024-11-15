# src/models/lstm_model.py
from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam


def create_lstm_model(input_shape: Tuple[int, int], learning_rate: float) -> Sequential:
    """
    Creates and compiles an LSTM-based Keras model.

    Parameters:
    - input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (Sequential): Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # Explicit Input layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # Test the create_lstm_model function
    model = create_lstm_model(input_shape=(10, 7), learning_rate=0.001)  # Corrected input_shape
    model.summary()
