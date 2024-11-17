# src/models/cnn_lstm_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from typing import Tuple


def create_cnn_lstm_model(input_shape: Tuple[int, int], learning_rate: float) -> Sequential:
    """
    Creates and compiles a CNN-LSTM hybrid Keras model.

    Parameters:
    - input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (Sequential): Compiled Keras CNN-LSTM model.
    """
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # Convolutional layers for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM layers for capturing temporal dependencies
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dropout(0.3))

    # Fully connected output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # Example usage: Create and summarize the model
    input_shape_example = (10, 7)  # (window_size, features)
    learning_rate_example = 0.001
    model = create_cnn_lstm_model(input_shape_example, learning_rate_example)
    model.summary()
