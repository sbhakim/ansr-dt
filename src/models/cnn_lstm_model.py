# src/models/cnn_lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # Added Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input # Added Input
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional

# Assuming attention_model.py is in the same directory or accessible via Python path
try:
    from .attention_model import AttentionBlock
    attention_available = True
except ImportError:
    # Fallback if attention_model.py is missing - Log a warning
    import logging
    # Use a basic logger setup if main logger isn't configured yet during direct script run
    _logger = logging.getLogger(__name__)
    if not _logger.hasHandlers():
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        _logger.addHandler(_handler)
        _logger.setLevel(logging.INFO)
    _logger.warning("AttentionBlock could not be imported from .attention_model. Attention layer will be skipped.")
    attention_available = False


def create_cnn_lstm_model(
        input_shape: Tuple[int, int],
        learning_rate: float,
        use_attention: bool = True, # Add flag to enable/disable attention
        attention_key_dim: int = 50, # Default attention param
        attention_heads: int = 4     # Default attention param
        ) -> tf.keras.Model: # Return type changed to base Model for flexibility
    """
    Creates and compiles a CNN-LSTM hybrid Keras model, optionally with Attention.

    Parameters:
    - input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).
    - learning_rate (float): Learning rate for the optimizer.
    - use_attention (bool): Whether to include the Attention layer.
    - attention_key_dim (int): Key dimension for the MultiHeadAttention layer.
    - attention_heads (int): Number of heads for the MultiHeadAttention layer.

    Returns:
    - model (tf.keras.Model): Compiled Keras CNN-LSTM model.
    """
    # Use functional API for more flexibility, especially with Attention
    inputs = Input(shape=input_shape)

    # Convolutional layers for feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # LSTM layers for capturing temporal dependencies
    # Need return_sequences=True for the layer feeding into Attention
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    # Second LSTM must return sequences if followed by Attention
    x = LSTM(50, return_sequences=True if (use_attention and attention_available) else False)(x)
    x = Dropout(0.3)(x)

    # --- Optional Attention Layer ---
    if use_attention and attention_available:
        # Use the imported AttentionBlock
        x = AttentionBlock(key_dim=attention_key_dim, num_heads=attention_heads)(x)
        # Flatten the output of Attention before the final Dense layer
        # Attention output is typically (batch, timesteps, features)
        x = Flatten()(x)
    elif use_attention and not attention_available:
         # Log if attention was requested but module not found
         import logging # Ensure logger is available
         logging.getLogger(__name__).warning("Attention requested but AttentionBlock module not available. Skipping attention layer.")
         # No Flatten needed here as the last LSTM has return_sequences=False
    # --- End Attention Layer ---

    # Fully connected output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs, name="cnn_lstm_attention_model" if (use_attention and attention_available) else "cnn_lstm_model")
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
    print("--- Model WITHOUT Attention ---")
    model_no_attention = create_cnn_lstm_model(input_shape_example, learning_rate_example, use_attention=False)
    model_no_attention.summary()

    if attention_available:
        print("\n--- Model WITH Attention ---")
        model_with_attention = create_cnn_lstm_model(
            input_shape=input_shape_example,
            learning_rate=learning_rate_example,
            use_attention=True,
            attention_key_dim=50, # Example value
            attention_heads=4     # Example value
        )
        model_with_attention.summary()
    else:
        print("\n--- Attention Module not available, skipping Attention model summary ---")