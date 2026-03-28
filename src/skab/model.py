# src/skab/model.py
# Defines the dedicated SKAB neural architecture, combining sequence-preserving convolutions, bidirectional LSTM encoding, and temporal attention for robust anomaly scoring behavior.

from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam


def build_skab_model(config: Dict[str, Any]) -> tf.keras.Model:
    model_cfg = config['model']
    input_shape = tuple(model_cfg['input_shape'])
    conv_filters = model_cfg.get('conv_filters', [64, 64])
    kernel_size = int(model_cfg.get('kernel_size', 3))
    lstm_units = int(model_cfg.get('lstm_units', 64))
    attention_heads = int(model_cfg.get('attention_heads', 4))
    attention_key_dim = int(model_cfg.get('attention_key_dim', 16))
    dense_units = int(model_cfg.get('dense_units', 64))
    dropout = float(model_cfg.get('dropout', 0.3))
    learning_rate = float(config['training']['learning_rate'])

    inputs = layers.Input(shape=input_shape, name='skab_input')
    # Keep the convolution stack sequence-preserving; unlike the synthetic model,
    # this branch avoids collapsing the temporal axis before attention.
    x = inputs
    for index, filters in enumerate(conv_filters):
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv_{index + 1}')(x)
        x = layers.BatchNormalization(name=f'bn_{index + 1}')(x)
        x = layers.Activation('relu', name=f'relu_{index + 1}')(x)
        x = layers.Dropout(dropout, name=f'conv_dropout_{index + 1}')(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm')(x)
    x = layers.Dropout(dropout, name='lstm_dropout')(x)

    # Temporal attention operates after BiLSTM encoding so the model can
    # reweight informative timesteps instead of isolated sensor channels.
    attention_output = layers.MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_key_dim,
        dropout=dropout,
        name='temporal_attention',
    )(x, x)
    x = layers.Add(name='attention_residual')([x, attention_output])
    x = layers.LayerNormalization(name='attention_norm')(x)
    x = layers.GlobalAveragePooling1D(name='temporal_pool')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout, name='dense_dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='anomaly_score')(x)

    model = Model(inputs=inputs, outputs=outputs, name='skab_cnn_bilstm_attention')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model
