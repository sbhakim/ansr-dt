# src/models/attention_model.py

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads=2):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layernorm = LayerNormalization()

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        return self.layernorm(inputs + attn_output)

# Add to CNN-LSTM model after conv layers