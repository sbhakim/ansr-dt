# src/models/attention_model.py

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Layer # Added Layer

# Make AttentionBlock serializable
@tf.keras.utils.register_keras_serializable(package='Custom', name='AttentionBlock') # Register for saving/loading
class AttentionBlock(Layer): # Inherit directly from Layer
    def __init__(self, key_dim, num_heads=2, **kwargs): # Accept **kwargs
        # Call the parent constructor, passing kwargs. This handles standard args like name, trainable, dtype.
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        # Define sub-layers here
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name='multi_head_attention') # Give sublayers names
        self.layernorm = LayerNormalization(name='layer_norm') # Give sublayers names

    def call(self, inputs):
        # Self-attention: query, value, key are all the same
        attn_output = self.mha(query=inputs, value=inputs, key=inputs)
        # Add & Norm
        # Ensure inputs + attn_output is the correct way to combine based on your architecture needs
        # Sometimes concatenation or other methods are used depending on the attention type
        output = self.layernorm(inputs + attn_output)
        return output

    def get_config(self):
        # Get the parent config first
        config = super().get_config()
        # Update with the layer's specific parameters
        config.update({
            'key_dim': self.key_dim,
            'num_heads': self.num_heads,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create layer instance from config dictionary
        return cls(**config)

# Optional: Add example usage if desired
# if __name__ == "__main__":
#     # Example: Instantiate the layer
#     input_shape = (None, 10, 50) # Example: (batch, timesteps, features)
#     inputs = tf.keras.Input(shape=input_shape[1:])
#     attention_block = AttentionBlock(key_dim=50, num_heads=4)
#     outputs = attention_block(inputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     model.summary()
#
#     # Example: Save and load (requires TensorFlow installation)
#     try:
#         model.save("temp_attention_model.keras")
#         loaded_model = tf.keras.models.load_model(
#             "temp_attention_model.keras",
#             custom_objects={'AttentionBlock': AttentionBlock} # Pass custom object
#         )
#         print("\nModel with custom AttentionBlock saved and loaded successfully.")
#         loaded_model.summary()
#         # Clean up temp file
#         import os
#         os.remove("temp_attention_model.keras")
#     except Exception as e:
#         print(f"\nError during save/load test: {e}")