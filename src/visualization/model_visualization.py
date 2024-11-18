import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import logging
from typing import Tuple, Optional, Dict
import os


class ModelVisualizer:
    """Class for visualizing and analyzing model behavior and features."""

    def __init__(self, model: tf.keras.Model, logger: Optional[logging.Logger] = None):
        """
        Initialize the model visualizer.

        Args:
            model: Trained Keras model
            logger: Optional logger instance
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)

        # Verify model has been called
        self.is_model_ready = hasattr(model, 'input_shape')
        if not self.is_model_ready:
            self.logger.warning("Model has not been called yet. Some visualizations may be unavailable.")

        # Define feature names for visualization
        self.feature_names = [
            'Temperature', 'Vibration', 'Pressure',
            'Op Hours', 'Efficiency', 'State', 'Performance'
        ]

    def _create_intermediate_model(self, layer_name: str) -> Tuple[Optional[Model], str]:
        """
        Create an intermediate model for feature extraction.

        Args:
            layer_name: Name of the layer to extract features from

        Returns:
            Tuple of (intermediate model, actual layer name used)
        """
        try:
            if not self.is_model_ready:
                self.logger.warning("Model not ready for intermediate layer extraction")
                return None, ""

            # Try to get the specified layer
            target_layer = None
            actual_layer_name = layer_name

            # If exact layer name not found, try to find first layer containing the name
            if layer_name not in [layer.name for layer in self.model.layers]:
                for layer in self.model.layers:
                    if layer_name in layer.name:
                        target_layer = layer
                        actual_layer_name = layer.name
                        break
            else:
                target_layer = self.model.get_layer(layer_name)

            if target_layer is None:
                self.logger.warning(f"Layer '{layer_name}' not found in model")
                return None, ""

            intermediate_model = Model(
                inputs=self.model.input,
                outputs=target_layer.output
            )
            return intermediate_model, actual_layer_name

        except Exception as e:
            self.logger.error(f"Error creating intermediate model: {e}")
            return None, ""

    def visualize_cnn_features(
            self,
            input_data: np.ndarray,
            layer_name: str = 'conv1d',
            save_path: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Visualize CNN layer activations.

        Args:
            input_data: Input data to visualize features for
            layer_name: Name of the CNN layer to visualize
            save_path: Optional path to save the visualization

        Returns:
            Tuple of (feature maps array, success boolean)
        """
        try:
            if not self.is_model_ready:
                self.logger.warning("Model not ready for CNN feature visualization")
                return None, False

            # Ensure input data has correct shape
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)

            # Create intermediate model
            intermediate_model, actual_layer_name = self._create_intermediate_model(layer_name)
            if intermediate_model is None:
                return None, False

            # Get feature maps
            feature_maps = intermediate_model.predict(input_data, verbose=0)

            # Create visualization
            n_features = min(8, feature_maps.shape[-1])
            fig = plt.figure(figsize=(15, 5))

            for i in range(n_features):
                ax = plt.subplot(2, 4, i + 1)
                ax.set_title(f'Feature Map {i + 1}')
                plt.plot(feature_maps[0, :, i])
                plt.grid(True)

            plt.suptitle(f'CNN Features from {actual_layer_name} Layer')
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Feature maps saved to {save_path}")

            plt.close(fig)
            return feature_maps, True

        except Exception as e:
            self.logger.error(f"Error visualizing CNN features: {e}")
            return None, False

    def get_feature_importance(
            self,
            input_data: np.ndarray,
            save_path: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Calculate and visualize feature importance using gradient-based analysis.

        Args:
            input_data: Input data to calculate feature importance for
            save_path: Optional path to save the visualization

        Returns:
            Tuple of (feature importance array, success boolean)
        """
        try:
            if not self.is_model_ready:
                self.logger.warning("Model not ready for feature importance calculation")
                return None, False

            # Calculate gradients
            with tf.GradientTape() as tape:
                input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                tape.watch(input_tensor)
                predictions = self.model(input_tensor, training=False)

            # Get gradients
            gradients = tape.gradient(predictions, input_tensor)
            importance = np.abs(gradients.numpy()).mean(axis=(0, 1))

            if save_path:
                # Create visualization
                plt.figure(figsize=(10, 5))
                plt.bar(self.feature_names, importance)
                plt.title('Feature Importance Analysis')
                plt.xlabel('Features')
                plt.ylabel('Importance Score')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save plot
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Save scores
                scores_path = os.path.join(
                    os.path.dirname(save_path),
                    'feature_importance_scores.txt'
                )
                with open(scores_path, 'w') as f:
                    for name, score in zip(self.feature_names, importance):
                        f.write(f"{name}: {score:.4f}\n")

                self.logger.info(f"Feature importance plot saved to {save_path}")
                self.logger.info(f"Feature importance scores saved to {scores_path}")

            return importance, True

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return None, False

    def visualize_model_architecture(self, save_path: Optional[str] = None) -> bool:
        """
        Visualize and save the model architecture summary.

        Args:
            save_path: Optional path to save the architecture summary

        Returns:
            Success boolean
        """
        try:
            if not self.is_model_ready:
                self.logger.warning("Model not ready for architecture visualization")
                return False

            # Get model summary as string
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(model_summary)
                self.logger.info(f"Model architecture saved to {save_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error visualizing model architecture: {e}")
            return False

    def visualize_attention_weights(
            self,
            input_data: np.ndarray,
            layer_name: str = 'attention',
            save_path: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Visualize attention weights if model contains attention layers.

        Args:
            input_data: Input data to visualize attention for
            layer_name: Name of the attention layer
            save_path: Optional path to save the visualization

        Returns:
            Tuple of (attention weights array, success boolean)
        """
        try:
            if not self.is_model_ready:
                self.logger.warning("Model not ready for attention visualization")
                return None, False

            # Create intermediate model for attention layer
            attention_model, actual_layer_name = self._create_intermediate_model(layer_name)
            if attention_model is None:
                return None, False

            # Get attention weights
            attention_weights = attention_model.predict(input_data, verbose=0)

            if save_path:
                plt.figure(figsize=(10, 8))
                plt.imshow(attention_weights[0], cmap='viridis', aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.title(f'Attention Weights from {actual_layer_name} Layer')
                plt.xlabel('Key Dimension')
                plt.ylabel('Query Dimension')
                plt.tight_layout()

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Attention weights visualization saved to {save_path}")

            return attention_weights, True

        except Exception as e:
            self.logger.error(f"Error visualizing attention weights: {e}")
            return None, False

    def get_layer_names(self) -> list:
        """Get names of all layers in the model."""
        if not self.is_model_ready:
            self.logger.warning("Model not ready for layer name extraction")
            return []
        return [layer.name for layer in self.model.layers]

    def get_layer_configs(self) -> Dict:
        """Get configurations of all layers in the model."""
        if not self.is_model_ready:
            self.logger.warning("Model not ready for layer configuration extraction")
            return {}

        configs = {}
        for layer in self.model.layers:
            configs[layer.name] = layer.get_config()
        return configs