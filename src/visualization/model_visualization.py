import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import logging
from typing import Tuple, Optional, Union, Dict
import os


class ModelVisualizer:
    def __init__(self, model: tf.keras.Model, logger: Optional[logging.Logger] = None):
        """
        Initialize the model visualizer.

        Args:
            model: Trained Keras model
            logger: Optional logger instance
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
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
                # Ensure directory exists
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
        Calculate and visualize feature importance.

        Args:
            input_data: Input data to calculate feature importance for
            save_path: Optional path to save the visualization

        Returns:
            Tuple of (feature importance array, success boolean)
        """
        try:
            # Ensure input data has correct shape
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)

            # Create gradient model
            with tf.GradientTape() as tape:
                input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                tape.watch(input_tensor)
                predictions = self.model(input_tensor)

            # Calculate gradients
            gradients = tape.gradient(predictions, input_tensor)
            importance = np.abs(gradients.numpy()).mean(axis=(0, 1))

            # Create visualization
            fig = plt.figure(figsize=(10, 5))
            plt.bar(self.feature_names, importance)
            plt.title('Feature Importance Analysis')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Feature importance plot saved to {save_path}")

            plt.close(fig)

            # Create and save importance scores dictionary
            if save_path:
                scores_path = os.path.splitext(save_path)[0] + '_scores.txt'
                importance_dict = dict(zip(self.feature_names, importance.tolist()))
                with open(scores_path, 'w') as f:
                    for feature, score in importance_dict.items():
                        f.write(f"{feature}: {score:.4f}\n")
                self.logger.info(f"Feature importance scores saved to {scores_path}")

            return importance, True

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return None, False

    def visualize_model_architecture(self, save_path: Optional[str] = None) -> bool:
        """
        Visualize the model architecture.

        Args:
            save_path: Optional path to save the visualization

        Returns:
            Success boolean
        """
        try:
            # Convert model architecture to string representation
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)

            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save summary to file
                with open(save_path, 'w') as f:
                    f.write(model_summary)
                self.logger.info(f"Model architecture saved to {save_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error visualizing model architecture: {e}")
            return False

    def get_layer_names(self) -> list:
        """Get names of all layers in the model."""
        return [layer.name for layer in self.model.layers]