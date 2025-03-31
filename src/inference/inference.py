# src/inference/inference.py

import os
import logging
import numpy as np
import json
import argparse
from typing import Dict, Tuple

from src.utils.model_utils import load_model, load_scaler
from src.reasoning.reasoning import SymbolicReasoner
from src.config.config_manager import load_config
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import preprocess_sequences  # Ensure this import exists


class InferencePipeline:
    """
    Pipeline for running inference on new data using trained ANSR-DT model.
    """

    def __init__(self, config_path: str, logger: logging.Logger):
        """Initialize inference pipeline."""
        self.config = load_config(config_path)
        self.logger = logger
        self.window_size = self.config['model']['window_size']

        # Determine the directory of the configuration file
        config_dir = os.path.dirname(os.path.abspath(config_path))

        # Resolve paths relative to the configuration directory
        results_dir = os.path.join(config_dir, self.config['paths']['results_dir'])
        model_path = os.path.join(results_dir, 'model.keras')
        scaler_path = os.path.join(results_dir, 'scaler.pkl')

        self.logger.info(f"Resolved model path: {model_path}")
        self.logger.info(f"Resolved scaler path: {scaler_path}")

        # Load model and scaler
        self.model = load_model(model_path, logger)
        self.scaler = load_scaler(scaler_path, logger)

        # Initialize symbolic reasoner if enabled
        if self.config['symbolic_reasoning']['enabled']:
            rules_path = os.path.join(config_dir, self.config['paths']['reasoning_rules_path'])
            self.reasoner = SymbolicReasoner(rules_path)

    def load_and_preprocess_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data.

        Parameters:
        - data_file (str): Path to the new data file.

        Returns:
        - X_scaled (np.ndarray): Scaled feature sequences.
        - y_seq (np.ndarray): Corresponding labels.
        """
        try:
            data_loader = DataLoader(data_file, self.window_size)
            X, y = data_loader.load_data()
            X_seq, y_seq = data_loader.create_sequences(X, y)

            # Scale data
            X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

            return X_scaled, y_seq

        except Exception as e:
            self.logger.error(f"Failed to load and preprocess data: {e}")
            raise

    def get_predictions(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions."""
        y_scores = self.model.predict(X_scaled).ravel()
        y_pred = (y_scores > 0.5).astype(int)
        return y_scores, y_pred

    def get_symbolic_insights(self, X_seq: np.ndarray) -> list:
        """Get symbolic reasoning insights."""
        insights = []
        if self.config['symbolic_reasoning']['enabled']:
            for i in range(len(X_seq)):
                sensor_dict = {
                    'temperature': X_seq[i][-1, 0],
                    'vibration': X_seq[i][-1, 1],
                    'pressure': X_seq[i][-1, 2],
                    'operational_hours': X_seq[i][-1, 3],
                    'efficiency_index': X_seq[i][-1, 4]
                }
                insight = self.reasoner.reason(sensor_dict)
                insights.append(insight)
        return insights

    def run_inference(self, data_file: str, output_path: str):
        """Run complete inference pipeline."""
        try:
            # Load and preprocess data
            X_scaled, X_seq = self.load_and_preprocess_data(data_file)
            self.logger.info(f"Data loaded and preprocessed with shape {X_scaled.shape}")

            # Get predictions
            y_scores, y_pred = self.get_predictions(X_scaled)
            self.logger.info(f"Predictions generated for {len(y_pred)} samples")

            # Get symbolic insights
            insights = self.get_symbolic_insights(X_seq)

            # Prepare results
            results = {
                'y_scores': y_scores.tolist(),
                'y_pred': y_pred.tolist(),
                'insights': insights,
                'metadata': {
                    'data_file': data_file,
                    'model_version': self.config.get('model_version', 'unknown'),
                    'timestamp': str(np.datetime64('now'))
                }
            }

            # Save results
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Inference results saved to {output_path}")
            return results

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='ANSR-DT Inference Module')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the new .npz data file')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', type=str, default='results/inference_results.json',
                        help='Path to save inference results')
    args = parser.parse_args()

    logger = logging.getLogger('InferenceLogger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    pipeline = InferencePipeline(args.config, logger)
    pipeline.run_inference(args.data_file, args.output)


if __name__ == "__main__":
    main()
