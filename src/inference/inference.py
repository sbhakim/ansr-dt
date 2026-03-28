# src/inference/inference.py

import os
import sys
import logging
import numpy as np
import json
import argparse
from typing import Dict, Tuple, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.model_utils import load_model_with_initialization, load_scaler
from src.reasoning.reasoning import SymbolicReasoner
from src.config.config_manager import load_config
from src.data.data_loader import DataLoader


class InferencePipeline:
    """Pipeline for running inference on data using trained ANSR-DT artifacts."""

    def __init__(self, config_path: str, logger: logging.Logger):
        self.config_path = os.path.abspath(config_path)
        self.config = load_config(self.config_path)
        self.logger = logger
        self.window_size = self.config['model']['window_size']
        self.feature_names = self.config['model']['feature_names']
        self.input_shape = tuple(self.config['model']['input_shape'])

        config_dir = os.path.dirname(self.config_path)
        self.project_root = os.path.dirname(config_dir)
        self.results_dir = self._resolve_path(self.config['paths'].get('results_dir', 'results'))
        self.rules_path = self._resolve_path(self.config['paths']['reasoning_rules_path'])

        model_path = os.path.join(self.results_dir, 'best_model.keras')
        scaler_path = os.path.join(self.results_dir, 'scaler.pkl')

        self.logger.info(f"Resolved model path: {model_path}")
        self.logger.info(f"Resolved scaler path: {scaler_path}")

        self.model = load_model_with_initialization(model_path, logger, input_shape=self.input_shape)
        self.scaler = load_scaler(scaler_path, logger)
        self.reasoner: Optional[SymbolicReasoner] = None

        if self.config.get('symbolic_reasoning', {}).get('enabled', False):
            self.reasoner = SymbolicReasoner(
                rules_path=self.rules_path,
                input_shape=self.input_shape,
                model=self.model,
                logger=self.logger,
            )

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(self.project_root, path_value))

    def load_and_preprocess_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load raw data, create sequences, and scale them."""
        try:
            resolved_data_file = data_file if os.path.isabs(data_file) else os.path.join(self.project_root, data_file)
            data_loader = DataLoader(resolved_data_file, self.window_size)
            X, y = data_loader.load_data()
            X_seq, y_seq = data_loader.create_sequences(X, y)
            X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
            return X_scaled, X_seq, y_seq
        except Exception as e:
            self.logger.error(f"Failed to load and preprocess data: {e}")
            raise

    def get_predictions(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions for scaled sequences."""
        y_scores = self.model.predict(X_scaled, verbose=0).ravel()
        y_pred = (y_scores > 0.5).astype(int)
        return y_scores, y_pred

    def get_symbolic_insights(self, X_seq: np.ndarray) -> List[list]:
        """Get symbolic reasoning insights over the last timestep of each sequence."""
        insights: List[list] = []
        if not self.reasoner:
            return insights

        for i in range(len(X_seq)):
            sensor_dict = {
                feature_name: float(X_seq[i][-1, feature_idx])
                for feature_idx, feature_name in enumerate(self.feature_names)
            }
            insight = self.reasoner.reason(sensor_dict)
            insights.append(insight)
        return insights

    def run_inference(self, data_file: str, output_path: str) -> Dict[str, object]:
        """Run complete inference pipeline and persist results."""
        try:
            X_scaled, X_seq, y_seq = self.load_and_preprocess_data(data_file)
            self.logger.info(f"Data loaded and preprocessed with shape {X_scaled.shape}")

            y_scores, y_pred = self.get_predictions(X_scaled)
            self.logger.info(f"Predictions generated for {len(y_pred)} samples")

            insights = self.get_symbolic_insights(X_seq)

            results = {
                'y_true': y_seq.tolist(),
                'y_scores': y_scores.tolist(),
                'y_pred': y_pred.tolist(),
                'insights': insights,
                'metadata': {
                    'data_file': data_file,
                    'resolved_results_dir': self.results_dir,
                    'resolved_rules_path': self.rules_path,
                    'timestamp': str(np.datetime64('now')),
                },
            }

            resolved_output_path = output_path if os.path.isabs(output_path) else os.path.join(self.project_root, output_path)
            os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
            with open(resolved_output_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Inference results saved to {resolved_output_path}")
            return results
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='ANSR-DT Inference Module')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the .npz data file')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', type=str, default='results/inference_results.json', help='Path to save inference results')
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
