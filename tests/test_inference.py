# tests/test_inference.py

import json
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from src.inference.inference import InferencePipeline


class IdentityScaler:
    def transform(self, X):
        return X


class TestInference(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestInference')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

    def test_inference_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, 'configs')
            results_dir = os.path.join(tmpdir, 'results')
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)

            config_path = os.path.join(config_dir, 'config.yaml')
            data_path = os.path.join(data_dir, 'synthetic_new_data.npz')
            output_path = os.path.join(tmpdir, 'inference_results.json')
            scaler_path = os.path.join(results_dir, 'scaler.pkl')
            rules_path = os.path.join(tmpdir, 'src', 'reasoning', 'rules.pl')
            os.makedirs(os.path.dirname(rules_path), exist_ok=True)
            open(rules_path, 'a').close()

            synthetic_data = np.random.rand(20, 7).astype(np.float32)
            np.savez(
                data_path,
                temperature=synthetic_data[:, 0],
                vibration=synthetic_data[:, 1],
                pressure=synthetic_data[:, 2],
                operational_hours=synthetic_data[:, 3],
                efficiency_index=synthetic_data[:, 4],
                system_state=synthetic_data[:, 5],
                performance_score=synthetic_data[:, 6],
                fused=np.random.rand(20),
                anomaly=np.random.randint(0, 2, size=(20,)),
            )
            joblib.dump(IdentityScaler(), scaler_path)
            open(config_path, 'a').close()

            config = {
                'model': {
                    'window_size': 10,
                    'feature_names': [
                        'temperature', 'vibration', 'pressure', 'operational_hours',
                        'efficiency_index', 'system_state', 'performance_score',
                    ],
                    'input_shape': [10, 7],
                },
                'paths': {
                    'results_dir': 'results',
                    'reasoning_rules_path': 'src/reasoning/rules.pl',
                },
                'symbolic_reasoning': {'enabled': True},
            }

            mock_model = MagicMock()
            mock_model.predict.return_value = np.linspace(0.1, 0.9, 11).reshape(-1, 1)
            mock_reasoner = MagicMock()
            mock_reasoner.reason.side_effect = lambda sensor_dict: ['symbolic_insight'] if sensor_dict['temperature'] >= 0 else []

            with patch('src.inference.inference.load_config', return_value=config), \
                 patch('src.inference.inference.load_model_with_initialization', return_value=mock_model), \
                 patch('src.inference.inference.SymbolicReasoner', return_value=mock_reasoner):
                pipeline = InferencePipeline(config_path, self.logger)
                results = pipeline.run_inference(data_path, output_path)

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, 'r') as f:
                persisted = json.load(f)

            self.assertIn('y_scores', persisted)
            self.assertIn('y_pred', persisted)
            self.assertIn('insights', persisted)
            self.assertEqual(len(results['y_scores']), 11)
            self.assertEqual(len(results['y_pred']), 11)
            self.assertEqual(len(results['insights']), 11)
            self.assertEqual(results['metadata']['resolved_results_dir'], results_dir)


if __name__ == '__main__':
    unittest.main()
