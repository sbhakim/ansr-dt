# tests/test_inference.py

import unittest
import os
import json
import numpy as np
import logging  # Ensure logging is imported
from unittest import mock  # Optional: For future enhancements

class TestInference(unittest.TestCase):

    def setUp(self):
        # Initialize logger
        self.logger = logging.getLogger('TestInference')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        # Create synthetic data
        self.synthetic_data = np.random.rand(20, 7)
        np.savez('synthetic_new_data.npz',  # Removed 'tests/' prefix
                 temperature=self.synthetic_data[:, 0],
                 vibration=self.synthetic_data[:, 1],
                 pressure=self.synthetic_data[:, 2],
                 operational_hours=self.synthetic_data[:, 3],
                 efficiency_index=self.synthetic_data[:, 4],
                 system_state=self.synthetic_data[:, 5],
                 performance_score=self.synthetic_data[:, 6],
                 fused=np.random.rand(20),
                 anomaly=np.random.randint(0, 2, size=(20,)))

        # Assume model and scaler are already trained and saved
        self.config_path = '../configs/config.yaml'  # Adjusted path to point to parent directory
        self.output_path = 'inference_results.json'  # Removed 'tests/' prefix

    def tearDown(self):
        # Remove the synthetic data and output files after tests
        if os.path.exists('synthetic_new_data.npz'):
            os.remove('synthetic_new_data.npz')
        if os.path.exists('inference_results.json'):
            os.remove('inference_results.json')

    def test_inference(self):
        # Run inference using subprocess
        import subprocess
        result = subprocess.run([
            'python', '../src/inference/inference.py',  # Adjusted path to inference.py
            '--data_file', 'synthetic_new_data.npz',  # Corrected path
            '--output', 'inference_results.json',      # Corrected path
            '--config', self.config_path
        ], capture_output=True, text=True)

        # Check if the inference script ran successfully
        self.assertEqual(result.returncode, 0, f"Inference script failed: {result.stderr}")
        self.assertTrue(os.path.exists('inference_results.json'), "Inference results file was not created.")

        # Load and verify the inference results
        with open('inference_results.json', 'r') as f:
            data = json.load(f)

        self.assertIn('y_scores', data)
        self.assertIn('y_pred', data)
        self.assertIn('insights', data)
        self.assertEqual(len(data['y_scores']), 11)  # 20 - 10 +1 (window_size=10)
        self.assertEqual(len(data['y_pred']), 11)
        self.assertEqual(len(data['insights']), 11)

if __name__ == '__main__':
    unittest.main()
