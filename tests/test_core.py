# tests/test_core.py

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.ansrdt.core import ANSRDTCore

class TestNEXUSDTCore(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.config_path = '/path/to/config.yaml'
        self.core = ANSRDTCore(self.config_path, logger=self.logger)

    @patch('src.ansrdt.core.load_config')
    @patch('src.ansrdt.core.load_model')
    @patch('src.ansrdt.core.PPO.load')
    @patch('src.ansrdt.core.SymbolicReasoner')
    def test_initialization(self, mock_reasoner, mock_ppo_load, mock_load_model, mock_load_config):
        mock_load_config.return_value = {
            'model': {'window_size': 10},
            'paths': {
                'results_dir': '/absolute/path/to/results',
                'reasoning_rules_path': 'src/reasoning/rules.pl'
            },
            'symbolic_reasoning': {
                'enabled': True,
                'rules_path': 'src/reasoning/rules.pl'
            }
        }
        mock_load_model.return_value = MagicMock()
        mock_ppo_load.return_value = MagicMock()
        mock_reasoner.return_value = MagicMock()

        core = ANSRDTCore('/path/to/config.yaml', logger=self.logger)

        mock_load_config.assert_called_once_with('/path/to/config.yaml')
        mock_load_model.assert_called_once_with('/absolute/path/to/results/best_model.keras', self.logger)
        mock_ppo_load.assert_called_once_with('/absolute/path/to/results/ppo_nexus_dt.zip')
        mock_reasoner.assert_called_once_with('/absolute/path/to/src/reasoning/rules.pl')
        self.assertIsNotNone(core.cnn_lstm)
        self.assertIsNotNone(core.ppo_agent)
        self.assertIsNotNone(core.reasoner)

    def test_preprocess_data_invalid_shape(self):
        with self.assertRaises(ValueError):
            self.core.preprocess_data(np.array([1, 2, 3]))  # Invalid shape

    def test_update_state(self):
        sensor_data = np.random.rand(10, 7)  # window_size=10, features=7
        with patch.object(self.core, 'cnn_lstm') as mock_cnn:
            mock_cnn.predict.return_value = np.array([0.6])  # Anomaly score > 0.5
            with patch.object(self.core, 'ppo_agent') as mock_ppo:
                mock_ppo.predict.return_value = ([0.1, 0.2, 0.3], None)
                with patch.object(self.core, 'reasoner') as mock_reasoner:
                    mock_reasoner.reason.return_value = ['High temperature', 'Low pressure']
                    state = self.core.update_state(sensor_data)
                    self.assertEqual(state['anomaly_score'], 0.6)
                    self.assertEqual(state['recommended_action'], [0.1, 0.2, 0.3])
                    self.assertEqual(state['insights'], ['High temperature', 'Low pressure'])
                    self.assertEqual(len(self.core.state_history), 1)

if __name__ == '__main__':
    unittest.main()
