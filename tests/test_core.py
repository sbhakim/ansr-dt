# tests/test_core.py

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.ansrdt.core import ANSRDTCore


class TestANSRDTCore(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.config = {
            'model': {
                'window_size': 10,
                'feature_names': [
                    'temperature',
                    'vibration',
                    'pressure',
                    'operational_hours',
                    'efficiency_index',
                    'system_state',
                    'performance_score',
                ],
            },
            'paths': {
                'results_dir': 'results',
                'reasoning_rules_path': 'src/reasoning/rules.pl',
            },
            'symbolic_reasoning': {'enabled': True},
            'logging': {'state_history_limit': 5},
        }

    def _build_core(self, cnn_lstm=None, ppo_agent=None, reasoner=None):
        cnn_lstm = cnn_lstm or MagicMock()
        ppo_agent = ppo_agent or MagicMock()
        with patch('src.ansrdt.core.load_config', return_value=self.config), \
             patch.object(ANSRDTCore, '_initialize_reasoner', return_value=reasoner):
            return ANSRDTCore('configs/config.yaml', logger=self.logger, cnn_lstm_model=cnn_lstm, ppo_agent=ppo_agent)

    def test_initialization_with_injected_components(self):
        reasoner = MagicMock()
        cnn_lstm = MagicMock()
        ppo_agent = MagicMock()

        core = self._build_core(cnn_lstm=cnn_lstm, ppo_agent=ppo_agent, reasoner=reasoner)

        self.assertIs(core.cnn_lstm, cnn_lstm)
        self.assertIs(core.ppo_agent, ppo_agent)
        self.assertIs(core.reasoner, reasoner)
        self.assertEqual(core.window_size, 10)
        self.assertEqual(len(core.feature_names), 7)
        self.assertTrue(core.results_dir.endswith('results'))

    def test_preprocess_data_invalid_shape(self):
        core = self._build_core()
        with self.assertRaises(ValueError):
            core.preprocess_data(np.array([1, 2, 3]))

    def test_update_state(self):
        cnn_lstm = MagicMock()
        cnn_lstm.predict.return_value = np.array([[0.6]], dtype=np.float32)

        ppo_agent = MagicMock()
        ppo_agent.predict.return_value = (np.array([0.1, 0.2, 0.3]), None)
        ppo_agent.action_space.shape = (3,)

        reasoner = MagicMock()
        reasoner.reason.return_value = ['High temperature', 'Low pressure']
        reasoner.get_rule_activations.return_value = [{'activated_rules_detailed': ['rule_a']}]
        reasoner.state_tracker.update.return_value = {'transition_matrix': [[0, 1], [1, 0]]}

        core = self._build_core(cnn_lstm=cnn_lstm, ppo_agent=ppo_agent, reasoner=reasoner)
        core.adaptive_controller.adapt_control_parameters = MagicMock(return_value={
            'temperature_adjustment': 0.5,
            'vibration_adjustment': -0.1,
            'pressure_adjustment': 0.2,
            'efficiency_target': 0.9,
        })

        sensor_data = np.random.rand(10, 7).astype(np.float32)
        state = core.update_state(sensor_data)

        self.assertAlmostEqual(state['anomaly_score'], 0.6, places=5)
        self.assertEqual(state['recommended_action'], [0.1, 0.2, 0.3])
        self.assertEqual(state['insights'], ['High temperature', 'Low pressure'])
        self.assertEqual(state['rule_activations'], ['rule_a'])
        self.assertEqual(len(core.state_history), 1)


if __name__ == '__main__':
    unittest.main()
