# tests/test_reasoning.py

import logging
import unittest

from src.reasoning.reasoning import SymbolicReasoner


class TestSymbolicReasoner(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger('TestReasoning')
        logger.addHandler(logging.NullHandler())
        self.reasoner = SymbolicReasoner('src/reasoning/rules.pl', input_shape=(10, 7), model=None, logger=logger)

    def test_reason_with_correct_operational_hours(self):
        sensor_data = {
            'temperature': 85.0,
            'vibration': 60.0,
            'pressure': 15.0,
            'operational_hours': 1000,
            'efficiency_index': 0.5,
            'system_state': 1.0,
            'performance_score': 90.0,
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertTrue(any(item.startswith('Degraded State Triggered') for item in insights))
        self.assertTrue(any(item.startswith('System Stress Triggered') for item in insights))
        self.assertTrue(any(item.startswith('Critical State Triggered') for item in insights))
        self.assertTrue(any(item.startswith('Maintenance Needed Soon') for item in insights))

    def test_reason_with_float_operational_hours(self):
        sensor_data = {
            'temperature': 85.0,
            'vibration': 60.0,
            'pressure': 15.0,
            'operational_hours': 1000.0,
            'efficiency_index': 0.5,
            'system_state': 1.0,
            'performance_score': 90.0,
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertTrue(any(item.startswith('Maintenance Needed Soon') for item in insights))


if __name__ == '__main__':
    unittest.main()
