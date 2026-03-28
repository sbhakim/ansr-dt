# tests/test_symbolic_reasoning.py

import logging
import unittest

from src.reasoning.reasoning import SymbolicReasoner


class TestSymbolicReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger('TestSymbolicReasoning')
        cls.logger.setLevel(logging.DEBUG)
        if not cls.logger.handlers:
            cls.logger.addHandler(logging.StreamHandler())
        cls.reasoner = SymbolicReasoner('src/reasoning/rules.pl', input_shape=(10, 7), model=None, logger=cls.logger)

    def test_degraded_state(self):
        insights = self.reasoner.reason({
            'temperature': 85, 'vibration': 60, 'pressure': 25, 'operational_hours': 1000,
            'efficiency_index': 0.7, 'system_state': 1, 'performance_score': 85,
        })
        self.assertTrue(any(item.startswith('Degraded State Triggered') for item in insights))

    def test_system_stress(self):
        insights = self.reasoner.reason({
            'temperature': 75, 'vibration': 50, 'pressure': 15, 'operational_hours': 2000,
            'efficiency_index': 0.8, 'system_state': 1, 'performance_score': 85,
        })
        self.assertTrue(any(item.startswith('System Stress Triggered') for item in insights))

    def test_critical_state(self):
        insights = self.reasoner.reason({
            'temperature': 70, 'vibration': 50, 'pressure': 25, 'operational_hours': 3000,
            'efficiency_index': 0.5, 'system_state': 2, 'performance_score': 85,
        })
        self.assertTrue(any(item.startswith('Critical State Triggered') for item in insights))

    def test_maintenance_required(self):
        insights = self.reasoner.reason({
            'temperature': 75, 'vibration': 50, 'pressure': 25, 'operational_hours': 4000,
            'efficiency_index': 0.8, 'system_state': 0, 'performance_score': 85,
        })
        self.assertTrue(any(item.startswith('Maintenance Needed Soon') for item in insights))

    def test_multiple_insights(self):
        insights = self.reasoner.reason({
            'temperature': 85, 'vibration': 60, 'pressure': 15, 'operational_hours': 5000,
            'efficiency_index': 0.5, 'system_state': 2, 'performance_score': 90,
        })
        expected_prefixes = [
            'Degraded State Triggered',
            'System Stress Triggered',
            'Critical State Triggered',
            'Maintenance Needed Soon',
        ]
        for prefix in expected_prefixes:
            self.assertTrue(any(item.startswith(prefix) for item in insights))


if __name__ == '__main__':
    unittest.main()
