# tests/test_reasoning.py

import unittest
from src.reasoning.reasoning import SymbolicReasoner

class TestSymbolicReasoner(unittest.TestCase):
    def setUp(self):
        self.rules_path = 'src/reasoning/rules.pl'
        self.reasoner = SymbolicReasoner(self.rules_path)

    def test_reason_with_correct_operational_hours(self):
        sensor_data = {
            'temperature': 85.0,
            'vibration': 60.0,
            'pressure': 15.0,
            'operational_hours': 1000,  # Integer
            'efficiency_index': 0.5,
            'system_state': 1.0,
            'performance_score': 90.0
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertIn("Degraded State (Base Rule)", insights)
        self.assertIn("System Stress (Base Rule)", insights)
        self.assertIn("Critical State (Base Rule)", insights)
        self.assertIn("Maintenance Needed (Base Rule)", insights)

    def test_reason_with_incorrect_operational_hours(self):
        sensor_data = {
            'temperature': 85.0,
            'vibration': 60.0,
            'pressure': 15.0,
            'operational_hours': 1000.0,  # Float
            'efficiency_index': 0.5,
            'system_state': 1.0,
            'performance_score': 90.0
        }
        insights = self.reasoner.reason(sensor_data)
        # Since operational_hours should be cast to int, it should work
        self.assertIn("Maintenance Needed (Base Rule)", insights)

if __name__ == '__main__':
    unittest.main()
