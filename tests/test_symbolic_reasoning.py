# tests/test_symbolic_reasoning.py

import unittest
import logging
from src.reasoning.reasoning import SymbolicReasoner
import os


class TestSymbolicReasoner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize logger
        cls.logger = logging.getLogger('TestSymbolicReasoner')
        cls.logger.setLevel(logging.DEBUG)
        if not cls.logger.handlers:
            stream_handler = logging.StreamHandler()
            cls.logger.addHandler(stream_handler)

        # Path to Prolog rules
        cls.rules_path = 'src/reasoning/rules.pl'
        if not os.path.exists(cls.rules_path):
            raise FileNotFoundError(f"Prolog rules file not found at: {cls.rules_path}")

        # Initialize SymbolicReasoner
        cls.reasoner = SymbolicReasoner(cls.rules_path)

    def test_degraded_state(self):
        sensor_data = {
            'temperature': 85,
            'vibration': 60,
            'pressure': 25,
            'operational_hours': 1000,
            'efficiency_index': 0.7
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertIn("Degraded State", insights)

    def test_system_stress(self):
        sensor_data = {
            'temperature': 75,
            'vibration': 50,
            'pressure': 15,
            'operational_hours': 2000,
            'efficiency_index': 0.8
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertIn("System Stress", insights)

    def test_critical_state(self):
        sensor_data = {
            'temperature': 70,
            'vibration': 50,
            'pressure': 25,
            'operational_hours': 3000,
            'efficiency_index': 0.5
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertIn("Critical State", insights)

    def test_maintenance_required(self):
        sensor_data = {
            'temperature': 75,
            'vibration': 50,
            'pressure': 25,
            'operational_hours': 4000,  # Multiple of 1000
            'efficiency_index': 0.8
        }
        insights = self.reasoner.reason(sensor_data)
        self.assertIn("Maintenance Required", insights)

    def test_multiple_insights(self):
        sensor_data = {
            'temperature': 85,
            'vibration': 60,
            'pressure': 15,
            'operational_hours': 5000,  # Multiple of 1000
            'efficiency_index': 0.5
        }
        insights = self.reasoner.reason(sensor_data)
        expected_insights = [
            "Degraded State",
            "System Stress",
            "Critical State",
            "Maintenance Required"
        ]
        for insight in expected_insights:
            self.assertIn(insight, insights)

        def test_no_insights(self):
            """Test when sensor data is within normal operating ranges."""
            sensor_data = {
                'temperature': 70,  # Normal temperature
                'vibration': 30,  # Low vibration
                'pressure': 25,  # Normal pressure
                'operational_hours': 500,  # Low operational hours
                'efficiency_index': 0.9  # High efficiency
            }
            insights = self.reasoner.reason(sensor_data)
            self.assertEqual(len(insights), 0, "Expected no insights for normal operating conditions")

        def test_boundary_conditions(self):
            """Test behavior at boundary conditions."""
            sensor_data = {
                'temperature': 80,  # At threshold
                'vibration': 55,  # At threshold
                'pressure': 20,  # At threshold
                'operational_hours': 1000,  # At maintenance threshold
                'efficiency_index': 0.6  # At threshold
            }
            insights = self.reasoner.reason(sensor_data)
            self.assertIsInstance(insights, list)
            self.assertTrue(any(isinstance(insight, str) for insight in insights))

        def test_missing_data_handling(self):
            """Test handling of missing sensor data."""
            incomplete_sensor_data = {
                'temperature': 75,
                'vibration': 50
                # Missing other fields
            }
            with self.assertRaises(KeyError):
                self.reasoner.reason(incomplete_sensor_data)

        def test_invalid_data_types(self):
            """Test handling of invalid data types."""
            invalid_sensor_data = {
                'temperature': "85",  # String instead of number
                'vibration': 60,
                'pressure': 25,
                'operational_hours': 1000,
                'efficiency_index': 0.7
            }
            with self.assertRaises((ValueError, TypeError)):
                self.reasoner.reason(invalid_sensor_data)

        def test_extreme_values(self):
            """Test handling of extreme sensor values."""
            extreme_sensor_data = {
                'temperature': 150,  # Very high temperature
                'vibration': 100,  # Very high vibration
                'pressure': 50,  # Very high pressure
                'operational_hours': 10000,  # Very high hours
                'efficiency_index': 0.1  # Very low efficiency
            }
            insights = self.reasoner.reason(extreme_sensor_data)
            self.assertGreater(len(insights), 0, "Expected insights for extreme conditions")
            self.assertIn("Critical State", insights)

        def test_zero_values(self):
            """Test handling of zero values."""
            zero_sensor_data = {
                'temperature': 0,
                'vibration': 0,
                'pressure': 0,
                'operational_hours': 0,
                'efficiency_index': 0
            }
            insights = self.reasoner.reason(zero_sensor_data)
            self.assertIsInstance(insights, list)

        @classmethod
        def tearDownClass(cls):
            """Clean up any resources after all tests are run."""
            cls.logger.handlers.clear()

    if __name__ == '__main__':
        unittest.main()

