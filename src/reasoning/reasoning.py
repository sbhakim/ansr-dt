# src/reasoning/reasoning.py

import logging
from pyswip import Prolog, Functor, Variable, Query
import os
import numpy as np

class SymbolicReasoner:
    def __init__(self, rules_path: str):
        """
        Initializes the Symbolic Reasoner with the specified Prolog rules file.

        Parameters:
        - rules_path (str): Path to the Prolog rules file.
        """
        self.logger = logging.getLogger(__name__)
        self.prolog = Prolog()
        if not os.path.exists(rules_path):
            self.logger.error(f"Prolog rules file not found at: {rules_path}")
            raise FileNotFoundError(f"Prolog rules file not found at: {rules_path}")
        self.prolog.consult(rules_path)
        self.logger.info(f"Prolog rules loaded from {rules_path}")

    def reason(self, sensor_data: dict) -> list:
        """
        Applies symbolic reasoning rules to the provided sensor data.

        Parameters:
        - sensor_data (dict): Dictionary containing sensor readings.
            Expected keys: 'temperature', 'vibration', 'pressure', 'operational_hours', 'efficiency_index'

        Returns:
        - insights (list): List of inferred states/actions based on rules.
        """
        insights = []
        try:
            # Extract and preprocess sensor values
            temperature = float(sensor_data.get('temperature', 0))
            vibration = float(sensor_data.get('vibration', 0))
            pressure = float(sensor_data.get('pressure', 0))
            # Convert operational hours to integer and ensure it's positive
            operational_hours = max(0, int(round(float(sensor_data.get('operational_hours', 0)))))
            efficiency_index = float(sensor_data.get('efficiency_index', 0))

            # Apply rules with proper type conversion
            # For degraded state (using float values)
            try:
                for solution in self.prolog.query(f"degraded_state({temperature}, {vibration})."):
                    insights.append("Degraded State")
                    break
            except Exception as e:
                self.logger.warning(f"Error in degraded_state query: {e}")

            # For system stress (using float values)
            try:
                for solution in self.prolog.query(f"system_stress({pressure})."):
                    insights.append("System Stress")
                    break
            except Exception as e:
                self.logger.warning(f"Error in system_stress query: {e}")

            # For critical state (using float values)
            try:
                for solution in self.prolog.query(f"critical_state({efficiency_index})."):
                    insights.append("Critical State")
                    break
            except Exception as e:
                self.logger.warning(f"Error in critical_state query: {e}")

            # For maintenance (using integer values)
            try:
                for solution in self.prolog.query(f"maintenance_needed({operational_hours})."):
                    insights.append("Maintenance Required")
                    break
            except Exception as e:
                self.logger.warning(f"Error in maintenance_needed query: {e}")

            return insights

        except Exception as e:
            self.logger.error(f"Error during symbolic reasoning: {e}")
            raise