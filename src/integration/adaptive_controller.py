import numpy as np
from typing import Dict, List, Tuple
import logging


class AdaptiveController:
    def __init__(self, window_size: int = 10):
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.control_history = []

    def adapt_control_parameters(self,
                                 current_state: Dict[str, float],
                                 predictions: np.ndarray,
                                 rule_activations: List[Dict]) -> Dict[str, float]:
        """Adapt control parameters based on system state and predictions."""
        try:
            # Extract current values with defaults
            temperature = current_state.get('temperature', 70.0)
            vibration = current_state.get('vibration', 50.0)
            pressure = current_state.get('pressure', 30.0)
            efficiency = current_state.get('efficiency_index', 0.8)

            # Calculate adjustments based on system state
            temp_adjust = self._calculate_adjustment(temperature, 70.0, 0.1)
            vib_adjust = self._calculate_adjustment(vibration, 50.0, 0.1)
            press_adjust = self._calculate_adjustment(pressure, 30.0, 0.1)

            # Modify adjustments based on rule activations
            if rule_activations:
                for rule in rule_activations:
                    if rule.get('confidence', 0.0) > 0.8:
                        temp_adjust *= 1.2
                        vib_adjust *= 1.2
                        press_adjust *= 1.2

            control_params = {
                'temperature_adjustment': temp_adjust,
                'vibration_adjustment': vib_adjust,
                'pressure_adjustment': press_adjust,
                'efficiency_target': max(0.8, efficiency)
            }

            self.control_history.append(control_params)
            if len(self.control_history) > self.window_size:
                self.control_history.pop(0)

            return control_params

        except Exception as e:
            self.logger.error(f"Error in control adaptation: {e}")
            return {
                'temperature_adjustment': 0.0,
                'vibration_adjustment': 0.0,
                'pressure_adjustment': 0.0,
                'efficiency_target': 0.8
            }

    def _calculate_adjustment(self, current: float, target: float, rate: float) -> float:
        """Calculate smooth adjustment towards target."""
        diff = target - current
        return rate * diff