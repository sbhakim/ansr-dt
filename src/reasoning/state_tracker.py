# src/reasoning/state_tracker.py

import numpy as np
from typing import Dict, List
import logging


class StateTracker:
    def __init__(self):
        """Initialize state tracking."""
        self.logger = logging.getLogger(__name__)
        self.state_transitions = np.zeros((3, 3))  # 3 states: normal, degraded, critical
        self.state_history = []

    def update(self, current_state: Dict[str, float]) -> Dict[str, any]:
        """
        Track state changes and detect significant transitions.
        """
        try:
            # Get current system state
            system_state = int(current_state['system_state'])

            # Update history
            if len(self.state_history) > 0:
                prev_state = self.state_history[-1]
                prev_system_state = int(prev_state['system_state'])

                # Update transition matrix
                self.state_transitions[prev_system_state][system_state] += 1

            # Keep last 1000 states
            self.state_history.append(current_state)
            if len(self.state_history) > 1000:
                self.state_history.pop(0)

            return {
                'current_state': system_state,
                'transition_matrix': self.state_transitions.tolist()
            }

        except Exception as e:
            self.logger.error(f"Error in state tracking: {e}")
            return {}

    def get_transition_probabilities(self) -> np.ndarray:
        """Calculate state transition probabilities."""
        try:
            # Avoid division by zero
            row_sums = self.state_transitions.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return self.state_transitions / row_sums

        except Exception as e:
            self.logger.error(f"Error calculating transition probabilities: {e}")
            return np.zeros((3, 3))