import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
import logging


class NeurosymbolicVisualizer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def visualize_rule_activations(self,
                                   activations: List[Dict],
                                   save_path: str):
        """Visualize pattern of rule activations over time."""
        try:
            timesteps = [a['timestep'] for a in activations]
            confidences = [a['confidence'] for a in activations]

            plt.figure(figsize=(10, 6))
            plt.scatter(timesteps, confidences, alpha=0.6)
            plt.xlabel('Timestep')
            plt.ylabel('Rule Confidence')
            plt.title('Rule Activation Patterns')
            plt.grid(True, alpha=0.3)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error in rule activation visualization: {e}")

    def plot_state_transitions(self,
                               transition_matrix: np.ndarray,
                               save_path: str):
        """Visualize state transition patterns."""
        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(transition_matrix, cmap='YlOrRd')
            plt.colorbar(label='Transition Count')

            states = ['Normal', 'Degraded', 'Critical']
            plt.xticks(range(3), states)
            plt.yticks(range(3), states)
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.title('System State Transitions')

            # Add text annotations
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, f'{transition_matrix[i, j]:.0f}',
                             ha='center', va='center')

            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error in state transition visualization: {e}")