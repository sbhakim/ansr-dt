import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import logging
from typing import Optional, Dict, Tuple, Any


class NexusDTEnv(gym.Env):
    """Custom Environment for NEXUS-DT Framework integrating PPO."""

    metadata = {'render_modes': ['human']}

    def __init__(self, data_file: str, window_size: int = 10, config: Optional[Dict] = None):
        """Initialize environment."""
        super().__init__()

        # Basic logging setup
        self.logger = logging.getLogger(__name__)

        # Load data
        try:
            self.data = np.load(data_file)
            self._validate_data()
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

        # Environment parameters
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(self.data['temperature']) - window_size

        # Initialize observation history with correct shape (window_size, features)
        self.history = np.zeros((self.window_size, 7))

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0]),
            high=np.array([5.0, 5.0, 5.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 7),
            dtype=np.float32
        )

        # Load weights from config or use defaults
        self.weights = self._load_weights(config)

        # Target values
        self.targets = {
            'temperature': 70.0,
            'vibration': 50.0,
            'pressure': 30.0
        }

    def _validate_data(self):
        """Validate required data fields."""
        required_fields = ['temperature', 'vibration', 'pressure',
                           'operational_hours', 'efficiency_index',
                           'system_state', 'performance_score']

        missing = [field for field in required_fields if field not in self.data]
        if missing:
            raise KeyError(f"Missing required fields: {missing}")

    def _load_weights(self, config: Optional[Dict]) -> Dict[str, float]:
        """Load reward weights from config."""
        if config is None:
            config = {}
        weights = config.get('reward_weights', {})
        return {
            'efficiency': weights.get('efficiency', 1.0),
            'satisfaction': weights.get('satisfaction', 1.0),
            'safety': weights.get('safety', 1.0)
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment state."""
        super().reset(seed=seed)

        # Reset tracking variables
        self.current_step = self.window_size

        # Initialize history with first window_size observations
        for i in range(self.window_size):
            self.history[i] = self._get_observation(i)

        return self.history, {}

    def _get_observation(self, index: int) -> np.ndarray:
        """Get observation vector for given index."""
        return np.array([
            self.data['temperature'][index],
            self.data['vibration'][index],
            self.data['pressure'][index],
            self.data['operational_hours'][index],
            self.data['efficiency_index'][index],
            self.data['system_state'][index],
            self.data['performance_score'][index]
        ])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute environment step."""
        # Apply actions
        adjusted_values = {
            'temperature': self.data['temperature'][self.current_step] + action[0],
            'vibration': self.data['vibration'][self.current_step] + action[1],
            'pressure': self.data['pressure'][self.current_step] + action[2]
        }

        # Update history
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = np.array([
            adjusted_values['temperature'],
            adjusted_values['vibration'],
            adjusted_values['pressure'],
            self.data['operational_hours'][self.current_step],
            self.data['efficiency_index'][self.current_step],
            self.data['system_state'][self.current_step],
            self.data['performance_score'][self.current_step]
        ])

        # Calculate reward components
        rewards = {
            'efficiency': 1.0 - abs(adjusted_values['temperature'] - self.targets['temperature']) / 100,
            'satisfaction': 1.0 - abs(adjusted_values['vibration'] - self.targets['vibration']) / 100,
            'safety': 1.0 - abs(adjusted_values['pressure'] - self.targets['pressure']) / 50
        }

        # Calculate total reward
        reward = sum(self.weights[k] * v for k, v in rewards.items())

        # Update state
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            'current_step': self.current_step,
            'reward_components': rewards,
            'adjusted_values': adjusted_values
        }

        return self.history, reward, terminated, truncated, info

    def get_current_state(self) -> Dict[str, float]:
        """Get current state values."""
        return {
            'temperature': float(self.history[-1, 0]),
            'vibration': float(self.history[-1, 1]),
            'pressure': float(self.history[-1, 2])
        }