# src/nexusdt/nexus_dt_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import logging
from typing import Optional, Dict, Tuple, Any


class NexusDTEnv(gym.Env):
    """Custom Environment for NEXUS-DT Framework integrating PPO."""

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, data_file: str, window_size: int = 10, config: Optional[Dict] = None):
        """
        Initialize the environment.

        Args:
            data_file: Path to the data file containing sensor readings
            window_size: Number of timesteps to include in each observation
            config: Additional configuration parameters
        """
        super(NexusDTEnv, self).__init__()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Load and validate data
        if not os.path.exists(data_file):
            self.logger.error(f"Data file not found at {data_file}")
            raise FileNotFoundError(f"Data file not found at {data_file}")

        try:
            self.data = np.load(data_file)
            required_keys = ['temperature', 'vibration', 'pressure', 'fused', 'anomaly']
            for key in required_keys:
                if key not in self.data.files:
                    raise KeyError(f"Required data field '{key}' not found in data file")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

        # Store sensor data
        self.temperature = self.data['temperature']
        self.vibration = self.data['vibration']
        self.pressure = self.data['pressure']
        self.anomaly = self.data['anomaly']

        # Environment parameters
        self.window_size = window_size
        self.current_step = window_size
        self.episode_steps = 0
        self.max_episode_steps = len(self.temperature) - window_size

        # Initialize observation history
        self.history = np.zeros((window_size, 7))  # [temperature, vibration, pressure]

        # Action space: control adjustments for temperature, vibration, pressure
        # Each action is a continuous value between -5.0 and 5.0
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0]),
            high=np.array([5.0, 5.0, 5.0]),
            dtype=np.float32
        )

        # Observation space: window_size timesteps of 3 sensor readings
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 7),  # Updated shape for all features
            dtype=np.float32
        )

        # Load reward weights from config
        if config is None:
            config = {}
        weights = config.get('reward_weights', {})
        self.alpha1 = weights.get('efficiency', 1.0)  # Efficiency weight
        self.alpha2 = weights.get('satisfaction', 1.0)  # Satisfaction weight
        self.alpha3 = weights.get('safety', 1.0)  # Safety weight

        # Target values for optimal operation
        self.target_temperature = 70.0
        self.target_vibration = 50.0
        self.target_pressure = 30.0

        # Additional tracking variables
        self.total_reward = 0.0
        self.n_steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options for reset

        Returns:
            observation (np.ndarray): The initial observation.
            info (dict): Additional information (empty in this case).
        """
        super().reset(seed=seed)

        # Reset internal state
        self.current_step = self.window_size
        self.episode_steps = 0
        self.total_reward = 0.0
        self.n_steps = 0

        # Initialize history with first window_size observations
        for i in range(self.window_size):
            self.history[i] = np.array([
                self.temperature[i],
                self.vibration[i],
                self.pressure[i],
                float(self.data['operational_hours'][i]),
                float(self.data['efficiency_index'][i]),
                float(self.data['system_state'][i]),
                float(self.data['performance_score'][i])
            ])

        # Optionally, include additional info
        info = {}

        return self.history, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of control adjustments [temp_adj, vib_adj, press_adj]

        Returns:
            observation: Next state observation
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.episode_steps += 1

        # Apply actions to controllable variables
        adjusted_temperature = self.temperature[self.current_step] + action[0]
        adjusted_vibration = self.vibration[self.current_step] + action[1]
        adjusted_pressure = self.pressure[self.current_step] + action[2]

        # Get current values for non-controllable variables
        current_operational_hours = float(self.data['operational_hours'][self.current_step])
        current_efficiency_index = float(self.data['efficiency_index'][self.current_step])
        current_system_state = float(self.data['system_state'][self.current_step])
        current_performance_score = float(self.data['performance_score'][self.current_step])

        # Update history with new observations (all 7 features)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = np.array([
            adjusted_temperature,
            adjusted_vibration,
            adjusted_pressure,
            current_operational_hours,
            current_efficiency_index,
            current_system_state,
            current_performance_score
        ])

        # Calculate component rewards
        efficiency = 1.0 - abs(adjusted_temperature - self.target_temperature) / 100
        satisfaction = 1.0 - abs(adjusted_vibration - self.target_vibration) / 100
        safety = 1.0 - abs(adjusted_pressure - self.target_pressure) / 50

        # Calculate total reward
        reward = (
                self.alpha1 * efficiency +
                self.alpha2 * satisfaction +
                self.alpha3 * safety
        )

        # Update internal state
        self.current_step += 1
        self.total_reward += reward
        self.n_steps += 1

        # Check for episode termination
        terminated = False
        truncated = False

        if self.current_step >= len(self.temperature) - 1:
            terminated = True
        elif self.episode_steps >= self.max_episode_steps:
            truncated = True

        # Compile info dictionary
        info = {
            'episode_steps': self.episode_steps,
            'total_reward': self.total_reward,
            'efficiency': efficiency,
            'satisfaction': satisfaction,
            'safety': safety,
            'current_temperature': adjusted_temperature,
            'current_vibration': adjusted_vibration,
            'current_pressure': adjusted_pressure,
            'current_operational_hours': current_operational_hours,
            'current_efficiency_index': current_efficiency_index,
            'current_system_state': current_system_state,
            'current_performance_score': current_performance_score
        }

        return self.history, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: Rendering mode ('human' for human consumption)

        Returns:
            Optional rendered frame
        """
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current environment state.

        Returns:
            Dictionary containing current sensor values and metrics
        """
        return {
            'temperature': float(self.history[-1, 0]),
            'vibration': float(self.history[-1, 1]),
            'pressure': float(self.history[-1, 2]),
            'total_reward': float(self.total_reward),
            'steps': self.n_steps
        }

    def is_terminal_state(self) -> bool:
        """
        Check if current state is terminal.

        Returns:
            Boolean indicating if state is terminal
        """
        return (
            self.current_step >= len(self.temperature) - 1 or
            self.episode_steps >= self.max_episode_steps
        )

    def get_state_description(self) -> str:
        """
        Get human-readable description of current state.

        Returns:
            String describing current state
        """
        current_state = self.get_current_state()
        return (
            f"Temperature: {current_state['temperature']:.2f}, "
            f"Vibration: {current_state['vibration']:.2f}, "
            f"Pressure: {current_state['pressure']:.2f}, "
            f"Total Reward: {current_state['total_reward']:.2f}, "
            f"Steps: {current_state['steps']}"
        )
