# src/ansrdt/ansr_dt_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import logging
from typing import Optional, Dict, Tuple, Any


class ANSRDTEnv(gym.Env):
    """
    Custom Environment for ANSR-DT Framework integrating PPO.
    This environment simulates the interaction of the PPO agent with the
    digital twin system, using pre-loaded synthetic data and calculating
    rewards based on efficiency, safety, and potentially action costs.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, data_file: str, window_size: int = 10, config: Optional[Dict] = None):
        """Initialize environment."""
        super(ANSRDTEnv, self).__init__()

        self.logger = logging.getLogger(__name__)

        # Load data
        try:
            # Allow loading from .npz files
            if not os.path.exists(data_file):
                 raise FileNotFoundError(f"Data file not found at {data_file}")
            self.data = np.load(data_file)
            self._validate_data()
            self.logger.info(f"Data loaded from {data_file}. Available keys: {list(self.data.keys())}")
        except Exception as e:
            self.logger.error(f"Error loading data from {data_file}: {e}", exc_info=True)
            raise

        # Environment parameters
        self.window_size = window_size
        if self.window_size <= 0:
             raise ValueError("window_size must be positive.")
        self.current_step = 0
        self.num_samples = len(self.data['temperature'])
        if self.num_samples <= self.window_size:
             raise ValueError(f"Data length ({self.num_samples}) must be greater than window size ({self.window_size}).")
        self.max_steps = self.num_samples - window_size # Max steps agent can take

        # Determine number of features (must match _get_observation)
        self.num_features = 7 # Hardcoded based on _get_observation - consider making dynamic

        # Initialize observation history with correct shape (window_size, features)
        self.history = np.zeros((self.window_size, self.num_features), dtype=np.float32)

        # Define action and observation spaces
        # Action: Adjustments to Temperature, Vibration, Pressure
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0]), # Allow adjustments within +/- 5 units
            high=np.array([5.0, 5.0, 5.0]),
            dtype=np.float32
        )

        # Observation: Window of past sensor/metric data
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features), # (timesteps, features)
            dtype=np.float32
        )

        # Load weights for reward calculation from config or use defaults
        self.weights = self._load_weights(config)
        self.logger.info(f"Reward weights loaded: {self.weights}")

        # Target values (used in old reward, kept for reference or potential future use)
        self.targets = {
            'temperature': 70.0,
            'vibration': 50.0,
            'pressure': 30.0
        }

        # Initialize history with first window_size observations
        self._initialize_history()

    def _initialize_history(self):
         """Fills the initial history buffer."""
         try:
             for i in range(self.window_size):
                 self.history[i] = self._get_observation(i)
             self.logger.debug("History initialized.")
         except IndexError as e:
              self.logger.error(f"Error initializing history: Index out of bounds. Data length {self.num_samples}, required index {i}. Error: {e}")
              raise ValueError("Not enough data points to initialize history.") from e
         except Exception as e:
              self.logger.error(f"Unexpected error initializing history: {e}", exc_info=True)
              raise


    def _validate_data(self):
        """Validate required data fields and shapes."""
        required_fields = [
            'temperature', 'vibration', 'pressure',
            'operational_hours', 'efficiency_index',
            'system_state', 'performance_score'
        ]

        missing = [field for field in required_fields if field not in self.data]
        if missing:
            raise KeyError(f"Missing required fields in data file: {missing}")

        # Check data length consistency
        lengths = {field: len(self.data[field]) for field in required_fields}
        first_len = lengths[required_fields[0]]
        if not all(l == first_len for l in lengths.values()):
             raise ValueError(f"Inconsistent data lengths found: {lengths}")
        self.logger.debug("Data fields validated.")

    def _load_weights(self, config: Optional[Dict]) -> Dict[str, float]:
        """Load reward weights from config."""
        if config is None:
            config = {}
        # Navigate potentially nested structure for ppo settings
        ppo_config = config.get('ppo', {})
        weights = ppo_config.get('reward_weights', {})
        # Use defaults if keys are missing
        return {
            'efficiency': weights.get('efficiency', 1.0), # Weight for efficiency_index
            'safety': weights.get('safety', 1.0),         # Weight for system_state penalty
            'satisfaction': weights.get('satisfaction', 0.1) # Weight for action cost/smoothness
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment state."""
        super().reset(seed=seed)

        # Reset tracking variables
        self.current_step = 0
        self._initialize_history() # Re-initialize the history buffer

        self.logger.info(f"Environment reset. Current step: {self.current_step}")
        # Return the initial observation and an empty info dict
        return self.history.copy(), {}

    def _get_observation(self, index: int) -> np.ndarray:
        """Get observation vector for a given data index."""
        # Check index bounds
        if not (0 <= index < self.num_samples):
             raise IndexError(f"Attempted to access index {index} outside of data bounds [0, {self.num_samples-1}]")
        try:
             # Order must match self.num_features and observation_space shape
             obs = np.array([
                 self.data['temperature'][index],
                 self.data['vibration'][index],
                 self.data['pressure'][index],
                 self.data['operational_hours'][index],
                 self.data['efficiency_index'][index],
                 self.data['system_state'][index],
                 self.data['performance_score'][index]
             ], dtype=np.float32)
             # Check for NaN/Inf values
             if np.isnan(obs).any() or np.isinf(obs).any():
                 self.logger.warning(f"NaN or Inf detected in observation at index {index}. Replacing with 0.")
                 obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
             return obs
        except KeyError as e:
             self.logger.error(f"KeyError accessing data at index {index}: {e}")
             raise
        except Exception as e:
             self.logger.error(f"Error creating observation at index {index}: {e}", exc_info=True)
             raise

    # <<< SUGGESTION 1: Implemented calculate_reward method >>>
    def calculate_reward(self, current_obs_vector: np.ndarray, action: np.ndarray) -> float:
        """
        Calculate reward based on current state metrics and agent action.
        Aligns with the multi-objective approach (Efficiency, Safety, Satisfaction).

        Args:
            current_obs_vector (np.ndarray): The observation vector representing the state *after* the action.
                                             Shape: (num_features,)
            action (np.ndarray): The action taken by the agent. Shape: (action_dim,)

        Returns:
            float: The calculated reward.
        """
        try:
            # Extract metrics from the observation vector representing the state AFTER the action
            # Indices match _get_observation: 4=efficiency, 5=state
            # Ensure casting to appropriate types
            efficiency_index = float(current_obs_vector[4])
            system_state = int(current_obs_vector[5])

            # --- 1. Efficiency Component ---
            eff_weight = self.weights.get('efficiency', 1.0)
            # Reward based on being close to ideal efficiency (1.0)
            # Using (1 - abs(1.0 - eff_index)) rewards values closer to 1
            efficiency_reward = eff_weight * (1.0 - abs(1.0 - efficiency_index))

            # --- 2. Safety Component ---
            safety_weight = self.weights.get('safety', 1.0)
            # Penalize undesirable states (Degraded=1, Critical=2)
            state_penalty = 0.0
            if system_state == 2: # Critical state
                state_penalty = -1.0  # Strong penalty
            elif system_state == 1: # Degraded state
                state_penalty = -0.5  # Moderate penalty
            safety_reward = safety_weight * state_penalty

            # --- 3. Satisfaction Component (Action Cost/Smoothness Proxy) ---
            sat_weight = self.weights.get('satisfaction', 0.1) # Lower weight for penalty
            # Penalize large magnitude actions to encourage smoother control
            action_magnitude = np.linalg.norm(action)
            # Example: quadratic penalty increases faster for larger actions
            action_cost_penalty = -sat_weight * (action_magnitude ** 2) * 0.1 # Scaled penalty

            # --- Combine Rewards ---
            total_reward = efficiency_reward + safety_reward + action_cost_penalty

            self.logger.debug(f"Reward Calc: Eff={efficiency_reward:.3f} (Idx:{efficiency_index:.3f}), Safety={safety_reward:.3f} (State:{system_state}), ActionCost={action_cost_penalty:.3f} (Mag:{action_magnitude:.2f}), Total={total_reward:.3f}")
            return total_reward

        except IndexError as e:
            self.logger.error(f"IndexError during reward calculation. Obs vector shape: {current_obs_vector.shape}. Error: {e}")
            return 0.0 # Return neutral reward on error
        except Exception as e:
            self.logger.error(f"Unexpected error during reward calculation: {e}", exc_info=True)
            return 0.0 # Return neutral reward on error

    # <<< SUGGESTION 1: Modified step method >>>
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        Applies the action, updates the state history, calculates the reward using the
        multi-objective function, and determines if the episode is done.
        """
        try:
            # --- 1. Determine Current Data Index ---
            # The action affects the transition *to* the next state, which corresponds
            # to the data point at the end of the *next* window.
            # current_step starts at 0. The first observation is indices 0..window_size-1.
            # The first step (current_step=0) uses action based on initial obs,
            # calculates reward based on state at index window_size.
            data_index = self.current_step + self.window_size
            if data_index >= self.num_samples:
                # If we would step beyond the data, terminate the episode
                self.logger.warning(f"Step {self.current_step}: Attempting to access data index {data_index} beyond data length {self.num_samples}. Terminating.")
                # Return the *last valid* history, 0 reward, terminated=True
                return self.history.copy(), 0.0, True, False, {'status': 'terminated_data_end'}

            # --- 2. Simulate Action Effect ---
            # Get the *actual* sensor readings for the *next* timestep (which the action influences)
            # Note: In a real system, the action would influence future readings. Here, we simulate
            # by adjusting the *target* state's sensor values conceptually for reward calc,
            # but the *observed* state uses the actual data metrics.
            # Let's get the actual observation vector for the *next* state (index `data_index`).
            next_state_obs_actual = self._get_observation(data_index)

            # Clipping action before using it in reward calculation or info
            clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

            # --- 3. Calculate Reward ---
            # The reward is based on the outcome state (next_state_obs_actual) and the action taken.
            reward = self.calculate_reward(next_state_obs_actual, clipped_action)

            # --- 4. Update History ---
            # Roll the history buffer and add the *actual* observation of the new state
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = next_state_obs_actual # Add the actual next state observation

            # --- 5. Update Step Counter ---
            self.current_step += 1

            # --- 6. Check Termination Conditions ---
            terminated = self.current_step >= self.max_steps # Reached the end of the data for stepping
            truncated = False # We are not truncating based on time limit within an episode here

            # --- 7. Populate Info Dictionary ---
            info = {
                'current_step': self.current_step,
                'reward_calculated': reward, # The reward returned
                'action_applied': clipped_action.tolist(), # Log the potentially clipped action
                'state_vector_used_for_reward': next_state_obs_actual.tolist(), # Log the state vector
                # Add actual metrics from the state for context
                'actual_efficiency': float(next_state_obs_actual[4]),
                'actual_system_state': int(next_state_obs_actual[5]),
            }

            self.logger.debug(f"Step {self.current_step}: Action={clipped_action.tolist()}, Reward={reward:.4f}, Terminated={terminated}")

            # Return: observation (new history), reward, terminated, truncated, info
            return self.history.copy(), reward, terminated, truncated, info

        except IndexError as e:
             self.logger.error(f"IndexError during step {self.current_step} accessing data index {data_index}: {e}", exc_info=True)
             # Terminate episode on critical data access error
             return self.history.copy(), 0.0, True, False, {'error': 'IndexError during step execution'}
        except Exception as e:
            self.logger.error(f"Unexpected error during step {self.current_step}: {e}", exc_info=True)
            # Terminate episode on critical error
            return self.history.copy(), 0.0, True, False, {'error': f'Unexpected error: {str(e)}'}

    def get_current_state(self) -> Dict[str, float]:
        """Get current state sensor values from the latest history entry."""
        # Indices: 0=temp, 1=vib, 2=press
        # Check if history is populated
        if self.history.shape[0] == self.window_size:
             latest_obs = self.history[-1]
             return {
                 'temperature': float(latest_obs[0]),
                 'vibration': float(latest_obs[1]),
                 'pressure': float(latest_obs[2])
                 # Add other metrics if needed by external callers
             }
        else:
             self.logger.warning("Attempted to get current state before history is fully initialized.")
             return { 'temperature': 0.0, 'vibration': 0.0, 'pressure': 0.0 }


    def render(self, mode='human'):
        """Render the environment (optional). Not implemented."""
        pass

    def close(self):
        """Cleanup when the environment is closed."""
        self.logger.info("Closing ANSRDTEnv environment.")
        pass # No specific resources to close in this version