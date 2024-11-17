# src/rl/train_ppo.py

import os
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.nexusdt.nexus_dt_env import NexusDTEnv  # Added import

def setup_logger(log_file: str = 'logs/ppo_training.log') -> logging.Logger:
    """
    Sets up the logger for PPO training.
    """
    logger = logging.getLogger('PPOTraining')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.getLogger('PPOTraining').error(f"Failed to load configuration: {e}")
        raise


def train_ppo_agent():
    """
    Trains the PPO agent and saves the trained model.
    """
    logger = setup_logger()
    config = load_config('configs/config.yaml')

    # Initialize environment
    env = DummyVecEnv([lambda: NexusDTEnv(
        data_file=config['paths']['data_file'],
        window_size=config['model']['window_size'],
        config=config.get('ppo', {})
    )])

    # Initialize PPO agent
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="logs/ppo_tensorboard/")

    # Force a small number of timesteps for quick development
    DEV_MODE_TIMESTEPS = 100  # Small number for quick testing
    total_timesteps = DEV_MODE_TIMESTEPS  # Override config value

    logger.info(f"Starting PPO training for {total_timesteps} timesteps (Development Mode).")
    model.learn(total_timesteps=total_timesteps)
    logger.info("PPO training completed.")

    # Save the trained agent
    ppo_model_path = os.path.join(config['paths']['results_dir'], 'ppo_nexus_dt')
    model.save(ppo_model_path)
    logger.info(f"PPO agent saved to {ppo_model_path}")

if __name__ == "__main__":
    train_ppo_agent()
