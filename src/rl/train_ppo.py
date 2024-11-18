import os
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.nexusdt.nexus_dt_env import NexusDTEnv


def setup_logger(log_file: str = 'logs/ppo_training.log') -> logging.Logger:
    """Setup basic logging configuration."""
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
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.getLogger('PPOTraining').error(f"Failed to load configuration: {e}")
        raise


def train_ppo_agent(config_path: str) -> bool:
    """
    Train PPO agent for NEXUS-DT system.

    Args:
        config_path (str): Path to configuration file

    Returns:
        bool: Success status of training
    """
    logger = setup_logger()

    try:
        # Load configuration
        config = load_config(config_path)

        # Setup directories
        results_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), '..', config['paths']['results_dir']))
        os.makedirs(results_dir, exist_ok=True)

        # Initialize environment with proper paths
        data_file = os.path.join(os.path.dirname(results_dir), config['paths']['data_file'])

        env = DummyVecEnv([lambda: NexusDTEnv(
            data_file=data_file,
            window_size=config['model']['window_size'],
            config=config.get('ppo', {})
        )])

        # Initialize PPO with optimal parameters for this task
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(results_dir, 'logs/ppo_tensorboard/')
        )

        logger.info("Starting PPO training...")

        # Train the agent
        total_timesteps = config.get('ppo', {}).get('total_timesteps', 100000)
        model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        model_path = os.path.join(results_dir, 'ppo_nexus_dt.zip')
        model.save(model_path)

        if not os.path.exists(model_path):
            raise RuntimeError("Failed to save PPO model")

        logger.info(f"PPO training completed. Model saved to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Error during PPO training: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO Agent for NEXUS-DT')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    success = train_ppo_agent(config_path)

    if not success:
        exit(1)