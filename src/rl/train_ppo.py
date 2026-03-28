import os
import sys
import logging
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ansrdt.ansr_dt_env import ANSRDTEnv
from src.config.config_manager import load_config


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


def _resolve_path(project_root: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(project_root, path_value))


def train_ppo_agent(config_path: str) -> bool:
    """Train PPO agent for ANSR-DT system."""
    logger = setup_logger()

    try:
        config_path = os.path.abspath(config_path)
        config = load_config(config_path)
        project_root = os.path.dirname(os.path.dirname(config_path))

        results_dir = _resolve_path(project_root, config['paths'].get('results_dir', 'results'))
        data_file = _resolve_path(project_root, config['paths']['data_file'])
        os.makedirs(results_dir, exist_ok=True)

        env = DummyVecEnv([
            lambda: ANSRDTEnv(
                data_file=data_file,
                window_size=config['model']['window_size'],
                config=config,
            )
        ])

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
            tensorboard_log=os.path.join(results_dir, 'logs', 'ppo_tensorboard'),
        )

        logger.info("Starting PPO training...")
        total_timesteps = config.get('ppo', {}).get('total_timesteps', 10000)
        model.learn(total_timesteps=total_timesteps)

        model_path = os.path.join(results_dir, 'ppo_ansr_dt.zip')
        model.save(model_path)
        if not os.path.exists(model_path):
            raise RuntimeError("Failed to save PPO model")

        logger.info(f"PPO training completed. Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error during PPO training: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO Agent for ANSR-DT')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    success = train_ppo_agent(args.config)
    if not success:
        raise SystemExit(1)
