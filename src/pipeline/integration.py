# src/pipeline/integration.py

import os
import json
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.ansrdt.ansr_dt_env import ANSRDTEnv  # Updated import
import yaml
from src.reasoning.reasoning import SymbolicReasoner


class RL_Symbolic_Integration:
    """
    Integrates Reinforcement Learning (PPO) with Symbolic Reasoning.
    """

    def __init__(self, config_path: str, logger: logging.Logger):
        """
        Initializes the integration class.
        """
        self.config = self.load_config(config_path)
        self.logger = logger

        # Determine base directories
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)

        # Load PPO model
        ppo_model_path = os.path.join(self.config['paths']['results_dir'], 'ppo_ansr_dt')
        if not os.path.exists(ppo_model_path + ".zip"):
            self.logger.error(f"PPO model not found at {ppo_model_path}. Please train the PPO agent first.")
            raise FileNotFoundError(f"PPO model not found at {ppo_model_path}.")
        self.ppo_agent = PPO.load(ppo_model_path)
        self.logger.info(f"PPO agent loaded from {ppo_model_path}")

        # Initialize Symbolic Reasoner
        rules_path = os.path.join(project_root, self.config['paths']['reasoning_rules_path'])
        self.reasoner = SymbolicReasoner(rules_path)
        self.logger.info(f"Symbolic Reasoner initialized with rules from {rules_path}")

        # Initialize Vectorized Environment
        self.env = DummyVecEnv([lambda: ANSRDTEnv(
            data_file=self.config['paths']['data_file'],
            window_size=self.config['model']['window_size'],
            config=self.config.get('ppo', {})
        )])
        self.logger.info("Vectorized environment initialized for integration.")

    def load_config(self, config_path: str) -> dict:
        """
        Loads the YAML configuration file.

        Parameters:
        - config_path (str): Path to config.yaml

        Returns:
        - config (dict): Configuration parameters.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def integrate(self, output_path: str = 'results/inference_with_rl_results.json'):
        try:
            # Reset vectorized environment and get initial observation
            obs = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            actions_list = []
            rewards_list = []
            insights_list = []

            while not done:
                # Get action from policy
                action, _states = self.ppo_agent.predict(obs, deterministic=True)
                actions_list.append(action.tolist())

                # Step environment with SB3 VecEnv format
                obs, rewards, dones, infos = self.env.step(action)
                reward = rewards[0]  # Extract scalar reward
                done = dones[0]  # Extract scalar done flag
                info = infos[0]  # Extract info dict

                rewards_list.append(reward)
                total_reward += reward
                steps += 1

                # Extract the latest observation for symbolic reasoning
                # obs shape is (1, window_size, features)
                latest_obs = obs[0, -1]  # Get latest timestep
                sensor_state = {
                    'temperature': float(latest_obs[0]),
                    'vibration': float(latest_obs[1]),
                    'pressure': float(latest_obs[2]),
                    'operational_hours': float(latest_obs[3]),
                    'efficiency_index': float(latest_obs[4]),
                    'system_state': float(latest_obs[5]),
                    'performance_score': float(latest_obs[6])
                }

                # Get symbolic reasoning insights
                insight = self.reasoner.reason(sensor_state)
                insights_list.append(insight)

                self.logger.info(f"Step {steps}: Action: {action}, Reward: {reward}, Insights: {insight}")

            # Save results
            results = {
                'total_reward': float(total_reward),
                'steps': steps,
                'actions': actions_list,
                'rewards': [float(r) for r in rewards_list],
                'insights': insights_list
            }

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Integration results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Integration failed: {e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description='RL and Symbolic Reasoning Integration')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', type=str, default='results/inference_with_rl_results.json',
                        help='Path to save integration results')
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger('RL_Symbolic_Integration')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs('logs', exist_ok=True)
        handler = logging.FileHandler('logs/rl_integration.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Initialize and run integration
    integration = RL_Symbolic_Integration(config_path=args.config, logger=logger)
    integration.integrate(output_path=args.output)


if __name__ == "__main__":
    main()
