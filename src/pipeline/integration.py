# src/pipeline/integration.py

import os
import json
import logging
from stable_baselines3 import PPO
from src.nexusdt.nexus_dt_env import NexusDTEnv  # Updated import
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

        # Load PPO model
        ppo_model_path = os.path.join(self.config['paths']['results_dir'], 'ppo_nexus_dt')
        if not os.path.exists(ppo_model_path + ".zip"):
            self.logger.error(f"PPO model not found at {ppo_model_path}. Please train the PPO agent first.")
            raise FileNotFoundError(f"PPO model not found at {ppo_model_path}.")
        self.ppo_agent = PPO.load(ppo_model_path)
        self.logger.info(f"PPO agent loaded from {ppo_model_path}")

        # Initialize Symbolic Reasoner
        rules_path = self.config['paths']['reasoning_rules_path']
        self.reasoner = SymbolicReasoner(rules_path)

        # Initialize Environment
        self.env = NexusDTEnv(
            data_file=self.config['paths']['data_file'],
            window_size=self.config['model']['window_size'],
            config=self.config.get('ppo', {})
        )

    def load_config(self, config_path: str) -> dict:
        """
        Loads the YAML configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def integrate(self, output_path: str = 'results/inference_with_rl_results.json'):
        """
        Runs the integration by performing inference with RL actions and symbolic reasoning.
        """
        try:
            # Reset environment and get initial observation
            obs = self.env.reset()
            done = False
            trunc = False
            total_reward = 0
            steps = 0
            actions_list = []
            rewards_list = []
            insights_list = []

            while not (done or trunc):
                # Get action from policy using just the observation
                action, _states = self.ppo_agent.predict(obs, deterministic=True)
                actions_list.append(action.tolist())

                # Step environment with SB3 format
                obs, reward, done, trunc, info = self.env.step(action)
                rewards_list.append(reward)
                total_reward += reward
                steps += 1

                # Use action directly for symbolic insights
                sensor_state = {
                    'temperature': obs[-1, 0],  # Latest temperature
                    'vibration': obs[-1, 1],  # Latest vibration
                    'pressure': obs[-1, 2]  # Latest pressure
                }
                insights = self.reasoner.reason(sensor_state)
                insights_list.append(insights)

                self.logger.info(f"Step {steps}: Action: {action}, Reward: {reward}, Insights: {insights}")

            # Compile results
            results = {
                'total_reward': float(total_reward),  # Convert to float for JSON serialization
                'steps': steps,
                'actions': actions_list,
                'rewards': [float(r) for r in rewards_list],  # Convert rewards to float
                'insights': insights_list
            }

            # Save results
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
