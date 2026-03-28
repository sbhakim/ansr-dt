# src/pipeline/integration.py

import os
import sys
import json
import logging
import argparse
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ansrdt.ansr_dt_env import ANSRDTEnv
from src.config.config_manager import load_config
from src.reasoning.reasoning import SymbolicReasoner


class RLSymbolicIntegration:
    """Run the PPO policy and symbolic reasoner together over the ANSR-DT environment."""

    def __init__(self, config_path: str, logger: logging.Logger):
        self.config_path = os.path.abspath(config_path)
        self.config = load_config(self.config_path)
        self.logger = logger

        config_dir = os.path.dirname(self.config_path)
        self.project_root = os.path.dirname(config_dir)
        self.results_dir = self._resolve_path(self.config['paths'].get('results_dir', 'results'))
        self.data_file = self._resolve_path(self.config['paths']['data_file'])
        self.rules_path = self._resolve_path(self.config['paths']['reasoning_rules_path'])
        self.input_shape = (
            self.config['model']['window_size'],
            len(self.config['model']['feature_names']),
        )

        ppo_model_path = os.path.join(self.results_dir, 'ppo_ansr_dt.zip')
        if not os.path.exists(ppo_model_path):
            self.logger.error(f"PPO model not found at {ppo_model_path}. Please train the PPO agent first.")
            raise FileNotFoundError(f"PPO model not found at {ppo_model_path}.")
        self.ppo_agent = PPO.load(ppo_model_path)
        self.logger.info(f"PPO agent loaded from {ppo_model_path}")

        self.reasoner = SymbolicReasoner(
            rules_path=self.rules_path,
            input_shape=self.input_shape,
            # Integration-time reasoning operates over environment state snapshots;
            # it does not perform neural rule extraction inside this rollout loop.
            model=None,
            logger=self.logger,
        )
        self.logger.info(f"Symbolic Reasoner initialized with rules from {self.rules_path}")

        self.env = DummyVecEnv([
            lambda: ANSRDTEnv(
                data_file=self.data_file,
                window_size=self.config['model']['window_size'],
                config=self.config,
            )
        ])
        self.logger.info("Vectorized environment initialized for integration.")

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(self.project_root, path_value))

    def integrate(self, output_path: str = 'results/inference_with_rl_results.json'):
        try:
            resolved_output_path = output_path if os.path.isabs(output_path) else os.path.join(self.project_root, output_path)
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            actions_list = []
            rewards_list = []
            insights_list = []

            while not done:
                action, _states = self.ppo_agent.predict(obs, deterministic=True)
                actions_list.append(action.tolist())

                obs, rewards, dones, infos = self.env.step(action)
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0]

                rewards_list.append(reward)
                total_reward += reward
                steps += 1

                latest_obs = obs[0, -1]
                # Map the final timestep in the environment window back to the
                # symbolic feature names expected by the Prolog reasoner.
                sensor_state = {
                    'temperature': float(latest_obs[0]),
                    'vibration': float(latest_obs[1]),
                    'pressure': float(latest_obs[2]),
                    'operational_hours': float(latest_obs[3]),
                    'efficiency_index': float(latest_obs[4]),
                    'system_state': float(latest_obs[5]),
                    'performance_score': float(latest_obs[6]),
                }

                # Query the symbolic layer on the semantically named state so
                # the PPO trajectory can be inspected alongside rule activations.
                insight = self.reasoner.reason(sensor_state)
                insights_list.append(insight)
                self.logger.info(f"Step {steps}: Reward={reward:.4f}, Action={action.tolist()}, Insights={len(insight)}")

            results = {
                'total_reward': float(total_reward),
                'steps': steps,
                'actions': actions_list,
                'rewards': rewards_list,
                'insights': insights_list,
                'metadata': {
                    'data_file': self.data_file,
                    'results_dir': self.results_dir,
                    'rules_path': self.rules_path,
                },
            }

            os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
            with open(resolved_output_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Integration results saved to {resolved_output_path}")
            return results
        except Exception as e:
            self.logger.error(f"Integration failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='RL and Symbolic Reasoning Integration')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', type=str, default='results/inference_with_rl_results.json', help='Path to save integration results')
    args = parser.parse_args()

    logger = logging.getLogger('RL_Symbolic_Integration')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs('logs', exist_ok=True)
        handler = logging.FileHandler('logs/rl_integration.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    integration = RLSymbolicIntegration(config_path=args.config, logger=logger)
    integration.integrate(output_path=args.output)


if __name__ == "__main__":
    main()
