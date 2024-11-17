# main.py

import os
import logging

from src.logging.logging_setup import setup_logging
from src.config.config_manager import load_config
from src.pipeline.pipeline import NEXUSDTPipeline
from src.rl.train_ppo import train_ppo_agent
from src.pipeline.integration import RL_Symbolic_Integration


def main():
    """
    Executes the NEXUS-DT pipeline sequentially:
    1. Initializes logging.
    2. Loads configuration.
    3. Runs data loading, preprocessing, splitting, training, and evaluation.
    4. Generates and saves metrics and visualizations.
    5. Trains the PPO RL agent.
    6. Performs RL and Symbolic Reasoning integration.
    """
    # Step 1: Initialize Logging
    logger = setup_logging(
        log_file='logs/nexus_dt.log',
        log_level=logging.INFO,
        max_bytes=5 * 1024 * 1024,  # 5 MB
        backup_count=5
    )
    logger.info("Starting NEXUS-DT Model Training Pipeline.")

    try:
        # Step 2: Load Configuration
        config = load_config('configs/config.yaml')
        logger.info("Configuration loaded.")

        # Step 3: Initialize and Run the Main Pipeline
        pipeline = NEXUSDTPipeline(config, logger)
        pipeline.run()
        logger.info("Main pipeline execution completed.")

        # Step 4: Train PPO RL Agent
        logger.info("Initiating PPO RL agent training.")
        train_ppo_agent()
        logger.info("PPO RL agent training completed.")

        # Step 5: RL and Symbolic Reasoning Integration
        logger.info("Initiating RL and Symbolic Reasoning Integration.")
        integration = RL_Symbolic_Integration(
            config_path='configs/config.yaml',
            logger=logger
        )
        integration.integrate(
            output_path=os.path.join(
                config['paths']['results_dir'], 'inference_with_rl_results.json'
            )
        )
        logger.info("RL and Symbolic Reasoning Integration completed.")

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")


if __name__ == '__main__':
    main()
