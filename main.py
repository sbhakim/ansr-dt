# main.py

import os
import logging
from utils.logging_setup import setup_logging
from utils.config_manager import load_config
from utils.pipeline import NEXUSDTPipeline

def main():
    # Initialize logging
    logger = setup_logging(
        log_file='logs/nexus_dt.log',
        log_level=logging.INFO,
        max_bytes=5 * 1024 * 1024,  # 5 MB
        backup_count=5
    )
    logger.info("Starting NEXUS-DT Model Training Pipeline.")

    try:
        # Load configuration
        config = load_config('configs/config.yaml')
        logger.info("Configuration loaded.")

        # Initialize and run the pipeline
        pipeline = NEXUSDTPipeline(config, logger)
        pipeline.run()

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
