# src/config/config_manager.py

import yaml
import logging

def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - config (dict): Configuration parameters.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.getLogger(__name__).info(f"Configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load configuration: {e}")
        raise
