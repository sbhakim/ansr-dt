# src/config/config_manager.py

import yaml
import logging


def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file with validation.

    Raises:
        KeyError: If required keys are missing.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate essential keys
        required_sections = ['model', 'training', 'paths']
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Missing required configuration section: '{section}'")

        # Validate model-specific keys
        model_required_keys = ['window_size', 'feature_names', 'input_shape', 'architecture']
        for key in model_required_keys:
            if key not in config['model']:
                raise KeyError(f"Missing required model configuration key: '{key}'")

        logging.getLogger(__name__).info(f"Configuration loaded and validated from {config_path}.")
        return config
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load configuration: {e}")
        raise
