# main.py
import json
import os
import logging
import numpy as np

from src.logging.logging_setup import setup_logging
from src.config.config_manager import load_config
from src.pipeline.pipeline import NEXUSDTPipeline
from src.rl.train_ppo import train_ppo_agent
from src.nexusdt.explainable import ExplainableNEXUSDT
from src.utils.model_utils import load_model_with_initialization
from stable_baselines3 import PPO

def setup_project_structure(project_root: str) -> None:
    """
    Create necessary project directories to ensure the pipeline runs smoothly.

    Parameters:
    - project_root (str): Root directory of the project.
    """
    required_dirs = [
        'src/reasoning',
        'results',
        'logs',
        'results/visualization',
        'results/visualization/model_visualization'
    ]

    for dir_path in required_dirs:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)


def prepare_sensor_window(data: np.lib.npyio.NpzFile, window_size: int) -> np.ndarray:
    """
    Prepare sensor window with correct shape for models.

    Args:
        data: Loaded NPZ data file
        window_size: Size of the sliding window

    Returns:
        np.ndarray: Properly shaped sensor window (window_size, n_features)
    """
    # Extract and stack features in correct order
    features = [
        data['temperature'][:window_size],
        data['vibration'][:window_size],
        data['pressure'][:window_size],
        data['operational_hours'][:window_size],
        data['efficiency_index'][:window_size],
        data['system_state'][:window_size],
        data['performance_score'][:window_size]
    ]

    # Stack features to shape (window_size, n_features)
    return np.stack(features, axis=1)


def main():
    """
    Execute the NEXUS-DT pipeline:
    1. Train CNN-LSTM for anomaly detection or load existing model
    2. Train PPO for adaptive control or load existing agent
    3. Run integrated system for reasoning and explanation
    """
    # Determine project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Setup project directory structure
    setup_project_structure(project_root)

    # Initialize logging
    logger = setup_logging(
        log_file=os.path.join(project_root, 'logs', 'nexus_dt.log'),
        log_level=logging.INFO
    )
    logger.info("Starting NEXUS-DT Pipeline")

    try:
        # Load configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}.")

        # Update paths in config to be absolute
        config['paths']['results_dir'] = os.path.join(project_root, 'results')
        config['paths']['data_file'] = os.path.join(project_root, config['paths']['data_file'])
        config['paths']['plot_config_path'] = os.path.join(project_root, config['paths']['plot_config_path'])
        config['paths']['reasoning_rules_path'] = os.path.join(project_root, config['paths']['reasoning_rules_path'])

        # Paths to the models
        model_paths = {
            'cnn_lstm': os.path.join(config['paths']['results_dir'], 'best_model.keras'),
            'ppo': os.path.join(config['paths']['results_dir'], 'ppo_nexus_dt.zip')
        }

        # Step 1: Check if CNN-LSTM model exists, else train
        if os.path.exists(model_paths['cnn_lstm']):
            logger.info("Loading existing CNN-LSTM model.")
            cnn_lstm_model = load_model_with_initialization(
                path=model_paths['cnn_lstm'],
                logger=logger,
                input_shape=tuple(config['model']['input_shape'])
            )
            logger.info("CNN-LSTM model loaded successfully.")

            # Ensure the model is built
            if not cnn_lstm_model.built:
                logger.info("Model is not built. Building model with dummy input.")
                dummy_input = np.zeros((1,) + tuple(config['model']['input_shape']), dtype=np.float32)
                cnn_lstm_model.predict(dummy_input)
                logger.info("Model built successfully.")
            else:
                logger.info("Model is already built.")
        else:
            logger.info("CNN-LSTM model not found. Starting training.")
            pipeline = NEXUSDTPipeline(config, config_path, logger)
            pipeline.run()
            logger.info("Pipeline run completed.")

            # Load the trained model
            cnn_lstm_model = load_model_with_initialization(
                path=model_paths['cnn_lstm'],
                logger=logger,
                input_shape=tuple(config['model']['input_shape'])
            )
            logger.info("CNN-LSTM training completed and model loaded.")

            # Ensure the model is built
            if not cnn_lstm_model.built:
                logger.info("Model is not built after loading. Building model with dummy input.")
                dummy_input = np.zeros((1,) + tuple(config['model']['input_shape']), dtype=np.float32)
                cnn_lstm_model.predict(dummy_input)
                logger.info("Model built successfully.")
            else:
                logger.info("Model is already built after loading.")

        # Step 2: Check if PPO agent exists, else train
        if os.path.exists(model_paths['ppo']):
            logger.info("Loading existing PPO agent.")
            ppo_agent = PPO.load(model_paths['ppo'])
            logger.info("PPO agent loaded successfully.")
        else:
            logger.info("PPO agent not found. Starting training.")
            ppo_success = train_ppo_agent(config_path)

            if not ppo_success:
                raise RuntimeError("PPO training failed.")
            # Load the trained agent
            ppo_agent = PPO.load(model_paths['ppo'])
            logger.info("PPO training completed and agent loaded.")

        # Verify trained models exist
        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} model not found at {path}")
            else:
                logger.info(f"{name} model confirmed at {path}")

        # Step 3: Initialize NEXUS-DT system with reasoning capabilities
        logger.info("Initializing NEXUS-DT reasoning system...")

        # Initialize ExplainableNEXUSDT with loaded models
        nexusdt = ExplainableNEXUSDT(
            config_path=config_path,
            logger=logger,
            cnn_lstm_model=cnn_lstm_model,
            ppo_agent=ppo_agent
        )
        logger.info("ExplainableNEXUSDT initialized successfully.")

        # Load test data and prepare window
        test_data = np.load(config['paths']['data_file'])
        sensor_window = prepare_sensor_window(
            data=test_data,
            window_size=config['model']['window_size']
        )

        # Dynamically determine the number of features
        num_features = len(config['model']['feature_names'])
        expected_shape = (config['model']['window_size'], num_features)
        if sensor_window.shape != expected_shape:
            raise ValueError(
                f"Incorrect sensor window shape: {sensor_window.shape}, "
                f"expected {expected_shape}"
            )
        logger.info(f"Sensor window prepared with shape: {sensor_window.shape}")

        # Run integrated inference and reasoning
        logger.info("Running integrated inference and reasoning...")
        result = nexusdt.adapt_and_explain(sensor_window)

        # Save neurosymbolic results
        neurosymbolic_results = {
            'neural_rules': nexusdt.reasoner.learned_rules,
            'rule_confidence': nexusdt.reasoner.rule_confidence,
            'symbolic_insights': result.get('insights', []),
            'neural_confidence': result.get('confidence', 0.0),
            'timestamp': str(np.datetime64('now'))
        }

        neurosymbolic_path = os.path.join(
            config['paths']['results_dir'],
            'neurosymbolic_results.json'
        )

        with open(neurosymbolic_path, 'w') as f:
            json.dump(neurosymbolic_results, f, indent=2)

        logger.info(f"Neurosymbolic results saved to {neurosymbolic_path}")

        # Log Neurosymbolic Analysis Summary
        logger.info("Neurosymbolic Analysis Summary:")
        logger.info(f"- Neural Rules Extracted: {len(neurosymbolic_results['neural_rules'])}")
        logger.info(f"- Symbolic Insights Generated: {len(neurosymbolic_results['symbolic_insights'])}")
        logger.info(f"- Neural Confidence: {neurosymbolic_results['neural_confidence']:.2%}")

        # Save final state (assuming `nexusdt` has a `save_state` method)
        results_file = os.path.join(config['paths']['results_dir'], 'final_state.json')
        nexusdt.save_state(results_file)

        logger.info(f"Pipeline completed successfully. Results saved to {results_file}")
        logger.info(f"System explanation: {result.get('explanation', '')}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
