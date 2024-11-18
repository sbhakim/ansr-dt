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


def setup_dirs(project_root: str) -> None:
    """Create necessary project directories."""
    dirs = ['logs', 'results', 'results/visualization']
    for dir_path in dirs:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)


def prepare_sensor_window(data: np.lib.npyio.NpzFile, window_size: int) -> np.ndarray:
    """
    Prepare sensor window with correct shape for models.

    Args:
        data: Loaded NPZ data file
        window_size: Size of the sliding window

    Returns:
        np.ndarray: Properly shaped sensor window
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


def setup_project_structure():
    """Set up project directory structure."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    required_dirs = [
        'src/reasoning',
        'results',
        'logs',
        'results/visualization',
        'results/visualization/model_visualization'
    ]

    for dir_path in required_dirs:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)


def main():
    """
    Execute the NEXUS-DT pipeline:
    1. Train CNN-LSTM for anomaly detection
    2. Train PPO for adaptive control
    3. Run integrated system
    """
    # Setup project structure
    project_root = os.path.dirname(os.path.abspath(__file__))
    setup_dirs(project_root)

    # Initialize logging
    logger = setup_logging(
        log_file='logs/nexus_dt.log',
        log_level=logging.INFO
    )
    logger.info("Starting NEXUS-DT Pipeline")

    try:
        # Load and update configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = load_config(config_path)

        # Update paths to be absolute
        config['paths']['results_dir'] = os.path.join(project_root, 'results')
        config['paths']['data_file'] = os.path.join(project_root, config['paths']['data_file'])

        # Step 1: Train CNN-LSTM model
        logger.info("Starting CNN-LSTM training...")
        pipeline = NEXUSDTPipeline(config, config_path, logger)
        pipeline.run()
        logger.info("CNN-LSTM training completed")

        # Step 2: Train PPO agent
        logger.info("Starting PPO training...")
        ppo_success = train_ppo_agent(config_path)

        if not ppo_success:
            raise RuntimeError("PPO training failed")
        logger.info("PPO training completed")

        # Verify trained models exist
        model_paths = {
            'cnn_lstm': os.path.join(config['paths']['results_dir'], 'best_model.keras'),
            'ppo': os.path.join(config['paths']['results_dir'], 'ppo_nexus_dt.zip')
        }

        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} model not found at {path}")

        # Step 3: Initialize NEXUS-DT system
        logger.info("Initializing NEXUS-DT system...")
        nexusdt = ExplainableNEXUSDT(config_path, logger)

        # Load test data and prepare window
        test_data = np.load(config['paths']['data_file'])
        sensor_window = prepare_sensor_window(
            data=test_data,
            window_size=config['model']['window_size']
        )

        # Verify sensor window shape
        expected_shape = (config['model']['window_size'], 7)
        if sensor_window.shape != expected_shape:
            raise ValueError(
                f"Incorrect sensor window shape: {sensor_window.shape}, "
                f"expected {expected_shape}"
            )

        # Run integrated inference
        logger.info("Running integrated inference...")
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

        # After saving neurosymbolic results
        logger.info("Neurosymbolic Analysis Summary:")
        logger.info(f"- Neural Rules Extracted: {len(neurosymbolic_results['neural_rules'])}")
        logger.info(f"- Symbolic Insights Generated: {len(neurosymbolic_results['symbolic_insights'])}")
        logger.info(f"- Neural Confidence: {neurosymbolic_results['neural_confidence']:.2%}")

        # Save results
        results_file = os.path.join(config['paths']['results_dir'], 'final_state.json')
        nexusdt.save_state(results_file)

        logger.info(f"Pipeline completed successfully. Results saved to {results_file}")
        logger.info(f"System explanation: {result.get('explanation', '')}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    setup_project_structure()
    main()