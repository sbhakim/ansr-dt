# main.py
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
import numpy as np
from datetime import datetime

from src.logging.logging_setup import setup_logging
from src.config.config_manager import load_config
from src.pipeline.pipeline import NEXUSDTPipeline
from src.rl.train_ppo import train_ppo_agent
from src.nexusdt.explainable import ExplainableNEXUSDT
from src.utils.model_utils import load_model_with_initialization
from stable_baselines3 import PPO
from src.visualization.neurosymbolic_visualizer import NeurosymbolicVisualizer
from src.integration.adaptive_controller import AdaptiveController
from src.reasoning.knowledge_graph import KnowledgeGraphGenerator


def setup_project_structure(project_root: str):
    """Create necessary project directories."""
    required_dirs = [
        'src/reasoning',
        'results',
        'results/models',
        'results/metrics',
        'logs',
        'results/visualization',
        'results/visualization/model_visualization',
        'results/visualization/neurosymbolic',
        'results/visualization/metrics',
        'results/visualization/pattern_analysis',
        'results/knowledge_graphs'  # Added for knowledge graphs
    ]

    for dir_path in required_dirs:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)


def prepare_sensor_window(data: np.lib.npyio.NpzFile, window_size: int) -> np.ndarray:
    """Prepare sensor window with correct shape for models."""
    try:
        features = [
            data['temperature'][:window_size],
            data['vibration'][:window_size],
            data['pressure'][:window_size],
            data['operational_hours'][:window_size],
            data['efficiency_index'][:window_size],
            data['system_state'][:window_size],
            data['performance_score'][:window_size]
        ]
        return np.stack(features, axis=1)
    except KeyError as e:
        raise KeyError(f"Missing required feature in data: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error preparing sensor window: {str(e)}")


def initialize_visualizers(logger: logging.Logger, figures_dir: str, config: dict):
    """Initialize visualization components."""
    try:
        os.makedirs(figures_dir, exist_ok=True)
        neurosymbolic_visualizer = NeurosymbolicVisualizer(logger)
        knowledge_graph_generator = KnowledgeGraphGenerator(logger=logger)

        return {
            'neurosymbolic': neurosymbolic_visualizer,
            'knowledge_graph': knowledge_graph_generator
        }
    except Exception as e:
        logger.error(f"Failed to initialize visualizers: {e}")
        raise


def save_results(results: dict, path: str, logger: logging.Logger):
    """Save results to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def save_knowledge_graph_state(graph_generator: KnowledgeGraphGenerator,
                               output_dir: str,
                               timestamp: str,
                               logger: logging.Logger):
    """Save knowledge graph state and visualization."""
    try:
        # Save visualization
        graph_path = os.path.join(output_dir, f'knowledge_graph_{timestamp}.png')
        graph_generator.visualize(graph_path)

        # Save graph state
        state_path = os.path.join(output_dir, f'graph_state_{timestamp}.json')
        graph_state = {
            'nodes': len(graph_generator.graph.nodes),
            'edges': len(graph_generator.graph.edges),
            'timestamp': timestamp
        }
        save_results(graph_state, state_path, logger)

    except Exception as e:
        logger.error(f"Failed to save knowledge graph state: {e}")
        raise


def main():
    """
    Execute the NEXUS-DT pipeline with enhanced neurosymbolic components:
    1. Train/load CNN-LSTM for anomaly detection
    2. Train/load PPO for adaptive control
    3. Initialize neurosymbolic components
    4. Run integrated system with visualization and knowledge graph generation
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
        # Load and validate configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        # Update paths to absolute
        config['paths']['results_dir'] = os.path.join(project_root, 'results')
        config['paths']['data_file'] = os.path.join(project_root, config['paths']['data_file'])
        config['paths']['plot_config_path'] = os.path.join(project_root, config['paths']['plot_config_path'])
        config['paths']['reasoning_rules_path'] = os.path.join(project_root, config['paths']['reasoning_rules_path'])
        config['paths']['knowledge_graphs_dir'] = os.path.join(project_root, 'results', 'knowledge_graphs')

        # Initialize model paths
        model_paths = {
            'cnn_lstm': os.path.join(config['paths']['results_dir'], 'best_model.keras'),
            'ppo': os.path.join(config['paths']['results_dir'], 'ppo_nexus_dt.zip')
        }

        # Load or train CNN-LSTM model
        if os.path.exists(model_paths['cnn_lstm']):
            logger.info("Loading existing CNN-LSTM model")
            cnn_lstm_model = load_model_with_initialization(
                path=model_paths['cnn_lstm'],
                logger=logger,
                input_shape=tuple(config['model']['input_shape'])
            )
        else:
            logger.info("Training new CNN-LSTM model")
            pipeline = NEXUSDTPipeline(config, config_path, logger)
            pipeline.run()
            cnn_lstm_model = load_model_with_initialization(
                path=model_paths['cnn_lstm'],
                logger=logger,
                input_shape=tuple(config['model']['input_shape'])
            )

        # Load or train PPO agent
        if os.path.exists(model_paths['ppo']):
            logger.info("Loading existing PPO agent")
            ppo_agent = PPO.load(model_paths['ppo'])
        else:
            logger.info("Training new PPO agent")
            ppo_success = train_ppo_agent(config_path)
            if not ppo_success:
                raise RuntimeError("PPO training failed")
            ppo_agent = PPO.load(model_paths['ppo'])

        # Verify models
        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} model not found at {path}")
            logger.info(f"{name} model verified at {path}")

        # Initialize NEXUS-DT system
        logger.info("Initializing NEXUS-DT system")
        nexusdt = ExplainableNEXUSDT(
            config_path=config_path,
            logger=logger,
            cnn_lstm_model=cnn_lstm_model,
            ppo_agent=ppo_agent
        )

        # Load test data
        test_data = np.load(config['paths']['data_file'])
        sensor_window = prepare_sensor_window(
            data=test_data,
            window_size=config['model']['window_size']
        )

        # Logging after loading test data
        logger.info(f"Loaded test data shape: {sensor_window.shape}")

        # Initialize visualizers and knowledge graph
        figures_dir = os.path.join(config['paths']['results_dir'], 'visualization')
        visualizers = initialize_visualizers(logger, figures_dir, config)

        # Logging before running inference
        logger.info("Input data validation:")
        logger.info(f"- Shape: {sensor_window.shape}")
        logger.info(f"- Range: [{np.min(sensor_window)}, {np.max(sensor_window)}]")
        logger.info(f"- Contains NaN: {np.isnan(sensor_window).any()}")

        # Run integrated inference
        logger.info("Running integrated inference")
        result = nexusdt.adapt_and_explain(sensor_window)

        # Logging after running inference
        logger.info("Inference results:")
        logger.info(f"- Result type: {type(result)}")
        logger.info(f"- Result keys: {result.keys() if isinstance(result, dict) else 'Not a dictionary'}")

        # Generate visualizations with error handling
        try:
            # Standard visualizations
            if hasattr(nexusdt.reasoner, 'get_rule_activations'):
                visualizers['neurosymbolic'].visualize_rule_activations(
                    activations=nexusdt.reasoner.get_rule_activations(),
                    save_path=os.path.join(figures_dir, 'neurosymbolic/rule_activations.png')
                )

            if hasattr(nexusdt.reasoner, 'state_tracker'):
                visualizers['neurosymbolic'].plot_state_transitions(
                    transition_matrix=nexusdt.reasoner.state_tracker.get_transition_probabilities(),
                    save_path=os.path.join(figures_dir, 'neurosymbolic/state_transitions.png')
                )

            # Knowledge graph generation and update
            if config['knowledge_graph']['enabled']:
                current_state = result.get('current_state', {})
                insights = result.get('insights', [])
                rules = [{'rule': rule, 'confidence': nexusdt.reasoner.rule_confidence.get(rule, 0.0)}
                         for rule in nexusdt.reasoner.learned_rules]

                # Update knowledge graph
                visualizers['knowledge_graph'].update_graph(
                    current_state=current_state,
                    insights=insights,
                    rules=rules,
                    anomalies=result.get('anomaly_status', {})
                )

                # Save knowledge graph state
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_knowledge_graph_state(
                    graph_generator=visualizers['knowledge_graph'],
                    output_dir=config['paths']['knowledge_graphs_dir'],
                    timestamp=timestamp,
                    logger=logger
                )

        except Exception as viz_error:
            logger.warning(f"Visualization error: {viz_error}")

        # Prepare results
        neurosymbolic_results = {
            'neural_rules': getattr(nexusdt.reasoner, 'learned_rules', []),
            'rule_confidence': getattr(nexusdt.reasoner, 'rule_confidence', {}),
            'symbolic_insights': result.get('insights', []),
            'neural_confidence': result.get('confidence', 0.0),
            'control_parameters': result.get('control_parameters', {}),
            'state_transitions': result.get('state_transitions', []),
            'timestamp': str(np.datetime64('now'))
        }

        # Save results
        save_results(
            results=neurosymbolic_results,
            path=os.path.join(config['paths']['results_dir'], 'neurosymbolic_results.json'),
            logger=logger
        )

        # Save final state
        save_results(
            results={
                'current_state': nexusdt.current_state,
                'state_history': nexusdt.state_history[-100:],
                'knowledge_graph_state': {
                    'nodes': len(visualizers['knowledge_graph'].graph.nodes),
                    'edges': len(visualizers['knowledge_graph'].graph.edges),
                    'timestamp': str(np.datetime64('now'))
                }
            },
            path=os.path.join(config['paths']['results_dir'], 'final_state.json'),
            logger=logger
        )

        # Log summary
        logger.info("\nNeurosymbolic Analysis Summary:")
        logger.info(f"- Neural Rules Extracted: {len(neurosymbolic_results['neural_rules'])}")
        logger.info(f"- Symbolic Insights Generated: {len(neurosymbolic_results['symbolic_insights'])}")
        logger.info(f"- Neural Confidence: {neurosymbolic_results['neural_confidence']:.2%}")
        logger.info(f"- Control Parameters Applied: {len(neurosymbolic_results['control_parameters'])}")
        logger.info(f"- Knowledge Graph Nodes: {len(visualizers['knowledge_graph'].graph.nodes)}")
        logger.info(f"- Knowledge Graph Edges: {len(visualizers['knowledge_graph'].graph.edges)}")
        logger.info(f"Final Explanation: {result.get('explanation', '')}\n")

        logger.info("Pipeline completed successfully")
        return True

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()