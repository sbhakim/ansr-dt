# main.py
import json
import os
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
from src.reasoning.rule_learning import RuleLearner
from src.reasoning.prob_log_interface import ProbLogInterface
from src.reasoning.reasoning import SymbolicReasoner


def setup_project_structure(project_root: str):
    """Create necessary project directories."""
    required_dirs = [
        'src/reasoning',
        'results/models',
        'results/metrics',
        'logs',
        'results/visualization/model_visualization',
        'results/visualization/neurosymbolic',
        'results/visualization/metrics',
        'results/visualization/pattern_analysis',
        'results/knowledge_graphs'
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

def initialize_visualizers(logger: logging.Logger, figures_dir: str):
    """Initialize visualization components."""
    os.makedirs(figures_dir, exist_ok=True)
    return {
        'neurosymbolic': NeurosymbolicVisualizer(logger),
        'knowledge_graph': KnowledgeGraphGenerator(logger=logger)
    }

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
        graph_path = os.path.join(output_dir, f'knowledge_graph_{timestamp}.png')
        graph_generator.visualize(graph_path)

        state_path = os.path.join(output_dir, f'graph_state_{timestamp}.json')
        state = {
            'nodes': len(graph_generator.graph.nodes),
            'edges': len(graph_generator.graph.edges),
            'timestamp': timestamp
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save knowledge graph state: {e}")
        raise


def main():
    """Execute the NEXUS-DT pipeline with neurosymbolic components."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    setup_project_structure(project_root)
    logger = setup_logging(
        log_file=os.path.join(project_root, 'logs', 'nexus_dt.log'),
        log_level=logging.INFO
    )
    logger.info("Starting NEXUS-DT Pipeline")

    try:
        # Load configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = load_config(config_path)

        # Update absolute paths
        for path_key in ['results_dir', 'data_file', 'plot_config_path', 'reasoning_rules_path',
                         'prob_log_rules_path', 'knowledge_graphs_dir']:
            config['paths'][path_key] = os.path.join(project_root, config['paths'][path_key])

        # Model paths
        model_paths = {
            'cnn_lstm': os.path.join(config['paths']['results_dir'], 'best_model.keras'),
            'ppo': os.path.join(config['paths']['results_dir'], 'ppo_nexus_dt.zip')
        }

        # Load or train models
        if os.path.exists(model_paths['cnn_lstm']):
            cnn_lstm_model = load_model_with_initialization(
                path=model_paths['cnn_lstm'],
                logger=logger,
                input_shape=tuple(config['model']['input_shape'])
            )
        else:
            pipeline = NEXUSDTPipeline(config, config_path, logger)
            pipeline.run()
            cnn_lstm_model = load_model_with_initialization(
                path=model_paths['cnn_lstm'],
                logger=logger,
                input_shape=tuple(config['model']['input_shape'])
            )

        if os.path.exists(model_paths['ppo']):
            ppo_agent = PPO.load(model_paths['ppo'])
        else:
            if not train_ppo_agent(config_path):
                raise RuntimeError("PPO training failed")
            ppo_agent = PPO.load(model_paths['ppo'])

        # Initialize components
        rule_learner = RuleLearner(
            base_threshold=config['reasoning']['rule_learning']['base_threshold'],
            window_size=config['reasoning']['rule_learning']['window_size'],
            rules_path=config['paths']['reasoning_rules_path'],
            logger=logger
        )

        prob_log_interface = ProbLogInterface(
            rules_path=config['paths']['prob_log_rules_path'],
            logger=logger
        )

        symbolic_reasoner = SymbolicReasoner(
            rules_path=config['paths']['reasoning_rules_path'],
            input_shape=tuple(config['model']['input_shape']),
            logger=logger
        )
        symbolic_reasoner.rule_learner = rule_learner
        symbolic_reasoner.prob_log_interface = prob_log_interface

        knowledge_graph = KnowledgeGraphGenerator(logger=logger)
        neurosymbolic_visualizer = NeurosymbolicVisualizer(logger)

        # Initialize NEXUS-DT
        nexusdt = ExplainableNEXUSDT(
            config_path=config_path,
            logger=logger,
            cnn_lstm_model=cnn_lstm_model,
            ppo_agent=ppo_agent,
            symbolic_reasoner=symbolic_reasoner,
            knowledge_graph=knowledge_graph
        )

        # Load and process test data
        test_data = np.load(config['paths']['data_file'])
        sensor_window = prepare_sensor_window(
            data=test_data,
            window_size=config['model']['window_size']
        )

        # Run inference
        result = nexusdt.adapt_and_explain(sensor_window)

        # Process results
        temporal_patterns = rule_learner.analyze_temporal_patterns(
            sequences=sensor_window,
            labels=result.get('labels', [])
        )
        new_rules = rule_learner.extract_rules(temporal_patterns)
        symbolic_reasoner.update_rules(new_rules, min_confidence=config['reasoning']['rule_learning']['min_confidence'])

        result = {
            'insights': [],
            'current_state': {},
            'anomaly_status': {},
            'labels': [],
            'control_parameters': {},
            'state_transitions': [],
            'confidence': 0.0
        }

        if config['reasoning']['prob_log']['enabled']:
            try:
                prob_log_results = prob_log_interface.run_single_query()
                if prob_log_results:
                    prob_log_insight = f"ProbLog Failure Risk: {prob_log_results.get('failure_risk', 0.0):.2f}"
                    if 'insights' not in result:
                        result['insights'] = []
                    result['insights'].append(prob_log_insight)
            except Exception as e:
                logger.warning(f"ProbLog processing warning: {str(e)}")
                # Continue execution even if ProbLog fails

        # Generate visualizations
        if hasattr(symbolic_reasoner, 'get_rule_activations'):
            neurosymbolic_visualizer.visualize_rule_activations(
                activations=symbolic_reasoner.get_rule_activations(),
                save_path=os.path.join(config['paths']['results_dir'],
                                       'visualization/neurosymbolic/rule_activations.png')
            )

        if config['knowledge_graph']['enabled']:
            knowledge_graph.update_graph(
                current_state=result.get('current_state', {}),
                insights=result.get('insights', []),
                rules=[{'rule': rule.id, 'confidence': rule.confidence} for rule in
                       rule_learner.learned_rules.values()],
                anomalies=result.get('anomaly_status', {})
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_knowledge_graph_state(
                graph_generator=knowledge_graph,
                output_dir=config['paths']['knowledge_graphs_dir'],
                timestamp=timestamp,
                logger=logger
            )

        # Save results
        neurosymbolic_results = {
            'neural_rules': [rule.id for rule in rule_learner.learned_rules.values()],
            'rule_confidence': {rule.id: rule.confidence for rule in rule_learner.learned_rules.values()},
            'symbolic_insights': result.get('insights', []),
            'prob_log_results': prob_log_results if config['reasoning']['prob_log']['enabled'] else {},
            'neural_confidence': result.get('confidence', 0.0),
            'control_parameters': result.get('control_parameters', {}),
            'state_transitions': result.get('state_transitions', []),
            'timestamp': str(np.datetime64('now'))
        }

        with open(os.path.join(config['paths']['results_dir'], 'neurosymbolic_results.json'), 'w') as f:
            json.dump(neurosymbolic_results, f, indent=2)

        with open(os.path.join(config['paths']['results_dir'], 'final_state.json'), 'w') as f:
            json.dump({
                'current_state': nexusdt.current_state,
                'state_history': nexusdt.state_history[-100:],
                'knowledge_graph_state': {
                    'nodes': len(knowledge_graph.graph.nodes),
                    'edges': len(knowledge_graph.graph.edges),
                    'timestamp': str(np.datetime64('now'))
                }
            }, f, indent=2)

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
