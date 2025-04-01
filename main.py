#!/usr/bin/env python3
import os
# Force CPU usage - Must be done before TensorFlow import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import json
import logging
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Setup relative path for project root
project_root = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path if necessary, though often better to run as a module
# sys.path.append(project_root) # Uncomment if running as a script and imports fail

# Import necessary components
# Ensure imports happen *after* setting CUDA_VISIBLE_DEVICES if TF is involved
from src.logging.logging_setup import setup_logging
from src.config.config_manager import load_config
from src.pipeline.pipeline import ANSRDTPipeline, NumpyEncoder
from src.rl.train_ppo import train_ppo_agent
from src.ansrdt.explainable import ExplainableANSRDT
from src.utils.model_utils import load_model_with_initialization
from stable_baselines3 import PPO
from src.visualization.neurosymbolic_visualizer import NeurosymbolicVisualizer
from src.reasoning.knowledge_graph import KnowledgeGraphGenerator

# Dynamically import clear_results from its relative path
import importlib.util
clear_results_path = os.path.join(project_root, "src", "utils", "clear_results.py")
# Check if clear_results.py exists before attempting import
if os.path.exists(clear_results_path):
    spec = importlib.util.spec_from_file_location("clear_results", clear_results_path)
    clear_results_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clear_results_module)
    clear_results_func = clear_results_module.clear_results
else:
    # Define a dummy function if the file doesn't exist to avoid errors
    # Or raise an error if it's critical
    logging.warning(f"clear_results.py not found at {clear_results_path}. Skipping result clearing.")
    def clear_results_func(path):
        logging.info(f"Skipping clearing results for path: {path}")

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
        'results/knowledge_graphs',
        'results/knowledge_graphs/visualizations',
        'results/knowledge_graphs/data'
    ]
    for dir_path in required_dirs:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)

def prepare_sensor_window(data: np.lib.npyio.NpzFile, window_size: int, feature_names: list) -> np.ndarray:
    """Prepare a sensor window with the correct shape and features for inference."""
    try:
        missing_features = [f for f in feature_names if f not in data.files]
        if missing_features:
            raise KeyError(f"Missing required features in data: {', '.join(missing_features)}")
        if len(data[feature_names[0]]) < window_size:
            raise ValueError(f"Not enough data points ({len(data[feature_names[0]])}) for window size {window_size}")
        features = [data[key][:window_size] for key in feature_names]
        sensor_window = np.stack(features, axis=1)
        return sensor_window.astype(np.float32)
    except KeyError as e:
        raise KeyError(f"Error preparing sensor window: Missing feature - {str(e)}")
    except ValueError as e:
        raise ValueError(f"Error preparing sensor window: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error preparing sensor window: {str(e)}")

def initialize_visualizers(logger: logging.Logger, figures_dir: str, config: dict) -> Dict[str, Any]:
    """Initialize visualization components."""
    try:
        os.makedirs(figures_dir, exist_ok=True)
        neurosymbolic_visualizer = NeurosymbolicVisualizer(logger)
        # Use KG config for max_nodes/history if available
        kg_config = config.get('knowledge_graph', {})
        knowledge_graph_generator = KnowledgeGraphGenerator(
            logger=logger,
            max_nodes=kg_config.get('max_nodes', 500), # Use default from KG class if not in config
            max_history=kg_config.get('max_history', 1000) # Use default from KG class
        )
        return {
            'neurosymbolic': neurosymbolic_visualizer,
            'knowledge_graph': knowledge_graph_generator
        }
    except Exception as e:
        logger.error(f"Failed to initialize visualizers: {e}")
        raise

def save_results(results: dict, path: str, logger: logging.Logger):
    """Save results to a JSON file, handling NumPy types."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Results saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save results to {path}: {e}")

def save_knowledge_graph_state(graph_generator: KnowledgeGraphGenerator,
                               output_dir: str,
                               timestamp: str,
                               logger: logging.Logger):
    """Save knowledge graph state and visualization."""
    try:
        kg_viz_dir = os.path.join(output_dir, 'visualizations')
        kg_data_dir = os.path.join(output_dir, 'data')
        os.makedirs(kg_viz_dir, exist_ok=True)
        os.makedirs(kg_data_dir, exist_ok=True)
        graph_path = os.path.join(kg_viz_dir, f'knowledge_graph_{timestamp}.png')
        graph_generator.visualize(graph_path) # visualize handles its own errors
        export_path = os.path.join(kg_data_dir, f'knowledge_graph_export_{timestamp}.json')
        graph_generator.export_graph(export_path) # export_graph handles its own errors
        # Log success separately
        logger.info(f"Knowledge graph visualization attempted. Path: {graph_path}")
        logger.info(f"Knowledge graph data export attempted. Path: {export_path}")
    except Exception as e:
        # Log error but don't stop main script
        logger.error(f"Failed to save knowledge graph state/visualization: {e}", exc_info=True)

def main():
    """
    Execute the ANSR-DT pipeline:
    1. Clear the results directory.
    2. Setup logging and directory structure.
    3. Load configuration.
    4. Train or load CNN-LSTM and PPO models.
    5. Initialize the ANSR-DT core system.
    6. Prepare sample input data.
    7. Run integrated inference and explanation.
    8. Generate visualizations and save results.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    # --- Initialization ---
    cnn_lstm_model = None
    ppo_agent = None
    ansrdt = None
    visualizers = None
    config = None

    # --- Setup Logging Early ---
    log_file_path = os.path.join(project_root, 'logs', 'ansr_dt_main.log')
    # Ensure logs directory exists before setting up logger
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = setup_logging(log_file=log_file_path, log_level=logging.INFO)
    logger.info(f"--- Starting ANSR-DT Pipeline Run ({datetime.now()}) ---")
    logger.info(f"Project Root: {project_root}")

    try:
        # --- Clear Results ---
        # Use function pointer defined earlier after checking existence
        results_dir_to_clear = os.path.join(project_root, "results")
        clear_results_func(results_dir_to_clear)

        # --- Setup Directories ---
        setup_project_structure(project_root)

        # --- Load Config ---
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        # --- Resolve and Validate Paths ---
        paths_to_resolve = {
            'results_dir': 'results',
            'data_file': config['paths']['data_file'],
            'plot_config_path': config['paths']['plot_config_path'],
            'reasoning_rules_path': config['paths']['reasoning_rules_path'],
            'knowledge_graphs_dir': 'results/knowledge_graphs'
        }
        resolved_paths = {}
        for key, rel_path in paths_to_resolve.items():
            # Make paths absolute relative to project_root
            abs_path = os.path.normpath(os.path.join(project_root, rel_path))
            resolved_paths[key] = abs_path
            logger.info(f"Resolved path '{key}': {abs_path}")
            # Validate existence (files) or ensure creation (dirs)
            if key.endswith('_file') or key.endswith('_path'):
                if not os.path.exists(abs_path):
                    logger.error(f"Required file/path '{key}' not found at: {abs_path}")
                    raise FileNotFoundError(f"File not found: {abs_path}")
            elif key.endswith('_dir'):
                os.makedirs(abs_path, exist_ok=True)
        config['paths'] = resolved_paths # Update config with absolute paths

        # --- Define Model Paths ---
        model_paths = {
            'cnn_lstm': os.path.join(config['paths']['results_dir'], 'best_model.keras'),
            'ppo': os.path.join(config['paths']['results_dir'], 'ppo_ansr_dt.zip')
        }

        # --- Train/Load CNN-LSTM Model ---
        input_shape_tuple = tuple(config['model']['input_shape'])
        # Check if model exists, if not, trigger training pipeline
        if not os.path.exists(model_paths['cnn_lstm']):
            logger.info(f"CNN-LSTM model not found at {model_paths['cnn_lstm']}. Training new model...")
            pipeline = ANSRDTPipeline(config, config_path, logger)
            pipeline.run() # This trains and saves 'best_model.keras'
            # Ensure model was actually created
            if not os.path.exists(model_paths['cnn_lstm']):
                 raise RuntimeError("Pipeline completed but CNN-LSTM model file still not found.")
            logger.info("CNN-LSTM training pipeline completed.")

        # Load the potentially newly trained or existing model
        logger.info(f"Loading CNN-LSTM model from {model_paths['cnn_lstm']}")
        cnn_lstm_model = load_model_with_initialization(
            path=model_paths['cnn_lstm'],
            logger=logger,
            input_shape=input_shape_tuple
        )
        if cnn_lstm_model is None:
            raise RuntimeError("Failed to load CNN-LSTM model.")
        logger.info("CNN-LSTM model loaded and initialized.")


        # --- Train/Load PPO Agent ---
        if not os.path.exists(model_paths['ppo']):
            logger.info(f"PPO model not found at {model_paths['ppo']}. Training new agent...")
            ppo_success = train_ppo_agent(config_path)
            if not ppo_success:
                raise RuntimeError("PPO training failed")
             # Ensure model was actually created
            if not os.path.exists(model_paths['ppo']):
                 raise RuntimeError("PPO training completed but model file still not found.")
            logger.info("PPO training completed.")

        # Load the potentially newly trained or existing PPO agent
        logger.info(f"Loading PPO agent from {model_paths['ppo']}")
        ppo_agent = PPO.load(model_paths['ppo'])
        if ppo_agent is None:
            raise RuntimeError("Failed to load PPO agent.")
        logger.info("PPO agent loaded.")


        # --- Final Model Check ---
        if cnn_lstm_model is None or ppo_agent is None:
            raise RuntimeError("Critical model (CNN-LSTM or PPO) failed to load or train.")
        logger.info("CNN-LSTM and PPO models are ready.")

        # --- Initialize ANSR-DT Core System ---
        logger.info("Initializing ANSR-DT core system...")
        ansrdt = ExplainableANSRDT(
            config_path=config_path,
            logger=logger,
            cnn_lstm_model=cnn_lstm_model, # Pass loaded model
            ppo_agent=ppo_agent          # Pass loaded agent
        )
        logger.info("ANSR-DT system initialized.")

        # --- Prepare Sample Input Window ---
        logger.info(f"Loading sample data from {config['paths']['data_file']}...")
        # Use try-except for file loading
        try:
            test_data = np.load(config['paths']['data_file'])
        except FileNotFoundError:
            logger.error(f"Sample data file not found at {config['paths']['data_file']}")
            raise
        feature_names = config['model'].get('feature_names')
        if not feature_names:
            raise ValueError("feature_names not found in model configuration.")
        sensor_window = prepare_sensor_window(
            data=test_data,
            window_size=config['model']['window_size'],
            feature_names=feature_names
        )
        logger.info(f"Prepared sample sensor window shape: {sensor_window.shape}")

        # --- Initialize Visualizers ---
        figures_dir = os.path.join(config['paths']['results_dir'], 'visualization')
        visualizers = initialize_visualizers(logger, figures_dir, config)
        knowledge_graph_enabled = config.get('knowledge_graph', {}).get('enabled', False)

        # --- Log Input Data Details ---
        logger.info("Input data validation for inference:")
        logger.info(f"- Shape: {sensor_window.shape}")
        logger.info(f"- Data Type: {sensor_window.dtype}")
        logger.info(f"- Range: [{np.min(sensor_window):.4f}, {np.max(sensor_window):.4f}]")
        logger.info(f"- Contains NaN/Inf: {np.isnan(sensor_window).any() or np.isinf(sensor_window).any()}")

        # --- Run Integrated Inference & Explanation ---
        logger.info("Running integrated inference and explanation on the sample window...")
        result = ansrdt.adapt_and_explain(sensor_window) # This updates ansrdt.current_state internally
        logger.info("Integrated inference step completed.")

        # --- Log Inference Results ---
        logger.info("Inference results overview:")
        if isinstance(result, dict):
            logger.info(f"- Timestamp: {result.get('timestamp', 'N/A')}")
            # Format confidence safely
            conf = result.get('confidence')
            conf_str = f"{conf:.3f}" if isinstance(conf, (float, int)) else 'N/A'
            logger.info(f"- Anomaly Confidence: {conf_str}")
            logger.info(f"- Recommended Action: {result.get('action', 'None')}") # Show 'None' if no action
            logger.info(f"- Explanation: {result.get('explanation', 'N/A')}")
            logger.info(f"- Symbolic Insights: {result.get('insights', [])}") # Show empty list if no insights
        else:
            logger.warning(f"- Result is not a dictionary, type: {type(result)}")

        # --- Generate Visualizations ---
        logger.info("Generating visualizations...")
        try:
            # Rule Activation Visualization
            if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner and hasattr(ansrdt.reasoner, 'get_rule_activations'):
                activation_history = ansrdt.reasoner.get_rule_activations()
                if activation_history: # Check if history is not empty
                    # Process activations for visualization
                    all_activations_for_viz = []
                    # Iterate through history records (each record is a dict for a timestep)
                    for record in activation_history:
                        # Each record has 'activated_rules_detailed' which is a list of dicts
                        for activation_detail in record.get('activated_rules_detailed', []):
                            # Ensure the detail has confidence before adding
                            if 'confidence' in activation_detail:
                                all_activations_for_viz.append(activation_detail)
                            else:
                                logger.warning(f"Skipping activation detail due to missing 'confidence': {activation_detail}")

                    if all_activations_for_viz: # Check if we collected any valid activations
                        save_path_activations = os.path.join(figures_dir, 'neurosymbolic', 'rule_activations.png')
                        # Ensure the subdirectory exists
                        os.makedirs(os.path.dirname(save_path_activations), exist_ok=True)
                        visualizers['neurosymbolic'].visualize_rule_activations(
                            activations=all_activations_for_viz,
                            save_path=save_path_activations
                        )
                    else:
                        logger.info("No valid rule activations with confidence found in history to visualize.")
                else:
                     logger.info("Rule activation history is empty.") # Log if history itself is empty
            else:
                logger.warning("Could not access reasoner or rule activations for visualization.")

            # State Transition Visualization
            if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner and hasattr(ansrdt.reasoner, 'state_tracker'):
                transition_probs = ansrdt.reasoner.state_tracker.get_transition_probabilities()
                if transition_probs is not None and transition_probs.size > 0:
                    save_path_transitions = os.path.join(figures_dir, 'neurosymbolic', 'state_transitions.png')
                    # Ensure the subdirectory exists
                    os.makedirs(os.path.dirname(save_path_transitions), exist_ok=True)
                    visualizers['neurosymbolic'].plot_state_transitions(
                        transition_matrix=transition_probs,
                        save_path=save_path_transitions
                    )
                else:
                     logger.info("No state transition data available from state tracker.")
            else:
                 logger.warning("Could not access state tracker or transition probabilities for visualization.")

            # Knowledge Graph Visualization
            if knowledge_graph_enabled:
                logger.info("Updating and visualizing knowledge graph...")
                # --- FIX: Get current_state from ansrdt object ---
                current_state_data = ansrdt.current_state if hasattr(ansrdt, 'current_state') and ansrdt.current_state is not None else {}
                # Ensure it's a dict even if it was None
                if not isinstance(current_state_data, dict):
                     logger.warning(f"ansrdt.current_state was not a dict ({type(current_state_data)}), using empty dict for KG.")
                     current_state_data = {}
                # --- END FIX ---

                # Get insights from the result dictionary (returned by adapt_and_explain)
                insights_data = result.get('insights', []) if isinstance(result, dict) else []

                # Get learned rules from the reasoner instance
                rules_data = []
                if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner and hasattr(ansrdt.reasoner, 'learned_rules'):
                    rules_data = [{'rule': r, 'confidence': m.get('confidence', 0.0)}
                                  for r, m in ansrdt.reasoner.learned_rules.items()]

                # Construct anomaly info based on inference result
                # Use boolean comparison for clarity
                anomaly_detected = (result.get('confidence', 0.0) > 0.5) if isinstance(result, dict) else False
                anomaly_confidence = result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
                # Determine severity (example: 1 for basic anomaly, could be more nuanced)
                anomaly_severity = 1 if anomaly_detected else 0
                anomaly_data = {
                    'severity': anomaly_severity,
                    'confidence': anomaly_confidence
                }

                visualizers['knowledge_graph'].update_graph(
                    current_state=current_state_data, # Pass the corrected state data
                    insights=insights_data,
                    rules=rules_data, # Pass the list of rule dicts
                    anomalies=anomaly_data # Pass the anomaly dict
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_knowledge_graph_state(
                    graph_generator=visualizers['knowledge_graph'],
                    output_dir=config['paths']['knowledge_graphs_dir'],
                    timestamp=timestamp,
                    logger=logger
                )
            else:
                 logger.info("Knowledge graph generation is disabled in config.")
        except Exception as viz_error:
            logger.error(f"Visualization failed: {viz_error}", exc_info=True) # Log full traceback

        # --- Save Results ---
        logger.info("Saving final results and state...")
        # Safely access reasoner attributes for final saving
        learned_rules_dict = {}
        rule_activation_history = []
        if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner:
            # Save learned rules currently in the reasoner
            learned_rules_dict = {r: m for r, m in ansrdt.reasoner.learned_rules.items()} if hasattr(ansrdt.reasoner, 'learned_rules') else {}
            # Save the entire activation history accumulated by the reasoner
            rule_activation_history = ansrdt.reasoner.get_rule_activations() if hasattr(ansrdt.reasoner, 'get_rule_activations') else []

        # Compile final results, including the inference outcome and KG stats
        final_results_data = {
            'inference_result': result if isinstance(result, dict) else {'error': 'Result not a dictionary', 'raw': str(result)},
            'learned_rules_snapshot': learned_rules_dict, # Rules present at the end
            'rule_activations_history': rule_activation_history, # Full history from reasoner
            'state_history_summary': ansrdt.state_history[-100:], # Summary from ANSRDTCore
            'knowledge_graph_stats': visualizers['knowledge_graph'].get_graph_statistics() if knowledge_graph_enabled and visualizers else None,
            'run_timestamp': str(datetime.now().isoformat())
        }

        # Save the compiled results
        save_results(
            results=final_results_data,
            path=os.path.join(config['paths']['results_dir'], 'final_run_results.json'),
            logger=logger
        )

        # --- Log Summary ---
        logger.info("\n--- ANSR-DT Run Summary ---")
        # Use data from final_results_data for consistency
        learned_rules_count = len(final_results_data['learned_rules_snapshot'])
        # Get insights specific to the *last inference step* from the saved inference_result
        insights_count = len(final_results_data['inference_result'].get('insights', [])) if isinstance(final_results_data['inference_result'], dict) else 0
        kg_stats = final_results_data.get('knowledge_graph_stats')
        kg_nodes = kg_stats.get('total_nodes', 'N/A') if kg_stats else 'Disabled'
        kg_edges = kg_stats.get('total_edges', 'N/A') if kg_stats else 'Disabled'

        final_conf = final_results_data['inference_result'].get('confidence', 0.0) if isinstance(final_results_data['inference_result'], dict) else 0.0
        final_conf_perc = f"{final_conf:.2%}" if isinstance(final_conf, (float, int)) else 'N/A'
        final_explanation = final_results_data['inference_result'].get('explanation', 'N/A') if isinstance(final_results_data['inference_result'], dict) else 'N/A'

        logger.info(f"- Learned Rules in Reasoner (End): {learned_rules_count}")
        logger.info(f"- Symbolic Insights Generated (Last Step): {insights_count}")
        logger.info(f"- Final Neural Confidence (Last Step): {final_conf_perc}")
        logger.info(f"- Knowledge Graph Nodes (Last Snapshot): {kg_nodes}")
        logger.info(f"- Knowledge Graph Edges (Last Snapshot): {kg_edges}")
        logger.info(f"- Final Explanation Provided (Last Step): {final_explanation}\n")

        logger.info("--- ANSR-DT Pipeline Completed Successfully ---")
        return True

    # --- Exception Handling ---
    except KeyboardInterrupt:
        logger.warning("--- Pipeline Interrupted by User ---") # Use warning level
        return False
    except FileNotFoundError as e:
         logger.error(f"Pipeline Failed: Required file not found - {e}", exc_info=True)
         return False
    except KeyError as e:
         logger.error(f"Pipeline Failed: Missing configuration key or data feature - {e}", exc_info=True)
         return False
    except ValueError as e:
         logger.error(f"Pipeline Failed: Invalid value or data shape - {e}", exc_info=True)
         return False
    except RuntimeError as e:
         logger.error(f"Pipeline Failed: Runtime error during execution - {e}", exc_info=True)
         return False
    except Exception as e: # Catch-all for any other unexpected errors
        logger.exception(f"--- Pipeline Failed Unexpectedly: {str(e)} ---") # Use logger.exception
        return False
    finally:
        # Log completion regardless of success/failure
        completion_status = "Successfully" if 'run_successful' in locals() and run_successful else "with Errors"
        logger.info(f"--- ANSR-DT Run Finalizing {completion_status} ({datetime.now()}) ---")


if __name__ == '__main__':
    run_successful = main()
    exit_code = 0 if run_successful else 1
    # Ensure logger exists before final message
    if 'logger' in globals() and logger:
         logger.info(f"--- ANSR-DT Run Finished with Exit Code {exit_code} ({datetime.now()}) ---")
    else:
         print(f"--- ANSR-DT Run Finished with Exit Code {exit_code} ({datetime.now()}) --- (Logger not initialized)")
    # Close logging handlers if necessary (usually handled on exit)
    # logging.shutdown()
    exit(exit_code)