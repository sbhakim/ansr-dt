#!/usr/bin/env python3
import os
# Force CPU usage - Must be done before TensorFlow import
# Comment out or remove this line if you want to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import json
import logging
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional # Added Optional

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
from src.reasoning.knowledge_graph import KnowledgeGraphGenerator # Already imported

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
        'results/knowledge_graphs/visualizations', # Directory for plot images
        'results/knowledge_graphs/data' # Directory for exported graph data
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
        # Select the *first* window_size elements for the sample
        features = [data[key][:window_size] for key in feature_names]
        sensor_window = np.stack(features, axis=1)
        # Ensure data type is float32 for TF models
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
        os.makedirs(figures_dir, exist_ok=True) # Ensure base visualization dir exists
        neurosymbolic_visualizer = NeurosymbolicVisualizer(logger)
        # Use KG config for max_nodes/history if available
        kg_config = config.get('knowledge_graph', {})
        knowledge_graph_generator = KnowledgeGraphGenerator(
            logger=logger,
            max_nodes=kg_config.get('max_nodes', 500), # Default from KG class if not in config
            max_history=kg_config.get('max_history', 1000) # Default from KG class
        )
        return {
            'neurosymbolic': neurosymbolic_visualizer,
            'knowledge_graph': knowledge_graph_generator
        }
    except Exception as e:
        logger.error(f"Failed to initialize visualizers: {e}")
        raise

def save_results(results: dict, path: str, logger: logging.Logger):
    """Save results to a JSON file, handling NumPy types using NumpyEncoder."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            # Use the imported NumpyEncoder from pipeline.py (or the local one if needed)
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Results saved to {path}")
    except TypeError as te:
         logger.error(f"Failed to serialize results to JSON: {te}. Check data types.", exc_info=True)
         # Optionally save raw dump on error
         try:
             import pickle
             with open(path + '.pkl', 'wb') as pf:
                  pickle.dump(results, pf)
             logger.warning(f"Saved raw results as pickle due to JSON error: {path + '.pkl'}")
         except Exception as pkl_e:
             logger.error(f"Failed to save pickle backup: {pkl_e}")
    except Exception as e:
        logger.error(f"Failed to save results to {path}: {e}")


# --- Updated Function Definition ---
def save_knowledge_graph_state(graph_generator: KnowledgeGraphGenerator,
                               output_dir: str,
                               timestamp: str,
                               logger: logging.Logger,
                               kg_config: dict): # Added kg_config argument
    """Save knowledge graph state (data export) and visualizations (full and focused)."""
    try:
        kg_viz_dir = os.path.join(output_dir, 'visualizations')
        kg_data_dir = os.path.join(output_dir, 'data')
        os.makedirs(kg_viz_dir, exist_ok=True)
        os.makedirs(kg_data_dir, exist_ok=True)

        # --- Get Plotting Settings from Config ---
        vis_config = kg_config.get('visualization', {}) # Get the visualization sub-dict
        # Provide defaults directly here if not found in config
        ieee_layout = vis_config.get('ieee_layout', 'dot')
        ieee_rule_threshold = vis_config.get('ieee_rule_threshold', 0.8)
        ieee_font_scale = vis_config.get('ieee_font_scale', 0.8)
        full_layout = vis_config.get('full_layout', 'spring')
        full_font_scale = vis_config.get('full_font_scale', 1.0)
        show_full_edge_labels = vis_config.get('show_full_edge_labels', True)
        dpi = vis_config.get('dpi', 300)

        # --- Plot 1: Full Snapshot ---
        graph_path_full = os.path.join(kg_viz_dir, f'knowledge_graph_full_{timestamp}.png')
        logger.info(f"Generating full KG plot ({full_layout} layout)...")
        graph_generator.visualize(
            output_path=graph_path_full,
            plot_type='full',
            layout_engine=full_layout,
            show_edge_labels=show_full_edge_labels,
            font_scale=full_font_scale,
            use_cache=True, # Caching useful for potentially large full plots
            dpi=dpi
        )
        logger.info(f"Full knowledge graph visualization attempt saved to: {graph_path_full}")

        # --- Plot 2: Focused IEEE Snapshot ---
        graph_path_ieee = os.path.join(kg_viz_dir, f'knowledge_graph_ieee_{timestamp}.png')
        logger.info(f"Generating focused IEEE KG plot ({ieee_layout} layout)...")
        graph_generator.visualize(
            output_path=graph_path_ieee,
            plot_type='focused_ieee',
            layout_engine=ieee_layout,
            rule_confidence_threshold=ieee_rule_threshold,
            show_edge_labels=False, # Typically False for focused plots
            font_scale=ieee_font_scale,
            use_cache=False, # Recalculate layout for potentially different subgraph
            dpi=dpi
        )
        logger.info(f"Focused IEEE knowledge graph visualization attempt saved to: {graph_path_ieee}")

        # --- Export Graph Data ---
        export_path = os.path.join(kg_data_dir, f'knowledge_graph_export_{timestamp}.json')
        graph_generator.export_graph(export_path) # export_graph handles its own errors
        logger.info(f"Knowledge graph data export attempted. Path: {export_path}")

    except ImportError as ie:
         # Catch missing optional dependencies for layout engines
         logger.error(f"Failed KG visualization: Missing dependency (e.g., pygraphviz or pydot for '{ieee_layout}' layout?). Error: {ie}", exc_info=False) # Show less traceback for ImportErrors
    except Exception as e:
        # Log other errors but don't stop main script
        logger.error(f"Failed during knowledge graph saving/visualization: {e}", exc_info=True)


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
    8. Generate visualizations (including KG) and save results.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_successful = False # Track success for finally block

    # --- Initialization ---
    cnn_lstm_model: Optional[Any] = None # Use type hint
    ppo_agent: Optional[PPO] = None
    ansrdt: Optional[ExplainableANSRDT] = None
    visualizers: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    logger: Optional[logging.Logger] = None # Define logger variable

    try:
        # --- Setup Logging Early ---
        log_file_path = os.path.join(project_root, 'logs', 'ansr_dt_main.log')
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logger = setup_logging(log_file=log_file_path, log_level=logging.INFO)
        logger.info(f"--- Starting ANSR-DT Pipeline Run ({datetime.now()}) ---")
        logger.info(f"Project Root: {project_root}")

        # --- Clear Results ---
        results_dir_to_clear = os.path.join(project_root, "results")
        clear_results_func(results_dir_to_clear) # Use function pointer

        # --- Setup Directories ---
        setup_project_structure(project_root)

        # --- Load Config ---
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = load_config(config_path) # load_config handles its own logging/errors
        logger.info(f"Configuration loaded from {config_path}")

        # --- Resolve and Validate Paths ---
        # Define paths relative to project root or config dir as appropriate
        paths_to_resolve = {
            'results_dir': (project_root, 'results'), # Tuple: (base_dir, relative_path_in_config_or_default)
            'data_file': (project_root, config['paths']['data_file']),
            'plot_config_path': (os.path.dirname(config_path), config['paths']['plot_config_path']), # Relative to config dir
            'reasoning_rules_path': (project_root, config['paths']['reasoning_rules_path']),
            'knowledge_graphs_dir': (project_root, 'results/knowledge_graphs') # Usually relative to results
        }
        resolved_paths = {}
        for key, (base_dir, rel_path) in paths_to_resolve.items():
            abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
            resolved_paths[key] = abs_path
            logger.info(f"Resolved path '{key}': {abs_path}")
            # Validate existence (files) or ensure creation (dirs)
            if key.endswith('_dir'):
                os.makedirs(abs_path, exist_ok=True)
            elif not os.path.exists(abs_path): # Check file existence
                 logger.error(f"Required file/path '{key}' not found at: {abs_path}")
                 raise FileNotFoundError(f"File not found: {abs_path}")

        # Update config with resolved, absolute paths BEFORE passing config around
        config['paths'].update(resolved_paths) # Update the paths sub-dictionary

        # --- Define Model Paths ---
        # Paths should now be absolute from the updated config
        model_paths = {
            'cnn_lstm': os.path.join(config['paths']['results_dir'], 'best_model.keras'),
            'ppo': os.path.join(config['paths']['results_dir'], 'ppo_ansr_dt.zip')
        }
        logger.debug(f"Using CNN-LSTM path: {model_paths['cnn_lstm']}")
        logger.debug(f"Using PPO path: {model_paths['ppo']}")

        # --- Train/Load CNN-LSTM Model ---
        input_shape_tuple = tuple(config['model']['input_shape'])
        if not os.path.exists(model_paths['cnn_lstm']):
            logger.info(f"CNN-LSTM model not found at {model_paths['cnn_lstm']}. Training new model...")
            # Pass the full config dict, as pipeline expects it
            pipeline = ANSRDTPipeline(config, config_path, logger)
            pipeline.run() # This trains and saves 'best_model.keras'
            if not os.path.exists(model_paths['cnn_lstm']):
                 raise RuntimeError("Pipeline completed but CNN-LSTM model file still not found.")
            logger.info("CNN-LSTM training pipeline completed.")
        else:
             logger.info(f"Found existing CNN-LSTM model at {model_paths['cnn_lstm']}")

        logger.info(f"Loading CNN-LSTM model from {model_paths['cnn_lstm']}")
        cnn_lstm_model = load_model_with_initialization(
            path=model_paths['cnn_lstm'],
            logger=logger,
            input_shape=input_shape_tuple # Provide shape for potential build
        )
        if cnn_lstm_model is None: # load_model raises error, but double-check
            raise RuntimeError("Failed to load CNN-LSTM model.")
        logger.info("CNN-LSTM model loaded and initialized.")


        # --- Train/Load PPO Agent ---
        if not os.path.exists(model_paths['ppo']):
            logger.info(f"PPO model not found at {model_paths['ppo']}. Training new agent...")
            # Pass config_path, train_ppo_agent loads config internally
            ppo_success = train_ppo_agent(config_path)
            if not ppo_success:
                raise RuntimeError("PPO training failed")
            if not os.path.exists(model_paths['ppo']):
                 raise RuntimeError("PPO training completed but model file still not found.")
            logger.info("PPO training completed.")
        else:
             logger.info(f"Found existing PPO model at {model_paths['ppo']}")

        logger.info(f"Loading PPO agent from {model_paths['ppo']}")
        ppo_agent = PPO.load(model_paths['ppo'])
        if ppo_agent is None: # PPO.load raises error, but double-check
            raise RuntimeError("Failed to load PPO agent.")
        logger.info("PPO agent loaded.")


        # --- Final Model Check ---
        if cnn_lstm_model is None or ppo_agent is None:
            # This condition might be redundant due to checks above, but serves as safeguard
            raise RuntimeError("Critical model (CNN-LSTM or PPO) failed to load or train.")
        logger.info("CNN-LSTM and PPO models are ready.")

        # --- Initialize ANSR-DT Core System ---
        logger.info("Initializing ANSR-DT core system...")
        # Pass config_path, core system loads config internally now
        ansrdt = ExplainableANSRDT(
            config_path=config_path, # Core loads config from this path
            logger=logger,
            cnn_lstm_model=cnn_lstm_model, # Pass loaded model
            ppo_agent=ppo_agent          # Pass loaded agent
        )
        logger.info("ANSR-DT system initialized.")

        # --- Prepare Sample Input Window ---
        # Use the absolute path from the resolved config
        logger.info(f"Loading sample data from {config['paths']['data_file']}...")
        try:
            test_data = np.load(config['paths']['data_file'])
        except FileNotFoundError:
            logger.error(f"Sample data file not found at {config['paths']['data_file']}")
            raise
        feature_names = config['model'].get('feature_names')
        if not feature_names:
            logger.error("feature_names not found in model configuration.")
            raise ValueError("feature_names not found in model configuration.")

        sensor_window = prepare_sensor_window(
            data=test_data,
            window_size=config['model']['window_size'],
            feature_names=feature_names
        )
        logger.info(f"Prepared sample sensor window shape: {sensor_window.shape}")

        # --- Initialize Visualizers ---
        figures_dir = os.path.join(config['paths']['results_dir'], 'visualization')
        visualizers = initialize_visualizers(logger, figures_dir, config) # Pass full config
        knowledge_graph_enabled = config.get('knowledge_graph', {}).get('enabled', False)

        # --- Log Input Data Details ---
        logger.info("Input data validation for inference:")
        logger.info(f"- Shape: {sensor_window.shape}")
        logger.info(f"- Data Type: {sensor_window.dtype}")
        # Add check for NaN/Inf before calculating min/max
        if np.isnan(sensor_window).any() or np.isinf(sensor_window).any():
             logger.warning("Input data contains NaN or Inf values.")
             logger.info(f"- Contains NaN/Inf: True")
             # Optionally handle/log range differently if NaNs are present
        else:
             logger.info(f"- Range: [{np.min(sensor_window):.4f}, {np.max(sensor_window):.4f}]")
             logger.info(f"- Contains NaN/Inf: False")


        # --- Run Integrated Inference & Explanation ---
        logger.info("Running integrated inference and explanation on the sample window...")
        # adapt_and_explain performs state update and returns the state dict
        result = ansrdt.adapt_and_explain(sensor_window) # This updates ansrdt.current_state internally
        logger.info("Integrated inference step completed.")

        # --- Log Inference Results ---
        logger.info("Inference results overview:")
        if isinstance(result, dict):
            # Check for error status from adapt_and_explain
            if result.get('status') == 'error':
                logger.error(f"Inference failed with error: {result.get('error', 'Unknown error')}")
                # Depending on severity, might want to raise an error here or stop processing
                # raise RuntimeError(f"Inference failed: {result.get('error')}")
            else:
                logger.info(f"- Timestamp: {result.get('timestamp', 'N/A')}")
                conf = result.get('confidence') # Use 'confidence' as primary key now
                conf_str = f"{conf:.3f}" if isinstance(conf, (float, int, np.number)) else 'N/A'
                logger.info(f"- Anomaly Confidence: {conf_str}")
                logger.info(f"- Recommended Action (RL): {result.get('recommended_action', 'None')}")
                logger.info(f"- Control Parameters (Adaptive): {result.get('control_parameters', 'None')}")
                logger.info(f"- Symbolic Insights: {result.get('insights', [])}")
                # Log the generated explanation string
                explanation = ansrdt.explain_decision(result) # Generate explanation from result dict
                logger.info(f"- Generated Explanation: {explanation}")
        else:
            logger.warning(f"Inference result is not a dictionary, type: {type(result)}")


        # --- Generate Visualizations ---
        logger.info("Generating visualizations...")
        try:
            # Rule Activation Visualization
            # Ensure reasoner and necessary methods/attributes exist
            if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner and hasattr(ansrdt.reasoner, 'get_rule_activations'):
                activation_history = ansrdt.reasoner.get_rule_activations() # Gets full history
                if activation_history:
                    # Check structure of activation_history elements
                    # Assuming it's List[Dict] where each dict has 'activated_rules_detailed': List[Dict]
                    all_activations_for_viz = []
                    for record in activation_history:
                         # Check if record is a dict and has the key
                         if isinstance(record, dict) and 'activated_rules_detailed' in record:
                              details = record['activated_rules_detailed']
                              if isinstance(details, list):
                                   for activation_detail in details:
                                       # Check if detail is dict and has confidence
                                       if isinstance(activation_detail, dict) and 'confidence' in activation_detail:
                                           all_activations_for_viz.append(activation_detail)
                                       else:
                                            logger.debug(f"Skipping activation detail due to missing 'confidence' or invalid format: {activation_detail}")
                         else:
                             logger.debug(f"Skipping activation history record due to unexpected format: {record}")

                    if all_activations_for_viz:
                        save_path_activations = os.path.join(figures_dir, 'neurosymbolic', 'rule_activations.png')
                        os.makedirs(os.path.dirname(save_path_activations), exist_ok=True)
                        # Ensure visualizer exists
                        if 'neurosymbolic' in visualizers:
                            visualizers['neurosymbolic'].visualize_rule_activations(
                                activations=all_activations_for_viz,
                                save_path=save_path_activations
                            )
                        else:
                            logger.warning("Neurosymbolic visualizer not initialized.")
                    else:
                        logger.info("No valid rule activations with confidence found in history to visualize.")
                else:
                     logger.info("Rule activation history is empty or unavailable.")
            else:
                logger.warning("Could not access reasoner or rule activations for visualization.")

            # State Transition Visualization
            # Ensure reasoner and state_tracker exist
            if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner and hasattr(ansrdt.reasoner, 'state_tracker') and hasattr(ansrdt.reasoner.state_tracker, 'get_transition_probabilities'):
                transition_probs = ansrdt.reasoner.state_tracker.get_transition_probabilities()
                # Check if the result is a valid numpy array
                if isinstance(transition_probs, np.ndarray) and transition_probs.size > 0:
                    save_path_transitions = os.path.join(figures_dir, 'neurosymbolic', 'state_transitions.png')
                    os.makedirs(os.path.dirname(save_path_transitions), exist_ok=True)
                    if 'neurosymbolic' in visualizers:
                        visualizers['neurosymbolic'].plot_state_transitions(
                            transition_matrix=transition_probs,
                            save_path=save_path_transitions
                        )
                    else:
                         logger.warning("Neurosymbolic visualizer not initialized.")
                else:
                     logger.info("No state transition data available from state tracker or data was invalid.")
            else:
                 logger.warning("Could not access state tracker or transition probabilities for visualization.")

            # --- Knowledge Graph Visualization ---
            if knowledge_graph_enabled:
                logger.info("Updating and visualizing knowledge graph...")
                # Get the latest state from the ansrdt object (which was updated by adapt_and_explain)
                current_state_data = ansrdt.current_state if hasattr(ansrdt, 'current_state') and isinstance(ansrdt.current_state, dict) else {}
                if not current_state_data:
                     logger.warning("ansrdt.current_state is not available or not a dictionary. Using empty data for KG.")

                # Extract necessary info directly from the latest state dictionary
                insights_data = current_state_data.get('insights', [])
                # Get learned rules snapshot from the reasoner instance
                rules_data = []
                if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner and hasattr(ansrdt.reasoner, 'learned_rules'):
                    # Ensure learned_rules is a dictionary {rule_str: metadata_dict}
                    if isinstance(ansrdt.reasoner.learned_rules, dict):
                         rules_data = [{'rule': r, 'confidence': m.get('confidence', 0.0)}
                                       for r, m in ansrdt.reasoner.learned_rules.items() if isinstance(m, dict)]
                    else:
                         logger.warning("ansrdt.reasoner.learned_rules is not a dictionary.")


                # Construct anomaly info using latest state data
                anomaly_confidence = current_state_data.get('confidence', 0.0)
                # Define anomaly severity based on state or confidence, e.g.:
                anomaly_detected = current_state_data.get('anomaly_detected', False)
                system_state_val = current_state_data.get('system_state', 0) # 0=N, 1=D, 2=C
                anomaly_severity = 0
                if system_state_val == 2: anomaly_severity = 2 # Critical
                elif system_state_val == 1: anomaly_severity = 1 # Degraded
                elif anomaly_detected: anomaly_severity = 1 # Basic anomaly if detected but not Degraded/Critical

                anomaly_data = {
                    'severity': anomaly_severity,
                    'confidence': anomaly_confidence
                }

                # Ensure visualizer exists
                if 'knowledge_graph' in visualizers:
                    visualizers['knowledge_graph'].update_graph(
                        current_state=current_state_data, # Pass the full current state
                        insights=insights_data,
                        rules=rules_data, # Pass the list of rule dicts
                        anomalies=anomaly_data # Pass the anomaly dict
                    )

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # --- Pass KG config section to save function ---
                    kg_config_section = config.get('knowledge_graph', {})
                    save_knowledge_graph_state(
                        graph_generator=visualizers['knowledge_graph'],
                        output_dir=config['paths']['knowledge_graphs_dir'], # Use absolute path
                        timestamp=timestamp,
                        logger=logger,
                        kg_config=kg_config_section # Pass the config subsection
                    )
                else:
                    logger.warning("Knowledge Graph visualizer not initialized.")
            else:
                 logger.info("Knowledge graph generation is disabled in config.")
        except Exception as viz_error:
            logger.error(f"Visualization generation failed: {viz_error}", exc_info=True) # Log full traceback


        # --- Save Final Results ---
        logger.info("Saving final results and state...")
        # Safely access reasoner attributes for final saving
        learned_rules_snapshot = {}
        rule_activation_history = []
        if hasattr(ansrdt, 'reasoner') and ansrdt.reasoner:
            # Ensure learned_rules is a dict before accessing items
            if hasattr(ansrdt.reasoner, 'learned_rules') and isinstance(ansrdt.reasoner.learned_rules, dict):
                learned_rules_snapshot = {r: m for r, m in ansrdt.reasoner.learned_rules.items()}
            if hasattr(ansrdt.reasoner, 'get_rule_activations'):
                rule_activation_history = ansrdt.reasoner.get_rule_activations() # Get full history

        # Compile final results, including the inference outcome and KG stats
        final_results_data = {
            'inference_result': result if isinstance(result, dict) else {'error': 'Result not a dictionary', 'raw': str(result)},
            'learned_rules_snapshot': learned_rules_snapshot, # Rules present at the end
            'rule_activations_history': rule_activation_history, # Full history from reasoner
            # Limit saved state history if it gets large
            'state_history_summary': ansrdt.state_history[-min(len(ansrdt.state_history), 100):],
            'knowledge_graph_stats': visualizers['knowledge_graph'].get_graph_statistics() if knowledge_graph_enabled and 'knowledge_graph' in visualizers else None,
            'run_timestamp': str(datetime.now().isoformat())
        }

        # Save the compiled results
        save_results(
            results=final_results_data,
            path=os.path.join(config['paths']['results_dir'], 'final_run_results.json'), # Use absolute path
            logger=logger
        )

        # --- Log Summary ---
        logger.info("\n--- ANSR-DT Run Summary ---")
        # Use data from final_results_data for consistency
        learned_rules_count = len(final_results_data['learned_rules_snapshot'])
        # Get insights specific to the *last inference step* from the saved inference_result
        last_inference = final_results_data.get('inference_result', {})
        insights_count = 0
        final_conf_perc = 'N/A'
        final_explanation = 'N/A'
        if isinstance(last_inference, dict) and last_inference.get('status') != 'error':
             insights_count = len(last_inference.get('insights', []))
             final_conf = last_inference.get('confidence', 0.0)
             final_conf_perc = f"{final_conf:.2%}" if isinstance(final_conf, (float, int, np.number)) else 'N/A'
             # Re-generate explanation for summary if needed, or get from result if stored
             final_explanation = ansrdt.explain_decision(last_inference) # Use the method for consistency

        kg_stats = final_results_data.get('knowledge_graph_stats')
        kg_nodes = kg_stats.get('total_nodes', 'N/A') if kg_stats else 'Disabled'
        kg_edges = kg_stats.get('total_edges', 'N/A') if kg_stats else 'Disabled'

        logger.info(f"- Learned Rules in Reasoner (End): {learned_rules_count}")
        logger.info(f"- Symbolic Insights Generated (Last Step): {insights_count}")
        logger.info(f"- Final Neural Confidence (Last Step): {final_conf_perc}")
        logger.info(f"- Knowledge Graph Nodes (Last Snapshot): {kg_nodes}")
        logger.info(f"- Knowledge Graph Edges (Last Snapshot): {kg_edges}")
        logger.info(f"- Final Explanation Provided (Last Step): {final_explanation}\n")

        logger.info("--- ANSR-DT Pipeline Completed Successfully ---")
        run_successful = True # Mark run as successful
        return True

    # --- Exception Handling ---
    except KeyboardInterrupt:
        # Use logger if initialized, otherwise print
        msg = "--- Pipeline Interrupted by User ---"
        if logger: logger.warning(msg)
        else: print(msg)
        return False # Indicate unsuccessful run
    except FileNotFoundError as e:
         # Use logger if initialized, otherwise print
         msg = f"Pipeline Failed: Required file not found - {e}"
         if logger: logger.error(msg, exc_info=True)
         else: print(f"ERROR: {msg}")
         return False
    except (KeyError, ValueError) as e: # Combine config/value errors
         msg = f"Pipeline Failed: Configuration error or invalid value/shape - {e}"
         if logger: logger.error(msg, exc_info=True)
         else: print(f"ERROR: {msg}")
         return False
    except RuntimeError as e:
         msg = f"Pipeline Failed: Runtime error during execution - {e}"
         if logger: logger.error(msg, exc_info=True)
         else: print(f"ERROR: {msg}")
         return False
    except Exception as e: # Catch-all for any other unexpected errors
        msg = f"--- Pipeline Failed Unexpectedly: {str(e)} ---"
        if logger: logger.exception(msg) # Use logger.exception to include traceback
        else: print(f"ERROR: {msg}")
        return False
    finally:
        # Log completion status regardless of success/failure
        completion_status = "Successfully" if run_successful else "with Errors"
        final_msg = f"--- ANSR-DT Run Finalizing {completion_status} ({datetime.now()}) ---"
        if logger: logger.info(final_msg)
        else: print(final_msg)


if __name__ == '__main__':
    # Initialize logger outside main() scope for final message
    main_logger = None
    try:
        # Attempt minimal logging setup for the final message if main fails early
        log_file_path_final = os.path.join(project_root, 'logs', 'ansr_dt_main.log')
        os.makedirs(os.path.dirname(log_file_path_final), exist_ok=True)
        main_logger = setup_logging(log_file=log_file_path_final, log_level=logging.INFO)
    except Exception as log_setup_error:
        print(f"Warning: Failed to setup logger for final message: {log_setup_error}")

    # Run the main application logic
    run_successful = main()
    exit_code = 0 if run_successful else 1

    # Log final exit message
    final_exit_msg = f"--- ANSR-DT Run Finished with Exit Code {exit_code} ({datetime.now()}) ---"
    if main_logger:
         main_logger.info(final_exit_msg)
    else:
         # Fallback to print if logger setup failed completely
         print(final_exit_msg)

    # Optional: Explicitly shut down logging
    # logging.shutdown()
    exit(exit_code)