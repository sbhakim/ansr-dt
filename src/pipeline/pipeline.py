# src/pipeline/pipeline.py

import json
import logging
import os
from typing import Tuple, Dict, Any, List
from datetime import datetime # Import datetime

import numpy as np
import tensorflow as tf

# Import components
from src.models.cnn_lstm_model import create_cnn_lstm_model
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import preprocess_sequences
from src.data.data_processing import map_labels
from src.training.trainer import train_model
from src.evaluation.evaluation import evaluate_model
from src.utils.model_utils import save_model, save_scaler
from src.visualization.plotting import load_plot_config, plot_metrics
from src.reasoning.reasoning import SymbolicReasoner

# Custom JSON Encoder to handle NumPy data types and datetime
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts NumPy data types and datetime to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle datetime objects if they appear in learned_rules metadata
        elif isinstance(obj, datetime):
             return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


def validate_config(config: dict, logger: logging.Logger, project_root: str, config_dir: str):
    """Validates configuration and paths, ensuring absolute paths are stored."""
    try:
        required_keys = ['model', 'training', 'paths']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                raise KeyError(f"Missing required configuration key: {key}")

        # Define required paths with their base directories for resolution
        required_paths_info = {
            'data_file': {'relative_to': project_root, 'default': 'data/synthetic_sensor_data_with_anomalies.npz'},
            'results_dir': {'relative_to': project_root, 'default': 'results'},
            'plot_config_path': {'relative_to': config_dir, 'default': 'plot_config.yaml'},
            'reasoning_rules_path': {'relative_to': project_root, 'default': 'src/reasoning/rules.pl'}
        }

        # Resolve paths and store absolute paths back into the config
        for key, path_info in required_paths_info.items():
            # Use path from config if specified, otherwise use default
            relative_path = config['paths'].get(key, path_info['default'])
            base_dir = path_info['relative_to']
            # Resolve to absolute path
            full_path = os.path.normpath(os.path.join(base_dir, relative_path))

            # Store the absolute path back into the config dictionary
            config['paths'][key] = full_path
            logger.debug(f"Validated and resolved path for '{key}': {full_path}")

            # Check existence for files, ensure directories exist
            if key.endswith('_dir'):
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Directory '{key}' ensured at: {full_path}")
            else: # Assume it's a file path
                if not os.path.exists(full_path):
                    logger.error(f"Required file '{key}' not found at resolved path: {full_path}")
                    raise FileNotFoundError(f"Required file '{key}' not found at: {full_path}")
                logger.info(f"File '{key}' found at: {full_path}")

        logger.info("Configuration paths validated and resolved.")

    except Exception as e:
        logger.exception(f"Configuration validation failed: {e}")
        raise


class ANSRDTPipeline:
    """ANSR-DT Pipeline focused on CNN-LSTM training and evaluation,
       delegating rule extraction/update to SymbolicReasoner."""

    # Make NumpyEncoder accessible as a class attribute if needed elsewhere
    NumpyEncoder = NumpyEncoder

    def __init__(self, config: dict, config_path: str, logger: logging.Logger):
        """Initialize pipeline."""
        self.config = config
        self.logger = logger
        # --- CHANGE 1: Store config_path ---
        self.config_path = os.path.abspath(config_path) # Store absolute path
        # --- End CHANGE 1 ---
        self.config_dir = os.path.dirname(self.config_path)
        self.project_root = os.path.dirname(self.config_dir)

        # --- Validate config and resolve paths immediately ---
        validate_config(self.config, self.logger, self.project_root, self.config_dir)
        # Now self.config['paths'] contains absolute paths

        # Initialize components using absolute paths from validated config
        self.data_loader = DataLoader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size']
        )

        # Model state tracking
        self.model = None
        self.model_built = False

        # Define feature names from config
        self.feature_names = self.config['model'].get('feature_names', [])
        if not self.feature_names:
             self.logger.error("feature_names missing in model configuration.")
             raise ValueError("feature_names missing in model configuration.")
        self.logger.info(f"Pipeline initialized with {len(self.feature_names)} features: {self.feature_names}")

        # Initialize SymbolicReasoner as None; will initialize after model is trained
        self.reasoner = None


    def extract_and_update_neural_rules(
            self,
            model: tf.keras.Model,
            input_data: np.ndarray,
            y_pred: np.ndarray,
            threshold: float = 0.8
    ) -> None:
        """
        Delegates rule extraction and updating to the SymbolicReasoner.

        Args:
            model: Trained neural network model (should be built).
            input_data: Input sequences used for extraction (e.g., X_test_scaled).
            y_pred: Model prediction scores for the input sequences.
            threshold: Confidence threshold for considering sequences for rule extraction.
        """
        if self.reasoner is None:
            self.logger.error("Symbolic Reasoner not initialized. Cannot extract/update rules.")
            return
        if not model.built:
             self.logger.error("Model is not built. Cannot extract rules.")
             return

        try:
            self.logger.info("Delegating rule extraction to SymbolicReasoner...")
            potential_new_rules = self.reasoner.extract_rules_from_neural_model(
                input_data=input_data,
                feature_names=self.feature_names,
                threshold=threshold,
                model=model
            )

            if potential_new_rules:
                self.logger.info(f"Passing {len(potential_new_rules)} potential rules to reasoner for update.")
                # Use configurations for update parameters
                reasoning_config = self.config.get('symbolic_reasoning', {})
                self.reasoner.update_rules(
                    potential_new_rules,
                    min_confidence=reasoning_config.get('min_confidence', 0.7),
                    max_learned_rules=reasoning_config.get('max_rules', 100),
                    pruning_strategy=reasoning_config.get('pruning_strategy', 'confidence')
                )

                # Save summary based on reasoner's current state
                stats = self.reasoner.get_rule_statistics()
                self.logger.info(f"Rule update process completed. Current reasoner stats: {stats}")

                rules_summary = {
                    'reasoner_statistics': stats,
                    'learned_rules_in_reasoner': self.reasoner.learned_rules,
                    'timestamp': str(datetime.now().isoformat())
                }
                summary_path = os.path.join(
                    self.config['paths']['results_dir'],
                    'neurosymbolic_rules_summary_pipeline.json'
                )
                # Use the class attribute NumpyEncoder for saving
                with open(summary_path, 'w') as f:
                    json.dump(rules_summary, f, indent=2, cls=ANSRDTPipeline.NumpyEncoder)
                self.logger.info(f"Neurosymbolic rules summary (from pipeline) saved to {summary_path}")
            else:
                self.logger.info("No potential rules extracted by reasoner in this step.")

        except AttributeError as ae:
             self.logger.error(f"Attribute error during rule extraction/update: {ae}. Is reasoner correctly initialized?", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error during neural rule extraction/update delegation: {e}", exc_info=True)


    def run(self):
        """Execute complete pipeline for CNN-LSTM training and evaluation."""
        try:
            self.logger.info("Starting ANSR-DT Training Pipeline execution.")
            # Config validation and path resolution now happens in __init__

            # 2. Load and process data
            X, y = self.data_loader.load_data()
            self.logger.info(f"Data loaded with shapes - X: {X.shape}, y: {y.shape}")

            # 3. Create sequences
            X_seq, y_seq = self.data_loader.create_sequences(X, y)
            self.logger.info(f"Sequences created with shapes - X_seq: {X_seq.shape}, y_seq shape: {y_seq.shape}")

            # 4. Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
                X_seq, y_seq,
                validation_split=self.config['training']['validation_split'],
                test_split=self.config['training']['test_split']
            )
            self.logger.info("Data split completed.")

            # 5. Map labels to binary
            y_train_binary = map_labels(y_train, self.logger)
            y_val_binary = map_labels(y_val, self.logger)
            y_test_binary = map_labels(y_test, self.logger)

            # 6. Preprocess sequences (Fit on Train, Transform Train/Val/Test)
            scaler, X_train_scaled = preprocess_sequences(X_train)
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            self.logger.info("Sequence scaling completed.")

            # Save scaler
            scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            save_scaler(scaler, scaler_path, self.logger)

            # 7. Create model
            model_input_shape = X_train_scaled.shape[1:]
            self.logger.info(f"Creating model with input shape: {model_input_shape}")
            if self.config['model'].get('architecture', 'cnn_lstm') == 'cnn_lstm':
                self.model = create_cnn_lstm_model(
                    input_shape=model_input_shape,
                    learning_rate=self.config['training']['learning_rate']
                )
            else:
                raise ValueError(f"Unsupported architecture: {self.config['model'].get('architecture')}")

            self.logger.info("Model architecture created.")
            self.model.summary(print_fn=self.logger.info)

            # Build model explicitly before training
            try:
                 self.model.build(input_shape=(None,) + model_input_shape)
                 self.model_built = True
                 self.logger.info("Model built successfully.")
            except Exception as build_error:
                 self.logger.error(f"Failed to explicitly build model: {build_error}", exc_info=True)
                 # Decide whether to proceed or raise error if build fails

            # 8. Train model
            history, trained_model = train_model(
                model=self.model,
                X_train=X_train_scaled,
                y_train=y_train_binary,
                X_val=X_val_scaled,
                y_val=y_val_binary,
                config=self.config,
                results_dir=self.config['paths']['results_dir'],
                logger=self.logger
            )
            self.logger.info("Model training completed.")
            self.model = trained_model # Ensure self.model refers to the trained instance
            self.model_built = self.model.built

            # 9. Initialize Symbolic Reasoner AFTER model training
            rules_path = self.config['paths']['reasoning_rules_path'] # Absolute path
            reasoner_input_shape = model_input_shape

            self.reasoner = SymbolicReasoner(
                rules_path=rules_path,
                input_shape=reasoner_input_shape,
                model=self.model, # Pass the trained model
                logger=self.logger
            )
            self.logger.info("Symbolic Reasoner initialized after model training.")

            # 10. Plot training metrics
            figures_dir = os.path.join(self.config['paths']['results_dir'], 'visualization')
            os.makedirs(figures_dir, exist_ok=True)

            plot_config = load_plot_config(self.config['paths']['plot_config_path'])
            plot_metrics(history, figures_dir, plot_config)

            # 11. Evaluate model on Test Set & Extract/Update Rules
            self.logger.info("Predicting on test set...")
            y_test_pred_scores = self.model.predict(X_test_scaled).ravel()
            y_test_pred_classes = (y_test_pred_scores > 0.5).astype(int)

            # Extract rules using the test set predictions
            self.logger.info("Extracting and updating rules based on test set predictions...")
            extraction_threshold = self.config.get('symbolic_reasoning', {}).get('extraction_threshold', 0.8)
            self.extract_and_update_neural_rules(
                model=self.model,
                input_data=X_test_scaled,
                y_pred=y_test_pred_scores,
                threshold=extraction_threshold
            )

            # Prepare sensor data for evaluation context (last timestep, scaled)
            sensor_data_test_last_step = X_test_scaled[:, -1, :]

            # Evaluate performance on the test set
            self.logger.info("Evaluating model performance on test set...")
            evaluate_model(
                y_true=y_test_binary,
                y_pred=y_test_pred_classes,
                y_scores=y_test_pred_scores,
                figures_dir=figures_dir,
                plot_config_path=self.config['paths']['plot_config_path'],
                # --- CHANGE 2: Pass the stored absolute config path ---
                config_path=self.config_path,
                # --- End CHANGE 2 ---
                sensor_data=sensor_data_test_last_step,
                model=self.model
            )

            # 12. Save final trained model
            final_model_path = os.path.join(self.config['paths']['results_dir'], 'final_pipeline_model.keras')
            save_model(self.model, final_model_path, self.logger)

            self.logger.info("ANSR-DT Training Pipeline completed successfully.")

        except Exception as e:
            self.logger.exception(f"ANSR-DT Training Pipeline failed: {e}")
            raise # Re-raise to signal failure


    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the rules managed by the reasoner.
        Delegates to the reasoner instance.
        """
        if self.reasoner and hasattr(self.reasoner, 'get_rule_statistics'):
            try:
                return self.reasoner.get_rule_statistics()
            except Exception as e:
                 self.logger.error(f"Failed to get rule statistics from reasoner: {e}")
                 return {}
        else:
            self.logger.warning("Reasoner not initialized, cannot get rule statistics.")
            return {}