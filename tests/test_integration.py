# tests/test_integration.py

import numpy as np
import pandas as pd
from src.nexusdt.explainable import ExplainableNEXUSDT
from src.logging.logging_setup import setup_logging
import logging
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Tuple
import json


class NEXUSDTIntegrationTest:
    def __init__(self, config_path: str):
        """Initialize integration test suite."""
        self.logger = setup_logging(
            'logs/integration_test.log',
            logging.INFO,
            max_bytes=5 * 1024 * 1024,
            backup_count=3
        )
        self.nexusdt = ExplainableNEXUSDT(config_path, self.logger)
        self.results = []

    def load_test_data(self, data_path: str) -> np.ndarray:
        """Load test data."""
        try:
            data = np.load(data_path)
            self.logger.info(f"Test data loaded from {data_path}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise

    def run_tests(self, test_data: np.ndarray, actual_labels: np.ndarray) -> pd.DataFrame:
        """Run integration tests."""
        try:
            self.logger.info("Starting integration tests")

            for i, scenario in enumerate(test_data):
                # Get system response
                decision = self.nexusdt.adapt_and_explain(scenario)

                # Track decision
                self.nexusdt.track_decision(decision)

                # Record results
                self.results.append({
                    'scenario': i,
                    'anomaly_detected': decision['action'] is not None,
                    'predicted_confidence': decision.get('confidence', 0),
                    'explanation': decision.get('explanation', ''),
                    'actual_label': actual_labels[i],
                    'has_insights': len(decision.get('insights', [])) > 0
                })

            # Convert to DataFrame
            results_df = pd.DataFrame(self.results)
            self.logger.info("Integration tests completed")

            return results_df

        except Exception as e:
            self.logger.error(f"Error running integration tests: {e}")
            raise

    def evaluate_performance(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            metrics = {}

            # Calculate detection metrics
            y_true = results_df['actual_label'].values
            y_pred = results_df['anomaly_detected'].values

            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )

            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            # Calculate interpretability metrics
            metrics['explanation_coverage'] = (
                    results_df['explanation'].str.len() > 0
            ).mean()

            metrics['insight_coverage'] = results_df['has_insights'].mean()

            # Calculate confidence metrics
            metrics['average_confidence'] = results_df['predicted_confidence'].mean()
            metrics['confidence_std'] = results_df['predicted_confidence'].std()

            # Log metrics
            self.logger.info("Performance metrics calculated:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise

    def save_results(self, results_df: pd.DataFrame, metrics: Dict[str, float],
                     output_dir: str):
        """Save test results and metrics."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save results DataFrame
            results_path = os.path.join(output_dir, 'test_results.csv')
            results_df.to_csv(results_path, index=False)

            # Save metrics
            metrics_path = os.path.join(output_dir, 'performance_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'timestamp': str(np.datetime64('now'))
                }, f, indent=2)

            # Save decision history
            history_path = os.path.join(output_dir, 'decision_history.json')
            self.nexusdt.save_decision_history(history_path)

            self.logger.info(f"Test results and metrics saved to {output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
            raise

    def generate_report(self, results_df: pd.DataFrame, metrics: Dict[str, float]) -> str:
        """Generate test report in markdown format."""
        try:
            report = ["# NEXUS-DT Integration Test Report\n"]
            report.append(f"Generated at: {str(np.datetime64('now'))}\n")

            # Add performance metrics section
            report.append("## Performance Metrics\n")
            for metric, value in metrics.items():
                report.append(f"- {metric}: {value:.4f}")
            report.append("\n")

            # Add detection analysis
            report.append("## Detection Analysis\n")
            total_anomalies = results_df['actual_label'].sum()
            detected_anomalies = results_df['anomaly_detected'].sum()
            report.append(f"- Total actual anomalies: {total_anomalies}")
            report.append(f"- Total detected anomalies: {detected_anomalies}")
            report.append(f"- False positives: {(results_df['anomaly_detected'] & ~results_df['actual_label']).sum()}")
            report.append(
                f"- False negatives: {(~results_df['anomaly_detected'] & results_df['actual_label']).sum()}\n")

            # Add interpretability analysis
            report.append("## Interpretability Analysis\n")
            report.append(f"- Scenarios with explanations: {(results_df['explanation'].str.len() > 0).sum()}")
            report.append(f"- Scenarios with insights: {results_df['has_insights'].sum()}")
            report.append(f"- Average confidence: {results_df['predicted_confidence'].mean():.4f}\n")

            # Add sample explanations
            report.append("## Sample Explanations\n")
            sample_explanations = results_df[results_df['explanation'].str.len() > 0].sample(
                min(5, len(results_df)), random_state=42
            )
            for _, row in sample_explanations.iterrows():
                report.append(f"Scenario {row['scenario']}:")
                report.append(f"- Actual label: {'Anomaly' if row['actual_label'] else 'Normal'}")
                report.append(f"- Prediction: {'Anomaly' if row['anomaly_detected'] else 'Normal'}")
                report.append(f"- Explanation: {row['explanation']}\n")

            return "\n".join(report)

        except Exception as e:
            self.logger.error(f"Error generating test report: {e}")
            raise


def run_integration_test(config_path: str, test_data_path: str, output_dir: str) -> Tuple[
    pd.DataFrame, Dict[str, float]]:
    """Run complete integration test suite."""
    try:
        # Initialize test suite
        test_suite = NEXUSDTIntegrationTest(config_path)

        # Load test data
        data = test_suite.load_test_data(test_data_path)
        test_scenarios = data['scenarios']
        test_labels = data['labels']

        # Run tests
        results_df = test_suite.run_tests(test_scenarios, test_labels)

        # Evaluate performance
        metrics = test_suite.evaluate_performance(results_df)

        # Generate and save report
        report = test_suite.generate_report(results_df, metrics)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'test_report.md'), 'w') as f:
            f.write(report)

        # Save all results
        test_suite.save_results(results_df, metrics, output_dir)

        return results_df, metrics

    except Exception as e:
        logging.error(f"Error in integration test: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run NEXUS-DT integration tests')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data .npz file')
    parser.add_argument('--output_dir', type=str, default='results/integration_tests',
                        help='Directory to save test results')

    args = parser.parse_args()

    results_df, metrics = run_integration_test(
        args.config,
        args.test_data,
        args.output_dir
    )