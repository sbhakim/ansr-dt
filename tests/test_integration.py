# tests/test_integration.py

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class NEXUSDTIntegrationTest:
    def __init__(self):
        self.logger = MagicMock()
        self.nexusdt = MagicMock()
        self.results = []

    def evaluate_performance(self, results_df: pd.DataFrame):
        metrics = {}
        y_true = results_df['actual_label'].values
        y_pred = results_df['anomaly_detected'].values
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics.update({'precision': precision, 'recall': recall, 'f1_score': f1})
        metrics['explanation_coverage'] = (results_df['explanation'].str.len() > 0).mean()
        metrics['insight_coverage'] = results_df['has_insights'].mean()
        metrics['average_confidence'] = results_df['predicted_confidence'].mean()
        metrics['confidence_std'] = results_df['predicted_confidence'].std()
        return metrics

    def save_results(self, results_df: pd.DataFrame, metrics, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump({'metrics': metrics, 'timestamp': str(np.datetime64('now'))}, f, indent=2)
        history_path = os.path.join(output_dir, 'decision_history.json')
        self.nexusdt.save_decision_history(history_path)

    def generate_report(self, results_df: pd.DataFrame, metrics) -> str:
        report = ['# NEXUS-DT Integration Test Report\n']
        report.append('## Performance Metrics\n')
        for metric, value in metrics.items():
            report.append(f'- {metric}: {value:.4f}')
        report.append('\n## Sample Explanations\n')
        explained = results_df[results_df['explanation'].str.len() > 0]
        sample_explanations = explained.sample(min(5, len(explained)), random_state=42) if not explained.empty else explained
        for _, row in sample_explanations.iterrows():
            report.append(f"Scenario {row['scenario']}:")
            report.append(f"- Explanation: {row['explanation']}\n")
        return '\n'.join(report)


class TestIntegrationHarness(unittest.TestCase):
    def setUp(self):
        self.harness = NEXUSDTIntegrationTest()

    def test_evaluate_performance(self):
        results_df = pd.DataFrame([
            {'actual_label': 1, 'anomaly_detected': 1, 'explanation': 'a', 'has_insights': True, 'predicted_confidence': 0.9},
            {'actual_label': 0, 'anomaly_detected': 0, 'explanation': '', 'has_insights': False, 'predicted_confidence': 0.1},
        ])
        metrics = self.harness.evaluate_performance(results_df)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertIn('explanation_coverage', metrics)
        self.assertIn('insight_coverage', metrics)

    def test_save_results_and_generate_report(self):
        results_df = pd.DataFrame([
            {'scenario': 0, 'actual_label': 1, 'anomaly_detected': 1, 'explanation': 'rule fired', 'has_insights': True, 'predicted_confidence': 0.8},
            {'scenario': 1, 'actual_label': 0, 'anomaly_detected': 0, 'explanation': '', 'has_insights': False, 'predicted_confidence': 0.2},
        ])
        metrics = {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            self.harness.save_results(results_df, metrics, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'test_results.csv')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'performance_metrics.json')))
            self.harness.nexusdt.save_decision_history.assert_called_once()

            with open(os.path.join(tmpdir, 'performance_metrics.json'), 'r') as f:
                payload = json.load(f)
            self.assertIn('metrics', payload)

            report = self.harness.generate_report(results_df, metrics)
            self.assertIn('Performance Metrics', report)
            self.assertIn('Scenario 0', report)


if __name__ == '__main__':
    unittest.main()
