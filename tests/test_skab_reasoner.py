# tests/test_skab_reasoner.py

import unittest

import numpy as np

from src.skab.data import DEFAULT_FEATURES
from src.skab.reasoner import SKABRuleReasoner


class TestSKABRuleReasoner(unittest.TestCase):
    def _make_windows(self, count: int, anomaly: bool) -> np.ndarray:
        windows = []
        for i in range(count):
            base = np.zeros((6, len(DEFAULT_FEATURES)), dtype=np.float32)
            trend = np.linspace(0.0, 1.0, 6, dtype=np.float32)
            for col in range(len(DEFAULT_FEATURES)):
                base[:, col] = 0.05 * col + 0.01 * i + 0.02 * trend
            if anomaly:
                base[:, DEFAULT_FEATURES.index('Pressure')] += 2.0
                base[:, DEFAULT_FEATURES.index('Volume Flow RateRMS')] -= 0.8
                base[:, DEFAULT_FEATURES.index('Accelerometer1RMS')] += 0.6
            windows.append(base)
        return np.asarray(windows, dtype=np.float32)

    def test_fit_predict_and_explain(self):
        X_train = np.concatenate([self._make_windows(8, False), self._make_windows(8, True)], axis=0)
        y_train = np.array([0] * 8 + [1] * 8, dtype=np.int32)
        X_val = np.concatenate([self._make_windows(4, False), self._make_windows(4, True)], axis=0)
        y_val = np.array([0] * 4 + [1] * 4, dtype=np.int32)

        reasoner = SKABRuleReasoner(max_rules=4, min_rule_precision=0.5, min_rule_recall=0.1)
        summary = reasoner.fit(X_train, y_train, X_val, y_val, DEFAULT_FEATURES)

        self.assertGreater(summary['selected_rule_count'], 0)
        self.assertLessEqual(summary['selected_rule_count'], 4)
        self.assertGreaterEqual(summary['validation_threshold'], 0.0)

        scores = reasoner.predict_scores(X_val, DEFAULT_FEATURES)
        self.assertEqual(scores.shape, (8,))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))

        evaluation = reasoner.evaluate(X_val, y_val, DEFAULT_FEATURES)
        self.assertIn('metrics', evaluation)
        self.assertIn('rules', evaluation)
        self.assertGreater(len(evaluation['rules']), 0)

        explanations = reasoner.explain_samples(X_val, DEFAULT_FEATURES, limit=3)
        self.assertEqual(len(explanations), 3)
        self.assertIn('triggered_rules', explanations[0])
        self.assertIn('score', explanations[0])


if __name__ == '__main__':
    unittest.main()
