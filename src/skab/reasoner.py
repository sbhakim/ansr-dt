from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import compute_metrics, select_threshold


@dataclass
class SymbolicRule:
    name: str
    feature: str
    direction: str
    threshold: float
    weight: float
    precision: float
    recall: float
    f1: float
    support: int

    def describe(self) -> str:
        operator = '>=' if self.direction == 'gte' else '<='
        return f"{self.name}: {self.feature} {operator} {self.threshold:.4f}"


class SKABRuleReasoner:
    """Interpretable SKAB-native rule scorer learned from labeled windows."""

    def __init__(
        self,
        max_rules: int = 8,
        min_rule_precision: float = 0.55,
        min_rule_recall: float = 0.02,
        selection_metric: str = 'f1',
    ):
        self.max_rules = int(max_rules)
        self.min_rule_precision = float(min_rule_precision)
        self.min_rule_recall = float(min_rule_recall)
        self.selection_metric = selection_metric
        self.rules: List[SymbolicRule] = []
        self.score_threshold = 0.5
        self.feature_columns: List[str] = []
        self.total_weight = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
        train_frame = self._window_to_feature_frame(X_train, feature_names)
        val_frame = self._window_to_feature_frame(X_val, feature_names)
        candidates = self._generate_candidates(train_frame, y_train, val_frame, y_val)
        self.rules = candidates[: self.max_rules]
        self.total_weight = float(sum(rule.weight for rule in self.rules)) or 1.0

        val_scores = self.predict_scores(X_val, feature_names)
        self.score_threshold, val_threshold_metrics = select_threshold(y_val, val_scores, metric_name=self.selection_metric)
        return {
            'selected_rule_count': len(self.rules),
            'selected_rules': [asdict(rule) for rule in self.rules],
            'validation_threshold': self.score_threshold,
            'validation_threshold_metrics': val_threshold_metrics,
        }

    def predict_scores(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        frame = self._window_to_feature_frame(X, feature_names)
        if not self.rules:
            return np.zeros(len(frame), dtype=float)

        scores = np.zeros(len(frame), dtype=float)
        for rule in self.rules:
            if rule.direction == 'gte':
                triggered = frame[rule.feature].to_numpy() >= rule.threshold
            else:
                triggered = frame[rule.feature].to_numpy() <= rule.threshold
            scores += triggered.astype(float) * rule.weight
        return scores / self.total_weight

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
        scores = self.predict_scores(X, feature_names)
        metrics = compute_metrics(y, scores, threshold=self.score_threshold)
        return {
            'scores': scores,
            'metrics': metrics,
            'rules': [asdict(rule) for rule in self.rules],
            'score_threshold': self.score_threshold,
        }

    def explain_samples(self, X: np.ndarray, feature_names: Sequence[str], limit: int = 25) -> List[Dict[str, Any]]:
        frame = self._window_to_feature_frame(X, feature_names)
        explanations: List[Dict[str, Any]] = []
        for index in range(min(limit, len(frame))):
            row = frame.iloc[index]
            triggered_rules: List[str] = []
            for rule in self.rules:
                value = float(row[rule.feature])
                if rule.direction == 'gte' and value >= rule.threshold:
                    triggered_rules.append(rule.describe())
                elif rule.direction == 'lte' and value <= rule.threshold:
                    triggered_rules.append(rule.describe())
            explanations.append({
                'index': index,
                'triggered_rules': triggered_rules,
                'score': float(self.predict_scores(X[index:index + 1], feature_names)[0]) if self.rules else 0.0,
            })
        return explanations

    def _window_to_feature_frame(self, X: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
        # Convert each temporal window into interpretable summary descriptors so
        # rule learning operates over transparent statistics rather than raw traces.
        feature_rows: List[Dict[str, float]] = []
        for window in X:
            row: Dict[str, float] = {}
            last_values = window[-1]
            prev_values = window[-2] if len(window) > 1 else window[-1]
            mean_values = np.mean(window, axis=0)
            std_values = np.std(window, axis=0)
            for idx, name in enumerate(feature_names):
                safe_name = self._safe_name(name)
                row[f'last_{safe_name}'] = float(last_values[idx])
                row[f'mean_{safe_name}'] = float(mean_values[idx])
                row[f'std_{safe_name}'] = float(std_values[idx])
                row[f'delta_{safe_name}'] = float(last_values[idx] - prev_values[idx])
                row[f'abs_delta_{safe_name}'] = float(abs(last_values[idx] - prev_values[idx]))

            acc1 = float(last_values[feature_names.index('Accelerometer1RMS')])
            acc2 = float(last_values[feature_names.index('Accelerometer2RMS')])
            temp = float(last_values[feature_names.index('Temperature')])
            thermo = float(last_values[feature_names.index('Thermocouple')])
            flow = float(last_values[feature_names.index('Volume Flow RateRMS')])
            current = float(last_values[feature_names.index('Current')])
            pressure = float(last_values[feature_names.index('Pressure')])
            voltage = float(last_values[feature_names.index('Voltage')])
            row['last_accel_max'] = max(acc1, acc2)
            row['last_accel_gap'] = abs(acc1 - acc2)
            row['last_temp_gap'] = abs(temp - thermo)
            row['flow_current_ratio'] = flow / (abs(current) + 1e-6)
            row['pressure_flow_ratio'] = pressure / (abs(flow) + 1e-6)
            row['voltage_current_ratio'] = voltage / (abs(current) + 1e-6)
            feature_rows.append(row)

        frame = pd.DataFrame(feature_rows)
        self.feature_columns = list(frame.columns)
        return frame

    def _generate_candidates(self, train_frame: pd.DataFrame, y_train: np.ndarray, val_frame: pd.DataFrame, y_val: np.ndarray) -> List[SymbolicRule]:
        normal_frame = train_frame[np.asarray(y_train) == 0]
        anomaly_frame = train_frame[np.asarray(y_train) == 1]
        candidates: List[SymbolicRule] = []

        if anomaly_frame.empty:
            return candidates

        for column in train_frame.columns:
            normal_values = normal_frame[column].to_numpy(dtype=float)
            anomaly_values = anomaly_frame[column].to_numpy(dtype=float)
            if len(np.unique(normal_values)) < 2 and len(np.unique(anomaly_values)) < 2:
                continue

            # Probe quantiles from both normal and anomalous windows to obtain a
            # compact threshold set that still covers the main separation regimes.
            candidate_thresholds = sorted(set(np.round(np.concatenate([
                np.quantile(normal_values, [0.90, 0.95, 0.975, 0.99]),
                np.quantile(normal_values, [0.01, 0.025, 0.05, 0.10]),
                np.quantile(anomaly_values, [0.10, 0.25, 0.50, 0.75, 0.90]),
            ]), 6).tolist()))

            directions = ['gte', 'lte']
            for direction in directions:
                best_rule = None
                best_score = -1.0
                for threshold in candidate_thresholds:
                    if direction == 'gte':
                        pred = (val_frame[column].to_numpy(dtype=float) >= threshold).astype(int)
                    else:
                        pred = (val_frame[column].to_numpy(dtype=float) <= threshold).astype(int)
                    support = int(np.sum(pred))
                    if support == 0:
                        continue
                    metrics = compute_metrics(y_val, pred.astype(float), threshold=0.5)
                    if metrics['precision'] < self.min_rule_precision or metrics['recall'] < self.min_rule_recall:
                        continue
                    score = float(metrics[self.selection_metric])
                    if score > best_score:
                        best_score = score
                        best_rule = SymbolicRule(
                            name=f'rule_{len(candidates) + 1}_{column}',
                            feature=column,
                            direction=direction,
                            threshold=float(threshold),
                            weight=max(float(metrics['precision']), 0.5),
                            precision=float(metrics['precision']),
                            recall=float(metrics['recall']),
                            f1=float(metrics['f1']),
                            support=support,
                        )
                if best_rule is not None:
                    candidates.append(best_rule)

        candidates.sort(key=lambda rule: (rule.f1, rule.precision, rule.recall), reverse=True)
        # Keep at most one rule per feature/direction pair so the final rule set
        # stays compact enough to inspect and report directly.
        deduped: List[SymbolicRule] = []
        seen = set()
        for rule in candidates:
            key = (rule.feature, rule.direction)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rule)
        return deduped

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.lower().replace(' ', '_').replace('-', '_')
