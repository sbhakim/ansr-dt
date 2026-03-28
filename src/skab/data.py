import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    'Accelerometer1RMS',
    'Accelerometer2RMS',
    'Current',
    'Pressure',
    'Temperature',
    'Thermocouple',
    'Voltage',
    'Volume Flow RateRMS',
]


class NativeSKABLoader:
    """Load native SKAB sensor streams without forcing the synthetic ANSR-DT schema."""

    def __init__(
        self,
        data_dir: str,
        feature_names: Optional[Sequence[str]] = None,
        window_size: int = 32,
        categories: Optional[Sequence[str]] = None,
        include_anomaly_free: bool = True,
    ):
        self.data_dir = data_dir
        self.feature_names = list(feature_names) if feature_names else list(DEFAULT_FEATURES)
        self.window_size = int(window_size)
        self.categories = list(categories) if categories else ['valve1', 'valve2', 'other']
        self.include_anomaly_free = bool(include_anomaly_free)
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(self, validation_split: float, test_split: float) -> Dict[str, Any]:
        labeled_series = self._load_labeled_series()
        anomaly_free_series = self._load_anomaly_free_series() if self.include_anomaly_free else []
        if not labeled_series:
            raise ValueError(f'No labeled SKAB series found under {self.data_dir}')

        train_series, val_series, test_series = self._split_series(
            labeled_series,
            validation_split=validation_split,
            test_split=test_split,
        )
        train_series = anomaly_free_series + train_series

        X_train_raw, y_train, train_meta = self._series_to_sequences(train_series)
        X_val_raw, y_val, val_meta = self._series_to_sequences(val_series)
        X_test_raw, y_test, test_meta = self._series_to_sequences(test_series)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape)
        X_val_scaled = scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape)
        X_test_scaled = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape)

        return {
            'feature_names': list(self.feature_names),
            'input_shape': X_train_scaled.shape[1:],
            'X_train_raw': X_train_raw,
            'X_val_raw': X_val_raw,
            'X_test_raw': X_test_raw,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train.astype(np.int32),
            'y_val': y_val.astype(np.int32),
            'y_test': y_test.astype(np.int32),
            'scaler': scaler,
            'metadata': {
                'train_series': train_meta,
                'val_series': val_meta,
                'test_series': test_meta,
            },
        }

    def _load_anomaly_free_series(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.data_dir, 'anomaly-free', 'anomaly-free.csv')
        if not os.path.exists(path):
            self.logger.warning('Anomaly-free SKAB file not found at %s', path)
            return []
        series = self._load_series_file(path, category='anomaly-free', anomaly_free=True)
        return [series] if series else []

    def _load_labeled_series(self) -> List[Dict[str, Any]]:
        series_list: List[Dict[str, Any]] = []
        for category in self.categories:
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_dir):
                self.logger.warning('SKAB category directory missing: %s', category_dir)
                continue
            csv_files = sorted(
                [os.path.join(category_dir, name) for name in os.listdir(category_dir) if name.endswith('.csv')],
                key=self._sort_key,
            )
            for csv_path in csv_files:
                series = self._load_series_file(csv_path, category=category, anomaly_free=False)
                if series is not None:
                    series_list.append(series)
        return series_list

    def _load_series_file(self, csv_path: str, category: str, anomaly_free: bool) -> Optional[Dict[str, Any]]:
        frame = pd.read_csv(csv_path, sep=';')
        if 'datetime' in frame.columns:
            frame['datetime'] = pd.to_datetime(frame['datetime'], errors='coerce')
            frame = frame.sort_values('datetime').reset_index(drop=True)

        missing = [feature for feature in self.feature_names if feature not in frame.columns]
        if missing:
            raise KeyError(f'Missing SKAB columns {missing} in {csv_path}')

        X = frame[self.feature_names].astype(float).to_numpy(dtype=np.float32)
        if len(X) < self.window_size:
            return None

        if anomaly_free:
            y = np.zeros(len(X), dtype=np.int32)
        else:
            if 'anomaly' not in frame.columns:
                raise KeyError(f"Labeled SKAB file missing 'anomaly' column: {csv_path}")
            y = frame['anomaly'].fillna(0).astype(int).to_numpy(dtype=np.int32)

        return {
            'source_file': csv_path,
            'category': category,
            'X': X,
            'y': y,
        }

    def _split_series(
        self,
        labeled_series: List[Dict[str, Any]],
        validation_split: float,
        test_split: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for series in labeled_series:
            grouped.setdefault(series['category'], []).append(series)

        train_series: List[Dict[str, Any]] = []
        val_series: List[Dict[str, Any]] = []
        test_series: List[Dict[str, Any]] = []

        for category in self.categories:
            category_series = grouped.get(category, [])
            if not category_series:
                continue
            total = len(category_series)
            if total < 3:
                if total == 1:
                    train_series.extend(category_series)
                elif total == 2:
                    train_series.append(category_series[0])
                    test_series.append(category_series[1])
                continue

            test_count = max(1, int(round(total * test_split)))
            val_count = max(1, int(round(total * validation_split)))
            if test_count + val_count >= total:
                test_count = 1
                val_count = 1

            test_series.extend(category_series[:test_count])
            val_series.extend(category_series[test_count:test_count + val_count])
            train_series.extend(category_series[test_count + val_count:])

        if not train_series or not val_series or not test_series:
            raise ValueError('Per-category SKAB split produced an empty split.')

        return train_series, val_series, test_series

    def _series_to_sequences(self, series_list: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        X_sequences: List[np.ndarray] = []
        y_sequences: List[int] = []
        metadata: List[Dict[str, Any]] = []

        for series in series_list:
            X = series['X']
            y = series['y']
            seq_count = 0
            for start in range(0, len(X) - self.window_size + 1):
                end = start + self.window_size
                X_sequences.append(X[start:end])
                y_sequences.append(int(y[end - 1]))
                seq_count += 1
            metadata.append({
                'source_file': series['source_file'],
                'category': series['category'],
                'rows': int(len(X)),
                'sequences': int(seq_count),
                'anomalies': int(np.sum(y)),
            })

        return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.int32), metadata

    @staticmethod
    def _sort_key(path: str) -> Tuple[int, str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(stem), stem
        except ValueError:
            return 10**9, stem
