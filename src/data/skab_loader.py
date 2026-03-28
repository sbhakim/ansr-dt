import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_MAP = {
    'temperature': 'Temperature',
    'vibration': '__accel_max__',
    'pressure': 'Volume Flow RateRMS',
    'operational_hours': '__constant__',
    'efficiency_index': 'Current',
    'system_state': '__constant__',
    'performance_score': 'Pressure',
}


class SKABLoader:
    """Load SKAB CSV files while preserving the existing ANSR-DT 7-feature contract.

    The synthetic pipeline expects seven semantic features. To keep that path untouched,
    this loader maps SKAB sensor columns into the same interface and generates sequences
    per file so windows never cross file boundaries.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 10,
        categories: Optional[Sequence[str]] = None,
        include_anomaly_free: bool = True,
        feature_map: Optional[Dict[str, str]] = None,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.categories = list(categories) if categories else ['valve1', 'valve2', 'other']
        self.include_anomaly_free = include_anomaly_free
        self.feature_map = dict(DEFAULT_FEATURE_MAP)
        if feature_map:
            self.feature_map.update(feature_map)
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(
        self,
        validation_split: float,
        test_split: float,
    ) -> Dict[str, Any]:
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

        X_train_raw, y_train_raw, train_meta = self._series_to_sequences(train_series)
        X_val_raw, y_val_raw, val_meta = self._series_to_sequences(val_series)
        X_test_raw, y_test_raw, test_meta = self._series_to_sequences(test_series)

        if X_train_raw.size == 0 or X_val_raw.size == 0 or X_test_raw.size == 0:
            raise ValueError(
                'SKAB split produced an empty split. '
                'Adjust categories, anomaly-free usage, or split fractions.'
            )

        return {
            'X_train_raw': X_train_raw,
            'X_val_raw': X_val_raw,
            'X_test_raw': X_test_raw,
            'y_train_raw': y_train_raw,
            'y_val_raw': y_val_raw,
            'y_test_raw': y_test_raw,
            'metadata': {
                'train_series': train_meta,
                'val_series': val_meta,
                'test_series': test_meta,
                'feature_map': dict(self.feature_map),
            },
        }

    def _load_anomaly_free_series(self) -> List[Dict[str, Any]]:
        anomaly_free_path = os.path.join(self.data_dir, 'anomaly-free', 'anomaly-free.csv')
        if not os.path.exists(anomaly_free_path):
            self.logger.warning('SKAB anomaly-free file not found at %s', anomaly_free_path)
            return []

        series = self._load_series_file(anomaly_free_path, category='anomaly-free', anomaly_free=True)
        return [series] if series else []

    def _load_labeled_series(self) -> List[Dict[str, Any]]:
        series_list: List[Dict[str, Any]] = []
        for category in self.categories:
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_dir):
                self.logger.warning('SKAB category directory missing: %s', category_dir)
                continue

            csv_files = sorted(
                [
                    os.path.join(category_dir, name)
                    for name in os.listdir(category_dir)
                    if name.endswith('.csv')
                ],
                key=self._sort_key,
            )

            for csv_path in csv_files:
                series = self._load_series_file(csv_path, category=category, anomaly_free=False)
                if series is not None:
                    series_list.append(series)

        return series_list

    def _load_series_file(
        self,
        csv_path: str,
        category: str,
        anomaly_free: bool,
    ) -> Optional[Dict[str, Any]]:
        try:
            frame = pd.read_csv(csv_path, sep=';')
        except Exception as exc:
            self.logger.error('Failed reading SKAB file %s: %s', csv_path, exc)
            return None

        if 'datetime' in frame.columns:
            frame['datetime'] = pd.to_datetime(frame['datetime'], errors='coerce')
            frame = frame.sort_values('datetime').reset_index(drop=True)

        feature_frame = self._build_feature_frame(frame)
        if feature_frame.empty or len(feature_frame) < self.window_size:
            self.logger.warning(
                'Skipping SKAB file %s because it has fewer than %d rows after preprocessing.',
                csv_path,
                self.window_size,
            )
            return None

        if anomaly_free:
            labels = np.zeros(len(feature_frame), dtype=np.int32)
        else:
            if 'anomaly' not in frame.columns:
                self.logger.warning('Skipping labeled SKAB file without anomaly column: %s', csv_path)
                return None
            labels = frame['anomaly'].fillna(0).astype(int).to_numpy()

        return {
            'source_file': csv_path,
            'category': category,
            'X': feature_frame.to_numpy(dtype=np.float32),
            'y': labels.astype(np.int32),
        }

    def _build_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        output: Dict[str, np.ndarray] = {}
        row_count = len(frame)

        for target_name, source_name in self.feature_map.items():
            if source_name == '__constant__':
                if target_name == 'operational_hours':
                    output[target_name] = np.ones(row_count, dtype=np.float32)
                elif target_name == 'system_state':
                    output[target_name] = np.zeros(row_count, dtype=np.float32)
                else:
                    output[target_name] = np.zeros(row_count, dtype=np.float32)
                continue

            if source_name == '__accel_max__':
                required = ['Accelerometer1RMS', 'Accelerometer2RMS']
                for column in required:
                    if column not in frame.columns:
                        raise KeyError(f"Required SKAB column '{column}' missing from CSV.")
                output[target_name] = frame[required].max(axis=1).to_numpy(dtype=np.float32)
                continue

            if source_name not in frame.columns:
                raise KeyError(f"Required SKAB column '{source_name}' missing from CSV.")

            output[target_name] = frame[source_name].astype(float).to_numpy(dtype=np.float32)

        return pd.DataFrame(output)

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

    def _series_to_sequences(
        self,
        series_list: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        X_sequences: List[np.ndarray] = []
        y_sequences: List[int] = []
        metadata: List[Dict[str, Any]] = []

        for series in series_list:
            X = series['X']
            y = series['y']

            sequence_count = 0
            for start in range(0, len(X) - self.window_size + 1):
                end = start + self.window_size
                X_sequences.append(X[start:end])
                y_sequences.append(int(y[end - 1]))
                sequence_count += 1

            metadata.append(
                {
                    'source_file': series['source_file'],
                    'category': series['category'],
                    'rows': int(len(X)),
                    'sequences': int(sequence_count),
                    'anomalies': int(np.sum(y)),
                }
            )

        if not X_sequences:
            feature_count = len(self.feature_map)
            return (
                np.empty((0, self.window_size, feature_count), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                metadata,
            )

        return (
            np.asarray(X_sequences, dtype=np.float32),
            np.asarray(y_sequences, dtype=np.int32),
            metadata,
        )

    @staticmethod
    def _sort_key(path: str) -> Tuple[int, str]:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            return int(stem), name
        return 10**9, name
