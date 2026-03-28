# tests/test_skab_data.py

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.skab.data import DEFAULT_FEATURES, NativeSKABLoader


class TestNativeSKABLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = self.temp_dir.name
        for category in ['valve1', 'valve2', 'other', 'anomaly-free']:
            os.makedirs(os.path.join(self.data_root, category), exist_ok=True)

        for category in ['valve1', 'valve2', 'other']:
            for idx in range(3):
                frame = self._make_series(rows=14, anomaly_start=10 + (idx % 2))
                frame.to_csv(os.path.join(self.data_root, category, f'{idx+1}.csv'), sep=';', index=False)

        anomaly_free = self._make_series(rows=14, anomaly_start=None)
        anomaly_free.to_csv(os.path.join(self.data_root, 'anomaly-free', 'anomaly-free.csv'), sep=';', index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_series(self, rows: int, anomaly_start=None):
        t = np.arange(rows, dtype=float)
        data = {
            'datetime': pd.date_range('2025-01-01', periods=rows, freq='min'),
            'Accelerometer1RMS': 0.1 + 0.01 * t,
            'Accelerometer2RMS': 0.2 + 0.01 * t,
            'Current': 1.0 + 0.02 * t,
            'Pressure': 2.0 + 0.03 * t,
            'Temperature': 30.0 + 0.2 * t,
            'Thermocouple': 29.5 + 0.2 * t,
            'Voltage': 220.0 + 0.1 * t,
            'Volume Flow RateRMS': 4.0 + 0.05 * t,
        }
        frame = pd.DataFrame(data)
        anomaly = np.zeros(rows, dtype=int)
        if anomaly_start is not None:
            anomaly[anomaly_start:] = 1
        frame['anomaly'] = anomaly
        return frame

    def test_prepare_dataset_shapes_and_metadata(self):
        loader = NativeSKABLoader(data_dir=self.data_root, window_size=4, include_anomaly_free=True)
        dataset = loader.prepare_dataset(validation_split=0.2, test_split=0.2)

        self.assertEqual(dataset['feature_names'], list(DEFAULT_FEATURES))
        self.assertEqual(dataset['input_shape'], (4, len(DEFAULT_FEATURES)))
        self.assertEqual(dataset['X_train_raw'].shape[1:], (4, len(DEFAULT_FEATURES)))
        self.assertEqual(dataset['X_val_raw'].shape[1:], (4, len(DEFAULT_FEATURES)))
        self.assertEqual(dataset['X_test_raw'].shape[1:], (4, len(DEFAULT_FEATURES)))
        self.assertGreater(len(dataset['metadata']['train_series']), 0)
        self.assertGreater(len(dataset['metadata']['val_series']), 0)
        self.assertGreater(len(dataset['metadata']['test_series']), 0)
        self.assertTrue(any(item['category'] == 'anomaly-free' for item in dataset['metadata']['train_series']))
        self.assertEqual(dataset['X_train_scaled'].shape, dataset['X_train_raw'].shape)
        self.assertEqual(dataset['y_train'].dtype, np.int32)


if __name__ == '__main__':
    unittest.main()
