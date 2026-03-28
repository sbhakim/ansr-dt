# tests/test_data_loader.py

import os
import tempfile
import unittest

import numpy as np

from src.data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.temp_dir.name, 'temp_data.npz')
        X = np.random.rand(100, 7)
        y = np.random.randint(0, 4, size=(100,))
        np.savez(
            self.data_path,
            temperature=X[:, 0],
            vibration=X[:, 1],
            pressure=X[:, 2],
            operational_hours=X[:, 3],
            efficiency_index=X[:, 4],
            system_state=X[:, 5],
            performance_score=X[:, 6],
            fused=np.random.rand(100),
            anomaly=y,
        )
        self.data_loader = DataLoader(self.data_path, window_size=10)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_data(self):
        X, y = self.data_loader.load_data()
        self.assertEqual(X.shape, (100, 7))
        self.assertEqual(y.shape, (100,))

    def test_create_sequences(self):
        X, y = self.data_loader.load_data()
        X_seq, y_seq = self.data_loader.create_sequences(X, y)
        self.assertEqual(X_seq.shape, (91, 10, 7))
        self.assertEqual(y_seq.shape, (91,))
        np.testing.assert_array_equal(y_seq, y[9:])


if __name__ == '__main__':
    unittest.main()
