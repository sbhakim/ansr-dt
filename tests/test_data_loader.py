# tests/test_data_loader.py

import unittest
import numpy as np
from utils.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create synthetic data for testing
        self.X = np.random.rand(100, 7)
        self.y = np.random.randint(0, 4, size=(100,))
        # Save to a temporary .npz file
        np.savez('tests/temp_data.npz', temperature=self.X[:, 0], vibration=self.X[:, 1],
                 pressure=self.X[:, 2], operational_hours=self.X[:, 3],
                 efficiency_index=self.X[:, 4], system_state=self.X[:, 5],
                 performance_score=self.X[:, 6], fused=np.random.rand(100),
                 anomaly=self.y)
        self.data_loader = DataLoader('tests/temp_data.npz', window_size=10)

    def tearDown(self):
        # Remove the temporary file after tests
        import os
        os.remove('tests/temp_data.npz')

    def test_load_data(self):
        X, y = self.data_loader.load_data()
        self.assertEqual(X.shape, (100, 7))
        self.assertEqual(y.shape, (100,))

    def test_create_sequences(self):
        X_seq, y_seq = self.data_loader.create_sequences(self.data_loader.X, self.data_loader.y)
        self.assertEqual(X_seq.shape, (91, 10, 7))
        self.assertEqual(y_seq.shape, (91,))


if __name__ == '__main__':
    unittest.main()
