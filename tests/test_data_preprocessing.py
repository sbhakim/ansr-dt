# tests/test_data_processing.py

import unittest
import numpy as np
import logging
from src.data.data_processing import map_labels, split_data


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestDataProcessing')
        self.logger.setLevel(logging.INFO)
        # Add a stream handler if you want to see logs during testing
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            self.logger.addHandler(stream_handler)

    def test_map_labels(self):
        y = np.array([0, 1, 2, 0, 3])
        y_binary = map_labels(y, self.logger)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(y_binary, expected)

    def test_split_data(self):
        X = np.arange(100).reshape(50, 2)  # 50 samples, 2 features
        y = np.arange(50)
        validation_split = 0.2
        X_train, X_val, y_train, y_val = split_data(X, y, validation_split, self.logger)

        self.assertEqual(X_train.shape, (40, 2))
        self.assertEqual(X_val.shape, (10, 2))
        self.assertEqual(y_train.shape, (40,))
        self.assertEqual(y_val.shape, (10,))


if __name__ == '__main__':
    unittest.main()
