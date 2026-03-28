# tests/test_data_preprocessing.py

import logging
import unittest

import numpy as np

from src.data.data_processing import map_labels, split_data


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestDataProcessing')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

    def test_map_labels(self):
        y = np.array([0, 1, 2, 0, 3])
        y_binary = map_labels(y, self.logger)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(y_binary, expected)

    def test_split_data(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        validation_split = 0.2
        test_split = 0.1

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, validation_split, test_split, self.logger
        )

        self.assertEqual(X_test.shape, (5, 2))
        self.assertEqual(X_val.shape, (10, 2))
        self.assertEqual(X_train.shape, (35, 2))
        self.assertEqual(y_test.shape, (5,))
        self.assertEqual(y_val.shape, (10,))
        self.assertEqual(y_train.shape, (35,))
        np.testing.assert_array_equal(y_test, y[:5])
        np.testing.assert_array_equal(y_val, y[5:15])
        np.testing.assert_array_equal(y_train, y[15:])


if __name__ == '__main__':
    unittest.main()
