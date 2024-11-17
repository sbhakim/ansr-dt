# tests/test_model.py

import unittest
from src.models.cnn_lstm_model import create_cnn_lstm_model
import numpy as np


class TestCNNLSTMModel(unittest.TestCase):
    def test_create_cnn_lstm_model(self):
        input_shape = (10, 7)
        learning_rate = 0.001
        model = create_cnn_lstm_model(input_shape, learning_rate)
        self.assertIsNotNone(model)
        # Expected number of layers: Conv1D x2, BatchNormalization x2, MaxPooling1D x2,
        # Dropout x4, LSTM x2, Dense x1 => Total: 13 layers
        # This may vary based on implementation; adjust accordingly
        expected_num_layers = 13
        self.assertEqual(len(model.layers), expected_num_layers)


if __name__ == '__main__':
    unittest.main()
