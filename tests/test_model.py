# tests/test_model.py

import unittest

from tensorflow.keras.layers import Conv1D, Dense, LSTM

from src.models.cnn_lstm_model import create_cnn_lstm_model


class TestCNNLSTMModel(unittest.TestCase):
    def test_create_cnn_lstm_model(self):
        model = create_cnn_lstm_model((10, 7), 0.001)

        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 10, 7))
        self.assertEqual(model.output_shape[-1], 1)
        self.assertEqual(model.loss, 'binary_crossentropy')
        self.assertTrue(any(isinstance(layer, Conv1D) for layer in model.layers))
        self.assertTrue(any(isinstance(layer, LSTM) for layer in model.layers))
        self.assertIsInstance(model.layers[-1], Dense)
        self.assertEqual(model.layers[-1].units, 1)
        self.assertEqual(model.layers[-1].activation.__name__, 'sigmoid')


if __name__ == '__main__':
    unittest.main()
