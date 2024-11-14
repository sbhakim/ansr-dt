import unittest
from src.models.lstm_model import create_lstm_model
import numpy as np

class TestLSTMModel(unittest.TestCase):
    def test_create_lstm_model(self):
        input_shape = (10, 7)
        learning_rate = 0.001
        model = create_lstm_model(input_shape, learning_rate)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 6)  # Corrected layer count

if __name__ == '__main__':
    unittest.main()
