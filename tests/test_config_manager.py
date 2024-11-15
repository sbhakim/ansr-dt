# tests/test_config_manager.py

import unittest
from src.config.config_manager import load_config
import os

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        # Path to the config file
        self.config_path = 'configs/config.yaml'
        self.assertTrue(os.path.exists(self.config_path), f"{self.config_path} does not exist.")

    def test_load_config(self):
        config = load_config(self.config_path)
        self.assertIsInstance(config, dict)
        # Check for essential keys
        self.assertIn('model', config)
        self.assertIn('training', config)
        self.assertIn('paths', config)
        # Further nested checks can be added as needed

if __name__ == '__main__':
    unittest.main()
