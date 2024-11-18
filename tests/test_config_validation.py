# tests/test_config_validation.py

import unittest
import logging
from src.pipeline.pipeline import validate_config

class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestLogger')
        self.logger.addHandler(logging.NullHandler())
        self.project_root = '/path/to/project'
        self.config_dir = '/path/to/project/configs'

    def test_validate_config_correct_paths(self):
        config = {
            'model': {},
            'training': {},
            'paths': {}
        }
        validate_config(config, self.logger, self.project_root, self.config_dir)
        self.assertEqual(config['paths']['plot_config_path'], os.path.join(self.config_dir, 'plot_config.yaml'))

    def test_validate_config_missing_key(self):
        config = {
            'model': {},
            'training': {}
            # 'paths' key is missing
        }
        with self.assertRaises(KeyError):
            validate_config(config, self.logger, self.project_root, self.config_dir)

if __name__ == '__main__':
    unittest.main()
