# tests/test_pipeline.py

import unittest
from unittest.mock import MagicMock
from src.pipeline.pipeline import validate_config

class TestValidateConfig(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.project_root = '/path/to/project'
        self.config_dir = '/path/to/project/configs'

    def test_valid_config(self):
        config = {
            'model': {},
            'training': {},
            'paths': {
                'plot_config_path': 'plot_config.yaml',
                'reasoning_rules_path': 'src/reasoning/rules.pl'
            }
        }
        # Mock os.path.exists to return True
        with unittest.mock.patch('os.path.exists', return_value=True):
            validate_config(config, self.logger, self.project_root, self.config_dir)
            self.logger.info.assert_called_with("Configuration validation passed.")

    def test_missing_key(self):
        config = {
            'model': {},
            'training': {}
            # 'paths' key missing
        }
        with self.assertRaises(KeyError):
            validate_config(config, self.logger, self.project_root, self.config_dir)

    def test_missing_plot_config(self):
        config = {
            'model': {},
            'training': {},
            'paths': {
                # 'plot_config_path' missing
                'reasoning_rules_path': 'src/reasoning/rules.pl'
            }
        }
        with self.assertRaises(KeyError):
            validate_config(config, self.logger, self.project_root, self.config_dir)

    def test_plot_config_not_found(self):
        config = {
            'model': {},
            'training': {},
            'paths': {
                'plot_config_path': 'plot_config.yaml',
                'reasoning_rules_path': 'src/reasoning/rules.pl'
            }
        }
        with unittest.mock.patch('os.path.exists', side_effect=lambda x: False if 'plot_config.yaml' in x else True):
            with self.assertRaises(FileNotFoundError):
                validate_config(config, self.logger, self.project_root, self.config_dir)

if __name__ == '__main__':
    unittest.main()
