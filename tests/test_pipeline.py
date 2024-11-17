import unittest
import os
from src.pipeline.pipeline import NEXUSDTPipeline
from src.logging.logging_setup import setup_logging
from src.config.config_manager import load_config


class TestNEXUSDTPipeline(unittest.TestCase):
    def setUp(self):
        self.logger = setup_logging('logs/test_pipeline.log', log_level=logging.DEBUG)
        self.config = load_config('configs/config.yaml')
        self.pipeline = NEXUSDTPipeline(self.config, self.logger)

    def test_plot_config_path_exists(self):
        config_dir = os.path.dirname(os.path.abspath('configs/config.yaml'))
        plot_config_path = os.path.join(config_dir, self.config['paths']['plot_config_path'])
        self.assertTrue(os.path.exists(plot_config_path), f"Plot config path does not exist: {plot_config_path}")

    def test_rules_path_exists(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath('configs/config.yaml')))
        rules_path = os.path.join(project_root, self.config['paths']['reasoning_rules_path'])
        self.assertTrue(os.path.exists(rules_path), f"Rules path does not exist: {rules_path}")


if __name__ == '__main__':
    unittest.main()
