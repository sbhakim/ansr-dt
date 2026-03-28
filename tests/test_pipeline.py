# tests/test_pipeline.py

import logging
import os
import tempfile
import unittest

from src.pipeline.pipeline import validate_config


class TestValidateConfig(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestValidateConfig')
        self.logger.addHandler(logging.NullHandler())

    def test_valid_config_resolves_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            config_dir = os.path.join(project_root, 'configs')
            data_dir = os.path.join(project_root, 'data')
            reasoning_dir = os.path.join(project_root, 'src', 'reasoning')
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(reasoning_dir, exist_ok=True)

            open(os.path.join(data_dir, 'synthetic_sensor_data_with_anomalies.npz'), 'a').close()
            open(os.path.join(reasoning_dir, 'rules.pl'), 'a').close()

            config = {'model': {}, 'training': {}, 'paths': {}}
            validate_config(config, self.logger, project_root, config_dir)

            self.assertTrue(config['paths']['data_file'].endswith('synthetic_sensor_data_with_anomalies.npz'))
            self.assertTrue(config['paths']['reasoning_rules_path'].endswith('src/reasoning/rules.pl'))
            self.assertTrue(os.path.isdir(config['paths']['results_dir']))

    def test_missing_key(self):
        with self.assertRaises(KeyError):
            validate_config({'model': {}, 'training': {}}, self.logger, '/tmp/project', '/tmp/project/configs')

    def test_missing_plot_config_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            config_dir = os.path.join(project_root, 'configs')
            data_dir = os.path.join(project_root, 'data')
            reasoning_dir = os.path.join(project_root, 'src', 'reasoning')
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(reasoning_dir, exist_ok=True)
            open(os.path.join(data_dir, 'synthetic_sensor_data_with_anomalies.npz'), 'a').close()
            open(os.path.join(reasoning_dir, 'rules.pl'), 'a').close()

            config = {'model': {}, 'training': {}, 'paths': {'reasoning_rules_path': 'src/reasoning/rules.pl'}}
            validate_config(config, self.logger, project_root, config_dir)
            self.assertTrue(config['paths']['plot_config_path'].endswith('plot_config.yaml'))

    def test_missing_required_rule_file_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            config_dir = os.path.join(project_root, 'configs')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            open(os.path.join(data_dir, 'synthetic_sensor_data_with_anomalies.npz'), 'a').close()

            config = {'model': {}, 'training': {}, 'paths': {'reasoning_rules_path': 'src/reasoning/rules.pl'}}
            with self.assertRaises(FileNotFoundError):
                validate_config(config, self.logger, project_root, config_dir)


if __name__ == '__main__':
    unittest.main()
