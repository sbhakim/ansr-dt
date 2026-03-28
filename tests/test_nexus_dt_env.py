# tests/test_nexus_dt_env.py

import os
import tempfile
import unittest

import numpy as np

from src.ansrdt.ansr_dt_env import ANSRDTEnv


class TestANSRDTEnv(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.temp_dir.name, 'env_data.npz')
        num_samples = 25
        np.savez(
            self.data_path,
            temperature=np.random.rand(num_samples),
            vibration=np.random.rand(num_samples),
            pressure=np.random.rand(num_samples),
            operational_hours=np.arange(num_samples),
            efficiency_index=np.random.rand(num_samples),
            system_state=np.random.randint(0, 3, size=num_samples),
            performance_score=np.random.rand(num_samples),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reset(self):
        env = ANSRDTEnv(data_file=self.data_path, window_size=10)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (10, 7))
        self.assertEqual(env.observation_space.shape, (10, 7))
        self.assertIsInstance(info, dict)

    def test_step(self):
        env = ANSRDTEnv(data_file=self.data_path, window_size=10)
        env.reset()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(next_obs.shape, (10, 7))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main()
