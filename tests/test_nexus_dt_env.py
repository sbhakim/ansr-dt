# tests/test_nexus_dt_env.py

import unittest
from src.ansrdt.ansr_dt_env import ANSRDTEnv


class TestNexusDTEnv(unittest.TestCase):
    def test_reset(self):
        env = ANSRDTEnv(data_file='data/synthetic_sensor_data_with_anomalies.npz', window_size=10)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (10, 3))
        self.assertIsInstance(info, dict)

    def test_step(self):
        env = ANSRDTEnv(data_file='data/synthetic_sensor_data_with_anomalies.npz', window_size=10)
        observation, info = env.reset()
        action = [0.0, 0.0, 0.0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(next_obs.shape, (10, 3))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main()
