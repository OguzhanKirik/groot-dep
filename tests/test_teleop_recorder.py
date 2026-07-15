import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from adam_u_groot.adapters.teleop_recorder import AdamUTeleopRecorder


class TestAdamUTeleopRecorder(unittest.TestCase):
    def sample(self):
        return {
            "observation.images.front": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation.state.body": np.zeros(19, dtype=np.float32),
            "observation.state.hands": np.zeros(12, dtype=np.float32),
            "action.body": np.zeros(19, dtype=np.float32),
            "action.hands": np.zeros(12, dtype=np.float32),
            "observation.right_wrist_pose": np.zeros(9, dtype=np.float32),
            "observation.object_pose": np.zeros(7, dtype=np.float32),
            "timestamp": np.asarray(0.0, dtype=np.float64),
        }

    def test_writes_native_adam_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demos.hdf5"
            recorder = AdamUTeleopRecorder(output, "pick cube", 1 / 30)
            recorder.append(self.sample())
            name = recorder.save_success()
            self.assertEqual(name, "episode_000000")
            with h5py.File(output, "r") as handle:
                episode = handle["episodes/episode_000000"]
                self.assertEqual(episode["action.body"].shape, (1, 19))
                self.assertEqual(episode["action.hands"].shape, (1, 12))
                self.assertEqual(episode["observation.images.front"].shape, (1, 8, 8, 3))

    def test_rejects_invalid_native_width(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = AdamUTeleopRecorder(Path(directory) / "demos.hdf5", "pick cube", 1 / 30)
            sample = self.sample()
            sample["action.body"] = np.zeros(18)
            with self.assertRaises(ValueError):
                recorder.append(sample)


if __name__ == "__main__":
    unittest.main()
