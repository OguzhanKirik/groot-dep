from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adam_u_groot"))

from adapters.eef_pose import G1AdamWorkspaceTransform
from adapters.virtual_g1_state import VirtualG1StateAdapter


class TestG1AdamWorkspaceTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = G1AdamWorkspaceTransform()

    def test_adam_initial_right_wrist_maps_into_g1_workspace(self):
        adam_pose = np.array(
            [-0.3016535, 0.13447498, 1.12449729, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=np.float32,
        )
        g1_pose = self.transform.world_to_g1_pose(adam_pose)
        np.testing.assert_allclose(g1_pose[:3], [0.3016535, -0.13447498, 0.12449729], atol=1e-6)

    def test_pose_round_trip_is_invertible(self):
        poses = np.array(
            [
                [-0.30, 0.13, 1.12, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [-0.25, -0.18, 1.05, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        reconstructed = self.transform.g1_to_world_pose(
            self.transform.world_to_g1_pose(poses)
        )
        np.testing.assert_allclose(reconstructed, poses, atol=1e-6)

    def test_invalid_rotation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            G1AdamWorkspaceTransform(
                world_to_g1_rotation=((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
            )


class TestVirtualG1StateAdapter(unittest.TestCase):
    def test_virtual_joints_match_transformed_wrist_positions(self):
        transform = G1AdamWorkspaceTransform()
        virtual = VirtualG1StateAdapter()
        right = transform.world_to_g1_pose(
            np.array([[-0.30165, 0.13447, 1.1245, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
        )
        left = transform.world_to_g1_pose(
            np.array([[-0.30165, -0.13447, 1.1245, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
        )
        left_joints, right_joints = virtual.update(left, right)
        self.assertEqual(left_joints.shape, (1, 7))
        self.assertEqual(right_joints.shape, (1, 7))
        self.assertLess(virtual.last_errors["left"], 0.01)
        self.assertLess(virtual.last_errors["right"], 0.01)
        self.assertTrue(np.all(np.isfinite(left_joints)))
        self.assertTrue(np.all(np.isfinite(right_joints)))
        np.testing.assert_allclose(virtual.last_eef_poses["left"][0, :3], left[0, :3], atol=0.01)
        np.testing.assert_allclose(virtual.last_eef_poses["right"][0, :3], right[0, :3], atol=0.01)
        # Null-space posture bias keeps shoulder roll near the REAL_G1 dataset.
        self.assertGreater(left_joints[0, 1], 0.16)
        self.assertLess(right_joints[0, 1], -0.15)


if __name__ == "__main__":
    unittest.main()
