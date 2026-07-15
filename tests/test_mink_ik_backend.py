from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adam_u_groot"))

from adapters.eef_pose import IDENTITY_EEF_9D, MinkAdamUIKSolver


class _SyntheticIsaacProvider:
    def __call__(self, _side, current_arm):
        pose = np.tile(IDENTITY_EEF_9D, (current_arm.shape[0], 1)).astype(np.float64)
        return pose, np.zeros((current_arm.shape[0], 6, 7), dtype=np.float64)


class MinkAdamUIKSolverTest(unittest.TestCase):
    def test_pnd_tuning_constants_are_preserved(self):
        self.assertEqual(MinkAdamUIKSolver.PND_SOLVER, "daqp")
        self.assertEqual(MinkAdamUIKSolver.PND_DAMPING, 0.3)
        self.assertEqual(MinkAdamUIKSolver.PND_ITERATIONS, 3)
        np.testing.assert_allclose(
            MinkAdamUIKSolver.PND_WRIST_OFFSET_WXYZ,
            [0.866, 0.0, -0.5, 0.0],
        )

    def test_missing_official_model_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            MinkAdamUIKSolver(_SyntheticIsaacProvider(), ROOT / "missing_adam_u.xml")

    def test_official_model_hold_and_safe_step(self):
        model = ROOT / "third_party" / "pnd_models" / "adam_u" / "adam_u.xml"
        if not model.is_file():
            self.skipTest("optional pnd_models checkout is not installed")
        solver = MinkAdamUIKSolver(_SyntheticIsaacProvider(), model)
        current = np.zeros((1, 7), dtype=np.float32)
        command = np.tile(IDENTITY_EEF_9D, (1, 1))
        hold = solver.solve("right", command, current, command_is_relative=False)
        self.assertEqual(hold.shape, (1, 7))
        np.testing.assert_allclose(hold, current, atol=1e-6)

        command[0, 2] += 0.08
        moved = solver.solve("right", command, current, command_is_relative=False)
        self.assertTrue(np.all(np.isfinite(moved)))
        self.assertLessEqual(float(np.max(np.abs(moved - current))), 0.005001)
        self.assertGreater(float(np.linalg.norm(moved - current)), 1e-5)

        # A persistent target must accumulate commanded position rather than
        # being permanently rebased to q_measured at one max_joint_delta.
        solver.sync_commanded_joint_pos("right", moved)
        moved_again = solver.solve("right", command, current, command_is_relative=False)
        self.assertGreater(
            float(np.linalg.norm(moved_again - current)),
            float(np.linalg.norm(moved - current)),
        )

    def test_relative_commands_are_rejected(self):
        model = ROOT / "third_party" / "pnd_models" / "adam_u" / "adam_u.xml"
        if not model.is_file():
            self.skipTest("optional pnd_models checkout is not installed")
        solver = MinkAdamUIKSolver(_SyntheticIsaacProvider(), model)
        with self.assertRaises(ValueError):
            solver.solve(
                "right",
                np.tile(IDENTITY_EEF_9D, (1, 1)),
                np.zeros((1, 7)),
                command_is_relative=True,
            )

    def test_translation_priority_softens_and_restores_orientation(self):
        model = ROOT / "third_party" / "pnd_models" / "adam_u" / "adam_u.xml"
        if not model.is_file():
            self.skipTest("optional pnd_models checkout is not installed")
        solver = MinkAdamUIKSolver(_SyntheticIsaacProvider(), model)
        current = np.zeros((1, 7))
        command = np.tile(IDENTITY_EEF_9D, (1, 1))
        solver.set_translation_priority(True, orientation_cost=2.0)
        solver.solve("right", command, current, command_is_relative=False)
        task = solver._states[("right", 0)]["task"]
        np.testing.assert_allclose(task.cost[3:], 2.0)
        solver.set_translation_priority(False)
        solver.solve("right", command, current, command_is_relative=False)
        np.testing.assert_allclose(task.cost[3:], 18.0)

    def test_gravity_compensation_is_finite_and_bounded(self):
        model = ROOT / "third_party" / "pnd_models" / "adam_u" / "adam_u.xml"
        if not model.is_file():
            self.skipTest("optional pnd_models checkout is not installed")
        solver = MinkAdamUIKSolver(_SyntheticIsaacProvider(), model)
        effort = solver.gravity_compensation(
            "right", np.asarray([[-0.35, 0.2, 0.0, -1.1, 0.0, -0.3, 0.0]])
        )
        self.assertEqual(effort.shape, (1, 7))
        self.assertTrue(np.all(np.isfinite(effort)))
        self.assertGreater(float(np.linalg.norm(effort)), 0.1)
        self.assertLessEqual(float(np.max(np.abs(effort))), 40.0)


if __name__ == "__main__":
    unittest.main()
