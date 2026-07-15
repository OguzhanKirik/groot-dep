from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adam_u_groot"))

from adapters.adam_u_action_adapter import RealG1ToAdamUAdapter
from adapters.adam_u_action_adapter import AdamUCommand
from adapters.groot_adapter import apply_execution_scope
from configs.adam_u_action_mapping import AdamUActionMappingConfig


def synthetic_action(batch: int = 1, horizon: int = 2) -> dict[str, np.ndarray]:
    def values(width: int, fill: float = 0.0):
        return np.full((batch, horizon, width), fill, dtype=np.float32)

    return {
        "left_arm": values(7, 0.01),
        "right_arm": values(7, -0.01),
        "waist": values(3, 0.02),
        "left_hand": values(7, -0.2),
        "right_hand": values(7, 0.2),
        "left_wrist_eef_9d": values(9, 0.03),
        "right_wrist_eef_9d": values(9, -0.03),
        "base_height_command": values(1, 0.8),
        "navigate_command": values(3, 1.0),
    }


def config(**kwargs) -> AdamUActionMappingConfig:
    return AdamUActionMappingConfig(
        log_commands=False,
        max_body_position_step=None,
        max_hand_position_step=None,
        **kwargs,
    )


class FakeIK:
    def __init__(self):
        self.calls: list[str] = []

    def solve(self, side, eef_command_9d, current_arm, *, command_is_relative):
        self.calls.append(side)
        if command_is_relative:
            raise AssertionError("PolicyClient outputs should be absolute EEF poses")
        result = np.full_like(current_arm, 0.25 if side == "left" else -0.25)
        result[:, 3] = -0.25  # Adam-U elbow range is [-2, 0].
        return result


class TestAdamUActionAdapter(unittest.TestCase):
    def test_g1_waist_order_is_converted_to_adam_order(self):
        action = synthetic_action()
        action["waist"][0, 0] = [0.3, 0.1, -0.2]  # G1: yaw, roll, pitch
        command = RealG1ToAdamUAdapter(config()).adapt(
            action, current_body=np.zeros((1, 19), dtype=np.float32)
        )
        np.testing.assert_allclose(command.body[0, :3], [0.1, -0.2, 0.3], atol=1e-6)

    def test_right_arm_hand_execution_scope_holds_everything_else(self):
        command = AdamUCommand(
            body=np.full((1, 19), 0.7, dtype=np.float32),
            hands=np.full((1, 12), 0.8, dtype=np.float32),
        )
        hold_body = np.full((1, 19), -0.2, dtype=np.float32)
        hold_hands = np.full((1, 12), 0.1, dtype=np.float32)
        masked = apply_execution_scope(
            command, scope="right_arm_hand", hold_body=hold_body, hold_hands=hold_hands
        )
        np.testing.assert_allclose(masked.body[:, :12], -0.2)
        np.testing.assert_allclose(masked.body[:, 12:], 0.7)
        np.testing.assert_allclose(masked.hands[:, :6], 0.1)
        np.testing.assert_allclose(masked.hands[:, 6:], 0.8)
        for key in ("waist", "left_arm", "left_hand"):
            self.assertIn(key, masked.ignored_outputs)

    def test_joint_space_dimensions_neck_and_ignored_outputs(self):
        adapter = RealG1ToAdamUAdapter(config(neck_neutral=(0.15, -0.2)))
        command = adapter.adapt(synthetic_action(), current_body=np.zeros((1, 19)))
        self.assertEqual(command.body.shape, (1, 19))
        self.assertEqual(command.hands.shape, (1, 12))
        np.testing.assert_allclose(command.body[:, 3:5], [[0.15, -0.2]])
        for key in ("navigate_command", "base_height_command", "left_wrist_eef_9d", "right_wrist_eef_9d"):
            self.assertIn(key, command.ignored_outputs)

    def test_relative_joint_commands_become_absolute_targets(self):
        adapter = RealG1ToAdamUAdapter(config(arm_commands_relative=True))
        current = np.zeros((1, 19))
        current[:, 5:12], current[:, 12:19] = -0.4, -0.4
        command = adapter.adapt(synthetic_action(), current_body=current)
        np.testing.assert_allclose(command.body[:, 5:12], -0.39, atol=1e-6)
        np.testing.assert_allclose(command.body[:, 12:19], -0.41, atol=1e-6)

    def test_hand_mapping_is_semantic_and_six_dimensional_per_side(self):
        action = synthetic_action()
        action["left_hand"][0, 0] = [-0.2, -0.3, -0.4, -0.5, 0.1, 0.6, 1.0]
        action["right_hand"][0, 0] = [0.2, 0.3, 0.4, 0.5, 0.1, -0.6, -1.0]
        command = RealG1ToAdamUAdapter(config()).adapt(action, current_body=np.zeros((1, 19)))
        expected = [0.1, 0.8, 0.2, 0.3, 0.4, 0.5]
        np.testing.assert_allclose(command.hands[0, :6], expected)
        np.testing.assert_allclose(command.hands[0, 6:], expected)

    def test_eef_mode_uses_only_eef_for_arms(self):
        action = synthetic_action()
        action["left_arm"][:], action["right_arm"][:] = 999.0, 999.0
        ik = FakeIK()
        command = RealG1ToAdamUAdapter(config(control_mode="eef_space"), ik_solver=ik).adapt(
            action, current_body=np.zeros((1, 19))
        )
        self.assertEqual(ik.calls, ["left", "right"])
        np.testing.assert_allclose(command.body[:, 5:12], [[0.25, 0.25, 0.25, -0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_allclose(command.body[:, 12:19], -0.25)
        self.assertIn("left_arm", command.ignored_outputs)
        self.assertIn("right_arm", command.ignored_outputs)

    def test_eef_mode_never_falls_back_without_ik(self):
        with self.assertRaisesRegex(ValueError, "requires an AdamUIKSolver"):
            RealG1ToAdamUAdapter(config(control_mode="eef_space"))

    def test_invalid_values_are_rejected(self):
        action = synthetic_action()
        action["right_arm"][0, 0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            RealG1ToAdamUAdapter(config()).adapt(action, current_body=np.zeros((1, 19)))

    def test_invalid_values_can_hold_last_safe_command(self):
        adapter = RealG1ToAdamUAdapter(config(invalid_value_policy="hold"))
        safe = adapter.adapt(synthetic_action(), current_body=np.zeros((1, 19)))
        invalid = synthetic_action()
        invalid["left_hand"][0, 0, 0] = np.inf
        held = adapter.adapt(invalid, current_body=safe.body)
        np.testing.assert_array_equal(held.body, safe.body)
        np.testing.assert_array_equal(held.hands, safe.hands)

    def test_wrong_dimensions_are_rejected(self):
        action = synthetic_action()
        action["waist"] = np.zeros((1, 2, 2))
        with self.assertRaisesRegex(ValueError, "waist"):
            RealG1ToAdamUAdapter(config()).adapt(action, current_body=np.zeros((1, 19)))

    def test_maximum_per_step_change_is_enforced(self):
        cfg = AdamUActionMappingConfig(
            log_commands=False,
            max_body_position_step=0.05,
            max_hand_position_step=0.04,
        )
        action = synthetic_action()
        action["left_arm"][:] = -1.0
        action["right_arm"][:] = -1.0
        action["left_hand"][:] = -1.0
        action["right_hand"][:] = 1.0
        command = RealG1ToAdamUAdapter(cfg).adapt(action, current_body=np.zeros((1, 19)))
        self.assertLessEqual(float(np.max(np.abs(command.body))), 0.05 + 1e-7)
        self.assertLessEqual(float(np.max(np.abs(command.hands))), 0.04 + 1e-7)


if __name__ == "__main__":
    unittest.main()
