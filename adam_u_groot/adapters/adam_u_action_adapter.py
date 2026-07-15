"""Safety adapter from GR00T REAL_G1 actions to Adam-U low-level commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from configs.adam_u_action_mapping import AdamUActionMappingConfig, G1_HAND_CHANNEL_NAMES
from configs.joint_state import BODY_JOINT_NAMES, HAND_COMMAND_NAMES

LOGGER = logging.getLogger(__name__)


class AdamUIKSolver(Protocol):
    """IK boundary used only in eef_space mode."""

    def solve(
        self,
        side: str,
        eef_command_9d: np.ndarray,
        current_arm: np.ndarray,
        *,
        command_is_relative: bool,
    ) -> np.ndarray:
        """Return absolute Adam-U arm targets with shape (batch, 7)."""


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    first = rot6d[..., :3]
    second = rot6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


class DampedLeastSquaresIKSolver:
    """Generic Adam-U 7-DOF IK using a named FK/Jacobian provider.

    The provider must return ``(eef_pose_9d, geometric_jacobian_6x7)`` for the
    requested side and current Adam-U joint vector. This keeps simulator and
    real-robot kinematics backends separate while sharing the safety adapter.
    """

    def __init__(
        self,
        kinematics_provider: Callable[[str, np.ndarray], tuple[np.ndarray, np.ndarray]],
        *,
        damping: float = 0.05,
        max_joint_delta: float = 0.10,
    ) -> None:
        self.kinematics_provider = kinematics_provider
        self.damping = float(damping)
        self.max_joint_delta = float(max_joint_delta)

    def solve(self, side, eef_command_9d, current_arm, *, command_is_relative):
        current_pose, jacobian = self.kinematics_provider(side, current_arm)
        current_pose = np.asarray(current_pose, dtype=np.float64)
        jacobian = np.asarray(jacobian, dtype=np.float64)
        command = np.asarray(eef_command_9d, dtype=np.float64)
        if current_pose.shape != command.shape or current_pose.shape[1] != 9:
            raise ValueError("IK FK provider and EEF command must both have shape (batch, 9)")
        if jacobian.shape != (command.shape[0], 6, 7):
            raise ValueError(f"IK Jacobian must have shape (batch, 6, 7), got {jacobian.shape}")

        current_rot = _rot6d_to_matrix(current_pose[:, 3:])
        command_rot = _rot6d_to_matrix(command[:, 3:])
        target_pos = current_pose[:, :3] + command[:, :3] if command_is_relative else command[:, :3]
        target_rot = command_rot @ current_rot if command_is_relative else command_rot
        pos_error = target_pos - current_pose[:, :3]
        relative_rot = target_rot @ np.swapaxes(current_rot, -1, -2)
        rot_error = 0.5 * np.stack(
            (
                relative_rot[:, 2, 1] - relative_rot[:, 1, 2],
                relative_rot[:, 0, 2] - relative_rot[:, 2, 0],
                relative_rot[:, 1, 0] - relative_rot[:, 0, 1],
            ), axis=1,
        )
        error = np.concatenate((pos_error, rot_error), axis=1)

        targets = np.asarray(current_arm, dtype=np.float64).copy()
        identity = np.eye(6)
        for index in range(targets.shape[0]):
            j = jacobian[index]
            delta = j.T @ np.linalg.solve(j @ j.T + self.damping**2 * identity, error[index])
            targets[index] += np.clip(delta, -self.max_joint_delta, self.max_joint_delta)
        return targets


@dataclass(frozen=True)
class AdamUCommand:
    """Commands matching the real Adam-U controller boundary."""

    body: np.ndarray  # [waist(3), neck(2), left arm(7), right arm(7)]
    hands: np.ndarray  # [left hand synergies(6), right hand synergies(6)]
    body_joint_names: tuple[str, ...] = BODY_JOINT_NAMES
    hand_command_names: tuple[str, ...] = (
        *(f"left_{name}" for name in HAND_COMMAND_NAMES),
        *(f"right_{name}" for name in HAND_COMMAND_NAMES),
    )
    ignored_outputs: tuple[str, ...] = ()
    clamped_fields: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.body.ndim != 2 or self.body.shape[1] != 19:
            raise ValueError(f"Adam-U body command must have shape (batch, 19), got {self.body.shape}")
        if self.hands.ndim != 2 or self.hands.shape[1] != 12:
            raise ValueError(f"Adam-U hand command must have shape (batch, 12), got {self.hands.shape}")
        if len(self.body_joint_names) != 19 or len(set(self.body_joint_names)) != 19:
            raise ValueError("Adam-U body joint names must contain 19 unique names")
        if len(self.hand_command_names) != 12 or len(set(self.hand_command_names)) != 12:
            raise ValueError("Adam-U hand command names must contain 12 unique names")
        if not np.isfinite(self.body).all() or not np.isfinite(self.hands).all():
            raise ValueError("Adam-U command contains NaN or infinite values")


class RealG1ToAdamUAdapter:
    """Convert named REAL_G1 action groups into safe Adam-U commands."""

    REQUIRED_COMMON = ("left_hand", "right_hand", "waist")
    IGNORED_FIXED_BASE = ("base_height_command", "navigate_command")

    def __init__(
        self,
        config: AdamUActionMappingConfig | None = None,
        *,
        ik_solver: AdamUIKSolver | None = None,
    ) -> None:
        self.config = config or AdamUActionMappingConfig()
        self.ik_solver = ik_solver
        if self.config.log_commands and not LOGGER.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[ADAM-U] %(levelname)s: %(message)s"))
            LOGGER.addHandler(handler)
            LOGGER.setLevel(logging.INFO)
            LOGGER.propagate = False
        if self.config.control_mode == "eef_space" and ik_solver is None:
            raise ValueError("eef_space requires an AdamUIKSolver; joint outputs will not be used as a fallback")
        self._previous_body: np.ndarray | None = None
        self._previous_hands: np.ndarray | None = None

    @staticmethod
    def _at_step(value: Any, step_index: int, width: int, key: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        elif array.ndim == 3:
            if not 0 <= step_index < array.shape[1]:
                raise IndexError(f"{key} step {step_index} outside horizon {array.shape[1]}")
            array = array[:, step_index, :]
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"GR00T {key!r} must resolve to shape (batch, {width}), got {array.shape}")
        return array.copy()

    @staticmethod
    def _as_batch(value: Any, width: int, key: str, batch: int | None = None) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{key} must have shape (batch, {width}), got {array.shape}")
        if batch is not None and array.shape[0] != batch:
            raise ValueError(f"{key} batch {array.shape[0]} does not match action batch {batch}")
        return array.copy()

    @staticmethod
    def _calibrate_absolute(values: np.ndarray, sign: tuple[float, ...], offset: tuple[float, ...]) -> np.ndarray:
        return values * np.asarray(sign) + np.asarray(offset)

    @staticmethod
    def _calibrate_relative(delta: np.ndarray, current: np.ndarray, sign: tuple[float, ...]) -> np.ndarray:
        return current + delta * np.asarray(sign)

    def _map_hand(self, source: np.ndarray, side: str) -> np.ndarray:
        matrix = self.config.left_hand_matrix if side == "left" else self.config.right_hand_matrix
        return source @ matrix.T

    @staticmethod
    def _limit_step(target: np.ndarray, previous: np.ndarray, max_step: float | None) -> tuple[np.ndarray, bool]:
        if max_step is None:
            return target, False
        limited = previous + np.clip(target - previous, -max_step, max_step)
        return limited, not np.array_equal(limited, target)

    @staticmethod
    def _clamp(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, bool]:
        clamped = np.clip(target, lower, upper)
        return clamped, not np.array_equal(clamped, target)

    def reset(self, body: np.ndarray | None = None, hands: np.ndarray | None = None) -> None:
        self._previous_body = None if body is None else self._as_batch(body, 19, "body")
        self._previous_hands = None if hands is None else self._as_batch(hands, 12, "hands")

    def adapt(
        self,
        groot_action: dict[str, Any],
        *,
        step_index: int = 0,
        current_body: np.ndarray,
    ) -> AdamUCommand:
        """Map one horizon step; returns absolute, limited Adam-U commands."""
        missing = [key for key in self.REQUIRED_COMMON if key not in groot_action]
        arm_keys = ("left_arm", "right_arm") if self.config.control_mode == "joint_space" else (
            "left_wrist_eef_9d", "right_wrist_eef_9d"
        )
        missing.extend(key for key in arm_keys if key not in groot_action)
        if missing:
            raise KeyError(f"Missing required GR00T action groups: {sorted(set(missing))}")

        current = self._as_batch(current_body, 19, "current_body")
        batch = current.shape[0]
        ignored = (
            ("left_wrist_eef_9d", "right_wrist_eef_9d", *self.IGNORED_FIXED_BASE)
            if self.config.control_mode == "joint_space"
            else ("left_arm", "right_arm", *self.IGNORED_FIXED_BASE)
        )
        try:
            waist_src = self._at_step(groot_action["waist"], step_index, 3, "waist")
            left_hand_src = self._at_step(groot_action["left_hand"], step_index, 7, "left_hand")
            right_hand_src = self._at_step(groot_action["right_hand"], step_index, 7, "right_hand")
            for key, value in (("waist", waist_src), ("left_hand", left_hand_src), ("right_hand", right_hand_src)):
                if value.shape[0] != batch:
                    raise ValueError(f"{key} batch does not match current_body")
                if not np.isfinite(value).all():
                    raise ValueError(f"GR00T {key!r} contains NaN or infinite values")

            if self.config.control_mode == "joint_space":
                left_src = self._at_step(groot_action["left_arm"], step_index, 7, "left_arm")
                right_src = self._at_step(groot_action["right_arm"], step_index, 7, "right_arm")
                if self.config.arm_commands_relative:
                    left_arm = self._calibrate_relative(left_src, current[:, 5:12], self.config.left_arm_sign)
                    right_arm = self._calibrate_relative(right_src, current[:, 12:19], self.config.right_arm_sign)
                else:
                    left_arm = self._calibrate_absolute(
                        left_src, self.config.left_arm_sign, self.config.left_arm_zero_offset
                    )
                    right_arm = self._calibrate_absolute(
                        right_src, self.config.right_arm_sign, self.config.right_arm_zero_offset
                    )
                ignored = ("left_wrist_eef_9d", "right_wrist_eef_9d", *self.IGNORED_FIXED_BASE)
            else:
                left_eef = self._at_step(groot_action["left_wrist_eef_9d"], step_index, 9, "left_wrist_eef_9d")
                right_eef = self._at_step(groot_action["right_wrist_eef_9d"], step_index, 9, "right_wrist_eef_9d")
                left_arm = self.ik_solver.solve(
                    "left", left_eef, current[:, 5:12],
                    command_is_relative=self.config.eef_commands_relative,
                )
                right_arm = self.ik_solver.solve(
                    "right", right_eef, current[:, 12:19],
                    command_is_relative=self.config.eef_commands_relative,
                )
                left_arm = self._as_batch(left_arm, 7, "left IK result", batch)
                right_arm = self._as_batch(right_arm, 7, "right IK result", batch)
                ignored = ("left_arm", "right_arm", *self.IGNORED_FIXED_BASE)

            waist = self._calibrate_absolute(
                waist_src, self.config.waist_sign, self.config.waist_zero_offset
            )
            neck = np.broadcast_to(np.asarray(self.config.neck_neutral), (batch, 2)).copy()
            body = np.concatenate((waist, neck, left_arm, right_arm), axis=1)
            hands = np.concatenate(
                (self._map_hand(left_hand_src, "left"), self._map_hand(right_hand_src, "right")), axis=1
            )
        except (ValueError, FloatingPointError):
            if self.config.invalid_value_policy != "hold" or self._previous_body is None:
                raise
            LOGGER.error("Rejected invalid GR00T output; holding previous Adam-U command")
            return AdamUCommand(
                self._previous_body.copy(), self._previous_hands.copy(), ignored_outputs=tuple(ignored)
            )

        if not np.isfinite(body).all() or not np.isfinite(hands).all():
            if self.config.invalid_value_policy == "hold" and self._previous_body is not None:
                LOGGER.error("GR00T output contains NaN/Inf; holding previous Adam-U command")
                return AdamUCommand(
                    self._previous_body.copy(), self._previous_hands.copy(), ignored_outputs=tuple(ignored)
                )
            raise ValueError("GR00T output contains NaN or infinite values")

        changed: list[str] = []
        body_lower, body_upper, _urdf_velocity = self.config.body_limits
        body, was_changed = self._clamp(body, body_lower, body_upper)
        if was_changed:
            changed.append("body_joint_limits")
        hand_lower = np.tile(np.asarray(self.config.hand_lower), 2)
        hand_upper = np.tile(np.asarray(self.config.hand_upper), 2)
        hands, was_changed = self._clamp(hands, hand_lower, hand_upper)
        if was_changed:
            changed.append("hand_limits")

        previous_body = current if self._previous_body is None else self._previous_body
        previous_hands = np.zeros_like(hands) if self._previous_hands is None else self._previous_hands
        max_body_step = self.config.max_body_position_step
        if self.config.max_body_velocity is not None:
            max_body_step = min(
                max_body_step if max_body_step is not None else np.inf,
                self.config.max_body_velocity * self.config.control_dt,
            )
        body, was_changed = self._limit_step(body, previous_body, max_body_step)
        if was_changed:
            changed.append("body_step_or_velocity")

        max_hand_step = self.config.max_hand_position_step
        if self.config.max_hand_velocity is not None:
            max_hand_step = min(
                max_hand_step if max_hand_step is not None else np.inf,
                self.config.max_hand_velocity * self.config.control_dt,
            )
        hands, was_changed = self._limit_step(hands, previous_hands, max_hand_step)
        if was_changed:
            changed.append("hand_step_or_velocity")

        alpha = self.config.smoothing_alpha
        body = previous_body + alpha * (body - previous_body)
        hands = previous_hands + alpha * (hands - previous_hands)

        command = AdamUCommand(
            body=body.astype(np.float32),
            hands=hands.astype(np.float32),
            ignored_outputs=tuple(key for key in ignored if key in groot_action),
            clamped_fields=tuple(dict.fromkeys(changed)),
        )
        command.validate()
        self._previous_body, self._previous_hands = command.body.copy(), command.hands.copy()

        if self.config.log_commands:
            LOGGER.info("Original GR00T action groups: %s", {k: np.shape(v) for k, v in groot_action.items()})
            LOGGER.info("Mapped Adam-U body[19] %s: %s", command.body_joint_names, command.body[0])
            LOGGER.info("Mapped Adam-U hands[12] %s: %s", command.hand_command_names, command.hands[0])
            LOGGER.info("Ignored GR00T outputs: %s", command.ignored_outputs)
            if command.clamped_fields:
                LOGGER.warning("Clamped/limited Adam-U fields: %s", command.clamped_fields)
        return command

    def validate_for_real_robot(
        self, command: AdamUCommand, previous_command: AdamUCommand | None = None
    ) -> None:
        """Final deployment gate; simulation does not call this automatically."""
        if not self.config.calibration_verified_for_real_robot:
            raise RuntimeError("Real-robot calibration is not marked verified; refusing to send command")
        command.validate()
        if command.body_joint_names != BODY_JOINT_NAMES:
            raise ValueError("Body command joint names/order do not match Adam-U")
        expected_hands = tuple(
            [*(f"left_{name}" for name in HAND_COMMAND_NAMES), *(f"right_{name}" for name in HAND_COMMAND_NAMES)]
        )
        if command.hand_command_names != expected_hands:
            raise ValueError("Hand command names/order do not match Adam-U")
        lower, upper, _ = self.config.body_limits
        if np.any(command.body < lower) or np.any(command.body > upper):
            raise ValueError("Body command violates Adam-U URDF limits")
        hand_lower = np.tile(np.asarray(self.config.hand_lower), 2)
        hand_upper = np.tile(np.asarray(self.config.hand_upper), 2)
        if np.any(command.hands < hand_lower) or np.any(command.hands > hand_upper):
            raise ValueError("Hand command violates Adam-U synergy limits")
        if previous_command is not None:
            if self.config.max_body_position_step is not None and np.any(
                np.abs(command.body - previous_command.body) > self.config.max_body_position_step + 1e-7
            ):
                raise ValueError("Body command exceeds maximum per-step position change")
            if self.config.max_hand_position_step is not None and np.any(
                np.abs(command.hands - previous_command.hands) > self.config.max_hand_position_step + 1e-7
            ):
                raise ValueError("Hand command exceeds maximum per-step position change")


def expand_hand_synergies_for_isaac(hand_command: np.ndarray) -> np.ndarray:
    """Expand public hands[12] into Adam-U's 24 named URDF finger-joint targets."""
    values = np.asarray(hand_command, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError(f"Expected Adam-U hands shape (batch, 12), got {values.shape}")

    expanded: list[np.ndarray] = []
    for offset in (0, 6):
        opposition, thumb, index, middle, ring, pinky = (values[:, offset + i] for i in range(6))
        expanded.extend(
            (
                opposition,
                thumb * (0.5 / 1.2),
                thumb * (1.0 / 1.2),
                thumb,
                index,
                index * (1.6 / 1.7),
                middle,
                middle * (1.6 / 1.7),
                ring,
                ring * (1.6 / 1.7),
                pinky,
                pinky * (1.6 / 1.7),
            )
        )
    return np.stack(expanded, axis=1).astype(np.float32)
