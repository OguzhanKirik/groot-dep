"""Verified joint ordering and configurable REAL_G1 -> Adam-U action mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from configs.joint_state import BODY_JOINT_NAMES, HAND_COMMAND_NAMES, parse_revolute_joint_limits

ControlMode = Literal["joint_space", "eef_space"]
InvalidValuePolicy = Literal["reject", "hold"]

# The REAL_G1 checkpoint does not publish per-channel labels, but its statistics
# expose four mirrored finger-flexion ranges followed by thumb opposition and two
# thumb-flexion ranges.  Keeping these names explicit makes the assumption auditable.
G1_HAND_CHANNEL_NAMES: tuple[str, ...] = (
    "index_flexion",
    "middle_flexion",
    "ring_flexion",
    "pinky_flexion",
    "thumb_opposition",
    "thumb_flexion_proximal",
    "thumb_flexion_distal",
)


def _default_hand_matrix(side: str) -> np.ndarray:
    """Map named G1 channels to six Adam-U synergies in HAND_COMMAND_NAMES order."""
    matrix = np.zeros((6, 7), dtype=np.float64)
    # G1 left flexion is negative while right flexion is positive. Adam-U's
    # low-level synergies use positive values for closing on both sides.
    flexion_sign = -1.0 if side == "left" else 1.0
    thumb_sign = 1.0 if side == "left" else -1.0
    matrix[0, 4] = 1.0  # thumb opposition; sign is calibrated independently below
    matrix[1, 5] = 0.5 * thumb_sign
    matrix[1, 6] = 0.5 * thumb_sign
    matrix[2, 0] = flexion_sign
    matrix[3, 1] = flexion_sign
    matrix[4, 2] = flexion_sign
    matrix[5, 3] = flexion_sign
    return matrix


def _body_limits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    urdf = parse_revolute_joint_limits()
    lower = np.asarray([urdf[name][0] for name in BODY_JOINT_NAMES], dtype=np.float64)
    upper = np.asarray([urdf[name][1] for name in BODY_JOINT_NAMES], dtype=np.float64)
    velocity = np.asarray([urdf[name][2] for name in BODY_JOINT_NAMES], dtype=np.float64)
    return lower, upper, velocity


@dataclass
class AdamUActionMappingConfig:
    """Safety and calibration settings for the Adam-U low-level command adapter."""

    control_mode: ControlMode = "joint_space"
    # PolicyClient postprocessing normally reconstructs absolute arm targets.
    # Set this True only when consuming raw relative model outputs.
    arm_commands_relative: bool = False
    # Gr00tPolicy decodes REAL_G1 relative EEF predictions back to absolute
    # poses using the supplied wrist state. Enable only for raw model outputs.
    eef_commands_relative: bool = False
    neck_neutral: tuple[float, float] = (0.0, -0.35)

    # Source-to-Adam calibration: adam = sign * source + zero_offset (radians).
    waist_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    waist_zero_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_arm_sign: tuple[float, ...] = (1.0,) * 7
    left_arm_zero_offset: tuple[float, ...] = (0.0,) * 7
    right_arm_sign: tuple[float, ...] = (1.0,) * 7
    right_arm_zero_offset: tuple[float, ...] = (0.0,) * 7

    left_hand_matrix: np.ndarray = field(default_factory=lambda: _default_hand_matrix("left"))
    right_hand_matrix: np.ndarray = field(default_factory=lambda: _default_hand_matrix("right"))
    # Adam-U synergy ranges: opposition, thumb flexion, then four finger flexions.
    hand_lower: tuple[float, ...] = (0.0,) * 6
    hand_upper: tuple[float, ...] = (1.1, 1.2, 1.7, 1.7, 1.7, 1.7)

    # Per-step safety. None disables that limiter.
    max_body_position_step: float | None = 0.10
    max_hand_position_step: float | None = 0.10
    max_body_velocity: float | None = None
    max_hand_velocity: float | None = None
    control_dt: float = 1.0 / 30.0
    smoothing_alpha: float = 1.0
    invalid_value_policy: InvalidValuePolicy = "reject"
    log_commands: bool = True

    # Must be deliberately enabled by deployment code after hardware calibration.
    calibration_verified_for_real_robot: bool = False

    body_joint_names: tuple[str, ...] = BODY_JOINT_NAMES
    hand_command_names: tuple[str, ...] = HAND_COMMAND_NAMES

    def __post_init__(self) -> None:
        if self.control_mode not in ("joint_space", "eef_space"):
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        if self.invalid_value_policy not in ("reject", "hold"):
            raise ValueError(f"Unsupported invalid-value policy: {self.invalid_value_policy}")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if tuple(self.body_joint_names) != BODY_JOINT_NAMES:
            raise ValueError("Adam-U body ordering must be waist, neck, left arm, right arm")
        if tuple(self.hand_command_names) != HAND_COMMAND_NAMES:
            raise ValueError("Adam-U hand command ordering does not match the controller contract")
        for side, matrix in (("left", self.left_hand_matrix), ("right", self.right_hand_matrix)):
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (6, 7):
                raise ValueError(f"{side}_hand_matrix must have shape (6, 7), got {matrix.shape}")
            setattr(self, f"{side}_hand_matrix", matrix)

    @property
    def body_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _body_limits()
