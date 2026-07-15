"""Adam-U joint definitions parsed from the URDF (single source of truth)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

# Repo root: adam_u_groot/configs -> adam_u_groot -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF_PATH = _REPO_ROOT / "assets" / "robots" / "adam_u" / "urdf" / "adam_u.urdf"


@lru_cache(maxsize=4)
def parse_revolute_joint_names(urdf_path: str | Path = DEFAULT_URDF_PATH) -> tuple[str, ...]:
    """Return revolute joint names in URDF document order."""
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Adam-U URDF not found: {path}")

    root = ET.parse(path).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        if name:
            names.append(name)
    return tuple(names)


# Joint groups used by GR00T / LeRobot (subset of full URDF).
WAIST_JOINT_NAMES: tuple[str, ...] = (
    "waistRoll",
    "waistPitch",
    "waistYaw",
)

LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulderPitch_Left",
    "shoulderRoll_Left",
    "shoulderYaw_Left",
    "elbow_Left",
    "wristYaw_Left",
    "wristPitch_Left",
    "wristRoll_Left",
)

RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulderPitch_Right",
    "shoulderRoll_Right",
    "shoulderYaw_Right",
    "elbow_Right",
    "wristYaw_Right",
    "wristPitch_Right",
    "wristRoll_Right",
)

LEFT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "L_thumb_MCP_joint1",
    "L_thumb_MCP_joint2",
    "L_thumb_PIP_joint",
    "L_thumb_DIP_joint",
    "L_index_MCP_joint",
    "L_index_DIP_joint",
    "L_middle_MCP_joint",
    "L_middle_DIP_joint",
    "L_ring_MCP_joint",
    "L_ring_DIP_joint",
    "L_pinky_MCP_joint",
    "L_pinky_DIP_joint",
)

RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "R_thumb_MCP_joint1",
    "R_thumb_MCP_joint2",
    "R_thumb_PIP_joint",
    "R_thumb_DIP_joint",
    "R_index_MCP_joint",
    "R_index_DIP_joint",
    "R_middle_MCP_joint",
    "R_middle_DIP_joint",
    "R_ring_MCP_joint",
    "R_ring_DIP_joint",
    "R_pinky_MCP_joint",
    "R_pinky_DIP_joint",
)

NECK_JOINT_NAMES: tuple[str, ...] = (
    "neckYaw",
    "neckPitch",
)

JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "waist": WAIST_JOINT_NAMES,
    "left_arm": LEFT_ARM_JOINT_NAMES,
    "left_hand": LEFT_HAND_JOINT_NAMES,
    "right_arm": RIGHT_ARM_JOINT_NAMES,
    "right_hand": RIGHT_HAND_JOINT_NAMES,
    "neck": NECK_JOINT_NAMES,
}


def get_joint_group(name: str) -> tuple[str, ...]:
    """Return joint names for a named group."""
    if name == "all":
        return parse_revolute_joint_names()
    if name not in JOINT_GROUPS:
        raise KeyError(f"Unknown joint group '{name}'. Available: {sorted(JOINT_GROUPS)} + ['all']")
    return JOINT_GROUPS[name]


def validate_joint_groups_against_urdf(urdf_path: str | Path = DEFAULT_URDF_PATH) -> None:
    """Ensure grouped joints match the URDF (helps catch renames early)."""
    urdf_joints = set(parse_revolute_joint_names(urdf_path))
    grouped: set[str] = set()
    for group_name, joints in JOINT_GROUPS.items():
        missing = [j for j in joints if j not in urdf_joints]
        if missing:
            raise ValueError(f"Group '{group_name}' references joints missing from URDF: {missing}")
        grouped.update(joints)

    extra = sorted(urdf_joints - grouped)
    if extra:
        raise ValueError(f"URDF joints not assigned to any group: {extra}")
