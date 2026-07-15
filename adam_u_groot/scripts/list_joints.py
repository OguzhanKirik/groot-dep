"""Print Adam-U joint names from the URDF (no Isaac Sim required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ADAM_U_GROOT_ROOT = _SCRIPT_DIR.parent
if str(_ADAM_U_GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADAM_U_GROOT_ROOT))

from configs.joint_state import JOINT_GROUPS, get_joint_group, parse_revolute_joint_names, validate_joint_groups_against_urdf


def main() -> None:
    parser = argparse.ArgumentParser(description="List Adam-U joint names from the URDF.")
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        help=f"Joint group: all, {', '.join(sorted(JOINT_GROUPS))}",
    )
    args = parser.parse_args()

    validate_joint_groups_against_urdf()
    joints = get_joint_group(args.group)

    print(f"Adam-U joints ({args.group}, count={len(joints)}):")
    for index, name in enumerate(joints):
        print(f"  [{index:02d}] {name}")

    if args.group == "all":
        print(f"\nTotal revolute joints in URDF: {len(parse_revolute_joint_names())}")


if __name__ == "__main__":
    main()
