"""Read Adam-U joint positions/velocities from an Isaac Lab environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from configs.joint_state import RIGHT_ARM_JOINT_NAMES, get_joint_group

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointStateReader:
    """Read joint states from the simulated Adam-U articulation."""

    def __init__(self, env: ManagerBasedEnv, group: str = "right_arm"):
        self.env = env
        self.group = group
        self.joint_names = list(get_joint_group(group))
        self._joint_ids = self._lookup_joint_ids(env, self.joint_names)

    @staticmethod
    def _lookup_joint_ids(env: ManagerBasedEnv, joint_names: list[str]) -> list[int]:
        robot = env.scene["robot"]
        sim_joint_names = robot.data.joint_names
        missing = [name for name in joint_names if name not in sim_joint_names]
        if missing:
            raise ValueError(
                f"Joints not found in simulation articulation: {missing}. "
                f"Available: {sim_joint_names}"
            )
        return [sim_joint_names.index(name) for name in joint_names]

    def _select(self, values) -> torch.Tensor:
        # Isaac Lab 6 may expose articulation buffers as Warp arrays. Warp only
        # accepts scalar/slice indexing, so convert through DLPack before the
        # multi-joint gather. This is zero-copy for CUDA buffers.
        if not isinstance(values, torch.Tensor):
            values = torch.utils.dlpack.from_dlpack(values)
        joint_ids = torch.as_tensor(self._joint_ids, dtype=torch.long, device=values.device)
        return torch.index_select(values, dim=1, index=joint_ids)

    def get_positions(self) -> torch.Tensor:
        """Joint positions, shape (num_envs, num_joints)."""
        robot = self.env.scene["robot"]
        return self._select(robot.data.joint_pos)

    def get_velocities(self) -> torch.Tensor:
        """Joint velocities, shape (num_envs, num_joints)."""
        robot = self.env.scene["robot"]
        return self._select(robot.data.joint_vel)

    def as_dict(self, env_index: int = 0) -> dict[str, float]:
        """Named joint positions for one environment (useful for logging/export)."""
        positions = self.get_positions()[env_index].detach().cpu().tolist()
        return dict(zip(self.joint_names, positions))

    def as_flat_vector(self, env_index: int = 0) -> list[float]:
        """Flat position vector in group joint order (LeRobot state layout)."""
        return self.get_positions()[env_index].detach().cpu().tolist()


def make_right_arm_reader(env: ManagerBasedEnv) -> JointStateReader:
    """Convenience helper for the 7-DoF right arm used by GR00T."""
    _ = RIGHT_ARM_JOINT_NAMES  # document expected group size
    return JointStateReader(env, group="right_arm")
