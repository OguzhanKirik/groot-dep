"""End-effector pose helpers for GR00T REAL_G1 schema shims."""

from __future__ import annotations

import numpy as np

# Valid identity rotation in GR00T's XYZ+rot6d layout (first two rows of I₃).
IDENTITY_EEF_9D = np.array(
    [0.3, 0.15, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float32,
)

_WRIST_BODY_NAMES = {
    "left": "wristRollLeft",
    "right": "wristRollRight",
}


class IsaacAdamUKinematicsProvider:
    """Expose Adam-U wrist FK and arm Jacobians in the environment world frame."""

    def __init__(self, env) -> None:
        from configs.joint_state import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

        self.env = env
        self.robot = env.scene["robot"]
        self._body_indices = {}
        for side, body_name in _WRIST_BODY_NAMES.items():
            if body_name not in self.robot.data.body_names:
                raise ValueError(f"Adam-U EEF body {body_name!r} is missing from the articulation")
            self._body_indices[side] = self.robot.data.body_names.index(body_name)
        self._joint_indices = {
            "left": [self.robot.data.joint_names.index(name) for name in LEFT_ARM_JOINT_NAMES],
            "right": [self.robot.data.joint_names.index(name) for name in RIGHT_ARM_JOINT_NAMES],
        }

    @staticmethod
    def _torch(value):
        import torch

        if isinstance(value, torch.Tensor):
            return value
        try:
            return torch.utils.dlpack.from_dlpack(value)
        except (TypeError, RuntimeError):
            return torch.as_tensor(value.numpy())

    def __call__(self, side: str, current_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if side not in self._body_indices:
            raise ValueError(f"Unknown Adam-U arm side: {side!r}")
        body_idx = self._body_indices[side]
        jacobian_body_idx = body_idx - 1 if self.robot.is_fixed_base else body_idx
        jacobians = self._torch(self.robot.root_view.get_jacobians())
        joint_ids = torch.as_tensor(self._joint_indices[side], dtype=torch.long, device=jacobians.device)
        jacobian = torch.index_select(jacobians[:, jacobian_body_idx], dim=2, index=joint_ids)

        body_pos_w = self._torch(self.robot.data.body_pos_w)[:, body_idx]
        body_quat_w = self._torch(self.robot.data.body_quat_w)[:, body_idx]
        env_origins = self._torch(self.env.scene.env_origins)
        pos = (body_pos_w - env_origins).detach().cpu().numpy()
        quat = body_quat_w.detach().cpu().numpy()
        pose = build_eef_9d(pos, quat).astype(np.float64)
        jacobian_np = jacobian.detach().cpu().numpy().astype(np.float64)
        if pose.shape[0] != np.asarray(current_arm).shape[0]:
            raise ValueError("Adam-U FK batch does not match current arm state")
        return pose, jacobian_np


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion(s) to rot6d (first two rows of R, flattened)."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    rot6d = rot[:, :2, :].reshape(-1, 6)
    return rot6d[0] if single else rot6d


def build_eef_9d(pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build XYZ+rot6d pose vector(s), shape (9,) or (N, 9)."""
    pos = np.asarray(pos_xyz, dtype=np.float32)
    rot6d = quat_wxyz_to_rot6d(quat_wxyz).astype(np.float32)
    if pos.ndim == 1:
        return np.concatenate([pos, rot6d], axis=0)
    return np.concatenate([pos, rot6d], axis=1)


def read_wrist_eef_9d(env, side: str, num_envs: int) -> np.ndarray:
    """Read wrist pose from Isaac Lab and pack as GR00T eef_9d."""
    import torch

    def as_torch(array):
        if isinstance(array, torch.Tensor):
            return array
        try:
            return torch.utils.dlpack.from_dlpack(array)
        except (TypeError, RuntimeError):
            return torch.as_tensor(array.numpy())

    body_name = _WRIST_BODY_NAMES[side]
    robot = env.scene["robot"]
    body_names = robot.data.body_names
    if body_name not in body_names:
        return np.tile(IDENTITY_EEF_9D, (num_envs, 1))

    body_idx = body_names.index(body_name)
    body_pos_w = as_torch(robot.data.body_pos_w)
    body_quat_w = as_torch(robot.data.body_quat_w)
    env_origins = as_torch(env.scene.env_origins)
    pos = body_pos_w[:, body_idx] - env_origins
    quat = body_quat_w[:, body_idx]
    poses = []
    for env_idx in range(num_envs):
        poses.append(build_eef_9d(pos[env_idx].detach().cpu().numpy(), quat[env_idx].detach().cpu().numpy()))
    return np.stack(poses, axis=0).astype(np.float32)
