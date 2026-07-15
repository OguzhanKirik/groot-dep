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
