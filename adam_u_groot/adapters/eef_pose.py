"""End-effector pose helpers for GR00T REAL_G1 schema shims."""

from __future__ import annotations

from dataclasses import dataclass

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


def _rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert the GR00T two-row 6D representation to rotation matrices."""
    values = np.asarray(rot6d, dtype=np.float64)
    first = values[..., :3]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = values[..., 3:]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


@dataclass(frozen=True)
class G1AdamWorkspaceTransform:
    """Invertible Adam world <-> REAL_G1 canonical workspace transform.

    Adam-U faces world -X and its imported articulation origin is one metre
    above the floor. REAL_G1 uses +X forward, opposite lateral sign, and a
    body-relative Z origin. The default mapping is therefore::

        p_g1 = diag(-1, -1, 1) @ (p_world - [0, 0, 1])

    A separate wrist/tool rotation offset is intentionally not guessed here.
    Until that calibration exists, GR00T EEF control uses position-only IK.
    """

    adam_world_origin: tuple[float, float, float] = (0.0, 0.0, 1.0)
    world_to_g1_rotation: tuple[tuple[float, float, float], ...] = (
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        rotation = np.asarray(self.world_to_g1_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("world_to_g1_rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("world_to_g1_rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("world_to_g1_rotation must be a proper rotation")

    @property
    def _rotation(self) -> np.ndarray:
        return np.asarray(self.world_to_g1_rotation, dtype=np.float64)

    @property
    def _origin(self) -> np.ndarray:
        return np.asarray(self.adam_world_origin, dtype=np.float64)

    @staticmethod
    def _as_pose(pose_9d: np.ndarray) -> tuple[np.ndarray, bool]:
        pose = np.asarray(pose_9d, dtype=np.float64)
        single = pose.ndim == 1
        if single:
            pose = pose[None, :]
        if pose.ndim != 2 or pose.shape[1] != 9 or not np.all(np.isfinite(pose)):
            raise ValueError(f"EEF pose must be finite with shape (9,) or (N, 9), got {pose.shape}")
        return pose, single

    def world_to_g1_pose(self, pose_9d: np.ndarray) -> np.ndarray:
        pose, single = self._as_pose(pose_9d)
        rotation = self._rotation
        position = (rotation @ (pose[:, :3] - self._origin).T).T
        eef_rotation = _rot6d_to_matrix_np(pose[:, 3:])
        transformed_rotation = rotation[None, :, :] @ eef_rotation
        result = np.concatenate((position, transformed_rotation[:, :2, :].reshape(-1, 6)), axis=1)
        return result[0].astype(np.float32) if single else result.astype(np.float32)

    def g1_to_world_pose(self, pose_9d: np.ndarray) -> np.ndarray:
        pose, single = self._as_pose(pose_9d)
        inverse = self._rotation.T
        position = (inverse @ pose[:, :3].T).T + self._origin
        eef_rotation = _rot6d_to_matrix_np(pose[:, 3:])
        transformed_rotation = inverse[None, :, :] @ eef_rotation
        result = np.concatenate((position, transformed_rotation[:, :2, :].reshape(-1, 6)), axis=1)
        return result[0].astype(np.float32) if single else result.astype(np.float32)


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

    def get_torch_state(self, side: str):
        """Return environment-local position, world quaternion, and world Jacobian."""
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
        return body_pos_w - env_origins, body_quat_w, jacobian

    def __call__(self, side: str, current_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_t, quat_t, jacobian_t = self.get_torch_state(side)
        pos = pos_t.detach().cpu().numpy()
        quat = quat_t.detach().cpu().numpy()
        pose = build_eef_9d(pos, quat).astype(np.float64)
        jacobian_np = jacobian_t.detach().cpu().numpy().astype(np.float64)
        if pose.shape[0] != np.asarray(current_arm).shape[0]:
            raise ValueError("Adam-U FK batch does not match current arm state")
        return pose, jacobian_np


class IsaacDifferentialIKSolver:
    """Adapter around Isaac Lab's trusted differential IK implementation."""

    def __init__(
        self,
        provider: IsaacAdamUKinematicsProvider,
        *,
        max_joint_delta: float = 0.01,
        command_type: str = "position",
        workspace_transform: G1AdamWorkspaceTransform | None = None,
    ):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.provider = provider
        self.max_joint_delta = float(max_joint_delta)
        if command_type not in ("position", "pose"):
            raise ValueError(f"Unsupported differential IK command type: {command_type!r}")
        self.command_type = command_type
        self.workspace_transform = workspace_transform
        cfg = DifferentialIKControllerCfg(
            command_type=command_type, use_relative_mode=False, ik_method="dls"
        )
        self.controller = DifferentialIKController(
            cfg, num_envs=provider.env.num_envs, device=provider.env.device
        )
        self._commanded_joint_pos = {}

    def solve(self, side, eef_command_9d, current_arm, *, command_is_relative):
        import torch

        command = np.asarray(eef_command_9d, dtype=np.float32)
        if self.workspace_transform is not None:
            if command_is_relative:
                raise ValueError(
                    "G1 workspace conversion expects PolicyClient-postprocessed absolute EEF poses"
                )
            command = self.workspace_transform.g1_to_world_pose(command)
        current = np.asarray(current_arm, dtype=np.float32)
        pos, quat, jacobian = self.provider.get_torch_state(side)
        target_pos = torch.as_tensor(command[:, :3], device=pos.device, dtype=pos.dtype)
        if command_is_relative:
            target_pos = pos + target_pos
        if self.command_type == "pose":
            from isaaclab.utils.math import quat_from_matrix, quat_mul

            rot6d = torch.as_tensor(command[:, 3:], device=pos.device, dtype=pos.dtype)
            first = torch.nn.functional.normalize(rot6d[:, :3], dim=1)
            second = rot6d[:, 3:] - torch.sum(first * rot6d[:, 3:], dim=1, keepdim=True) * first
            second = torch.nn.functional.normalize(second, dim=1)
            third = torch.linalg.cross(first, second, dim=1)
            target_quat = quat_from_matrix(torch.stack((first, second, third), dim=1))
            if command_is_relative:
                target_quat = quat_mul(target_quat, quat)
            controller_command = torch.cat((target_pos, target_quat), dim=1)
        else:
            controller_command = target_pos
        self.controller.set_command(controller_command, ee_pos=pos, ee_quat=quat)
        joint_pos = torch.as_tensor(current, device=pos.device, dtype=pos.dtype)
        target = self.controller.compute(pos, quat, jacobian, joint_pos)
        delta = torch.clamp(target - joint_pos, -self.max_joint_delta, self.max_joint_delta)
        # Accumulate safe IK increments on the last actuator target rather than
        # on the measured position. Re-basing on the measured pose every cycle
        # follows gravity sag and removes the position error needed to hold the
        # arm up.
        commanded = self._commanded_joint_pos.get(side)
        if commanded is None or commanded.shape != joint_pos.shape:
            commanded = joint_pos.detach().clone()
        commanded = commanded + delta
        self._commanded_joint_pos[side] = commanded.detach().clone()
        return commanded.detach().cpu().numpy()


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
