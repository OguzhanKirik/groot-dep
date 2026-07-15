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
        """Return EEF pose and Jacobian consistently in the robot-root frame."""
        import torch
        from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

        if side not in self._body_indices:
            raise ValueError(f"Unknown Adam-U arm side: {side!r}")
        body_idx = self._body_indices[side]
        jacobian_body_idx = body_idx - 1 if self.robot.is_fixed_base else body_idx
        jacobians = self._torch(self.robot.root_view.get_jacobians())
        joint_ids = torch.as_tensor(self._joint_indices[side], dtype=torch.long, device=jacobians.device)
        jacobian = torch.index_select(jacobians[:, jacobian_body_idx], dim=2, index=joint_ids)

        # PhysX returns the Jacobian in world axes. Isaac's differential IK
        # controller expects it in the same robot-root frame as the EEF pose.
        root_rotation_b = matrix_from_quat(quat_inv(self._torch(self.robot.data.root_quat_w)))
        jacobian = torch.cat(
            (
                torch.bmm(root_rotation_b, jacobian[:, :3, :]),
                torch.bmm(root_rotation_b, jacobian[:, 3:, :]),
            ),
            dim=1,
        )

        body_pos_w = self._torch(self.robot.data.body_pos_w)[:, body_idx]
        body_quat_w = self._torch(self.robot.data.body_quat_w)[:, body_idx]
        root_pos_w = self._torch(self.robot.data.root_pos_w)
        root_quat_w = self._torch(self.robot.data.root_quat_w)
        body_pos_b, body_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, body_pos_w, body_quat_w
        )
        return body_pos_b, body_quat_b, jacobian

    def world_position_to_root(self, position_env):
        """Convert environment-local world positions to robot-root coordinates."""
        import torch
        from isaaclab.utils.math import matrix_from_quat, quat_inv

        position = torch.as_tensor(position_env, device=self.env.device, dtype=torch.float32)
        env_origins = self._torch(self.env.scene.env_origins)
        root_pos_env = self._torch(self.robot.data.root_pos_w) - env_origins
        root_rotation_b = matrix_from_quat(quat_inv(self._torch(self.robot.data.root_quat_w)))
        return torch.bmm(root_rotation_b, (position - root_pos_env).unsqueeze(-1)).squeeze(-1)

    def __call__(self, side: str, current_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        body_idx = self._body_indices[side]
        body_pos_w = self._torch(self.robot.data.body_pos_w)[:, body_idx]
        quat_t = self._torch(self.robot.data.body_quat_w)[:, body_idx]
        pos_t = body_pos_w - self._torch(self.env.scene.env_origins)
        _, _, jacobian_t = self.get_torch_state(side)
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
        accumulate_joint_targets: bool = True,
        max_commanded_joint_error: float | None = None,
    ):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.provider = provider
        self.max_joint_delta = float(max_joint_delta)
        if command_type not in ("position", "pose"):
            raise ValueError(f"Unsupported differential IK command type: {command_type!r}")
        self.command_type = command_type
        self.workspace_transform = workspace_transform
        self.accumulate_joint_targets = bool(accumulate_joint_targets)
        self.max_commanded_joint_error = (
            None if max_commanded_joint_error is None else float(max_commanded_joint_error)
        )
        if self.max_commanded_joint_error is not None and self.max_commanded_joint_error <= 0:
            raise ValueError("max_commanded_joint_error must be positive")
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
        else:
            target_pos = self.provider.world_position_to_root(target_pos).to(dtype=pos.dtype)
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
        # GR00T may accumulate safe increments on the last actuator target to
        # resist gravity sag. Native absolute-pose teleoperation disables that
        # behavior below to avoid repeatedly integrating the same pose error.
        if self.accumulate_joint_targets:
            commanded = self._commanded_joint_pos.get(side)
            if commanded is None or commanded.shape != joint_pos.shape:
                commanded = joint_pos.detach().clone()
            commanded = commanded + delta
        else:
            # Native teleoperation supplies a persistent absolute Cartesian
            # target. Re-accumulating the same pose error into the actuator
            # target causes integral windup and large oscillations; standard
            # differential IK instead commands q_measured + dq each cycle.
            commanded = joint_pos + delta
        if self.max_commanded_joint_error is not None:
            # Bound persistent target lead so an unreachable Cartesian target
            # or a near-singular Jacobian cannot wind the joints indefinitely.
            lead = torch.clamp(
                commanded - joint_pos,
                -self.max_commanded_joint_error,
                self.max_commanded_joint_error,
            )
            commanded = joint_pos + lead
        self._commanded_joint_pos[side] = commanded.detach().clone()
        return commanded.detach().cpu().numpy()

    def sync_commanded_joint_pos(self, side: str, commanded_joint_pos: np.ndarray) -> None:
        """Synchronize integrator state with the final clamped/masked command."""
        import torch

        values = np.asarray(commanded_joint_pos, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
            raise ValueError("Synchronized IK joint target must be finite with shape (batch, 7)")
        self._commanded_joint_pos[side] = torch.as_tensor(
            values, device=self.provider.env.device, dtype=torch.float32
        ).detach().clone()


class PinkAdamUIKSolver:
    """Pink/Pinocchio QP IK backend for Adam-U absolute wrist-pose targets.

    Pink solves on the complete fixed-base URDF model, so its configuration and
    velocity limits are enforced by the QP. A low-weight posture task resolves
    Adam-U's redundant seventh arm DOF without allowing unrelated joints to
    drift. The caller still applies actuator lead, slew-rate, and smoothing
    limits after this solver, matching the Isaac differential-IK path.
    """

    def __init__(
        self,
        provider: "PinocchioAdamUKinematicsProvider",
        *,
        max_joint_delta: float = 0.01,
        position_cost: float = 1.0,
        orientation_cost: float = 0.25,
        posture_cost: float = 1e-3,
        damping: float = 1e-6,
        solver: str = "daqp",
        max_commanded_joint_error: float | None = None,
    ) -> None:
        import pink
        import qpsolvers
        from pink.tasks import PostureTask, Task

        if solver not in qpsolvers.available_solvers:
            raise ValueError(
                f"Pink QP solver {solver!r} is unavailable; installed: {qpsolvers.available_solvers}"
            )
        if max_joint_delta <= 0 or position_cost <= 0 or orientation_cost < 0:
            raise ValueError("Pink IK step/cost parameters are invalid")
        self.provider = provider
        self.pink = pink
        self.PostureTask = PostureTask
        self.Task = Task
        self.max_joint_delta = float(max_joint_delta)
        self.position_cost = float(position_cost)
        self.orientation_cost = float(orientation_cost)
        self.posture_cost = float(posture_cost)
        self.damping = float(damping)
        self.qp_solver = solver
        self.max_commanded_joint_error = (
            None
            if max_commanded_joint_error is None
            else float(max_commanded_joint_error)
        )
        self._commanded_joint_pos = {}
        from configs.joint_state import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

        self._models = {}
        self._q_indices = {}
        for side, arm_names in (
            ("left", LEFT_ARM_JOINT_NAMES),
            ("right", RIGHT_ARM_JOINT_NAMES),
        ):
            arm_set = set(arm_names)
            locked_joint_ids = [
                joint_id
                for joint_id in range(1, self.provider.model.njoints)
                if self.provider.model.names[joint_id] not in arm_set
            ]
            reduced = self.provider.pin.buildReducedModel(
                self.provider.model,
                locked_joint_ids,
                self.provider.pin.neutral(self.provider.model),
            )
            self._models[side] = reduced
            self._q_indices[side] = {
                name: reduced.joints[reduced.getJointId(name)].idx_q for name in arm_names
            }

    def _model_configuration(self, side: str, env_index: int, current_arm: np.ndarray):
        model = self._models[side]
        q = self.provider.pin.neutral(model)
        arm_names = tuple(self._q_indices[side])
        for arm_index, name in enumerate(arm_names):
            q[self._q_indices[side][name]] = current_arm[env_index, arm_index]
        return self.pink.Configuration(model, model.createData(), q)

    def solve(self, side, eef_command_9d, current_arm, *, command_is_relative):
        from configs.joint_state import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

        command = np.asarray(eef_command_9d, dtype=np.float64)
        current = np.asarray(current_arm, dtype=np.float64)
        if command.ndim != 2 or command.shape[1] != 9 or current.shape[1] != 7:
            raise ValueError("Pink IK expects EEF[batch,9] and arm[batch,7]")
        target_rotations = _rot6d_to_matrix_np(command[:, 3:])
        measured_pose, world_jacobians = self.provider(side, current)
        measured_rotations = _rot6d_to_matrix_np(measured_pose[:, 3:])
        arm_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        results = []

        for env_index in range(command.shape[0]):
            model = self._models[side]
            configuration = self._model_configuration(side, env_index, current)
            posture_task = self.PostureTask(cost=self.posture_cost)
            posture_task.set_target(configuration.q.copy())

            if command_is_relative:
                target_position_world = measured_pose[env_index, :3] + command[env_index, :3]
                target_rotation_world = target_rotations[env_index] @ measured_rotations[env_index]
            else:
                target_position_world = command[env_index, :3]
                target_rotation_world = target_rotations[env_index]

            task_error = np.concatenate(
                (
                    target_position_world - measured_pose[env_index, :3],
                    self.provider.pin.log3(
                        target_rotation_world @ measured_rotations[env_index].T
                    ),
                )
            )

            # Pink's QP machinery and Pinocchio limits are retained, while the
            # task uses the calibrated Adam-world Jacobian already validated
            # against Isaac's imported articulation. Raw URDF frame tasks do
            # not match this asset's imported wrist axes closely enough.
            task_base = self.Task

            class AdamWorldFrameTask(task_base):
                def __init__(self, error, jacobian, costs):
                    super().__init__(cost=costs, gain=1.0, lm_damping=1e-6)
                    self._error = error
                    self._jacobian = jacobian

                def compute_error(self, _configuration):
                    return self._error

                def compute_jacobian(self, _configuration):
                    # Pink's QP objective applies its own task-error sign.
                    # The calibrated provider Jacobian is supplied directly;
                    # negating it makes requested +Z motion execute as -Z.
                    return self._jacobian

                def __repr__(self):
                    return "AdamWorldFrameTask(6D)"

            world_task = AdamWorldFrameTask(
                task_error,
                world_jacobians[env_index],
                np.asarray(
                    [self.position_cost] * 3 + [self.orientation_cost] * 3,
                    dtype=np.float64,
                ),
            )

            velocity = self.pink.solve_ik(
                configuration,
                (world_task, posture_task),
                self.provider.env.step_dt,
                solver=self.qp_solver,
                damping=self.damping,
                safety_break=False,
            )
            q_next = self.provider.pin.integrate(
                model,
                configuration.q,
                velocity * self.provider.env.step_dt,
            )
            arm_target = np.asarray(
                [q_next[self._q_indices[side][name]] for name in arm_names],
                dtype=np.float64,
            )
            arm_target = current[env_index] + np.clip(
                arm_target - current[env_index],
                -self.max_joint_delta,
                self.max_joint_delta,
            )
            results.append(arm_target)
        result = np.asarray(results, dtype=np.float32)
        commanded = self._commanded_joint_pos.get(side)
        if commanded is None or commanded.shape != result.shape:
            commanded = current.astype(np.float32).copy()
        # Accumulate Pink's safe differential correction on the last actuator
        # target so gravity sag does not erase a keyboard jog every frame.
        result = commanded + (result - current)
        if self.max_commanded_joint_error is not None:
            result = current + np.clip(
                result - current,
                -self.max_commanded_joint_error,
                self.max_commanded_joint_error,
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("Pink IK produced a non-finite arm target")
        self._commanded_joint_pos[side] = result.copy()
        return result

    def sync_commanded_joint_pos(self, side: str, commanded_joint_pos: np.ndarray) -> None:
        """Pink is stateless; validate compatibility with the shared interface."""
        values = np.asarray(commanded_joint_pos)
        if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
            raise ValueError("Synchronized IK joint target must be finite with shape (batch, 7)")
        self._commanded_joint_pos[side] = values.astype(np.float32).copy()


class MinkAdamUIKSolver:
    """PND-tuned MuJoCo/Mink QP IK while Isaac Lab remains the simulator.

    The solver owns a persistent Mink configuration, just like PND's
    ``AdamMinkBase``.  Isaac supplies measured wrist poses and joint feedback;
    only the resulting Adam-U arm joint targets are returned to Isaac.

    Direct teleoperation commands already describe ``wristRoll*`` poses, so
    the mocap/controller-frame wrist offset is disabled by default.  Set
    ``apply_controller_frame_offset`` only when the input pose is in PND's
    controller/tracker frame.
    """

    PND_SOLVER = "daqp"
    PND_DAMPING = 3e-1
    PND_ITERATIONS = 3
    PND_ERROR_THRESHOLD = 1e-3
    PND_WRIST_POSITION_COST = 20.0
    PND_WRIST_ORIENTATION_COST = 18.0
    PND_SOFT_ORIENTATION_COST = 2.0
    PND_LM_DAMPING = 1.0
    PND_WRIST_OFFSET_WXYZ = np.asarray((0.866, 0.0, -0.5, 0.0), dtype=np.float64)

    def __init__(
        self,
        provider: "IsaacAdamUKinematicsProvider",
        model_path,
        *,
        max_joint_delta: float = 0.005,
        max_commanded_joint_error: float | None = 0.10,
        solver: str = PND_SOLVER,
        damping: float = PND_DAMPING,
        iterations: int = PND_ITERATIONS,
        apply_controller_frame_offset: bool = False,
        collision_avoidance: bool = True,
    ) -> None:
        from pathlib import Path

        import mink
        import mujoco
        import qpsolvers

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Official Adam-U MJCF not found: {path}. Clone pndbotics/pnd_models "
                "and pass --mink-model /path/to/pnd_models/adam_u/adam_u.xml"
            )
        if solver not in qpsolvers.available_solvers:
            raise ValueError(
                f"Mink QP solver {solver!r} is unavailable; installed: "
                f"{qpsolvers.available_solvers}"
            )
        if max_joint_delta <= 0 or damping < 0 or iterations < 1:
            raise ValueError("Mink step, damping, or iteration count is invalid")

        self.provider = provider
        self.mink = mink
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.max_joint_delta = float(max_joint_delta)
        self.max_commanded_joint_error = (
            None if max_commanded_joint_error is None else float(max_commanded_joint_error)
        )
        self.qp_solver = solver
        self.damping = float(damping)
        self.iterations = int(iterations)
        self.apply_controller_frame_offset = bool(apply_controller_frame_offset)
        self.collision_avoidance = bool(collision_avoidance)
        self._states: dict[tuple[str, int], dict[str, object]] = {}
        self._translation_priority = False
        self._translation_orientation_cost = self.PND_SOFT_ORIENTATION_COST

        from configs.joint_state import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

        self._arm_names = {
            "left": LEFT_ARM_JOINT_NAMES,
            "right": RIGHT_ARM_JOINT_NAMES,
        }
        self._qpos_indices: dict[str, int] = {}
        for names in self._arm_names.values():
            for name in names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    raise ValueError(f"Official Adam-U MJCF is missing joint {name!r}")
                self._qpos_indices[name] = int(self.model.jnt_qposadr[joint_id])
        for body in ("wristRollLeft", "wristRollRight"):
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
                raise ValueError(f"Official Adam-U MJCF is missing body {body!r}")

    def _geom_group(self, names):
        """Resolve PND config labels as geom names or all geoms on a body.

        PND's current MJCF leaves collision geoms unnamed while its YAML uses
        link/body labels.  Mink itself accepts geom IDs, so expand body labels
        explicitly instead of depending on an older named-geom export.
        """
        geom_ids = []
        for name in names:
            geom_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id >= 0:
                geom_ids.append(int(geom_id))
                continue
            body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                raise ValueError(f"PND collision label {name!r} is neither a geom nor body")
            geom_ids.extend(np.flatnonzero(self.model.geom_bodyid == body_id).tolist())
        if not geom_ids:
            raise ValueError(f"PND collision group {tuple(names)!r} contains no geometry")
        return tuple(geom_ids)

    def _limits(self, side: str):
        limits = [self.mink.ConfigurationLimit(self.model)]
        if self.collision_avoidance:
            collision_groups = (
                (
                    ("wristYawLeft", "wristYawRight", "shoulderYawLeft", "shoulderYawRight"),
                    ("torso",),
                ),
                (("wristYawLeft",), ("wristYawRight",)),
            )
            try:
                limits.extend(
                    self.mink.CollisionAvoidanceLimit(
                        self.model,
                        geom_pairs=[
                            (self._geom_group(pair[0]), self._geom_group(pair[1]))
                        ],
                        minimum_distance_from_collisions=0.02,
                        collision_detection_distance=0.03,
                    )
                    for pair in collision_groups
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "PND collision geometry groups do not match this MJCF; "
                    "use the official pnd_models/adam_u/adam_u.xml"
                ) from exc

        # PND enables 10 rad/s for shoulder pitch/roll and elbow and leaves
        # yaw/wrist entries commented.  We enable the documented 10/9 values
        # for the complete active arm, while freezing the opposite arm.
        velocity = {}
        active_names = set(self._arm_names[side])
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == self.mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not name:
                continue
            # Right-arm-only teleoperation must not let the internal QP solve
            # through an uncommanded waist, neck, hand, or opposite arm.
            velocity[name] = (
                (9.0 if name.startswith("wrist") else 10.0)
                if name in active_names
                else 1e-6
            )
        limits.append(self.mink.VelocityLimit(self.model, velocity))
        return limits

    def _new_state(self, side: str, env_index: int, current: np.ndarray, measured_pose: np.ndarray):
        configuration = self.mink.Configuration(self.model)
        for arm_name, value in zip(self._arm_names[side], current, strict=True):
            configuration.data.qpos[self._qpos_indices[arm_name]] = float(value)
        self.mujoco.mj_forward(self.model, configuration.data)
        body_name = "wristRollLeft" if side == "left" else "wristRollRight"
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        mink_pos = configuration.data.xpos[body_id].copy()
        mink_quat_wxyz = configuration.data.xquat[body_id].copy()
        from scipy.spatial.transform import Rotation

        mink_rot = Rotation.from_quat(mink_quat_wxyz, scalar_first=True).as_matrix()
        isaac_rot = _rot6d_to_matrix_np(measured_pose[3:])
        rotation_mink_from_isaac = mink_rot @ isaac_rot.T
        translation_mink_from_isaac = mink_pos - rotation_mink_from_isaac @ measured_pose[:3]
        task = self.mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=self.PND_WRIST_POSITION_COST,
            orientation_cost=self.PND_WRIST_ORIENTATION_COST,
            lm_damping=self.PND_LM_DAMPING,
        )
        state = {
            "configuration": configuration,
            "task": task,
            "limits": self._limits(side),
            "R_mi": rotation_mink_from_isaac,
            "t_mi": translation_mink_from_isaac,
        }
        self._states[(side, env_index)] = state
        return state

    def solve(self, side, eef_command_9d, current_arm, *, command_is_relative):
        from scipy.spatial.transform import Rotation

        if command_is_relative:
            raise ValueError("Mink backend expects persistent absolute Cartesian targets")
        command = np.asarray(eef_command_9d, dtype=np.float64)
        current = np.asarray(current_arm, dtype=np.float64)
        if command.ndim != 2 or command.shape[1] != 9 or current.shape != (command.shape[0], 7):
            raise ValueError("Mink IK expects EEF[batch,9] and arm[batch,7]")
        if not np.all(np.isfinite(command)) or not np.all(np.isfinite(current)):
            raise ValueError("Mink IK rejected NaN or infinite input")
        measured_pose, _ = self.provider(side, current)
        results = []
        for env_index in range(command.shape[0]):
            state = self._states.get((side, env_index))
            if state is None:
                state = self._new_state(side, env_index, current[env_index], measured_pose[env_index])
            configuration = state["configuration"]
            state["task"].set_orientation_cost(
                self._translation_orientation_cost
                if self._translation_priority
                else self.PND_WRIST_ORIENTATION_COST
            )
            target_rot_i = _rot6d_to_matrix_np(command[env_index, 3:])
            target_pos_m = state["R_mi"] @ command[env_index, :3] + state["t_mi"]
            target_rot_m = state["R_mi"] @ target_rot_i
            if self.apply_controller_frame_offset:
                offset = Rotation.from_quat(
                    self.PND_WRIST_OFFSET_WXYZ, scalar_first=True
                ).as_matrix()
                target_rot_m = target_rot_m @ offset
            target_quat = Rotation.from_matrix(target_rot_m).as_quat(scalar_first=True)
            state["task"].set_target(
                self.mink.SE3.from_rotation_and_translation(
                    self.mink.SO3(target_quat), target_pos_m
                )
            )
            q_before = np.asarray(
                [
                    configuration.data.qpos[self._qpos_indices[n]]
                    for n in self._arm_names[side]
                ],
                dtype=np.float64,
            )
            for _ in range(self.iterations):
                velocity = self.mink.solve_ik(
                    configuration,
                    (state["task"],),
                    self.model.opt.timestep,
                    solver=self.qp_solver,
                    damping=self.damping,
                    limits=state["limits"],
                    safety_break=False,
                )
                configuration.integrate_inplace(velocity, self.model.opt.timestep)
            raw = np.asarray(
                [configuration.data.qpos[self._qpos_indices[n]] for n in self._arm_names[side]]
            )
            # Advance from Mink's persistent commanded state, not from the
            # gravity-sagged measured state. Rebasing on q_measured capped the
            # actuator lead at max_joint_delta forever, so the arm could not
            # generate enough PD effort to follow vertical keyboard commands.
            target = q_before + np.clip(
                raw - q_before, -self.max_joint_delta, self.max_joint_delta
            )
            if self.max_commanded_joint_error is not None:
                target = current[env_index] + np.clip(
                    target - current[env_index],
                    -self.max_commanded_joint_error,
                    self.max_commanded_joint_error,
                )
            results.append(target)
        result = np.asarray(results, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("Mink IK produced a non-finite arm target")
        return result

    def set_translation_priority(
        self, active: bool, *, orientation_cost: float | None = None
    ) -> None:
        """Soften wrist orientation while the operator requests XYZ motion."""
        if orientation_cost is not None:
            if orientation_cost < 0:
                raise ValueError("Translation orientation cost must be non-negative")
            self._translation_orientation_cost = float(orientation_cost)
        self._translation_priority = bool(active)

    def gravity_compensation(self, side: str, current_arm: np.ndarray) -> np.ndarray:
        """Return MJCF gravity/bias torques for the active Adam-U arm."""
        current = np.asarray(current_arm, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != 7 or not np.all(np.isfinite(current)):
            raise ValueError("Gravity compensation expects finite arm[batch,7]")
        efforts = []
        for arm in current:
            data = self.mujoco.MjData(self.model)
            for name, value in zip(self._arm_names[side], arm, strict=True):
                data.qpos[self._qpos_indices[name]] = float(value)
            self.mujoco.mj_forward(self.model, data)
            torque = []
            for name in self._arm_names[side]:
                joint_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
                )
                dof_id = int(self.model.jnt_dofadr[joint_id])
                torque.append(float(data.qfrc_bias[dof_id]))
            efforts.append(torque)
        result = np.asarray(efforts, dtype=np.float32)
        # Match Adam-U actuator capabilities; Isaac applies its own final cap.
        limits = np.asarray((40.0, 40.0, 40.0, 30.0, 6.4, 6.4, 6.4), dtype=np.float32)
        return np.clip(result, -limits, limits)

    def sync_commanded_joint_pos(self, side: str, commanded_joint_pos: np.ndarray) -> None:
        values = np.asarray(commanded_joint_pos, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
            raise ValueError("Synchronized IK joint target must be finite with shape (batch, 7)")
        # Preserve Cartesian frame calibration, but synchronize the persistent
        # Mink configuration to the command actually accepted by Isaac.
        for env_index, arm in enumerate(values):
            state = self._states.get((side, env_index))
            if state is None:
                continue
            for name, value in zip(self._arm_names[side], arm, strict=True):
                state["configuration"].data.qpos[self._qpos_indices[name]] = float(value)
            self.mujoco.mj_forward(self.model, state["configuration"].data)


class PinocchioAdamUKinematicsProvider(IsaacAdamUKinematicsProvider):
    """Adam-U FK pose from Isaac plus an independent URDF Jacobian in world axes.

    This is useful for teleoperation on Isaac/PhysX versions whose articulation
    Jacobian frame is ambiguous for an imported, rotated fixed-base robot.
    """

    def __init__(self, env, urdf_path=None) -> None:
        super().__init__(env)
        import pinocchio as pin
        from configs.joint_state import DEFAULT_URDF_PATH

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path or DEFAULT_URDF_PATH))
        self.data = self.model.createData()
        self._pin_q_indices = {}
        for name in self.robot.data.joint_names:
            joint_id = self.model.getJointId(name)
            if joint_id and self.model.joints[joint_id].nq == 1:
                self._pin_q_indices[name] = self.model.joints[joint_id].idx_q
        self._pin_frame_ids = {}
        for side, frame_name in _WRIST_BODY_NAMES.items():
            frame_id = self.model.getFrameId(frame_name)
            if frame_id >= len(self.model.frames):
                raise ValueError(f"Adam-U Pinocchio wrist frame is missing: {frame_name}")
            self._pin_frame_ids[side] = frame_id

    def world_position_to_root(self, position_env):
        """Pinocchio Jacobian below is rotated into environment-world axes."""
        import torch

        return torch.as_tensor(position_env, device=self.env.device, dtype=torch.float32)

    def get_torch_state(self, side: str):
        import torch
        from isaaclab.utils.math import matrix_from_quat
        from configs.joint_state import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

        body_idx = self._body_indices[side]
        body_pos_w = self._torch(self.robot.data.body_pos_w)[:, body_idx]
        body_quat_w = self._torch(self.robot.data.body_quat_w)[:, body_idx]
        env_origins = self._torch(self.env.scene.env_origins)

        all_joint_pos = self._torch(self.robot.data.joint_pos).detach().cpu().numpy()
        jacobians = []
        for env_index in range(all_joint_pos.shape[0]):
            q = self.pin.neutral(self.model)
            for sim_index, name in enumerate(self.robot.data.joint_names):
                q_index = self._pin_q_indices.get(name)
                if q_index is not None:
                    q[q_index] = float(all_joint_pos[env_index, sim_index])
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            jacobian_model = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self._pin_frame_ids[side],
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            arm_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
            columns = [self._pin_q_indices[name] for name in arm_names]
            jacobians.append(jacobian_model[:, columns])
        jacobian = torch.as_tensor(np.stack(jacobians), device=self.env.device, dtype=torch.float32)
        root_rotation_w = matrix_from_quat(self._torch(self.robot.data.root_quat_w))
        jacobian = torch.cat(
            (
                torch.bmm(root_rotation_w, jacobian[:, :3, :]),
                torch.bmm(root_rotation_w, jacobian[:, 3:, :]),
            ),
            dim=1,
        )
        return body_pos_w - env_origins, body_quat_w, jacobian


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Isaac Lab 6 xyzw quaternion(s) to first-two-rows rot6d."""
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
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


def build_eef_9d(pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build XYZ+rot6d pose vector(s), shape (9,) or (N, 9)."""
    pos = np.asarray(pos_xyz, dtype=np.float32)
    rot6d = quat_xyzw_to_rot6d(quat_xyzw).astype(np.float32)
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
