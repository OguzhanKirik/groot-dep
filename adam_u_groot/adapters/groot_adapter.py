"""Convert Isaac Lab observations/actions to the GR00T N1.7 Policy API format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from adapters.eef_pose import G1AdamWorkspaceTransform, IDENTITY_EEF_9D, read_wrist_eef_9d
from adapters.adam_u_action_adapter import (
    AdamUCommand,
    RealG1ToAdamUAdapter,
    expand_hand_synergies_for_isaac,
)
from adapters.joint_state_reader import JointStateReader, make_right_arm_reader
from adapters.virtual_g1_state import VirtualG1StateAdapter
from configs.adam_u_action_mapping import AdamUActionMappingConfig
from configs.constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEFAULT_TASK_INSTRUCTION,
    FRONT_CAMERA_NAME,
    GROOT_VIDEO_KEYS,
    WRIST_CAMERA_NAME,
)
from configs.groot_schemas import GrootSchema, get_groot_schema

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_execution_scope(
    command: AdamUCommand,
    *,
    scope: str,
    hold_body: np.ndarray,
    hold_hands: np.ndarray,
) -> AdamUCommand:
    """Mask policy outputs while preserving the fixed Adam-U interface."""
    if scope == "full":
        return command
    if scope != "right_arm_hand":
        raise ValueError(f"Unsupported Adam-U execution scope: {scope!r}")
    body = np.asarray(command.body, dtype=np.float32).copy()
    hands = np.asarray(command.hands, dtype=np.float32).copy()
    held_body = np.asarray(hold_body, dtype=np.float32)
    held_hands = np.asarray(hold_hands, dtype=np.float32)
    if held_body.shape != body.shape or held_hands.shape != hands.shape:
        raise ValueError("Held Adam-U command shapes must match the policy command")
    body[:, :12] = held_body[:, :12]  # waist, neck, left arm
    hands[:, :6] = held_hands[:, :6]  # left hand
    masked = AdamUCommand(
        body=body,
        hands=hands,
        body_joint_names=command.body_joint_names,
        hand_command_names=command.hand_command_names,
        ignored_outputs=tuple(
            dict.fromkeys((*command.ignored_outputs, "waist", "left_arm", "left_hand"))
        ),
        clamped_fields=command.clamped_fields,
    )
    masked.validate()
    return masked


class GrootAdapter:
    """Bridge between Adam-U Isaac Lab env and GR00T Policy API."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
        execution_horizon: int = 8,
        schema: str | GrootSchema = "real_g1",
        action_mapping_config: AdamUActionMappingConfig | None = None,
        ik_solver=None,
        legacy_g1_real: bool = False,
        workspace_transform: G1AdamWorkspaceTransform | None = None,
        virtual_g1_state: VirtualG1StateAdapter | None = None,
        execution_scope: str = "right_arm_hand",
    ):
        self.env = env
        self.task_instruction = task_instruction
        self.execution_horizon = execution_horizon
        self.schema = schema if isinstance(schema, GrootSchema) else get_groot_schema(schema)
        self.legacy_g1_real = bool(legacy_g1_real)
        self.workspace_transform = workspace_transform or G1AdamWorkspaceTransform()
        self.virtual_g1_state = virtual_g1_state
        if execution_scope not in ("right_arm_hand", "full"):
            raise ValueError(f"Unsupported Adam-U execution scope: {execution_scope!r}")
        self.execution_scope = execution_scope
        self._hold_body: np.ndarray | None = None
        self._hold_hands: np.ndarray | None = None
        self.joint_state = make_right_arm_reader(env)
        self._waist_state = JointStateReader(env, group="waist")
        self._neck_state = JointStateReader(env, group="neck")
        self._left_arm_state = JointStateReader(env, group="left_arm")
        self._left_hand_state = JointStateReader(env, group="left_hand")
        self._right_hand_state = JointStateReader(env, group="right_hand")
        self.action_adapter = (
            RealG1ToAdamUAdapter(action_mapping_config, ik_solver=ik_solver)
            if self.schema.name == "real_g1"
            else None
        )
        self._logged_video_stats = False

    def get_right_arm_state(self) -> torch.Tensor:
        """Return right-arm joint positions, shape (num_envs, 7)."""
        return self.joint_state.get_positions()

    def get_joint_state(self, group: str = "right_arm") -> torch.Tensor:
        """Return joint positions for any URDF group."""
        if group == self.joint_state.group:
            return self.joint_state.get_positions()
        return JointStateReader(self.env, group=group).get_positions()

    def get_camera_rgb(self, camera_name: str) -> torch.Tensor:
        """Return RGB images as uint8 tensor, shape (num_envs, H, W, 3)."""
        camera = self.env.scene[camera_name]
        rgb = camera.data.output["rgb"]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb

    def _pack_video(self, num_envs: int) -> dict[str, np.ndarray]:
        video: dict[str, np.ndarray] = {}
        if self.schema.name == "adam_u":
            for sensor_name, groot_key in GROOT_VIDEO_KEYS.items():
                rgb = self.get_camera_rgb(sensor_name).detach().cpu().numpy()
                video[groot_key] = rgb[:, np.newaxis, ...]
            return video

        # REAL_G1: single ego_view, two-frame horizon (duplicate current front frame).
        if FRONT_CAMERA_NAME in self.env.scene.keys():
            front = self.get_camera_rgb(FRONT_CAMERA_NAME).detach().cpu().numpy()
            if not self._logged_video_stats:
                print(
                    f"[INFO] GR00T ego_view: real RGB camera, shape={front.shape}, "
                    f"min={front.min()}, max={front.max()}, mean={front.mean():.1f}",
                    flush=True,
                )
                self._logged_video_stats = True
        else:
            # Preserve the required tensor contract for camera-free diagnostic
            # runs; normal GR00T evaluation creates the real front RGB sensor.
            front = np.zeros(
                (num_envs, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                dtype=np.uint8,
            )
        ego = np.repeat(front[:, np.newaxis, ...], self.schema.video_horizon, axis=1)
        video["ego_view"] = ego
        return video

    def _pack_state(self, num_envs: int) -> dict[str, np.ndarray]:
        right_arm = self.get_right_arm_state().detach().cpu().numpy().astype(np.float32)
        left_arm = self._left_arm_state.get_positions().detach().cpu().numpy().astype(np.float32)
        waist = self._waist_state.get_positions().detach().cpu().numpy().astype(np.float32)
        neck = self._neck_state.get_positions().detach().cpu().numpy().astype(np.float32)
        left_hand_joints = self._left_hand_state.get_positions().detach().cpu().numpy().astype(np.float32)
        right_hand_joints = self._right_hand_state.get_positions().detach().cpu().numpy().astype(np.float32)
        if self.schema.name == "adam_u":
            # Exact inverse of expand_hand_synergies_for_isaac for the primary
            # joint in each Adam-U hand synergy.
            left_hand = left_hand_joints[:, (0, 3, 4, 6, 8, 10)]
            right_hand = right_hand_joints[:, (0, 3, 4, 6, 8, 10)]
            native = {
                "waist": waist,
                "neck": neck,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "left_hand": left_hand,
                "right_hand": right_hand,
            }
            return {key: native[key][:, np.newaxis, :] for key in self.schema.state_keys}

        # REAL_G1 hand state order mirrors the action semantics documented in
        # adam_u_action_mapping.py: four finger flexions, thumb opposition, and
        # two thumb-flexion components. Adam-U exposes 12 physical finger joints.
        left_hand = np.stack(
            (
                -left_hand_joints[:, 4], -left_hand_joints[:, 6],
                -left_hand_joints[:, 8], -left_hand_joints[:, 10],
                left_hand_joints[:, 0], left_hand_joints[:, 2], left_hand_joints[:, 3],
            ), axis=1,
        )
        right_hand = np.stack(
            (
                right_hand_joints[:, 4], right_hand_joints[:, 6],
                right_hand_joints[:, 8], right_hand_joints[:, 10],
                right_hand_joints[:, 0], -right_hand_joints[:, 2], -right_hand_joints[:, 3],
            ), axis=1,
        )
        left_wrist_eef = read_wrist_eef_9d(self.env, "left", num_envs)
        right_wrist_eef = read_wrist_eef_9d(self.env, "right", num_envs)
        # REAL_G1 normalization and relative-action postprocessing operate in
        # the G1 body-relative workspace, not Adam-U's Isaac world frame.
        left_wrist_eef = self.workspace_transform.world_to_g1_pose(left_wrist_eef)
        right_wrist_eef = self.workspace_transform.world_to_g1_pose(right_wrist_eef)
        if self.virtual_g1_state is not None:
            left_arm, right_arm = self.virtual_g1_state.update(left_wrist_eef, right_wrist_eef)
            # Use G1 FK for the complete EEF state so virtual G1 joints and
            # wrist position/orientation describe one internally consistent
            # configuration. Adam tool-frame orientation is not a G1 state.
            left_wrist_eef = self.virtual_g1_state.last_eef_poses["left"]
            right_wrist_eef = self.virtual_g1_state.last_eef_poses["right"]
            max_error = max(self.virtual_g1_state.last_errors.values())
            if max_error > 0.05:
                print(
                    "[WARN] Virtual G1 wrist-state consistency error: "
                    f"left={self.virtual_g1_state.last_errors['left']:.3f} m, "
                    f"right={self.virtual_g1_state.last_errors['right']:.3f} m",
                    flush=True,
                )

        state: dict[str, np.ndarray] = {}
        for key, dim in self.schema.state_dims.items():
            if key == "right_arm":
                values = right_arm
            elif key == "left_arm":
                values = left_arm
            elif key == "waist":
                # Adam: roll, pitch, yaw. REAL_G1: yaw, roll, pitch.
                values = waist[:, (2, 0, 1)]
            elif key == "right_hand":
                values = right_hand[:, :dim] if right_hand.shape[1] >= dim else np.pad(
                    right_hand, ((0, 0), (0, dim - right_hand.shape[1]))
                )
            elif key == "left_hand":
                values = left_hand[:, :dim] if left_hand.shape[1] >= dim else np.pad(
                    left_hand, ((0, 0), (0, dim - left_hand.shape[1]))
                )
            elif key == "left_wrist_eef_9d":
                values = left_wrist_eef[:, :dim]
            elif key == "right_wrist_eef_9d":
                values = right_wrist_eef[:, :dim]
            elif key.endswith("_eef_9d"):
                values = np.tile(IDENTITY_EEF_9D[:dim], (num_envs, 1))
            else:
                values = np.zeros((num_envs, dim), dtype=np.float32)
            state[key] = values[:, np.newaxis, :]
        return state

    def build_observation(self, task_instruction: str | None = None) -> dict[str, Any]:
        """Build GR00T Policy API observation dict with batch and time dims."""
        task = task_instruction or self.task_instruction
        num_envs = self.env.num_envs

        video = self._pack_video(num_envs)
        state = self._pack_state(num_envs)
        language = {self.schema.language_api_key: [[task] for _ in range(num_envs)]}

        return {"video": video, "state": state, "language": language}

    def _current_body(self) -> np.ndarray:
        groups = (
            self._waist_state.get_positions(),
            self._neck_state.get_positions(),
            self._left_arm_state.get_positions(),
            self.joint_state.get_positions(),
        )
        return torch.cat(groups, dim=1).detach().cpu().numpy().astype(np.float32)

    def action_to_adam_u(self, groot_action: dict[str, Any], step_index: int = 0) -> AdamUCommand:
        """Return the public Adam-U body[19] and hands[12] low-level command."""
        if self.action_adapter is None:
            raise RuntimeError("The body[19]+hands[12] adapter is defined for the REAL_G1 compatibility schema")
        current_body = self._current_body()
        if self._hold_body is None:
            self._hold_body = current_body.copy()
            left = self._left_hand_state.get_positions().detach().cpu().numpy()
            right = self._right_hand_state.get_positions().detach().cpu().numpy()
            # Primary physical joint of each six-channel Adam-U hand synergy.
            indices = (0, 3, 4, 6, 8, 10)
            self._hold_hands = np.concatenate((left[:, indices], right[:, indices]), axis=1).astype(
                np.float32
            )
        command = self.action_adapter.adapt(
            groot_action,
            step_index=step_index,
            current_body=current_body,
        )
        masked = apply_execution_scope(
            command,
            scope=self.execution_scope,
            hold_body=self._hold_body,
            hold_hands=self._hold_hands,
        )
        ik_solver = self.action_adapter.ik_solver
        if ik_solver is not None and hasattr(ik_solver, "sync_commanded_joint_pos"):
            # Prevent integrator windup when URDF limits or the execution mask
            # alter the raw IK result downstream.
            ik_solver.sync_commanded_joint_pos("left", masked.body[:, 5:12])
            ik_solver.sync_commanded_joint_pos("right", masked.body[:, 12:19])
        return masked

    def action_to_env(self, groot_action: dict[str, Any], step_index: int = 0) -> torch.Tensor:
        """Convert body/hands commands into Isaac's body[19]+finger-joints[24] vector."""
        if self.legacy_g1_real:
            value = groot_action["right_arm"]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3:
                value = value[:, step_index, :]
            if value.ndim == 1:
                value = value[None, :]
            if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
                raise ValueError("g1_real right_arm action must be finite with shape (batch, 7)")
            return torch.as_tensor(value.copy(), device=self.env.device, dtype=torch.float32)
        if self.action_adapter is None:
            def at_step(key: str, width: int) -> np.ndarray:
                value = groot_action[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value, dtype=np.float32)
                if value.ndim == 3:
                    value = value[:, step_index, :]
                if value.ndim == 1:
                    value = value[None, :]
                if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
                    raise ValueError(f"Native Adam-U action {key!r} must be finite with shape (batch, {width})")
                return value

            body = np.concatenate(
                (at_step("waist", 3), at_step("neck", 2), at_step("left_arm", 7), at_step("right_arm", 7)),
                axis=1,
            )
            hands = np.concatenate((at_step("left_hand", 6), at_step("right_hand", 6)), axis=1)
            lower, upper, _ = AdamUActionMappingConfig().body_limits
            body = np.clip(body, lower, upper)
            hands = np.clip(hands, np.zeros(12), np.tile(np.asarray((1.1, 1.2, 1.7, 1.7, 1.7, 1.7)), 2))
            sim_action = np.concatenate((body, expand_hand_synergies_for_isaac(hands)), axis=1)
            return torch.as_tensor(sim_action.copy(), device=self.env.device, dtype=torch.float32)
        command = self.action_to_adam_u(groot_action, step_index)
        finger_targets = expand_hand_synergies_for_isaac(command.hands)
        sim_action = np.concatenate((command.body, finger_targets), axis=1)
        return torch.as_tensor(sim_action.copy(), device=self.env.device, dtype=torch.float32)

    def get_execution_horizon(self, groot_action: dict[str, Any]) -> int:
        """Return how many action steps are available in the current GR00T chunk."""
        if self.legacy_g1_real:
            action_key = "right_arm"
        elif self.action_adapter is None:
            action_key = self.schema.action_keys[0]
        else:
            action_key = (
                "left_arm"
                if self.action_adapter.config.control_mode == "joint_space"
                else "left_wrist_eef_9d"
            )
        action_arr = groot_action[action_key]
        if isinstance(action_arr, torch.Tensor):
            if action_arr.ndim == 3:
                return min(action_arr.shape[1], self.execution_horizon)
            return 1
        if action_arr.ndim == 3:
            return min(action_arr.shape[1], self.execution_horizon)
        return 1
