"""Convert Isaac Lab observations/actions to the GR00T N1.7 Policy API format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from adapters.eef_pose import IDENTITY_EEF_9D, read_wrist_eef_9d
from adapters.joint_state_reader import JointStateReader, make_right_arm_reader
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


class GrootAdapter:
    """Bridge between Adam-U Isaac Lab env and GR00T Policy API."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
        execution_horizon: int = 8,
        schema: str | GrootSchema = "real_g1",
    ):
        self.env = env
        self.task_instruction = task_instruction
        self.execution_horizon = execution_horizon
        self.schema = schema if isinstance(schema, GrootSchema) else get_groot_schema(schema)
        self.joint_state = make_right_arm_reader(env)
        self._waist_state = JointStateReader(env, group="waist")
        self._left_arm_state = JointStateReader(env, group="left_arm")
        self._right_hand_state = JointStateReader(env, group="right_hand")

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
        else:
            # REAL_G1 is a base-checkpoint compatibility shim, not the native
            # Adam-U visual policy. Keep its required tensor contract without
            # creating the RTX sensor that currently deadlocks PhysX startup.
            front = np.zeros(
                (num_envs, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                dtype=np.uint8,
            )
        ego = np.repeat(front[:, np.newaxis, ...], self.schema.video_horizon, axis=1)
        video["ego_view"] = ego
        return video

    def _pack_state(self, num_envs: int) -> dict[str, np.ndarray]:
        if self.schema.name == "adam_u":
            state_np = self.get_right_arm_state().detach().cpu().numpy().astype(np.float32)
            return {"right_arm": state_np[:, np.newaxis, :]}

        right_arm = self.get_right_arm_state().detach().cpu().numpy().astype(np.float32)
        left_arm = self._left_arm_state.get_positions().detach().cpu().numpy().astype(np.float32)
        waist = self._waist_state.get_positions().detach().cpu().numpy().astype(np.float32)
        right_hand = self._right_hand_state.get_positions().detach().cpu().numpy().astype(np.float32)
        left_wrist_eef = read_wrist_eef_9d(self.env, "left", num_envs)
        right_wrist_eef = read_wrist_eef_9d(self.env, "right", num_envs)

        state: dict[str, np.ndarray] = {}
        for key, dim in self.schema.state_dims.items():
            if key == "right_arm":
                values = right_arm
            elif key == "left_arm":
                values = left_arm
            elif key == "waist":
                values = waist
            elif key == "right_hand":
                values = right_hand[:, :dim] if right_hand.shape[1] >= dim else np.pad(
                    right_hand, ((0, 0), (0, dim - right_hand.shape[1]))
                )
            elif key == "left_hand":
                values = np.zeros((num_envs, dim), dtype=np.float32)
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

    def action_to_env(self, groot_action: dict[str, Any], step_index: int = 0) -> torch.Tensor:
        """Convert GR00T action dict to Isaac Lab action tensor."""
        action_key = self.schema.env_action_key
        if action_key not in groot_action:
            raise KeyError(
                f"Expected action key '{action_key}' in GR00T output. Got keys: {list(groot_action.keys())}"
            )

        action_arr = groot_action[action_key]
        if isinstance(action_arr, torch.Tensor):
            action_arr = action_arr.detach().cpu().numpy()

        if action_arr.ndim == 3:
            arm_action = action_arr[:, step_index, :]
        elif action_arr.ndim == 2:
            arm_action = action_arr
        else:
            raise ValueError(f"Unexpected GR00T action shape: {action_arr.shape}")

        device = self.env.device
        return torch.as_tensor(np.array(arm_action, copy=True), device=device, dtype=torch.float32)

    def get_execution_horizon(self, groot_action: dict[str, Any]) -> int:
        """Return how many action steps are available in the current GR00T chunk."""
        action_key = self.schema.env_action_key
        action_arr = groot_action[action_key]
        if isinstance(action_arr, torch.Tensor):
            if action_arr.ndim == 3:
                return min(action_arr.shape[1], self.execution_horizon)
            return 1
        if action_arr.ndim == 3:
            return min(action_arr.shape[1], self.execution_horizon)
        return 1
