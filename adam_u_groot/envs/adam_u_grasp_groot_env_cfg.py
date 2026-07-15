# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Adam-U grasp environment configured for GR00T N1.7 evaluation."""

from __future__ import annotations

import os
import sys

import importlib.util

# Allow imports from adam_u_rl and adam_u_groot when this module is loaded directly.
_ADAM_U_GROOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ADAM_U_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "adam_u_rl"))
if _ADAM_U_RL_ROOT not in sys.path:
    sys.path.insert(0, _ADAM_U_RL_ROOT)
if _ADAM_U_GROOT_ROOT not in sys.path:
    sys.path.insert(0, _ADAM_U_GROOT_ROOT)


def _load_adam_u_rl_env_cfg_module():
    import importlib

    saved_path = list(sys.path)
    popped_modules = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == "envs" or key.startswith("envs.")
    }
    try:
        sys.path[:] = [p for p in sys.path if p not in {_ADAM_U_GROOT_ROOT, _ADAM_U_RL_ROOT}]
        sys.path.insert(0, _ADAM_U_RL_ROOT)
        return importlib.import_module("envs.adam_u_grasp_env_cfg")
    finally:
        sys.path[:] = saved_path
        for key, module in popped_modules.items():
            if key not in sys.modules:
                sys.modules[key] = module
        if _ADAM_U_RL_ROOT not in sys.path:
            sys.path.insert(0, _ADAM_U_RL_ROOT)
        if _ADAM_U_GROOT_ROOT not in sys.path:
            sys.path.insert(0, _ADAM_U_GROOT_ROOT)


_adam_u_rl_env_cfg = _load_adam_u_rl_env_cfg_module()
AdamUGraspSceneCfg = _adam_u_rl_env_cfg.AdamUGraspSceneCfg
EventsCfg = _adam_u_rl_env_cfg.EventsCfg

from isaaclab.assets import RigidObjectCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from configs.constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FRONT_CAMERA_NAME,
    RIGHT_ARM_JOINT_NAMES,
    WRIST_CAMERA_NAME,
)
from envs.scene_layout import (
    FRONT_CAMERA_POS,
    FRONT_CAMERA_ROT,
    PLACE_TARGET_POS,
    VIEWER_EYE,
    VIEWER_LOOKAT,
)

##
# Scene
##


@configclass
class AdamUGraspGrootSceneCfg(AdamUGraspSceneCfg):
    """Grasp scene with RGB cameras and a placement target for GR00T pick-and-place."""

    # Visual placement target on the table (kinematic marker).
    place_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PLACE_TARGET_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
                metallic=0.0,
                roughness=0.9,
            ),
        ),
    )

    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        offset=CameraCfg.OffsetCfg(
            pos=FRONT_CAMERA_POS,
            rot=FRONT_CAMERA_ROT,
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wristRollRight/wrist_cam",
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )


@configclass
class AdamUGraspGrootSceneNoCamCfg(AdamUGraspSceneCfg):
    """Grasp scene without RTX camera sensors (viewport-only; avoids PhysX invalidation on startup)."""

    place_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PLACE_TARGET_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
                metallic=0.0,
                roughness=0.9,
            ),
        ),
    )


##
# MDP
##


@configclass
class GrootActionsCfg:
    """Right-arm joint targets only (minimal GR00T control)."""

    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(RIGHT_ARM_JOINT_NAMES),
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class GrootObservationsCfg:
    """Proprioceptive observations for debugging; images are read via GrootAdapter."""

    @configclass
    class ProprioCfg(ObsGroup):
        right_arm_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(RIGHT_ARM_JOINT_NAMES))},
        )
        right_arm_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(RIGHT_ARM_JOINT_NAMES))},
        )
        object_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ProprioCfg = ProprioCfg()


@configclass
class AdamUGraspGrootEnvCfg(ManagerBasedEnvCfg):
    """Adam-U grasp task for GR00T closed-loop evaluation (no RL rewards)."""

    scene: AdamUGraspGrootSceneCfg = AdamUGraspGrootSceneCfg(num_envs=1, env_spacing=4.0)
    observations: GrootObservationsCfg = GrootObservationsCfg()
    actions: GrootActionsCfg = GrootActionsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0

        self.viewer.eye = VIEWER_EYE
        self.viewer.lookat = VIEWER_LOOKAT

        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.device = "cuda:0"
        self.sim.disable_contact_processing = False


def make_groot_env_cfg(
    *,
    num_envs: int = 1,
    env_spacing: float = 4.0,
    include_cameras: bool = True,
    include_wrist_camera: bool = True,
):
    """Build env cfg; omit scene cameras for zero/random viewport-only runs."""
    cfg = AdamUGraspGrootEnvCfg()
    if include_cameras:
        cfg.scene = AdamUGraspGrootSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        if not include_wrist_camera:
            # REAL_G1 consumes only ``ego_view`` (the front camera), so avoid the
            # cost of creating an unused second RTX render product.
            cfg.scene.wrist_camera = None
        # Safer startup when attaching Replicator render products alongside PhysX views.
        cfg.scene.replicate_physics = False
    else:
        cfg.scene = AdamUGraspGrootSceneNoCamCfg(num_envs=num_envs, env_spacing=env_spacing)
    return cfg


__all__ = [
    "AdamUGraspGrootEnvCfg",
    "AdamUGraspGrootSceneCfg",
    "AdamUGraspGrootSceneNoCamCfg",
    "make_groot_env_cfg",
]
