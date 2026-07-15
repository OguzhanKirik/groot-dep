#!/usr/bin/env python3
"""Teleoperate Adam-U's right EEF/hand and record successful demonstrations."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_GROOT_ROOT = os.path.join(_REPO_ROOT, "adam_u_groot")
_RL_ROOT = os.path.join(_REPO_ROOT, "adam_u_rl")
for _path in (_REPO_ROOT, _GROOT_ROOT, _RL_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_env_factory():
    path = os.path.join(_GROOT_ROOT, "envs", "adam_u_grasp_groot_env_cfg.py")
    spec = importlib.util.spec_from_file_location("adam_u_teleop_env_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Adam-U environment config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.make_groot_env_cfg


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim window.")
parser.add_argument("--teleop-device", choices=("keyboard", "spacemouse"), default="keyboard")
parser.add_argument("--output", default="logs/teleop/adam_u_demos.hdf5")
parser.add_argument("--task", default="pick up the cube and place it on the green target")
parser.add_argument("--max-steps", type=int, default=100000)
parser.add_argument(
    "--startup-stabilization-steps",
    type=int,
    default=60,
    help="Ignore teleop motion while the arm settles into its initial target.",
)
parser.add_argument(
    "--translation-self-test",
    action="store_true",
    help="Automatically exercise W/S/A/D/Q/E-equivalent world-axis commands and exit.",
)
parser.add_argument(
    "--translation-self-test-keys",
    default="WSADQE",
    help="Subset/order of WSADQE used by --translation-self-test.",
)
parser.add_argument("--position-sensitivity", type=float, default=0.001, help="Metres per simulation step while a motion key is held.")
parser.add_argument("--z-sensitivity-scale", type=float, default=2.0, help="Additional world-Z gain applied to Q/E translation.")
parser.add_argument(
    "--axis-locked-translation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="While jogging XYZ, rebase non-commanded axes to the measured wrist position.",
)
parser.add_argument("--rotation-sensitivity", type=float, default=0.01, help="Radians per simulation step while a rotation key is held.")
parser.add_argument("--ik-joint-step", type=float, default=0.03)
parser.add_argument("--ik-backend", choices=("isaac", "mink", "pink"), default="isaac")
parser.add_argument(
    "--mink-model",
    default=os.environ.get(
        "ADAM_U_MINK_MODEL",
        os.path.join(_REPO_ROOT, "third_party", "pnd_models", "adam_u", "adam_u.xml"),
    ),
    help="PND's official Adam-U MJCF used only for host-side Mink kinematics.",
)
parser.add_argument("--mink-damping", type=float, default=0.01)
parser.add_argument("--mink-iterations", type=int, default=3)
parser.add_argument(
    "--mink-translation-orientation-cost",
    type=float,
    default=0.0,
    help="Soft wrist-orientation cost while an XYZ key is held; PND's 18 is restored afterward.",
)
parser.add_argument(
    "--mink-controller-frame-offset",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply PND's [0.866,0,-0.5,0] WXYZ tracker-to-wrist offset. Keep off for direct wrist poses.",
)
parser.add_argument(
    "--mink-collision-avoidance",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--mink-gravity-compensation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply official-MJCF gravity torque as feed-forward effort in Isaac.",
)
parser.add_argument("--pink-position-cost", type=float, default=1.0)
parser.add_argument("--pink-orientation-cost", type=float, default=0.25)
parser.add_argument("--pink-posture-cost", type=float, default=0.001)
parser.add_argument("--pink-damping", type=float, default=1e-6)
parser.add_argument("--pink-qp-solver", choices=("daqp", "osqp"), default="daqp")
parser.add_argument("--max-eef-error", type=float, default=0.08, help="Maximum target-to-measured wrist distance in metres.")
parser.add_argument("--max-joint-lead", type=float, default=0.10, help="Maximum persistent IK target lead over a measured joint in radians.")
parser.add_argument("--joint-target-smoothing", type=float, default=1.0, help="EMA weight for each new IK joint target (0, 1].")
parser.add_argument("--max-joint-target-step", type=float, default=0.03, help="Maximum commanded joint-target change per simulation step in radians.")
parser.add_argument("--workspace-min", type=float, nargs=3, default=(-0.85, -0.50, 0.72))
parser.add_argument("--workspace-max", type=float, nargs=3, default=(-0.08, 0.50, 1.55))
parser.add_argument(
    "--startup-target-delta",
    type=float,
    nargs=6,
    default=(0.0,) * 6,
    metavar=("DX", "DY", "DZ", "RX", "RY", "RZ"),
    help="Diagnostic one-time XYZ/rotation-vector offset applied to each reset target.",
)
parser.add_argument("--include-wrist-camera", action="store_true")
parser.add_argument(
    "--fixed-cube",
    action="store_true",
    help="Disable cube X/Y reset randomization and always use scene_layout.OBJECT_POS.",
)
parser.add_argument(
    "--startup-over-cube-palm-down",
    action="store_true",
    help="Start manual teleop by moving the grasp center 10 cm above the cube surface with palm down.",
)
parser.add_argument("--scripted-grasp", action="store_true", help="Run and record the six-waypoint cube grasp instead of manual teleoperation.")
parser.add_argument(
    "--scripted-hover-height",
    type=float,
    default=0.10,
    help="Collision-free centering height above the cube top surface in metres.",
)
parser.add_argument(
    "--scripted-grasp-clearance",
    type=float,
    default=0.040,
    help="Pre-grasp center height above the cube center in metres (0.040 = 1.5 cm above a 5 cm cube's top).",
)
parser.add_argument("--scripted-lift-height", type=float, default=0.10)
parser.add_argument("--scripted-cube-size", type=float, default=0.05)
parser.add_argument("--scripted-seed", type=int, default=42, help="Fixed environment seed for reproducible scripted trials.")
parser.add_argument("--scripted-exit-after-attempt", action="store_true", help="Exit after the first scripted success or failure (useful for sweeps).")
parser.add_argument(
    "--scripted-max-contact-center-correction",
    "--scripted-max-thumb-center-correction",
    dest="scripted_max_contact_center_correction",
    type=float,
    default=0.03,
    help="Maximum XY correction used to center the thumb/finger pinch corridor.",
)
parser.add_argument(
    "--scripted-contact-center-tolerance",
    type=float,
    default=0.008,
    help="Required thumb/middle pinch-midpoint XY error before recenter/descent may advance.",
)
parser.add_argument("--scripted-grasp-rotvec", type=float, nargs=3, default=(0.0, -0.50, 0.0))
parser.add_argument(
    "--scripted-use-relative-grasp-rotation",
    action="store_true",
    help="Use --scripted-grasp-rotvec instead of the calibrated absolute palm-down orientation.",
)
parser.add_argument(
    "--scripted-grasp-frame-offset",
    type=float,
    nargs=3,
    default=(-0.005, 0.0, -0.15),
    metavar=("X", "Y", "Z"),
    help="Grasp-center position in the wristRollRight local frame, in metres.",
)
parser.add_argument("--scripted-position-tolerance", type=float, default=0.025)
parser.add_argument(
    "--scripted-contact-tolerance",
    type=float,
    default=0.045,
    help="Position tolerance at the collision-constrained grasp waypoint.",
)
parser.add_argument("--scripted-rotation-tolerance", type=float, default=0.12)
parser.add_argument(
    "--scripted-palm-tilt-tolerance",
    type=float,
    default=0.10,
    help="Maximum palm-normal tilt from table-down direction in radians.",
)
parser.add_argument("--scripted-hold-steps", type=int, default=10)
parser.add_argument("--scripted-close-steps", type=int, default=30)
parser.add_argument("--scripted-post-close-hold-steps", type=int, default=20)
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)

if "--gui" in sys.argv:
    sys.argv.remove("--gui")
    if not any(arg in ("--viz", "--visualizer") for arg in sys.argv):
        sys.argv.extend(("--viz", "kit"))
args_cli = parser.parse_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedEnv

from adapters.adam_u_action_adapter import expand_hand_synergies_for_isaac
from adapters.eef_pose import (
    IsaacDifferentialIKSolver,
    MinkAdamUIKSolver,
    PinocchioAdamUKinematicsProvider,
    PinkAdamUIKSolver,
)
from adapters.teleop_recorder import AdamUTeleopRecorder
from configs.adam_u_action_mapping import AdamUActionMappingConfig
from configs.joint_state import (
    BODY_JOINT_NAMES,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)
from envs.scene_layout import FRONT_CAMERA_LOOKAT, FRONT_CAMERA_POS, VIEWER_EYE, VIEWER_LOOKAT

# Adam-U's mesh-heavy URDF invalidates the first PhysX tensor view on this
# Isaac Sim 6 setup. Reuse the same startup rebind required by eval_groot.py.
_recovery_path = os.path.join(_GROOT_ROOT, "envs", "isaac_physics_recovery.py")
_recovery_spec = importlib.util.spec_from_file_location("adam_u_teleop_physics_recovery", _recovery_path)
if _recovery_spec is None or _recovery_spec.loader is None:
    raise ImportError(f"Could not load physics recovery helper: {_recovery_path}")
_recovery_module = importlib.util.module_from_spec(_recovery_spec)
_recovery_spec.loader.exec_module(_recovery_module)
_recovery_module.install_manager_env_physics_recovery_patch()


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        return value.numpy()
    return np.asarray(value)


def _ordered_joint_positions(robot, names: tuple[str, ...]) -> np.ndarray:
    indices = [robot.data.joint_names.index(name) for name in names]
    return _numpy(robot.data.joint_pos)[:, indices].astype(np.float32)


def _hand_synergies(robot) -> np.ndarray:
    left = _ordered_joint_positions(robot, LEFT_HAND_JOINT_NAMES)
    right = _ordered_joint_positions(robot, RIGHT_HAND_JOINT_NAMES)
    # Primary URDF joint for opposition, thumb flexion, and four finger synergies.
    return np.concatenate((left[:, (0, 3, 4, 6, 8, 10)], right[:, (0, 3, 4, 6, 8, 10)]), axis=1)


def _rgb(camera) -> np.ndarray:
    image = _numpy(camera.data.output["rgb"])[0, ..., :3]
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return image


def _matrix_from_pose9(pose: np.ndarray) -> np.ndarray:
    first = pose[3:6] / max(np.linalg.norm(pose[3:6]), 1e-8)
    second = pose[6:9] - np.dot(first, pose[6:9]) * first
    second /= max(np.linalg.norm(second), 1e-8)
    return np.stack((first, second, np.cross(first, second)), axis=0)


def _pose9(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.concatenate((position, rotation[:2].reshape(6))).astype(np.float32)


def _step_vector(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    delta = goal - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_step:
        return goal.copy()
    return current + delta * (max_step / distance)


def _step_rotation(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    error = Rotation.from_matrix(goal @ current.T).as_rotvec()
    magnitude = float(np.linalg.norm(error))
    if magnitude <= max_step:
        return goal.copy()
    return Rotation.from_rotvec(error * (max_step / magnitude)).as_matrix() @ current


def _rotation_distance(current: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(goal @ current.T).as_rotvec()))


def _adam_right_palm_down_rotation() -> np.ndarray:
    """Right palm parallel to the table: local +Y down, fingers -Z forward."""
    return np.asarray(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )


class Signals:
    reset = False
    save = False
    paused = False


def main() -> None:
    if (
        args_cli.ik_joint_step <= 0
        or args_cli.position_sensitivity <= 0
        or args_cli.z_sensitivity_scale <= 0
    ):
        raise ValueError("IK step and device sensitivity must be positive")
    if args_cli.max_eef_error <= 0 or args_cli.max_joint_lead <= 0:
        raise ValueError("EEF error and joint lead limits must be positive")
    if not 0.0 < args_cli.joint_target_smoothing <= 1.0:
        raise ValueError("--joint-target-smoothing must be in (0, 1]")
    if args_cli.max_joint_target_step <= 0:
        raise ValueError("--max-joint-target-step must be positive")
    if args_cli.startup_stabilization_steps < 0:
        raise ValueError("--startup-stabilization-steps must be non-negative")
    if (
        args_cli.pink_position_cost <= 0
        or args_cli.pink_orientation_cost < 0
        or args_cli.pink_posture_cost < 0
        or args_cli.pink_damping < 0
    ):
        raise ValueError("Pink costs/damping must be non-negative and position cost positive")
    if (
        args_cli.mink_damping < 0
        or args_cli.mink_iterations < 1
        or args_cli.mink_translation_orientation_cost < 0
    ):
        raise ValueError("Mink damping must be non-negative and iterations positive")
    if (
        args_cli.scripted_hold_steps < 1
        or args_cli.scripted_close_steps < 1
        or args_cli.scripted_post_close_hold_steps < 0
    ):
        raise ValueError("Scripted hold/close steps must be positive and post-close hold non-negative")
    if (
        args_cli.scripted_cube_size <= 0
        or args_cli.scripted_hover_height <= 0
        or args_cli.scripted_max_contact_center_correction <= 0
        or args_cli.scripted_contact_center_tolerance <= 0
        or args_cli.scripted_palm_tilt_tolerance <= 0
    ):
        raise ValueError("Scripted cube size and contact-centering limit must be positive")
    workspace_min = np.asarray(args_cli.workspace_min, dtype=np.float32)
    workspace_max = np.asarray(args_cli.workspace_max, dtype=np.float32)
    grasp_frame_offset = np.asarray(args_cli.scripted_grasp_frame_offset, dtype=np.float64)
    startup_delta = np.asarray(args_cli.startup_target_delta, dtype=np.float32)
    if np.any(workspace_min >= workspace_max):
        raise ValueError("Every --workspace-min value must be below --workspace-max")
    make_env_cfg = _load_env_factory()
    cfg = make_env_cfg(
        num_envs=1,
        include_cameras=True,
        include_wrist_camera=args_cli.include_wrist_camera,
        full_body_actions=True,
    )
    if args_cli.scripted_grasp:
        cfg.seed = args_cli.scripted_seed
    if args_cli.fixed_cube or args_cli.startup_over_cube_palm_down:
        cfg.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
    cfg.sim.device = os.environ.get("ADAM_U_PHYSICS_DEVICE", "cpu")
    cfg.scene.robot.spawn.asset_path = os.path.join(_REPO_ROOT, "assets/robots/adam_u/urdf/adam_u.urdf")
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    env = ManagerBasedEnv(cfg=cfg)
    robot = env.scene["robot"]
    thumb_body_index = (
        robot.data.body_names.index("R_thumb_distal")
        if "R_thumb_distal" in robot.data.body_names
        else None
    )
    middle_body_index = (
        robot.data.body_names.index("R_middle_distal")
        if "R_middle_distal" in robot.data.body_names
        else None
    )
    if args_cli.scripted_grasp and (
        thumb_body_index is None or middle_body_index is None
    ):
        print(
            "[SCRIPTED] WARNING: opposing thumb/middle contact links unavailable; "
            "using palm-center alignment"
        )
    env.reset()
    if env.viewport_camera_controller is not None:
        env.viewport_camera_controller.update_view_to_world()
        env.sim.set_camera_view(eye=VIEWER_EYE, target=VIEWER_LOOKAT)
    camera = env.scene["front_camera"]
    camera.set_world_poses_from_view(
        torch.tensor([FRONT_CAMERA_POS], device=env.device),
        torch.tensor([FRONT_CAMERA_LOOKAT], device=env.device),
    )
    env.sim.render()

    provider = PinocchioAdamUKinematicsProvider(env)
    if args_cli.ik_backend == "mink":
        solver = MinkAdamUIKSolver(
            provider,
            args_cli.mink_model,
            max_joint_delta=args_cli.ik_joint_step,
            max_commanded_joint_error=args_cli.max_joint_lead,
            damping=args_cli.mink_damping,
            iterations=args_cli.mink_iterations,
            apply_controller_frame_offset=args_cli.mink_controller_frame_offset,
            collision_avoidance=args_cli.mink_collision_avoidance,
        )
        print(
            "[TELEOP] IK backend=Mink/DAQP with PND Adam-U tuning; "
            f"model={os.path.abspath(os.path.expanduser(args_cli.mink_model))}, "
            f"costs position={solver.PND_WRIST_POSITION_COST}, "
            f"orientation={solver.PND_WRIST_ORIENTATION_COST}, "
            f"damping={args_cli.mink_damping}, iterations={args_cli.mink_iterations}, "
            f"controller_frame_offset={args_cli.mink_controller_frame_offset}"
        )
    elif args_cli.ik_backend == "pink":
        solver = PinkAdamUIKSolver(
            provider,
            max_joint_delta=args_cli.ik_joint_step,
            position_cost=args_cli.pink_position_cost,
            orientation_cost=args_cli.pink_orientation_cost,
            posture_cost=args_cli.pink_posture_cost,
            damping=args_cli.pink_damping,
            solver=args_cli.pink_qp_solver,
            max_commanded_joint_error=args_cli.max_joint_lead,
        )
        print(
            f"[TELEOP] IK backend=Pink ({args_cli.pink_qp_solver}); "
            f"costs position={args_cli.pink_position_cost}, "
            f"orientation={args_cli.pink_orientation_cost}, "
            f"posture={args_cli.pink_posture_cost}"
        )
    else:
        solver = IsaacDifferentialIKSolver(
            provider,
            max_joint_delta=args_cli.ik_joint_step,
            # Native Adam-U teleoperation keeps pose, quaternion, and the
            # Pinocchio Jacobian in the same world frame, so all six EEF dimensions
            # can be solved together without any G1 tool-frame conversion.
            command_type="pose",
            # Persistent targets provide the PD position error required to resist
            # gravity. The Pinocchio Jacobian avoids the PhysX frame ambiguity that
            # previously made accumulated corrections diverge.
            accumulate_joint_targets=True,
            max_commanded_joint_error=args_cli.max_joint_lead,
        )
        print("[TELEOP] IK backend=Isaac differential IK (DLS)")
    device_cfg = dict(
        pos_sensitivity=args_cli.position_sensitivity,
        rot_sensitivity=args_cli.rotation_sensitivity,
        sim_device=env.device,
    )
    teleop = (
        Se3Keyboard(Se3KeyboardCfg(**device_cfg))
        if args_cli.teleop_device == "keyboard"
        else Se3SpaceMouse(Se3SpaceMouseCfg(**device_cfg))
    )
    signals = Signals()
    teleop.add_callback("R", lambda: setattr(signals, "reset", True))
    teleop.add_callback("ENTER", lambda: setattr(signals, "save", True))
    teleop.add_callback("P", lambda: setattr(signals, "paused", not signals.paused))

    output = args_cli.output
    if not os.path.isabs(output):
        output = os.path.join(_REPO_ROOT, output)
    recorder = AdamUTeleopRecorder(output, args_cli.task, env.step_dt)
    limits = AdamUActionMappingConfig()
    body_lower, body_upper, _ = limits.body_limits
    hand_upper = np.tile(np.asarray(limits.hand_upper, dtype=np.float32), 2)

    hold_body = _ordered_joint_positions(robot, BODY_JOINT_NAMES)
    hold_hands = _hand_synergies(robot)
    current_right = hold_body[:, 12:19]
    target_pose, _ = provider("right", current_right)
    target_position = target_pose[0, :3].copy()
    target_rotation = _matrix_from_pose9(target_pose[0])
    initial_wrist_position = target_position.copy()
    target_position += startup_delta[:3]
    target_rotation = Rotation.from_rotvec(startup_delta[3:]).as_matrix() @ target_rotation
    if args_cli.startup_over_cube_palm_down and not args_cli.scripted_grasp:
        startup_cube = (
            _numpy(env.scene["object"].data.root_pos_w)[0]
            - _numpy(env.scene.env_origins)[0]
        )
        startup_grasp_center = startup_cube.copy()
        startup_grasp_center[2] += 0.5 * args_cli.scripted_cube_size + 0.15
        target_rotation = _adam_right_palm_down_rotation()
        target_position = startup_grasp_center - target_rotation @ grasp_frame_offset
        print(
            "[TELEOP] Startup preset: palm-down grasp center 15 cm above cube surface "
            f"at {startup_grasp_center}"
        )
    manual_palm_position = target_position + target_rotation @ grasp_frame_offset
    solver.sync_commanded_joint_pos("right", current_right)
    previous_right_target = current_right.copy()

    scripted_phase = 0
    scripted_hold_count = 0
    scripted_close_count = 0
    scripted_done = False
    scripted_pending_save = False
    scripted_initial_cube_z = 0.0
    scripted_waypoints: list[tuple[np.ndarray, np.ndarray, str]] = []
    scripted_position_error = float("inf")
    scripted_rotation_error = float("inf")
    scripted_contact_center_error = float("inf")
    scripted_palm_tilt_error = float("inf")
    scripted_close_wrist_position: np.ndarray | None = None
    scripted_close_wrist_rotation: np.ndarray | None = None

    def reset_scripted_state() -> None:
        nonlocal scripted_phase, scripted_hold_count, scripted_close_count
        nonlocal scripted_done, scripted_pending_save, scripted_initial_cube_z, scripted_waypoints
        nonlocal scripted_position_error, scripted_rotation_error
        nonlocal scripted_contact_center_error
        nonlocal scripted_palm_tilt_error
        nonlocal scripted_close_wrist_position, scripted_close_wrist_rotation
        cube = _numpy(env.scene["object"].data.root_pos_w)[0] - _numpy(env.scene.env_origins)[0]
        scripted_initial_cube_z = float(cube[2])
        if args_cli.scripted_use_relative_grasp_rotation:
            grasp_rotation = (
                Rotation.from_rotvec(args_cli.scripted_grasp_rotvec).as_matrix()
                @ target_rotation
            )
        else:
            # Adam-U right-hand convention from the URDF:
            #   local +Y = palm normal, local -Z = fingers-forward direction.
            # Point the palm normal down toward the table (-world Z) and the
            # fingers toward the front of the robot (-world X). Columns are
            # the local X/Y/Z axes expressed in the Adam world frame.
            grasp_rotation = _adam_right_palm_down_rotation()
        hover = cube.copy()
        # Center and orient at a collision-free height measured from the top
        # surface, not from the cube center. Descent is forbidden until both
        # pinch-midpoint centering and palm parallelism checks pass.
        hover[2] += 0.5 * args_cli.scripted_cube_size + args_cli.scripted_hover_height
        grasp = cube.copy()
        grasp[2] += args_cli.scripted_grasp_clearance
        lifted = grasp.copy()
        lifted[2] += args_cli.scripted_lift_height
        # Rotate waypoint 2 about a fixed wrist origin. This intentionally
        # allows the palm center to shift during rotation; waypoint 3 then
        # performs a visible, explicit recentering over the cube.
        hover_wrist = hover - target_rotation @ grasp_frame_offset
        rotated_grasp_center = hover_wrist + grasp_rotation @ grasp_frame_offset
        scripted_waypoints = [
            (hover.copy(), target_rotation.copy(), "approach_10cm"),
            (rotated_grasp_center.copy(), grasp_rotation.copy(), "rotate_palm_down"),
            (hover.copy(), grasp_rotation.copy(), "recenter_palm_over_cube"),
            (grasp.copy(), grasp_rotation.copy(), "lower_to_grasp"),
            (grasp.copy(), grasp_rotation.copy(), "close_hand"),
            (lifted.copy(), grasp_rotation.copy(), "lift_10cm"),
        ]
        scripted_phase = 0
        scripted_hold_count = 0
        scripted_close_count = 0
        scripted_done = False
        scripted_pending_save = False
        scripted_position_error = float("inf")
        scripted_rotation_error = float("inf")
        scripted_contact_center_error = float("inf")
        scripted_palm_tilt_error = float("inf")
        scripted_close_wrist_position = None
        scripted_close_wrist_rotation = None
        if args_cli.scripted_grasp:
            print(
                f"[SCRIPTED] waypoint 1/6: {scripted_waypoints[0][2]} "
                f"grasp_center_target={scripted_waypoints[0][0]}"
            )

    reset_scripted_state()

    print(teleop)
    print("[TELEOP] K/button: toggle hand | P: pause | R: discard/reset | ENTER: save success")
    print("[TELEOP] Keys update one 6D palm-center target; it is converted to wristRollRight for IK")
    print("[TELEOP] Click once inside the Isaac simulation viewport before using the keyboard")
    if args_cli.scripted_grasp:
        print("[SCRIPTED] Automatic six-waypoint grasp collection enabled; R aborts/restarts")
    print(f"[TELEOP] Recording native Adam-U demonstrations to {output}")
    step = 0
    input_was_active = False
    translation_was_active = False
    manual_pose_was_active = False
    all_translation_test_cases = (
        ("W", np.asarray((1.0, 0.0, 0.0), dtype=np.float32)),
        ("S", np.asarray((-1.0, 0.0, 0.0), dtype=np.float32)),
        ("A", np.asarray((0.0, 1.0, 0.0), dtype=np.float32)),
        ("D", np.asarray((0.0, -1.0, 0.0), dtype=np.float32)),
        ("Q", np.asarray((0.0, 0.0, 1.0), dtype=np.float32)),
        ("E", np.asarray((0.0, 0.0, -1.0), dtype=np.float32)),
    )
    invalid_test_keys = set(args_cli.translation_self_test_keys) - set("WSADQE")
    if invalid_test_keys or not args_cli.translation_self_test_keys:
        raise ValueError("--translation-self-test-keys must be a non-empty subset of WSADQE")
    cases_by_key = dict(all_translation_test_cases)
    translation_test_cases = tuple(
        (key, cases_by_key[key]) for key in args_cli.translation_self_test_keys
    )
    translation_test_index = 0
    translation_test_step = 0
    translation_test_start = None
    translation_test_joint_start = None
    translation_test_results = []
    while simulation_app.is_running() and step < args_cli.max_steps:
        started = time.time()
        if signals.save:
            signals.save = False
            if recorder.sample_count:
                name = recorder.save_success()
                print(f"[TELEOP] Saved {name}; resetting scene")
                signals.reset = True
        if signals.reset:
            signals.reset = False
            recorder.clear()
            env.reset()
            teleop.reset()
            hold_body = _ordered_joint_positions(robot, BODY_JOINT_NAMES)
            hold_hands = _hand_synergies(robot)
            current_right = hold_body[:, 12:19]
            target_pose, _ = provider("right", current_right)
            target_position = target_pose[0, :3].copy()
            target_rotation = _matrix_from_pose9(target_pose[0])
            initial_wrist_position = target_position.copy()
            target_position += startup_delta[:3]
            target_rotation = Rotation.from_rotvec(startup_delta[3:]).as_matrix() @ target_rotation
            if args_cli.startup_over_cube_palm_down and not args_cli.scripted_grasp:
                startup_cube = (
                    _numpy(env.scene["object"].data.root_pos_w)[0]
                    - _numpy(env.scene.env_origins)[0]
                )
                startup_grasp_center = startup_cube.copy()
                startup_grasp_center[2] += 0.5 * args_cli.scripted_cube_size + 0.15
                target_rotation = _adam_right_palm_down_rotation()
                target_position = startup_grasp_center - target_rotation @ grasp_frame_offset
            manual_palm_position = target_position + target_rotation @ grasp_frame_offset
            solver.sync_commanded_joint_pos("right", current_right)
            previous_right_target = current_right.copy()
            translation_was_active = False
            manual_pose_was_active = False
            reset_scripted_state()
            print("[TELEOP] Episode discarded/reset")

        command = _numpy(teleop.advance()).reshape(-1).copy()
        input_is_active = bool(np.any(np.abs(command) > 1e-9))
        if input_is_active and not input_was_active:
            print(
                "[TELEOP INPUT] received "
                f"translation={command[:3]} rotation={command[3:6]} "
                f"gripper={command[6]:+.1f}"
            )
        input_was_active = input_is_active
        current_body = _ordered_joint_positions(robot, BODY_JOINT_NAMES)
        measured_pose_before, _ = provider("right", current_body[:, 12:19])
        measured_rotation_before = _matrix_from_pose9(measured_pose_before[0])
        translation_is_active = False
        stabilizing = step < args_cli.startup_stabilization_steps
        if stabilizing:
            command[:6] = 0.0
        elif step == args_cli.startup_stabilization_steps:
            print("[TELEOP] Startup stabilization complete; manual XYZ control enabled")
        if args_cli.translation_self_test and not stabilizing:
            command[:6] = 0.0
            key, axis = translation_test_cases[translation_test_index]
            if translation_test_step == 0:
                translation_test_start = measured_pose_before[0, :3].copy()
                translation_test_joint_start = current_body[0, 12:19].copy()
                translation_was_active = False
                manual_pose_was_active = False
                print(
                    f"[TRANSLATION TEST] {key} start={translation_test_start} "
                    f"requested_axis={axis}"
                )
            if translation_test_step < 30:
                command[:3] = axis * args_cli.position_sensitivity
        if args_cli.scripted_grasp and not signals.paused and not scripted_done:
            scripted_contact_center_error = 0.0
            grasp_goal_position, goal_rotation, phase_name = scripted_waypoints[scripted_phase]
            grasp_goal_position = grasp_goal_position.copy()
            # Recenter and descent use the current cube position rather than
            # the reset-time position, so minor contact motion cannot leave
            # the palm following a stale X/Y target.
            if scripted_phase in (2, 3):
                live_cube = (
                    _numpy(env.scene["object"].data.root_pos_w)[0]
                    - _numpy(env.scene.env_origins)[0]
                )
                grasp_goal_position[:2] = live_cube[:2]
                if thumb_body_index is not None and middle_body_index is not None:
                    body_positions = _numpy(robot.data.body_pos_w)[0]
                    env_origin = _numpy(env.scene.env_origins)[0]
                    measured_thumb = body_positions[thumb_body_index] - env_origin
                    measured_middle = body_positions[middle_body_index] - env_origin
                    # Local +X points from the palm center toward the thumb.
                    # Target the thumb and middle finger at opposite cube sides.
                    # Their averaged error translates the pinch midpoint onto
                    # the cube without favoring either contact.
                    thumb_side_xy = (goal_rotation @ np.asarray((1.0, 0.0, 0.0)))[:2]
                    thumb_side_xy /= max(float(np.linalg.norm(thumb_side_xy)), 1e-8)
                    desired_thumb_xy = (
                        live_cube[:2]
                        + 0.5 * args_cli.scripted_cube_size * thumb_side_xy
                    )
                    desired_middle_xy = (
                        live_cube[:2]
                        - 0.5 * args_cli.scripted_cube_size * thumb_side_xy
                    )
                    contact_center_error = 0.5 * (
                        (desired_thumb_xy - measured_thumb[:2])
                        + (desired_middle_xy - measured_middle[:2])
                    )
                    scripted_contact_center_error = float(
                        np.linalg.norm(contact_center_error)
                    )
                    correction_norm = float(np.linalg.norm(contact_center_error))
                    if correction_norm > args_cli.scripted_max_contact_center_correction:
                        contact_center_error *= (
                            args_cli.scripted_max_contact_center_correction / correction_norm
                        )
                    grasp_goal_position[:2] += contact_center_error
            # During closure, stop driving farther into the object. Hold the
            # measured wrist pose latched at the end of descent and let the
            # fingers establish contact around the cube.
            if (
                scripted_phase == 4
                and scripted_close_wrist_position is not None
                and scripted_close_wrist_rotation is not None
            ):
                goal_position = scripted_close_wrist_position
                goal_rotation = scripted_close_wrist_rotation
                grasp_goal_position = goal_position + goal_rotation @ grasp_frame_offset
            else:
                # IK controls wristRollRight. Convert the desired palm/finger
                # grasp center into the corresponding wrist origin target.
                goal_position = grasp_goal_position - goal_rotation @ grasp_frame_offset
            target_position = _step_vector(
                target_position, goal_position, args_cli.position_sensitivity
            )
            target_rotation = _step_rotation(
                target_rotation, goal_rotation, args_cli.rotation_sensitivity
            )
            measured_grasp_position = (
                measured_pose_before[0, :3] + measured_rotation_before @ grasp_frame_offset
            )
            position_error = float(np.linalg.norm(measured_grasp_position - grasp_goal_position))
            rotation_error = _rotation_distance(measured_rotation_before, goal_rotation)
            measured_palm_normal = measured_rotation_before @ np.asarray((0.0, 1.0, 0.0))
            palm_alignment = float(
                np.clip(np.dot(measured_palm_normal, (0.0, 0.0, -1.0)), -1.0, 1.0)
            )
            scripted_palm_tilt_error = float(np.arccos(palm_alignment))
            scripted_position_error = position_error
            scripted_rotation_error = rotation_error
            waypoint_reached = False
            if scripted_phase == 4:
                scripted_close_count += 1
                if scripted_close_count >= (
                    args_cli.scripted_close_steps + args_cli.scripted_post_close_hold_steps
                ):
                    scripted_phase = 5
                    scripted_hold_count = 0
                    print(
                        f"[SCRIPTED] waypoint 6/6: {scripted_waypoints[5][2]} "
                        f"grasp_center_target={scripted_waypoints[5][0]}"
                    )
            else:
                position_tolerance = (
                    args_cli.scripted_contact_tolerance
                    if scripted_phase == 3
                    else args_cli.scripted_position_tolerance
                )
                waypoint_reached = (
                    position_error <= position_tolerance
                    and rotation_error <= args_cli.scripted_rotation_tolerance
                )
                if scripted_phase in (2, 3):
                    waypoint_reached = (
                        waypoint_reached
                        and scripted_contact_center_error
                        <= args_cli.scripted_contact_center_tolerance
                        and scripted_palm_tilt_error
                        <= args_cli.scripted_palm_tilt_tolerance
                    )
            if waypoint_reached:
                scripted_hold_count += 1
                if scripted_hold_count >= args_cli.scripted_hold_steps:
                    if scripted_phase == 5:
                        scripted_done = True
                    else:
                        scripted_phase += 1
                        scripted_hold_count = 0
                        if scripted_phase == 4:
                            scripted_close_wrist_position = measured_pose_before[0, :3].copy()
                            scripted_close_wrist_rotation = measured_rotation_before.copy()
                            target_position = scripted_close_wrist_position.copy()
                            target_rotation = scripted_close_wrist_rotation.copy()
                            print(
                                "[SCRIPTED] Contact pose accepted; wrist target latched "
                                "while fingers close"
                            )
                        print(
                            f"[SCRIPTED] waypoint {scripted_phase + 1}/6: "
                            f"{scripted_waypoints[scripted_phase][2]} "
                            f"grasp_center_target={scripted_waypoints[scripted_phase][0]}"
                        )
            elif scripted_phase != 4:
                scripted_hold_count = 0
        elif not signals.paused:
            # Isaac's keyboard emits Q=+Z and E=-Z. Apply an independent
            # world-Z gain because gravity and the arm configuration make the
            # vertical response weaker than equal-sized X/Y increments.
            command[2] *= args_cli.z_sensitivity_scale
            translation_is_active = bool(np.any(np.abs(command[:3]) > 1e-9))
            rotation_is_active = bool(np.any(np.abs(command[3:6]) > 1e-9))
            manual_pose_is_active = translation_is_active or rotation_is_active
            measured_palm_position = (
                measured_pose_before[0, :3]
                + measured_rotation_before @ grasp_frame_offset
            )
            if manual_pose_is_active and not manual_pose_was_active:
                # Take over at the visible palm pose, not at wristRollRight.
                # This also clears any unfinished startup target and stale
                # actuator-filter state before a new keyboard gesture.
                manual_palm_position = measured_palm_position.copy()
                target_rotation = measured_rotation_before.copy()
                measured_right = current_body[:, 12:19].astype(np.float32)
                solver.sync_commanded_joint_pos("right", measured_right)
                previous_right_target = measured_right.copy()
                print(
                    "[TELEOP] Manual palm-pose takeover latched at "
                    f"{manual_palm_position}"
                )
            if args_cli.axis_locked_translation and translation_is_active:
                active_translation_axes = np.abs(command[:3]) > 1e-9
                if manual_pose_was_active:
                    # During a held key, preserve progress only on the active
                    # axis and eliminate cross-axis drift each frame.
                    manual_palm_position[~active_translation_axes] = measured_palm_position[
                        ~active_translation_axes
                    ]
            manual_palm_position += command[:3]
            manual_palm_position = np.clip(
                manual_palm_position, workspace_min, workspace_max
            )
            # Rotation keys are tool-local roll/pitch/yaw. Right multiplication
            # is essential: at a palm-down pose the wrist-roll axis is not
            # world X, and world-axis rotation makes Mink recruit the shoulder.
            target_rotation = target_rotation @ Rotation.from_rotvec(
                command[3:6]
            ).as_matrix()
            # IK controls wristRollRight, so compensate the palm/tool offset.
            # Keeping the palm target fixed while orientation changes makes
            # rotations occur about the palm rather than the wrist or shoulder.
            target_position = (
                manual_palm_position - target_rotation @ grasp_frame_offset
            )
            translation_was_active = translation_is_active
            manual_pose_was_active = manual_pose_is_active
        target_offset = target_position - measured_pose_before[0, :3]
        target_distance = float(np.linalg.norm(target_offset))
        if target_distance > args_cli.max_eef_error:
            target_position = measured_pose_before[0, :3] + target_offset * (
                args_cli.max_eef_error / target_distance
            )
        if isinstance(solver, MinkAdamUIKSolver):
            solver.set_translation_priority(
                translation_is_active,
                orientation_cost=args_cli.mink_translation_orientation_cost,
            )
        raw_right_target = solver.solve(
            "right", _pose9(target_position, target_rotation)[None], current_body[:, 12:19],
            command_is_relative=False,
        ).astype(np.float32)
        raw_right_target = np.clip(
            raw_right_target,
            current_body[:, 12:19] - args_cli.max_joint_lead,
            current_body[:, 12:19] + args_cli.max_joint_lead,
        )
        # Rate-limit first, then low-pass filter in command space. This keeps
        # alternating IK corrections from becoming visible arm oscillation.
        limited_right_target = np.clip(
            raw_right_target,
            previous_right_target - args_cli.max_joint_target_step,
            previous_right_target + args_cli.max_joint_target_step,
        )
        alpha = args_cli.joint_target_smoothing
        right_target = (
            previous_right_target
            + alpha * (limited_right_target - previous_right_target)
        ).astype(np.float32)
        previous_right_target = right_target.copy()
        body_action = hold_body.copy()
        body_action[:, 12:19] = right_target
        body_action = np.clip(body_action, body_lower, body_upper).astype(np.float32)
        solver.sync_commanded_joint_pos("right", body_action[:, 12:19])

        hand_action = hold_hands.copy()
        if args_cli.scripted_grasp:
            close_fraction = min(1.0, scripted_close_count / args_cli.scripted_close_steps)
            if scripted_phase >= 5 or scripted_done:
                close_fraction = 1.0
            hand_action[:, 6:12] = close_fraction * hand_upper[6:12]
        else:
            # Device convention is +1 open, -1 closed.
            hand_action[:, 6:12] = 0.0 if command[6] > 0 else hand_upper[6:12]
        hand_action = np.clip(hand_action, 0.0, hand_upper).astype(np.float32)
        sim_action = np.concatenate((body_action, expand_hand_synergies_for_isaac(hand_action)), axis=1)
        if isinstance(solver, MinkAdamUIKSolver) and args_cli.mink_gravity_compensation:
            gravity_effort = solver.gravity_compensation("right", current_body[:, 12:19])
            right_joint_ids = [
                robot.data.joint_names.index(name) for name in RIGHT_ARM_JOINT_NAMES
            ]
            robot.set_joint_effort_target_index(
                target=torch.as_tensor(
                    gravity_effort, device=env.device, dtype=torch.float32
                ),
                joint_ids=right_joint_ids,
            )
        env.step(torch.as_tensor(sim_action, device=env.device, dtype=torch.float32))

        measured_body = _ordered_joint_positions(robot, BODY_JOINT_NAMES)[0]
        measured_hands = _hand_synergies(robot)[0]
        measured_wrist, _ = provider("right", measured_body[None, 12:19])
        if args_cli.translation_self_test and not stabilizing:
            key, axis = translation_test_cases[translation_test_index]
            if translation_test_step == 29:
                displacement = measured_wrist[0, :3] - translation_test_start
                measured_joint_delta = measured_body[12:19] - translation_test_joint_start
                commanded_joint_delta = body_action[0, 12:19] - translation_test_joint_start
                along = float(np.dot(displacement, axis))
                cross = float(np.linalg.norm(displacement - along * axis))
                passed = along > 0.003
                translation_test_results.append((key, along, cross, passed))
                print(
                    f"[TRANSLATION TEST] {key} displacement={displacement} "
                    f"dq_command={np.round(commanded_joint_delta, 5).tolist()} "
                    f"dq_measured={np.round(measured_joint_delta, 5).tolist()} "
                    f"along={along:.5f}m cross={cross:.5f}m "
                    f"result={'PASS' if passed else 'FAIL'}"
                )
            translation_test_step += 1
            if translation_test_step >= 40:
                translation_test_step = 0
                translation_test_index += 1
                translation_was_active = False
                if translation_test_index >= len(translation_test_cases):
                    print("[TRANSLATION TEST] summary:")
                    for key_name, along, cross, passed in translation_test_results:
                        print(
                            f"  {key_name}: along={along:.5f}m cross={cross:.5f}m "
                            f"{'PASS' if passed else 'FAIL'}"
                        )
                    break
        object_position = _numpy(env.scene["object"].data.root_pos_w)[0] - _numpy(env.scene.env_origins)[0]
        object_quat = _numpy(env.scene["object"].data.root_quat_w)[0]
        sample = {
            "observation.images.front": _rgb(camera),
            "observation.state.body": measured_body,
            "observation.state.hands": measured_hands,
            "action.body": body_action[0],
            "action.hands": hand_action[0],
            "observation.right_wrist_pose": measured_wrist[0].astype(np.float32),
            "observation.object_pose": np.concatenate((object_position, object_quat)).astype(np.float32),
            "timestamp": np.asarray(step * env.step_dt, dtype=np.float64),
        }
        if args_cli.include_wrist_camera:
            sample["observation.images.wrist"] = _rgb(env.scene["wrist_camera"])
        if not signals.paused:
            recorder.append(sample)
        if args_cli.scripted_grasp and scripted_done and not scripted_pending_save:
            lifted = float(object_position[2] - scripted_initial_cube_z)
            scripted_pending_save = True
            if lifted >= 0.07:
                name = recorder.save_success()
                print(f"[SCRIPTED] SUCCESS: cube lifted {lifted:.3f} m; saved {name}")
                break
            else:
                print(
                    f"[SCRIPTED] FAILED: wrist completed lift but cube rose only {lifted:.3f} m; "
                    "episode remains unsaved. Press R to retry."
                )
                if args_cli.scripted_exit_after_attempt:
                    break
        step += 1
        if step % 30 == 0:
            eef_error = float(np.linalg.norm(measured_wrist[0, :3] - target_position))
            axis_error = target_position - measured_wrist[0, :3]
            measured_wrist_rotation = _matrix_from_pose9(measured_wrist[0])
            measured_grasp_position = (
                measured_wrist[0, :3] + measured_wrist_rotation @ grasp_frame_offset
            )
            joint_error = float(np.max(np.abs(measured_body[12:19] - body_action[0, 12:19])))
            hold_drift = float(np.linalg.norm(measured_wrist[0, :3] - initial_wrist_position))
            print(
                f"[TELEOP] samples={recorder.sample_count} input_xyz={command[:3]} "
                f"scripted_waypoint={scripted_phase + 1 if args_cli.scripted_grasp else '-'} "
                f"goal_position_error={scripted_position_error:.4f}m "
                f"goal_rotation_error={scripted_rotation_error:.4f}rad "
                f"pinch_center_error={scripted_contact_center_error:.4f}m "
                f"palm_tilt_error={scripted_palm_tilt_error:.4f}rad "
                f"target_xyz={target_position} measured_xyz={measured_wrist[0, :3]} "
                f"measured_grasp_xyz={measured_grasp_position} "
                f"axis_error={axis_error} eef_error={eef_error:.4f}m "
                f"max_joint_error={joint_error:.4f}rad "
                f"wrist_drift_from_reset={hold_drift:.4f}m"
            )
        if args_cli.real_time:
            delay = env.step_dt - (time.time() - started)
            if delay > 0:
                time.sleep(delay)

    if recorder.sample_count:
        print(f"[TELEOP] Exiting with {recorder.sample_count} unsaved samples (not marked successful).")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
