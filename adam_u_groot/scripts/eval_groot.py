"""Evaluate Adam-U grasp env with GR00T N1.7 (or zero/random baselines)."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

# Repo paths — must be set before local imports.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_ADAM_U_GROOT_ROOT = os.path.join(_REPO_ROOT, "adam_u_groot")
_ADAM_U_RL_ROOT = os.path.join(_REPO_ROOT, "adam_u_rl")

for _path in (_REPO_ROOT, _ADAM_U_GROOT_ROOT, _ADAM_U_RL_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_groot_env_cfg_class():
    cfg_path = os.path.join(_ADAM_U_GROOT_ROOT, "envs", "adam_u_grasp_groot_env_cfg.py")
    spec = importlib.util.spec_from_file_location("adam_u_grasp_groot_env_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GR00T env config from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AdamUGraspGrootEnvCfg, module.make_groot_env_cfg

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate GR00T N1.7 on Adam-U grasp environment.")
parser.add_argument(
    "--gui",
    action="store_true",
    default=False,
    help="Open the Isaac Sim window (Isaac Lab 6 default is headless; this sets --viz kit).",
)
parser.add_argument(
    "--mode",
    type=str,
    default="zero",
    choices=["zero", "random", "scripted_reach", "groot"],
    help="Policy mode: zero, random, Jacobian-IK cube reach, or GR00T PolicyClient.",
)
parser.add_argument("--task", type=str, default="pick up the cube and place it on the green target", help="Language instruction for GR00T.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps.")
parser.add_argument(
    "--execution_horizon",
    type=int,
    default=8,
    help="Number of predicted actions to execute per GR00T inference call.",
)
parser.add_argument("--groot_host", type=str, default="localhost", help="GR00T policy server host.")
parser.add_argument("--groot_port", type=int, default=5555, help="GR00T policy server port.")
parser.add_argument("--groot-launch-server", action="store_true", default=False)
parser.add_argument("--groot-server-conda-env", type=str, default="adam-u-groot-unified")
parser.add_argument(
    "--groot-server-script",
    type=str,
    default=os.path.expanduser("~/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"),
)
parser.add_argument("--groot-server-log", type=str, default="/tmp/gr00t_server_5555.log")
parser.add_argument(
    "--groot-inprocess",
    action="store_true",
    default=False,
    help=(
        "Load GR00T in the same Python process after Isaac Sim starts (recommended for "
        "adam-u-groot-unified). Avoids GPU/PhysX conflicts from a background server."
    ),
)
parser.add_argument(
    "--groot-model-path",
    type=str,
    default="/home/revel/models/GR00T-N1.7-3B",
    help="Checkpoint path for --groot-inprocess mode.",
)
parser.add_argument(
    "--groot-schema",
    type=str,
    default="real_g1",
    choices=["real_g1", "adam_u"],
    help=(
        "GR00T I/O schema. 'real_g1' shims Adam-U obs to the base REAL_G1 checkpoint for pipeline tests. "
        "'adam_u' uses native keys and requires a finetuned NEW_EMBODIMENT checkpoint."
    ),
)
parser.add_argument(
    "--groot-modality-config-path",
    type=str,
    default=os.path.join(_ADAM_U_GROOT_ROOT, "examples", "adam_u_modality_register.py"),
    help="Python modality registration used by the native Adam-U NEW_EMBODIMENT checkpoint.",
)
parser.add_argument(
    "--control-mode",
    choices=["joint_space", "eef_space", "g1_real"],
    default="joint_space",
    help=(
        "Arm control path. g1_real reproduces the former simulation-only right-arm "
        "double-offset behavior and is unsafe for real hardware."
    ),
)
parser.add_argument(
    "--execution-scope",
    choices=["right_arm_hand", "full"],
    default="right_arm_hand",
    help=(
        "Which GR00T outputs reach Adam-U. right_arm_hand holds waist, neck, left arm, "
        "and left hand at their reset targets while still observing their full state."
    ),
)
parser.add_argument(
    "--raw-relative-arm-actions",
    action="store_true",
    help="Treat received arm-joint commands as deltas. Leave off for PolicyClient-postprocessed outputs.",
)
parser.add_argument(
    "--neck-neutral", nargs=2, type=float, default=(0.0, -0.35), metavar=("YAW", "PITCH"),
    help="Held Adam-U neck pose in radians; negative pitch looks down toward the table.",
)
parser.add_argument("--max-position-step", type=float, default=0.10)
parser.add_argument(
    "--eef-ik-joint-step",
    type=float,
    default=0.01,
    help="Maximum EEF-IK change per Adam-U arm joint per simulation step (radians).",
)
parser.add_argument("--action-smoothing-alpha", type=float, default=1.0)
parser.add_argument(
    "--camera-debug-image",
    type=str,
    default="",
    help="Optional path for saving the first rendered GR00T front-camera RGB frame.",
)
parser.add_argument(
    "--reach-hover-height", type=float, default=0.12,
    help="Scripted reach target height in meters above the cube center.",
)
parser.add_argument(
    "--reach-joint-step", type=float, default=0.03,
    help="Maximum scripted IK change per right-arm joint and simulation step (radians).",
)
parser.add_argument(
    "--reach-ik-backend", choices=["custom", "isaac"], default="isaac",
    help="IK implementation for scripted_reach; isaac is the trusted baseline.",
)
parser.add_argument(
    "--joint-state-group",
    type=str,
    default="right_arm",
    help="URDF joint group to print/log (right_arm, right_hand, waist, all, ...).",
)
parser.add_argument(
    "--print-joint-states",
    action="store_true",
    default=False,
    help="Print named joint positions each log step.",
)
parser.add_argument(
    "--print-groot-actions",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print GR00T action chunks and per-step env actions to the terminal (groot mode).",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time.")
parser.add_argument("--video", action="store_true", default=False, help="Record rollout video.")

AppLauncher.add_app_launcher_args(parser)

# Isaac Lab 6 runs headless unless a visualizer is selected. Map --gui -> --viz kit.
if "--gui" in sys.argv:
    gui_index = sys.argv.index("--gui")
    sys.argv.pop(gui_index)
    if not any(arg == "--viz" or arg == "--visualizer" for arg in sys.argv):
        sys.argv.extend(["--viz", "kit"])

args_cli = parser.parse_args()

need_scene_cameras = args_cli.mode == "groot" or args_cli.video
if need_scene_cameras:
    args_cli.enable_cameras = True
elif args_cli.enable_cameras:
    print(
        "[INFO] --enable_cameras ignored for this mode/schema. "
        "Scene cameras are enabled only for GR00T evaluation or video recording.",
        flush=True,
    )
    args_cli.enable_cameras = False

sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedEnv

from adapters.groot_adapter import GrootAdapter
from adapters.adam_u_action_adapter import DampedLeastSquaresIKSolver
from adapters.eef_pose import (
    G1AdamWorkspaceTransform,
    IsaacAdamUKinematicsProvider,
    IsaacDifferentialIKSolver,
)
from adapters.joint_state_reader import JointStateReader
from adapters.virtual_g1_state import VirtualG1StateAdapter
from configs.constants import DEFAULT_TASK_INSTRUCTION
from configs.adam_u_action_mapping import AdamUActionMappingConfig
from configs.groot_schemas import GrootSchema, get_groot_schema
from configs.joint_state import RIGHT_ARM_JOINT_NAMES
from envs.scene_layout import TABLE_SURFACE_Z

AdamUGraspGrootEnvCfg, make_groot_env_cfg = _load_groot_env_cfg_class()

# The recovery works in the Isaac-only environment. Its synchronous Kit
# update/rebind sequence deadlocks in the unified process, with or without RTX
# camera render products.
_conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
_use_physics_recovery = "groot-unified" not in _conda_env
if _use_physics_recovery:
    _recovery_spec = importlib.util.spec_from_file_location(
        "isaac_physics_recovery",
        os.path.join(_ADAM_U_GROOT_ROOT, "envs", "isaac_physics_recovery.py"),
    )
    _recovery_mod = importlib.util.module_from_spec(_recovery_spec)
    assert _recovery_spec.loader is not None
    _recovery_spec.loader.exec_module(_recovery_mod)
    _recovery_mod.install_manager_env_physics_recovery_patch()


def _runs_headless(args) -> bool:
    if getattr(args, "headless", False):
        return True
    visualizers = getattr(args, "visualizer", None)
    if not visualizers:
        return True
    return "kit" not in visualizers


def _log_scene_spawn_status(env) -> None:
    """Print USD prim + physics positions so empty-viewport issues are easy to diagnose."""
    stage = env.sim.stage
    prim_paths = (
        "/World/envs/env_0/Robot",
        "/World/envs/env_0/TableTop",
        "/World/envs/env_0/TableLeg",
        "/World/envs/env_0/Object",
        "/World/GroundPlane",
    )
    print("[INFO] Scene spawn check:", flush=True)
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        valid = prim.IsValid()
        print(f"  {path}: {'OK' if valid else 'MISSING'}", flush=True)

    robot_pos = _to_numpy_array(env.scene["robot"].data.root_pos_w)
    table_pos = _to_numpy_array(env.scene["table_top"].data.root_pos_w)
    object_pos = _to_numpy_array(env.scene["object"].data.root_pos_w)
    print(
        f"[INFO] robot pos={robot_pos.reshape(-1)[:3]}, "
        f"table pos={table_pos.reshape(-1)[:3]}, "
        f"object pos={object_pos.reshape(-1)[:3]}",
        flush=True,
    )


def _frame_viewport(env) -> None:
    """Point the kit viewport at the robot/table after env init (GUI mode)."""
    if env.viewport_camera_controller is not None:
        env.viewport_camera_controller.update_view_to_world()
        env.sim.set_camera_view(
            eye=list(env.cfg.viewer.eye),
            target=list(env.cfg.viewer.lookat),
        )
        env.sim.render()


def _frame_front_camera(env) -> None:
    """Aim the GR00T policy camera at the current task workspace."""
    if "front_camera" not in env.scene.keys():
        return
    from envs.scene_layout import FRONT_CAMERA_LOOKAT, FRONT_CAMERA_POS

    camera = env.scene["front_camera"]
    eyes = torch.tensor([FRONT_CAMERA_POS], device=env.device, dtype=torch.float32)
    targets = torch.tensor([FRONT_CAMERA_LOOKAT], device=env.device, dtype=torch.float32)
    camera.set_world_poses_from_view(eyes, targets)
    # Refresh the render product before the first GR00T observation is packed.
    env.sim.render()
    print(
        f"[INFO] GR00T front camera eye={FRONT_CAMERA_POS}, lookat={FRONT_CAMERA_LOOKAT}",
        flush=True,
    )


def _save_front_camera_debug_image(env, path: str) -> None:
    """Save the exact first RGB tensor supplied to GR00T for visual diagnosis."""
    if not path or "front_camera" not in env.scene.keys():
        return
    from pathlib import Path
    from PIL import Image

    rgb = env.scene["front_camera"].data.output["rgb"][0]
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.detach().cpu().numpy()).save(output)
    print(f"[INFO] Saved GR00T front-camera debug frame: {output}", flush=True)


def _make_zero_action(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def _make_random_action(env) -> torch.Tensor:
    return torch.empty(env.num_envs, env.action_manager.total_action_dim, device=env.device).uniform_(
        -0.05, 0.05
    )


def _make_scripted_reach_action(
    env, adapter, provider, solver, hold_body: np.ndarray, hold_hands: np.ndarray
) -> tuple[torch.Tensor, float, float]:
    """Move only Adam-U's right arm toward a hover point above the cube."""
    current_body = adapter._current_body()
    current_right_arm = current_body[:, 12:19]
    current_pose, _ = provider("right", current_right_arm)

    object_pos_w = _to_numpy_array(env.scene["object"].data.root_pos_w)
    env_origins = _to_numpy_array(env.scene.env_origins)
    object_pos = object_pos_w - env_origins
    target_pose = current_pose.copy()
    target_pose[:, :3] = object_pos + np.asarray((0.0, 0.0, args_cli.reach_hover_height))
    right_arm_target = solver.solve(
        "right", target_pose, current_right_arm, command_is_relative=False
    ).astype(np.float32)

    # Keep every non-controlled joint at the pose captured immediately after
    # reset. Following measured values would ratchet gravity/actuator drift into
    # the next command and make the supposedly fixed body slowly collapse.
    body_target = hold_body.copy()
    body_target[:, 12:19] = right_arm_target
    body_lower, body_upper, _ = AdamUActionMappingConfig().body_limits
    body_target = np.clip(body_target, body_lower, body_upper).astype(np.float32)
    sim_action = np.concatenate((body_target, hold_hands), axis=1)
    current_distance = float(np.linalg.norm(current_pose[0, :3] - object_pos[0]))
    hover_error = float(np.linalg.norm(current_pose[0, :3] - target_pose[0, :3]))
    return (
        torch.as_tensor(sim_action, device=env.device, dtype=torch.float32),
        current_distance,
        hover_error,
    )


def _to_numpy_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    # Isaac Lab 6 returns some sim buffers (e.g. root_pos_w) as warp arrays, which
    # don't support item indexing or .item(); .numpy() copies device->host.
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        return value.numpy()
    return np.asarray(value)


def _format_action_vector(values: np.ndarray, joint_names: tuple[str, ...]) -> str:
    flat = values.reshape(-1)
    if len(joint_names) == len(flat):
        return ", ".join(f"{name}={float(v):+.4f}" for name, v in zip(joint_names, flat))
    return np.array2string(flat, precision=4, separator=", ")


def _print_groot_action_chunk(
    groot_action: dict,
    schema: GrootSchema,
    inference_index: int,
    execution_horizon: int,
) -> None:
    print(f"[GROOT] inference #{inference_index} — action keys from server:")
    for key, arr in sorted(groot_action.items()):
        np_arr = _to_numpy_array(arr)
        print(f"  {key}: shape={tuple(np_arr.shape)} dtype={np_arr.dtype}")

    env_key = schema.env_action_key
    if env_key in groot_action:
        chunk = _to_numpy_array(groot_action[env_key])
        steps = min(chunk.shape[1] if chunk.ndim == 3 else 1, execution_horizon)
        print(f"[GROOT] env action key '{env_key}' (first {steps} steps, env 0):")
        for t in range(steps):
            step_vals = chunk[0, t, :] if chunk.ndim == 3 else chunk[0, :]
            print(f"  t={t}: {_format_action_vector(step_vals, RIGHT_ARM_JOINT_NAMES)}")


def _print_groot_env_step(
    step: int,
    chunk_index: int,
    env_action: torch.Tensor,
) -> None:
    vals = env_action[0].detach().cpu().numpy()
    print(
        f"[GROOT] step={step} chunk_index={chunk_index} "
        f"applied={_format_action_vector(vals, RIGHT_ARM_JOINT_NAMES)}"
    )


def _load_groot_policy_remote(host: str, port: int):
    try:
        from adapters.groot_policy_client import GrootPolicyClient
    except ImportError as exc:
        raise ImportError(
            "GR00T client dependencies missing in this env. Install with: "
            "pip install pyzmq msgpack-numpy"
        ) from exc

    policy = GrootPolicyClient(host=host, port=port, timeout_ms=15000)
    if not policy.ping():
        raise RuntimeError(
            f"Cannot connect to GR00T policy server at {host}:{port}. "
            "Stop stale servers with: pkill -f run_gr00t_server.py "
            "or use --groot-inprocess in adam-u-groot-unified."
        )
    return policy


def _load_groot_policy_inprocess(model_path: str, embodiment_tag: str, device: str):
    if args_cli.groot_schema == "adam_u":
        config_path = os.path.abspath(args_cli.groot_modality_config_path)
        spec = importlib.util.spec_from_file_location("adam_u_modality_register_runtime", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Adam-U modality config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"[INFO] Registered native Adam-U modality config: {config_path}")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    print(f"[INFO] Loading GR00T in-process from {model_path} ({embodiment_tag})...")
    return Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
        model_path=model_path,
        device=device,
    )


def _stop_groot_server(process: subprocess.Popen, log_file=None) -> None:
    """Terminate the complete conda/server process group, including orphaned children."""
    process_group_id = getattr(process, "_groot_process_group_id", process.pid)
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    if log_file is not None and not log_file.closed:
        log_file.close()


def _launch_groot_server_after_isaac(embodiment_tag: str) -> tuple[subprocess.Popen, object]:
    """Start GR00T only after PhysX owns its CUDA context."""
    log_file = open(args_cli.groot_server_log, "w", buffering=1)
    command = [
        "conda", "run", "-n", args_cli.groot_server_conda_env, "--no-capture-output",
        "python", args_cli.groot_server_script,
        "--model-path", args_cli.groot_model_path,
        "--embodiment-tag", embodiment_tag,
        "--device", "cuda:0",
        "--port", str(args_cli.groot_port),
    ]
    if args_cli.groot_schema == "adam_u":
        command.extend(("--modality-config-path", os.path.abspath(args_cli.groot_modality_config_path)))
    print("[INFO] Isaac is ready; starting the GR00T server now...", flush=True)
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._groot_process_group_id = process.pid
    shutdown_callback = lambda: _stop_groot_server(process, log_file)
    process._groot_shutdown_callback = shutdown_callback
    atexit.register(shutdown_callback)
    for _ in range(300):
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"GR00T server exited early; see {args_cli.groot_server_log}"
            )
        try:
            with socket.create_connection((args_cli.groot_host, args_cli.groot_port), timeout=0.2):
                print("[INFO] GR00T server is listening.", flush=True)
                return process, log_file
        except OSError:
            time.sleep(1)
    _stop_groot_server(process, log_file)
    raise TimeoutError(f"Timed out waiting for GR00T server; see {args_cli.groot_server_log}")


def main():
    if args_cli.eef_ik_joint_step <= 0.0:
        raise ValueError("--eef-ik-joint-step must be positive")
    if args_cli.max_position_step <= 0.0:
        raise ValueError("--max-position-step must be positive")
    if args_cli.control_mode == "g1_real" and (
        args_cli.mode != "groot" or args_cli.groot_schema != "real_g1"
    ):
        raise ValueError("--control-mode g1_real requires --mode groot --groot-schema real_g1")
    # GPU PhysX 110 intermittently blocks in initialize_physics() for the
    # high-link-count Adam-U articulation on this Isaac Sim 6/Blackwell setup.
    # A single evaluation environment does not benefit materially from the GPU
    # pipeline; RTX rendering and the separate GR00T server still use the GPU.
    default_physics_device = "cpu"
    device = os.environ.get("ADAM_U_PHYSICS_DEVICE", default_physics_device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested physics device {device!r} is unavailable; using CPU.", flush=True)
        device = "cpu"
    print(f"[INFO] Isaac physics device: {device}", flush=True)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    if args_cli.mode == "zero" and "groot-unified" in conda_env:
        print(
            "[ERROR] zero-mode sim is unreliable in 'adam-u-groot-unified' (torch 2.9 breaks PhysX).\n"
            "  Use the Isaac-only env instead:\n"
            "    conda activate adam-u-isaac-6\n"
            "    python adam_u_groot/scripts/eval_groot.py --mode zero --gui --max_steps 1000\n"
            "  Do not pass --enable_cameras for zero mode (GUI viewport is enough).",
            flush=True,
        )
        simulation_app.close()
        return

    env_cfg = make_groot_env_cfg(
        num_envs=args_cli.num_envs,
        include_cameras=need_scene_cameras,
        include_wrist_camera=args_cli.groot_schema == "adam_u",
        full_body_actions=(
            args_cli.groot_schema in ("real_g1", "adam_u") and args_cli.control_mode != "g1_real"
        ),
        legacy_g1_real_actions=args_cli.control_mode == "g1_real",
    )
    env_cfg.sim.device = device
    # Resolve URDF path relative to the repo root.
    urdf_path = os.path.abspath(os.path.join(_REPO_ROOT, "assets/robots/adam_u/urdf/adam_u.urdf"))
    env_cfg.scene.robot.spawn.asset_path = urdf_path
    # Full-body self-collision makes PhysX 110 intermittently deadlock while
    # initializing this high-link-count URDF. External contacts (table/cube)
    # remain enabled, which is what the grasp task requires.
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    render_mode = "rgb_array" if args_cli.video else None
    if render_mode is not None:
        env_cfg.video_recorder.env_render_mode = render_mode
    print("[INFO] Creating Isaac environment (this can take 30-60s)...", flush=True)
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[INFO] Environment ready.", flush=True)
    _log_scene_spawn_status(env)
    if not _runs_headless(args_cli):
        _frame_viewport(env)

    if args_cli.video:
        video_path = os.path.join(_REPO_ROOT, "logs", "groot", "videos")
        os.makedirs(video_path, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_path,
            step_trigger=lambda step: step == 0,
            video_length=min(args_cli.max_steps, 200),
            disable_logger=True,
        )

    groot_schema = get_groot_schema(args_cli.groot_schema)
    action_mapping_config = AdamUActionMappingConfig(
        control_mode="joint_space" if args_cli.control_mode == "g1_real" else args_cli.control_mode,
        arm_commands_relative=args_cli.raw_relative_arm_actions,
        neck_neutral=tuple(args_cli.neck_neutral),
        max_body_position_step=args_cli.max_position_step,
        max_hand_position_step=args_cli.max_position_step,
        smoothing_alpha=args_cli.action_smoothing_alpha,
        log_commands=args_cli.print_groot_actions,
    )
    ik_solver = None
    workspace_transform = G1AdamWorkspaceTransform()
    virtual_g1_state = VirtualG1StateAdapter() if args_cli.groot_schema == "real_g1" else None
    if args_cli.groot_schema == "real_g1" and args_cli.control_mode == "eef_space":
        ik_solver = IsaacDifferentialIKSolver(
            IsaacAdamUKinematicsProvider(env),
            max_joint_delta=min(args_cli.eef_ik_joint_step, args_cli.max_position_step),
            # G1 and Adam-U wrist/tool axes have not been calibrated yet.
            # Position control is safe after workspace conversion; pose control
            # would make IK chase an unknown tool-frame rotation offset.
            command_type="position",
            workspace_transform=workspace_transform,
        )
        print(
            "[INFO] Adam-U EEF control enabled: G1-canonical <-> Adam-world conversion, "
            "Isaac position IK, and persistent joint targets; GR00T arm joints will be ignored.",
            flush=True,
        )
    adapter = GrootAdapter(
        env,
        task_instruction=args_cli.task or DEFAULT_TASK_INSTRUCTION,
        execution_horizon=args_cli.execution_horizon,
        schema=groot_schema,
        action_mapping_config=action_mapping_config,
        ik_solver=ik_solver,
        legacy_g1_real=args_cli.control_mode == "g1_real",
        workspace_transform=workspace_transform,
        virtual_g1_state=virtual_g1_state,
        execution_scope=args_cli.execution_scope,
    )
    if args_cli.execution_scope == "right_arm_hand":
        print(
            "[INFO] Execution scope=right_arm_hand: waist, neck, left arm, and left hand "
            "will hold their post-reset targets; full state is still sent to GR00T.",
            flush=True,
        )
    if args_cli.control_mode == "g1_real":
        print(
            "[WARN] g1_real compatibility mode: right arm only, default pose offset intentionally added; "
            "SIMULATION ONLY.",
            flush=True,
        )
    scripted_provider = None
    scripted_solver = None
    if args_cli.mode == "scripted_reach":
        scripted_provider = IsaacAdamUKinematicsProvider(env)
        if args_cli.reach_ik_backend == "isaac":
            scripted_solver = IsaacDifferentialIKSolver(
                scripted_provider, max_joint_delta=args_cli.reach_joint_step
            )
        else:
            scripted_solver = DampedLeastSquaresIKSolver(
                scripted_provider, damping=0.05, max_joint_delta=args_cli.reach_joint_step
            )
        print(
            f"[INFO] Scripted reach enabled: right arm only, IK backend={args_cli.reach_ik_backend}; "
            "GR00T is not loaded.",
            flush=True,
        )
    joint_state_reader = JointStateReader(env, group=args_cli.joint_state_group)

    groot_policy = None
    groot_server_process = None
    groot_server_log = None
    if args_cli.mode == "groot":
        if args_cli.groot_inprocess:
            print(
                f"[INFO] GR00T in-process mode: Isaac Sim started; loading model next "
                f"(schema='{groot_schema.name}', embodiment={groot_schema.embodiment_tag})."
            )
            groot_policy = _load_groot_policy_inprocess(
                args_cli.groot_model_path,
                groot_schema.embodiment_tag,
                device,
            )
        else:
            if args_cli.groot_launch_server:
                groot_server_process, groot_server_log = _launch_groot_server_after_isaac(
                    groot_schema.embodiment_tag
                )
            groot_policy = _load_groot_policy_remote(args_cli.groot_host, args_cli.groot_port)
            print(f"[INFO] Connected to GR00T server at {args_cli.groot_host}:{args_cli.groot_port}")
            print(
                f"[INFO] GR00T schema='{groot_schema.name}' "
                f"(server embodiment: {groot_schema.embodiment_tag})"
            )

    env.reset()
    _frame_front_camera(env)
    _save_front_camera_debug_image(env, args_cli.camera_debug_image)
    scripted_hold_body = None
    scripted_hold_hands = None
    if args_cli.mode == "scripted_reach":
        scripted_hold_body = adapter._current_body().copy()
        scripted_hold_hands = np.concatenate(
            (
                adapter._left_hand_state.get_positions().detach().cpu().numpy(),
                adapter._right_hand_state.get_positions().detach().cpu().numpy(),
            ),
            axis=1,
        ).astype(np.float32)
        initial_wrist_pose, _ = scripted_provider("right", scripted_hold_body[:, 12:19])
        wrist_z = float(initial_wrist_pose[0, 2])
        clearance = wrist_z - float(TABLE_SURFACE_Z)
        print(
            f"[REACH] initial right wrist xyz={initial_wrist_pose[0, :3]}, "
            f"table clearance={clearance:.3f} m",
            flush=True,
        )
        if clearance <= 0.0:
            raise RuntimeError(
                "Right wrist reset pose is not above the tabletop; "
                f"clearance={clearance:.3f} m"
            )
        print("[INFO] Latched fixed targets for waist, neck, left arm, and both hands.", flush=True)
    dt = env.step_dt
    print(f"[INFO] Running eval mode='{args_cli.mode}' for up to {args_cli.max_steps} steps.")

    step = 0
    chunk_index = 0
    groot_action = None
    inference_index = 0

    while step < args_cli.max_steps:
        headless = _runs_headless(args_cli)
        if not headless and not simulation_app.is_running():
            break
        start_time = time.time()

        with torch.inference_mode():
            if args_cli.mode == "zero":
                actions = _make_zero_action(env)
            elif args_cli.mode == "random":
                actions = _make_random_action(env)
            elif args_cli.mode == "scripted_reach":
                actions, wrist_cube_distance, hover_error = _make_scripted_reach_action(
                    env,
                    adapter,
                    scripted_provider,
                    scripted_solver,
                    scripted_hold_body,
                    scripted_hold_hands,
                )
            else:
                if groot_action is None or chunk_index >= adapter.get_execution_horizon(groot_action):
                    obs_groot = adapter.build_observation()
                    groot_action, info = groot_policy.get_action(obs_groot)
                    chunk_index = 0
                    inference_index += 1
                    if args_cli.print_groot_actions:
                        _print_groot_action_chunk(
                            groot_action,
                            groot_schema,
                            inference_index,
                            args_cli.execution_horizon,
                        )
                        if info:
                            print(f"[GROOT] server info: {info}")

                actions = adapter.action_to_env(groot_action, step_index=chunk_index)
                if args_cli.print_groot_actions:
                    _print_groot_env_step(step + 1, chunk_index, actions)
                chunk_index += 1

            env.step(actions)

        step += 1
        if step % 10 == 0 or step == args_cli.max_steps:
            obj_pos = _to_numpy_array(env.scene["object"].data.root_pos_w)
            obj_z = float(obj_pos[0, 2] if obj_pos.ndim >= 2 else obj_pos[2])
            print(f"[INFO] step={step}, object height={obj_z:.3f}")
            if args_cli.mode == "scripted_reach":
                print(
                    f"[REACH] right wrist-to-cube={wrist_cube_distance:.3f} m, "
                    f"hover-target error={hover_error:.3f} m",
                    flush=True,
                )
            if args_cli.print_joint_states:
                joint_dict = joint_state_reader.as_dict(env_index=0)
                joint_line = ", ".join(f"{name}={value:.4f}" for name, value in joint_dict.items())
                print(f"[JOINTS:{args_cli.joint_state_group}] {joint_line}")

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    if groot_server_process is not None:
        _stop_groot_server(groot_server_process, groot_server_log)
        atexit.unregister(groot_server_process._groot_shutdown_callback)
    simulation_app.close()
    print(f"[INFO] Finished eval after {step} steps.")


if __name__ == "__main__":
    main()
