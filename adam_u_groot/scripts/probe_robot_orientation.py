"""Probe which root quaternion makes Adam-U stand upright in Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "adam_u_groot"), os.path.join(_REPO_ROOT, "adam_u_rl")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_groot_env_cfg_class():
    cfg_path = os.path.join(_REPO_ROOT, "adam_u_groot", "envs", "adam_u_grasp_groot_env_cfg.py")
    spec = importlib.util.spec_from_file_location("adam_u_grasp_groot_env_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AdamUGraspGrootEnvCfg


def _span(body_pos, axis: int) -> float:
    return float(body_pos[:, axis].max() - body_pos[:, axis].min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    args, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args(["--headless", "--enable_cameras"])
    sys.argv = [sys.argv[0]]
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedEnv

    AdamUGraspGrootEnvCfg = _load_groot_env_cfg_class()
    urdf_path = os.path.abspath(os.path.join(_REPO_ROOT, "assets/robots/adam_u/urdf/adam_u.urdf"))

    candidates = {
        "identity": (1.0, 0.0, 0.0, 0.0),
        "flip180x": (0.0, 1.0, 0.0, 0.0),
        "flip180y": (0.0, 0.0, 1.0, 0.0),
        "flip180z": (0.0, 0.0, 0.0, 1.0),
        "pitch90x": (0.7071068, 0.7071068, 0.0, 0.0),
        "pitch-90x": (0.7071068, -0.7071068, 0.0, 0.0),
        "yaw90z": (0.7071068, 0.0, 0.0, 0.7071068),
    }

    def _score(env) -> tuple[float, float]:
        robot = env.scene["robot"]
        body_pos = robot.data.body_pos_w[0].cpu()
        z_span = _span(body_pos, 2)
        names = robot.data.body_names
        # Prefer head above base: neck/head links should be higher than lifting column root.
        head_z = float(body_pos.max(dim=0).values[2])
        base_z = float(robot.data.root_pos_w[0, 2].item())
        return z_span, head_z - base_z

    print("Rotation probe (larger z_span and head_above_base => upright):")
    best_name = None
    best_score = -1.0
    for name, rot in candidates.items():
        env_cfg = AdamUGraspGrootEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.scene.robot.init_state.rot = rot
        env_cfg.scene.robot.spawn.asset_path = urdf_path
        env = ManagerBasedEnv(cfg=env_cfg)
        env.reset()
        body_pos = env.scene["robot"].data.body_pos_w[0].cpu()
        z_span = _span(body_pos, 2)
        x_span = _span(body_pos, 0)
        y_span = _span(body_pos, 1)
        z_span2, head_above = _score(env)
        print(
            f"  {name:10s} rot={rot}  "
            f"z_span={z_span:.3f}  head_above={head_above:.3f}  "
            f"x_span={x_span:.3f}  y_span={y_span:.3f}"
        )
        score = z_span2 if head_above > 0 else -z_span2
        if score > best_score:
            best_score = score
            best_name = name
        env.close()

    print(f"Best upright candidate: {best_name} (score={best_score:.3f})")

    from envs.scene_layout import ROBOT_BASE_Z, ROBOT_POS, ROBOT_ROT

    env_cfg = AdamUGraspGrootEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.init_state.pos = ROBOT_POS
    env_cfg.scene.robot.init_state.rot = ROBOT_ROT
    env_cfg.scene.robot.spawn.asset_path = urdf_path
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    body = env.scene["robot"].data.body_pos_w[0].cpu()
    print(
        f"Configured spawn pos={ROBOT_POS} rot={ROBOT_ROT} -> "
        f"body_z=[{float(body[:,2].min()):.3f}, {float(body[:,2].max()):.3f}]"
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
