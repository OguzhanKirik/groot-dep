"""Generate state-consistent virtual REAL_G1 arm joints from Adam-U wrist poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np


G1_URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "g1_29dof_with_hand_rev_1_0.urdf"

G1_ARM_JOINT_NAMES = {
    side: tuple(
        f"{side}_{suffix}"
        for suffix in (
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
        )
    )
    for side in ("left", "right")
}


class VirtualG1StateAdapter:
    """Track continuous pseudo-G1 joints matching G1-canonical wrist positions.

    The solve is position-only until the G1/Adam-U tool-frame rotation is
    calibrated. Each arm is seeded from its previous virtual solution, which
    prevents elbow flips and makes the state supplied to GR00T continuous.
    """

    def __init__(
        self,
        urdf_path: str | Path = G1_URDF_PATH,
        *,
        damping: float = 0.03,
        max_iterations: int = 80,
        max_joint_update: float = 0.08,
        tolerance: float = 0.01,
    ) -> None:
        import pinocchio as pin

        path = Path(urdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"Official G1 URDF is missing: {path}")
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.max_joint_update = float(max_joint_update)
        self.tolerance = float(tolerance)
        self.q = pin.neutral(self.model)
        self._joint_q_ids: dict[str, list[int]] = {}
        self._joint_v_ids: dict[str, list[int]] = {}
        self._frame_ids: dict[str, int] = {}
        for side in ("left", "right"):
            q_ids, v_ids = [], []
            for name in G1_ARM_JOINT_NAMES[side]:
                joint_id = self.model.getJointId(name)
                if joint_id == 0:
                    raise ValueError(f"G1 arm joint is absent from URDF: {name}")
                joint = self.model.joints[joint_id]
                if joint.nq != 1 or joint.nv != 1:
                    raise ValueError(f"G1 arm joint must be scalar: {name}")
                q_ids.append(joint.idx_q)
                v_ids.append(joint.idx_v)
            self._joint_q_ids[side] = q_ids
            self._joint_v_ids[side] = v_ids
            frame_name = f"{side}_wrist_yaw_link"
            frame_id = self.model.getFrameId(frame_name)
            if frame_id >= len(self.model.frames):
                raise ValueError(f"G1 wrist frame is absent from URDF: {frame_name}")
            self._frame_ids[side] = frame_id

        # Isaac Lab's published G1 ready posture is a stable initial IK seed.
        ready = {
            "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "left_elbow_joint": 0.87,
            "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_elbow_joint": 0.87,
        }
        for name, value in ready.items():
            joint = self.model.joints[self.model.getJointId(name)]
            self.q[joint.idx_q] = value
        self.last_errors = {"left": np.inf, "right": np.inf}

    @staticmethod
    def _positions(pose_9d: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_9d, dtype=np.float64)
        if pose.ndim == 1:
            pose = pose[None, :]
        if pose.ndim != 2 or pose.shape[1] != 9 or not np.all(np.isfinite(pose)):
            raise ValueError(f"G1 wrist pose must be finite with shape (N, 9), got {pose.shape}")
        return pose[:, :3]

    def _solve_side(self, side: str, target: np.ndarray) -> float:
        pin = self.pin
        frame_id = self._frame_ids[side]
        v_ids = self._joint_v_ids[side]
        identity = np.eye(3)
        error_norm = np.inf
        for _ in range(self.max_iterations):
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[frame_id].translation
            error = target - current
            error_norm = float(np.linalg.norm(error))
            if error_norm <= self.tolerance:
                break
            full_jacobian = pin.computeFrameJacobian(
                self.model, self.data, self.q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            jacobian = full_jacobian[:3, v_ids]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.damping**2 * identity, error
            )
            delta = np.clip(delta, -self.max_joint_update, self.max_joint_update)
            velocity = np.zeros(self.model.nv, dtype=np.float64)
            velocity[v_ids] = delta
            self.q = pin.integrate(self.model, self.q, velocity)
            self.q = np.clip(self.q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        return error_norm

    def update(
        self, left_g1_pose_9d: np.ndarray, right_g1_pose_9d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        left_positions = self._positions(left_g1_pose_9d)
        right_positions = self._positions(right_g1_pose_9d)
        if left_positions.shape[0] != 1 or right_positions.shape[0] != 1:
            raise ValueError("Virtual G1 state currently supports exactly one evaluation environment")
        self.last_errors["left"] = self._solve_side("left", left_positions[0])
        self.last_errors["right"] = self._solve_side("right", right_positions[0])
        left = self.q[self._joint_q_ids["left"]][None, :].astype(np.float32)
        right = self.q[self._joint_q_ids["right"]][None, :].astype(np.float32)
        return left, right

