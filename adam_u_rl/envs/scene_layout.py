"""Shared scene layout for Adam-U grasp environments (Isaac Lab, Z-up world)."""

from __future__ import annotations

# Table geometry (table_top cuboid center + half-height = surface).
# Adam-U faces -X, so the table sits 0.5 m in front of the robot.
TABLE_X = -0.5
TABLE_Y = 0.0
TABLE_TOP_POS = (TABLE_X, TABLE_Y, 1.0)
TABLE_TOP_SIZE = (0.6, 0.5, 0.05)
TABLE_SURFACE_Z = TABLE_TOP_POS[2] + TABLE_TOP_SIZE[2] * 0.5  # 1.025 m

TABLE_LEG_POS = (TABLE_X, TABLE_Y, 0.5)
TABLE_LEG_HEIGHT = 1.0

# URDF root (lifting_Columns) sits above the feet; lift so lowest geometry clears z=0 floor.
ROBOT_BASE_Z = 1.00

# Robot at world origin; table is ahead along -X.
ROBOT_POS = (0.0, 0.0, ROBOT_BASE_Z)
# Upright and right-side up. Identity is upside-down in Isaac; flip 180 deg about Y fixes it.
ROBOT_ROT = (0.0, 0.0, 1.0, 0.0)

# Manipulation targets on the table surface.
OBJECT_POS = (TABLE_X, TABLE_Y, TABLE_SURFACE_Z + 0.025)  # 5 cm cube half-height
PLACE_TARGET_POS = (TABLE_X + 0.25, TABLE_Y + 0.15, TABLE_SURFACE_Z + 0.005)

# Default viewer framing (robot at origin, table in front).
VIEWER_EYE = (1.6, -0.85, 1.6)
VIEWER_LOOKAT = (0.0, 0.35, 1.05)

# Front camera (world frame) for GR00T eval — third-person view of table and robot.
FRONT_CAMERA_POS = (1.2, -0.65, 1.55)
FRONT_CAMERA_ROT = (0.9238795, 0.0, 0.3826834, 0.0)  # look toward table
