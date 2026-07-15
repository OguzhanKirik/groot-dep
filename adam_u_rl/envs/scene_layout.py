"""Shared scene layout for Adam-U grasp environments (Isaac Lab, Z-up world)."""

from __future__ import annotations

# Table geometry (table_top cuboid center + half-height = surface).
# Adam-U faces -X. Keep the cube inside the fixed-waist arm workspace.
TABLE_X = -0.50
TABLE_Y = 0.0
TABLE_TOP_POS = (TABLE_X, TABLE_Y, 1.00)
TABLE_TOP_SIZE = (0.6, 0.5, 0.05)
TABLE_SURFACE_Z = TABLE_TOP_POS[2] + TABLE_TOP_SIZE[2] * 0.5  # 0.875 m

TABLE_LEG_HEIGHT = TABLE_TOP_POS[2]
TABLE_LEG_POS = (TABLE_X, TABLE_Y, TABLE_LEG_HEIGHT * 0.5)

# URDF root (lifting_Columns) sits above the feet; lift so lowest geometry clears z=0 floor.
ROBOT_BASE_Z = 1.00

# Robot at world origin; table is ahead along -X.
ROBOT_POS = (0.0, 0.0, ROBOT_BASE_Z)
# Upright and right-side up. Identity is upside-down in Isaac; flip 180 deg about Y fixes it.
ROBOT_ROT = (0.0, 0.0, 1.0, 0.0)

# Manipulation targets on the table surface.
# Keep the cube distinct from, but close to, the green placement target so the
# fixed-cube teleop episode requires a short transfer within right-arm reach.
OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_POS = (TABLE_X + 0.14, TABLE_Y + 0.05, TABLE_SURFACE_Z + OBJECT_SIZE[2] * 0.5)
PLACE_TARGET_POS = (TABLE_X + 0.25, TABLE_Y + 0.15, TABLE_SURFACE_Z + 0.005)

# Default viewer framing (robot at origin, table in front).
# GUI viewport from the far side of the table, looking back toward Adam-U.
# This does not change the separate front-camera sensor supplied to GR00T.
VIEWER_EYE = (-1.2, 0.9, 1.5)
VIEWER_LOOKAT = (-0.25, 0.0, TABLE_SURFACE_Z + 0.08)

# Front camera (world frame) for GR00T eval — third-person view of table and robot.
FRONT_CAMERA_POS = (0.65, -0.85, 1.55)
# This initial rotation only permits sensor creation.  eval_groot.py sets the
# exact world-space look-at pose after the Camera has initialized, so moving the
# table cannot silently leave the policy camera aimed at an obsolete location.
FRONT_CAMERA_ROT = (1.0, 0.0, 0.0, 0.0)
FRONT_CAMERA_LOOKAT = (TABLE_X, TABLE_Y, TABLE_SURFACE_Z + 0.08)
