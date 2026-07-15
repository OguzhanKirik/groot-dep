"""Shared constants for Adam-U GR00T integration."""

from configs.joint_state import RIGHT_ARM_JOINT_NAMES

# Camera sensor names in the Isaac Lab scene.
FRONT_CAMERA_NAME = "front_camera"
WRIST_CAMERA_NAME = "wrist_camera"

# GR00T Policy API modality keys (used by the adapter and future finetune config).
GROOT_VIDEO_KEYS = {
    FRONT_CAMERA_NAME: "front",
    WRIST_CAMERA_NAME: "wrist",
}
GROOT_STATE_KEY = "right_arm"
GROOT_ACTION_KEY = "right_arm"
GROOT_LANGUAGE_KEY = "task"

# Default language instruction for the pick-and-place task.
DEFAULT_TASK_INSTRUCTION = "pick up the cube and place it on the green target"

# Camera resolution (GR00T commonly uses 256x256 after preprocessing).
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256

# Number of sim steps between policy inference calls.
DEFAULT_EXECUTION_HORIZON = 8
