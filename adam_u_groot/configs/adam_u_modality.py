"""Adam-U modality schema for GR00T N1.7 (NEW_EMBODIMENT).

For finetune / server registration (requires ``lerobot-groot`` env), use:

    adam_u_groot/examples/adam_u_modality_register.py

For LeRobot dataset layout, see ``adam_u_groot/configs/modality.json``.
"""

from configs.constants import DEFAULT_EXECUTION_HORIZON
from configs.groot_schemas import ADAM_U_SCHEMA

# Documented layout (matches ADAM_U_SCHEMA and modality.json).
ADAM_U_MODALITY_CONFIG = {
    "video": {
        "delta_indices": [0],
        "modality_keys": list(ADAM_U_SCHEMA.video_keys),
    },
    "state": {
        "delta_indices": [0],
        "modality_keys": list(ADAM_U_SCHEMA.state_keys),
    },
    "action": {
        "delta_indices": list(range(DEFAULT_EXECUTION_HORIZON)),
        "modality_keys": list(ADAM_U_SCHEMA.action_keys),
    },
    "language": {
        "delta_indices": [0],
        "modality_keys": [ADAM_U_SCHEMA.language_key],
    },
}

LEROBOT_FEATURES = {
    "observation.images.front": {"dtype": "video", "shape": (256, 256, 3)},
    "observation.images.wrist": {"dtype": "video", "shape": (256, 256, 3)},
    "observation.state.waist": {"dtype": "float32", "shape": (3,)},
    "observation.state.neck": {"dtype": "float32", "shape": (2,)},
    "observation.state.left_arm": {"dtype": "float32", "shape": (7,)},
    "observation.state.right_arm": {"dtype": "float32", "shape": (7,)},
    "observation.state.left_hand": {"dtype": "float32", "shape": (6,)},
    "observation.state.right_hand": {"dtype": "float32", "shape": (6,)},
    "action.waist": {"dtype": "float32", "shape": (3,)},
    "action.neck": {"dtype": "float32", "shape": (2,)},
    "action.left_arm": {"dtype": "float32", "shape": (7,)},
    "action.right_arm": {"dtype": "float32", "shape": (7,)},
    "action.left_hand": {"dtype": "float32", "shape": (6,)},
    "action.right_hand": {"dtype": "float32", "shape": (6,)},
    "task": {"dtype": "string"},
}
