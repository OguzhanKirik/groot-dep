"""Register Adam-U modality config inside Isaac-GR00T (finetune / NEW_EMBODIMENT server).

Usage from Isaac-GR00T repo:

    python gr00t/experiment/launch_finetune.py \\
      --base-model-path /home/revel/models/GR00T-N1.7-3B \\
      --dataset-path /path/to/adam_u_lerobot_dataset \\
      --embodiment-tag NEW_EMBODIMENT \\
      --modality-config-path /home/revel/adam/adam_u_isaac_lab/adam_u_groot/examples/adam_u_modality_register.py

After finetune, start the server with the finetuned checkpoint and ``--embodiment-tag NEW_EMBODIMENT``.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ADAM_U_ACTION_HORIZON = 8
ADAM_U_GROUPS = ["waist", "neck", "left_arm", "right_arm", "left_hand", "right_hand"]

adam_u_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=ADAM_U_GROUPS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ADAM_U_ACTION_HORIZON)),
        modality_keys=ADAM_U_GROUPS,
        action_configs=[
            # Native Adam-U low-level commands are absolute joint/synergy
            # targets. Dataset actions must use radians for body joints and the
            # calibrated dimensionless hand-synergy units.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ] * len(ADAM_U_GROUPS),
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(adam_u_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
