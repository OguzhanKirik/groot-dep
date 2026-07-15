"""GR00T policy I/O schemas supported by the Adam-U integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrootSchema:
    name: str
    embodiment_tag: str
    video_keys: tuple[str, ...]
    video_horizon: int
    state_keys: tuple[str, ...]
    state_dims: dict[str, int]
    action_keys: tuple[str, ...]
    env_action_key: str
    language_key: str
    language_api_key: str
    notes: str = ""


# Native Adam-U layout (used after finetune with NEW_EMBODIMENT).
ADAM_U_SCHEMA = GrootSchema(
    name="adam_u",
    embodiment_tag="NEW_EMBODIMENT",
    video_keys=("front", "wrist"),
    video_horizon=1,
    state_keys=("waist", "neck", "left_arm", "right_arm", "left_hand", "right_hand"),
    state_dims={
        "waist": 3,
        "neck": 2,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 6,
        "right_hand": 6,
    },
    action_keys=("waist", "neck", "left_arm", "right_arm", "left_hand", "right_hand"),
    env_action_key="body_hands",
    language_key="annotation.human.action.task_description",
    language_api_key="annotation.human.action.task_description",
    notes="Native Adam-U body[19]+hands[12]; requires an Adam-U NEW_EMBODIMENT checkpoint.",
)

# Shim for the base nvidia/GR00T-N1.7-3B REAL_G1 checkpoint (pipeline testing only).
REAL_G1_SCHEMA = GrootSchema(
    name="real_g1",
    embodiment_tag="REAL_G1",
    video_keys=("ego_view",),
    video_horizon=2,
    state_keys=(
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
    ),
    state_dims={
        "left_wrist_eef_9d": 9,
        "right_wrist_eef_9d": 9,
        "left_hand": 7,
        "right_hand": 7,
        "left_arm": 7,
        "right_arm": 7,
        "waist": 3,
    },
    action_keys=(
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
        "base_height_command",
        "navigate_command",
    ),
    env_action_key="right_arm",
    language_key="annotation.human.task_description",
    language_api_key="annotation.human.task_description",
    notes="Maps Adam-U cameras/state into REAL_G1 keys. Motion is not meaningful.",
)

GROOT_SCHEMAS: dict[str, GrootSchema] = {
    ADAM_U_SCHEMA.name: ADAM_U_SCHEMA,
    REAL_G1_SCHEMA.name: REAL_G1_SCHEMA,
}


def get_groot_schema(name: str) -> GrootSchema:
    key = name.lower()
    if key not in GROOT_SCHEMAS:
        valid = ", ".join(sorted(GROOT_SCHEMAS))
        raise ValueError(f"Unknown GR00T schema '{name}'. Valid options: {valid}")
    return GROOT_SCHEMAS[key]
