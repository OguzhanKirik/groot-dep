"""Contract tests for the native two-camera Adam-U GR00T embodiment."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adam_u_groot"))

from configs.adam_u_modality import ADAM_U_MODALITY_CONFIG, LEROBOT_FEATURES
from configs.groot_schemas import ADAM_U_SCHEMA


class TestAdamUNativeSchema(unittest.TestCase):
    def test_two_camera_contract(self):
        self.assertEqual(ADAM_U_SCHEMA.video_keys, ("front", "wrist"))
        self.assertIn("observation.images.front", LEROBOT_FEATURES)
        self.assertIn("observation.images.wrist", LEROBOT_FEATURES)

    def test_state_and_action_groups_total_31(self):
        expected = ("waist", "neck", "left_arm", "right_arm", "left_hand", "right_hand")
        self.assertEqual(ADAM_U_SCHEMA.state_keys, expected)
        self.assertEqual(ADAM_U_SCHEMA.action_keys, expected)
        self.assertEqual(sum(ADAM_U_SCHEMA.state_dims.values()), 31)
        self.assertEqual(ADAM_U_MODALITY_CONFIG["state"]["modality_keys"], list(expected))
        self.assertEqual(ADAM_U_MODALITY_CONFIG["action"]["modality_keys"], list(expected))

    def test_dataset_ranges_are_contiguous(self):
        modality = json.loads((ROOT / "adam_u_groot/configs/modality.json").read_text())
        for section in ("state", "action"):
            cursor = 0
            for key in ADAM_U_SCHEMA.state_keys:
                self.assertEqual(modality[section][key]["start"], cursor)
                cursor = modality[section][key]["end"]
            self.assertEqual(cursor, 31)


if __name__ == "__main__":
    unittest.main()
