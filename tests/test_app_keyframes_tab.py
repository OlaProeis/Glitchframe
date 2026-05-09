"""Smoke checks for the Background keyframes Gradio tab helpers in ``app``."""

import json
import unittest


class TestAppKeyframesTabHandlers(unittest.TestCase):
    def test_kf_collect_regenerate_entry_ids_order_and_dedupe(self) -> None:
        import app

        payload = json.dumps(
            {
                "selected_target_id": "kf-a",
                "regen_batch_ids": ["kf-b", "kf-a", "kf-c"],
            }
        )
        got = app._kf_collect_regenerate_entry_ids("kf-z", payload)
        self.assertEqual(got, ["kf-a", "kf-b", "kf-c"])

    def test_kf_collect_regenerate_fallback_slot_id(self) -> None:
        import app

        self.assertEqual(app._kf_collect_regenerate_entry_ids("kf-z", "{}"), ["kf-z"])

    def test_kf_collect_regenerate_batch_only(self) -> None:
        import app

        payload = json.dumps({"regen_batch_ids": ["kf-1", "kf-2"]})
        self.assertEqual(app._kf_collect_regenerate_entry_ids(None, payload), ["kf-1", "kf-2"])

    def test_keyframes_handler_symbols_exist(self) -> None:
        import app

        for name in (
            "_load_keyframes_editor",
            "_save_keyframes_editor",
            "_prep_keyframe_crop",
            "_apply_keyframe_crop",
            "_generate_keyframes_sdxl",
            "_kf_resolve_prompt",
            "_kf_show_crop_section",
            "_kf_prep_keyframe_crop_still",
            "_regenerate_selected_keyframe_sdxl",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(app, name), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
