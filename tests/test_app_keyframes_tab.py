"""Smoke checks for the Background keyframes Gradio tab helpers in ``app``."""

import unittest


class TestAppKeyframesTabHandlers(unittest.TestCase):
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
