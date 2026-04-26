"""Smoke checks for the Effects timeline Gradio tab helpers in ``app``."""

import unittest


class TestAppEffectsTabHandlers(unittest.TestCase):
    def test_effects_handler_symbols_exist(self) -> None:
        import app

        for name in (
            "_load_effects_editor",
            "_save_effects_editor",
            "_bake_effects_editor",
            "_clear_effects_editor",
            "_resolve_wav_path_for_effects_editor",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(app, name), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
