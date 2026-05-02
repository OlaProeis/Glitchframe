"""Shader-first visual defaults (``pipeline.visual_style``)."""

from __future__ import annotations

import unittest

from pipeline.visual_style import (
    canonical_reactive_shader_stem,
    motion_flavor_for_style_preset,
    shader_style_bundle,
    style_preset_id,
)
from pipeline.builtin_shaders import BUILTIN_SHADERS


class TestVisualStyle(unittest.TestCase):
    def test_builtin_shaders_match_bundles(self) -> None:
        for stem in BUILTIN_SHADERS:
            b = shader_style_bundle(stem)
            self.assertTrue(b.example_prompt.strip())
            self.assertTrue(b.typo_style.strip())
            self.assertGreaterEqual(len(b.colors), 1)
            self.assertTrue(b.motion_flavor.strip())

    def test_style_preset_id_stable(self) -> None:
        self.assertEqual(style_preset_id("spectral_milkdrop"), "style-spectral_milkdrop")

    def test_motion_flavor_registered_per_style_preset(self) -> None:
        for stem in BUILTIN_SHADERS:
            pid = style_preset_id(stem)
            mf = motion_flavor_for_style_preset(pid)
            self.assertIsNotNone(mf)
            assert mf is not None
            self.assertTrue(mf.strip())

    def test_canonical_fallback(self) -> None:
        self.assertEqual(canonical_reactive_shader_stem(None), "none")
        self.assertEqual(canonical_reactive_shader_stem("__nope"), "none")


if __name__ == "__main__":
    unittest.main()
