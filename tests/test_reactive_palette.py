"""Hermetic tests for the palette helpers in :mod:`pipeline.reactive_shader`.

These exercise the pure-Python path (hex parsing, default fallback, padding,
truncation, strict validation) so they can run anywhere without a GPU or GL
context. GL-level smoke that a ``vec3[5]`` uniform lands in the shader is
covered by running ``ReactiveShader`` at the app level, not here.
"""

from __future__ import annotations

import unittest

from pipeline.reactive_shader import (
    DEFAULT_PALETTE,
    PALETTE_SLOTS,
    _build_palette_uniform,
    _parse_hex_color,
)


class TestParseHexColor(unittest.TestCase):
    def test_uppercase_hex_parses_to_unit_floats(self) -> None:
        r, g, b = _parse_hex_color("#FF8000")
        self.assertAlmostEqual(r, 1.0, places=6)
        self.assertAlmostEqual(g, 128 / 255, places=6)
        self.assertAlmostEqual(b, 0.0, places=6)

    def test_lowercase_hex_is_accepted(self) -> None:
        r, g, b = _parse_hex_color("#4cc9f0")
        self.assertAlmostEqual(r, 0x4C / 255, places=6)
        self.assertAlmostEqual(g, 0xC9 / 255, places=6)
        self.assertAlmostEqual(b, 0xF0 / 255, places=6)

    def test_whitespace_is_stripped(self) -> None:
        r, g, b = _parse_hex_color("  #000000  ")
        self.assertEqual((r, g, b), (0.0, 0.0, 0.0))

    def test_missing_hash_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_hex_color("FF8000")

    def test_wrong_length_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_hex_color("#FFF")
        with self.assertRaises(ValueError):
            _parse_hex_color("#FF80000")

    def test_non_hex_digits_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_hex_color("#ZZZZZZ")


class TestBuildPaletteUniform(unittest.TestCase):
    def test_none_falls_back_to_default_palette(self) -> None:
        flat, size = _build_palette_uniform(None)
        self.assertEqual(size, len(DEFAULT_PALETTE))
        self.assertEqual(len(flat), PALETTE_SLOTS * 3)

    def test_empty_list_falls_back_to_default_palette(self) -> None:
        flat, size = _build_palette_uniform([])
        self.assertEqual(size, len(DEFAULT_PALETTE))
        self.assertEqual(len(flat), PALETTE_SLOTS * 3)

    def test_exact_slot_count_preserved(self) -> None:
        palette = ["#000000", "#111111", "#222222", "#333333", "#444444"]
        flat, size = _build_palette_uniform(palette)
        self.assertEqual(size, PALETTE_SLOTS)
        self.assertEqual(len(flat), PALETTE_SLOTS * 3)
        self.assertAlmostEqual(flat[0], 0.0, places=6)
        self.assertAlmostEqual(flat[-1], 0x44 / 255, places=6)

    def test_short_palette_pads_by_repeating_last_color(self) -> None:
        flat, size = _build_palette_uniform(["#FF0000", "#00FF00"])
        self.assertEqual(size, 2)
        self.assertEqual(len(flat), PALETTE_SLOTS * 3)
        # First two slots are red, green; remaining slots repeat green.
        self.assertAlmostEqual(flat[0], 1.0, places=6)
        self.assertAlmostEqual(flat[1], 0.0, places=6)
        self.assertAlmostEqual(flat[2], 0.0, places=6)
        for slot in range(1, PALETTE_SLOTS):
            base = slot * 3
            self.assertAlmostEqual(flat[base + 0], 0.0, places=6)
            self.assertAlmostEqual(flat[base + 1], 1.0, places=6)
            self.assertAlmostEqual(flat[base + 2], 0.0, places=6)

    def test_long_palette_is_truncated_to_slot_count(self) -> None:
        palette = [f"#{i:02X}0000" for i in range(PALETTE_SLOTS + 3)]
        flat, size = _build_palette_uniform(palette)
        # Effective size clamps at PALETTE_SLOTS even if the user supplies more.
        self.assertEqual(size, PALETTE_SLOTS)
        self.assertEqual(len(flat), PALETTE_SLOTS * 3)
        # Last truncated slot's red channel must equal (PALETTE_SLOTS - 1) / 255.
        last_red = flat[(PALETTE_SLOTS - 1) * 3]
        self.assertAlmostEqual(last_red, (PALETTE_SLOTS - 1) / 255, places=6)

    def test_bad_hex_surfaces_original_error(self) -> None:
        with self.assertRaises(ValueError):
            _build_palette_uniform(["#FF0000", "not-a-color"])

    def test_default_palette_is_valid(self) -> None:
        # The bundled fallback must itself pass validation — otherwise any
        # ReactiveShader constructed without a palette would crash the pipeline.
        flat, size = _build_palette_uniform(list(DEFAULT_PALETTE))
        self.assertEqual(size, len(DEFAULT_PALETTE))
        self.assertEqual(len(flat), PALETTE_SLOTS * 3)
        for component in flat:
            self.assertGreaterEqual(component, 0.0)
            self.assertLessEqual(component, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
