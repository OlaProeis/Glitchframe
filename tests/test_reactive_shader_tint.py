"""Hermetic tests for the shader peak-tint helper in :mod:`pipeline.reactive_shader`.

Same shape as :mod:`tests.test_reactive_palette`: tests the pure-Python parsing
+ clamping path for ``shader_tint`` / ``shader_tint_strength`` so CI can run
without a GPU or GL context. The GL-level smoke that the uniforms actually
land on the program is exercised when a real ``ReactiveShader`` is built at
the app level, which is out of scope for this hermetic file.
"""

from __future__ import annotations

import math
import unittest

from pipeline.reactive_shader import (
    DEFAULT_SHADER_TINT_RGB,
    DEFAULT_SHADER_TINT_STRENGTH,
    _resolve_shader_tint,
)


class TestResolveShaderTint(unittest.TestCase):
    def test_none_hex_falls_back_to_white(self) -> None:
        rgb, strength = _resolve_shader_tint(None, 0.0)
        self.assertEqual(rgb, DEFAULT_SHADER_TINT_RGB)
        self.assertEqual(strength, 0.0)

    def test_empty_hex_falls_back_to_white(self) -> None:
        rgb, strength = _resolve_shader_tint("", 0.5)
        self.assertEqual(rgb, DEFAULT_SHADER_TINT_RGB)
        self.assertAlmostEqual(strength, 0.5, places=6)

    def test_whitespace_hex_falls_back_to_white(self) -> None:
        rgb, strength = _resolve_shader_tint("   ", 1.0)
        self.assertEqual(rgb, DEFAULT_SHADER_TINT_RGB)
        self.assertAlmostEqual(strength, 1.0, places=6)

    def test_uppercase_hex_parses_to_unit_floats(self) -> None:
        rgb, _ = _resolve_shader_tint("#FF8000", 0.5)
        self.assertAlmostEqual(rgb[0], 1.0, places=6)
        self.assertAlmostEqual(rgb[1], 128 / 255, places=6)
        self.assertAlmostEqual(rgb[2], 0.0, places=6)

    def test_lowercase_hex_is_accepted(self) -> None:
        rgb, _ = _resolve_shader_tint("#4cc9f0", 0.0)
        self.assertAlmostEqual(rgb[0], 0x4C / 255, places=6)
        self.assertAlmostEqual(rgb[1], 0xC9 / 255, places=6)
        self.assertAlmostEqual(rgb[2], 0xF0 / 255, places=6)

    def test_strength_below_zero_is_clamped(self) -> None:
        _, strength = _resolve_shader_tint("#FFFFFF", -1.5)
        self.assertEqual(strength, 0.0)

    def test_strength_above_one_is_clamped(self) -> None:
        _, strength = _resolve_shader_tint("#FFFFFF", 4.2)
        self.assertEqual(strength, 1.0)

    def test_strength_at_boundary_passes_through(self) -> None:
        _, lo = _resolve_shader_tint("#FFFFFF", 0.0)
        _, hi = _resolve_shader_tint("#FFFFFF", 1.0)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 1.0)

    def test_invalid_hex_raises(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_shader_tint("not-a-color", 0.0)

    def test_short_hex_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_shader_tint("#FFF", 0.0)

    def test_strength_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_shader_tint("#FFFFFF", math.nan)

    def test_strength_non_numeric_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_shader_tint("#FFFFFF", "loud")  # type: ignore[arg-type]

    def test_default_strength_constant_is_disabled(self) -> None:
        # Documents the no-op-on-default contract: the bundled constant
        # used by ``ReactiveShader.__init__`` must round-trip to a strength
        # of exactly 0 so existing renders behave as before this change.
        _, strength = _resolve_shader_tint(None, DEFAULT_SHADER_TINT_STRENGTH)
        self.assertEqual(strength, 0.0)

    def test_default_tint_rgb_constant_is_white(self) -> None:
        # Same contract for the colour: white * any-strength = white peaks,
        # which preserves the historical visual exactly.
        self.assertEqual(DEFAULT_SHADER_TINT_RGB, (1.0, 1.0, 1.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
