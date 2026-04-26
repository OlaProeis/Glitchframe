"""Tests for :mod:`pipeline.scanline_tear`."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.scanline_tear import apply_scanline_tear


def _clip(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: object,
) -> EffectClip:
    return EffectClip(
        id=id_,
        kind=EffectKind.SCANLINE_TEAR,
        t_start=t_start,
        duration_s=duration_s,
        settings=dict(settings),
    )


class TestScanlineTear(unittest.TestCase):
    def test_identity_outside_clip(self) -> None:
        f = np.zeros((8, 16, 3), dtype=np.uint8)
        c = _clip("a", 1.0, 1.0, intensity=1.0, band_count=2, wrap_mode="wrap")
        out = apply_scanline_tear(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_identity_no_active_scanline_in_list(self) -> None:
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        other = EffectClip(
            id="b",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.5, "color_hex": "#fff"},
        )
        c = _clip("s", 0.0, 1.0, intensity=1.0, band_count=1, wrap_mode="wrap")
        out = apply_scanline_tear(f, 0.5, [other], "h")
        self.assertIs(out, f)
        out2 = apply_scanline_tear(f, 0.5, [c, other], "h")
        self.assertIsInstance(out2, np.ndarray)

    def test_intensity_zero_is_identity(self) -> None:
        f = np.random.default_rng(0).integers(0, 256, size=(8, 16, 3), dtype=np.uint8)
        c = _clip("z", 0.0, 2.0, intensity=0.0, band_count=4, wrap_mode="wrap")
        for t in (0.0, 0.1, 0.5):
            out = apply_scanline_tear(f, t, [c], "song")
            self.assertIs(out, f)

    def test_non_finite_t_identity(self) -> None:
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        c = _clip("n", 0.0, 2.0, intensity=1.0, band_count=2, wrap_mode="wrap")
        out = apply_scanline_tear(f, float("nan"), [c], "h")
        self.assertIs(out, f)

    def test_deterministic(self) -> None:
        f = np.random.default_rng(1).integers(0, 256, size=(12, 24, 3), dtype=np.uint8)
        c = _clip("d", 0.0, 5.0, intensity=0.8, band_count=3, wrap_mode="wrap")
        t = 1.2345
        h = "stable"
        a = apply_scanline_tear(f, t, [c], h)
        b = apply_scanline_tear(f, t, [c], h)
        np.testing.assert_array_equal(a, b)

    def test_rgb_move_together_uniform_row(self) -> None:
        """Horizontal shift keeps R/G/B identical on a flat-colour band."""
        h, w = 12, 24
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[4:6, :, :] = 40
        c = _clip("u", 0.0, 2.0, intensity=1.0, band_count=2, band_height_px=2)
        out = apply_scanline_tear(f, 0.5, [c], "rgbtest")
        for y in range(h):
            for x in range(w):
                p = out[y, x]
                self.assertEqual(int(p[0]), int(p[1]))
                self.assertEqual(int(p[1]), int(p[2]))

    def test_active_clip_produces_change_large_frame(self) -> None:
        f = np.zeros((32, 64, 3), dtype=np.uint8)
        f[:, :20, 0] = 200
        f[:, 20:40, 1] = 200
        f[:, 40:, 2] = 200
        c = _clip("p", 0.0, 1.0, intensity=1.0, band_count=4, wrap_mode="wrap")
        out = apply_scanline_tear(f, 0.2, [c], "chg")
        self.assertFalse(np.array_equal(out, f))

    def test_wrap_mode_black(self) -> None:
        f = np.full((8, 16, 3), 100, dtype=np.uint8)
        c = _clip("k", 0.0, 1.0, intensity=1.0, band_count=1, band_height_px=4, wrap_mode="black")
        out = apply_scanline_tear(f, 0.25, [c], "blk")
        self.assertFalse(np.array_equal(out, f))
        self.assertTrue(np.any(out == 0))

    def test_distinct_clips_differ(self) -> None:
        """Different clip.id → different per-frame seed → different single-clip output."""
        f = np.arange(1 * 24 * 3, dtype=np.uint8).reshape(1, 24, 3)
        a = _clip("clip_a", 0.0, 1.0, intensity=1.0, band_count=2, band_height_px=1)
        b = _clip("clip_b", 0.0, 1.0, intensity=1.0, band_count=2, band_height_px=1)
        oa = apply_scanline_tear(f, 0.1, [a], "h")
        ob = apply_scanline_tear(f, 0.1, [b], "h")
        self.assertFalse(np.array_equal(oa, f))
        self.assertFalse(np.array_equal(ob, f))
        self.assertFalse(np.array_equal(oa, ob))

    def test_two_active_clips_stack_differs_from_first_alone(self) -> None:
        """Two clips at the same t: second pass reads the first pass output (t=0.0 chosen so both apply)."""
        f = np.zeros((4, 16, 3), dtype=np.uint8)
        f[:, 4:8, :] = 200
        a = _clip("aa", 0.0, 1.0, intensity=1.0, band_count=1, band_height_px=1, wrap_mode="wrap")
        b = _clip("bb", 0.0, 1.0, intensity=1.0, band_count=1, band_height_px=1, wrap_mode="wrap")
        t = 0.0
        stacked = apply_scanline_tear(f, t, [a, b], "stk")
        one = apply_scanline_tear(f, t, [a], "stk")
        self.assertFalse(np.array_equal(stacked, f))
        self.assertFalse(np.array_equal(stacked, one))

    def test_invalid_frame_shape_raises(self) -> None:
        f = np.zeros((4, 4, 4), dtype=np.uint8)
        c = _clip("x", 0.0, 1.0, intensity=1.0, band_count=1)
        with self.assertRaises(ValueError):
            apply_scanline_tear(f, 0.0, [c], "h")

