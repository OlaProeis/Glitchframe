"""Tests for :mod:`pipeline.fade`."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.fade import apply_fade, fade_alpha


def _make_fade(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: object,
) -> EffectClip:
    return EffectClip(
        id=id_,
        kind=EffectKind.FADE,
        t_start=t_start,
        duration_s=duration_s,
        settings=dict(settings),
    )


class TestFadeAlpha(unittest.TestCase):
    def test_outside_window_is_zero(self) -> None:
        c = _make_fade("a", 1.0, 1.0, direction_mode="out")
        self.assertEqual(fade_alpha(0.0, [c]), 0.0)
        self.assertEqual(fade_alpha(2.0, [c]), 0.0)
        self.assertEqual(fade_alpha(2.5, [c]), 0.0)

    def test_fade_in_starts_black_ends_clear(self) -> None:
        c = _make_fade("in", 0.0, 1.0, direction_mode="in", ease_mode="linear")
        # At t == t_start the screen is fully black.
        self.assertAlmostEqual(fade_alpha(0.0, [c]), 1.0, places=5)
        # Mid clip → half alpha (linear ramp, 1 → 0).
        self.assertAlmostEqual(fade_alpha(0.5, [c]), 0.5, places=5)
        # Just before the end → near zero.
        self.assertLess(fade_alpha(0.99, [c]), 0.05)

    def test_fade_out_starts_clear_ends_black(self) -> None:
        c = _make_fade("out", 0.0, 1.0, direction_mode="out", ease_mode="linear")
        self.assertAlmostEqual(fade_alpha(0.0, [c]), 0.0, places=5)
        self.assertAlmostEqual(fade_alpha(0.5, [c]), 0.5, places=5)
        self.assertGreater(fade_alpha(0.99, [c]), 0.95)

    def test_smoothstep_default_at_midpoint(self) -> None:
        # Smoothstep(0.5) == 0.5 at the midpoint of a fade-out ramp.
        c = _make_fade("def", 0.0, 1.0, direction_mode="out")
        self.assertAlmostEqual(fade_alpha(0.5, [c]), 0.5, places=5)

    def test_peak_alpha_caps_max_blackness(self) -> None:
        c = _make_fade(
            "cap",
            0.0,
            1.0,
            direction_mode="out",
            peak_alpha=0.4,
            ease_mode="linear",
        )
        # At the end of the ramp, alpha is capped to peak_alpha.
        self.assertAlmostEqual(fade_alpha(0.99, [c]), 0.4, delta=0.02)

    def test_overlap_takes_max(self) -> None:
        a = _make_fade("a", 0.0, 1.0, direction_mode="out", ease_mode="linear")
        b = _make_fade("b", 0.0, 1.0, direction_mode="in", ease_mode="linear")
        # At t=0.5 both linear contributions are 0.5; max is 0.5.
        self.assertAlmostEqual(fade_alpha(0.5, [a, b]), 0.5, places=5)

    def test_unknown_direction_falls_back_to_in(self) -> None:
        c = _make_fade(
            "junk", 0.0, 1.0, direction_mode="diagonal", ease_mode="linear"
        )
        # Default direction is "in" → at t=0 alpha is 1.0.
        self.assertAlmostEqual(fade_alpha(0.0, [c]), 1.0, places=5)

    def test_non_fade_clips_ignored(self) -> None:
        z = _make_fade("z", 0.0, 1.0, direction_mode="out", ease_mode="linear")
        other = EffectClip(
            id="b1",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.7},
        )
        self.assertEqual(fade_alpha(0.5, [z, other]), fade_alpha(0.5, [z]))

    def test_nonfinite_t_returns_zero(self) -> None:
        c = _make_fade("n", 0.0, 1.0, direction_mode="out")
        self.assertEqual(fade_alpha(float("nan"), [c]), 0.0)
        self.assertEqual(fade_alpha(float("inf"), [c]), 0.0)

    def test_empty_clips(self) -> None:
        self.assertEqual(fade_alpha(0.0, []), 0.0)


class TestApplyFade(unittest.TestCase):
    def _frame(self, h: int = 4, w: int = 4) -> np.ndarray:
        return np.full((h, w, 3), 200, dtype=np.uint8)

    def test_zero_alpha_passthrough(self) -> None:
        f = self._frame()
        out = apply_fade(f, 0.0)
        self.assertIs(out, f)

    def test_full_alpha_returns_black(self) -> None:
        f = self._frame()
        out = apply_fade(f, 1.0)
        self.assertEqual(out.shape, f.shape)
        np.testing.assert_array_equal(out, np.zeros_like(f))

    def test_half_alpha_dims_proportionally(self) -> None:
        f = self._frame()
        out = apply_fade(f, 0.5)
        # 200 * (1 - 0.5) == 100
        self.assertTrue(np.all(out == 100))

    def test_invalid_shape_raises(self) -> None:
        bad = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_fade(bad, 0.5)


if __name__ == "__main__":
    unittest.main()
