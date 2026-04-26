"""Tests for :mod:`pipeline.chromatic_aberration`."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.chromatic_aberration import apply_chromatic_aberration
from pipeline.effects_timeline import EffectClip, EffectKind


def _make_chromatic(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: float,
) -> EffectClip:
    s = {k: v for k, v in settings.items()}
    return EffectClip(
        id=id_,
        kind=EffectKind.CHROMATIC_ABERRATION,
        t_start=t_start,
        duration_s=duration_s,
        settings=s,
    )


class TestChromaticAberration(unittest.TestCase):
    def test_identity_outside_clip(self) -> None:
        f = np.zeros((8, 8, 3), dtype=np.uint8)
        c = _make_chromatic("a", 1.0, 1.0, shift_px=5.0, jitter=0.0, direction_deg=0.0)
        out = apply_chromatic_aberration(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_shift_px_zero_is_identity(self) -> None:
        f = np.random.default_rng(0).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
        c = _make_chromatic("z", 0.0, 2.0, shift_px=0.0, jitter=0.5, direction_deg=0.0)
        for t in (0.0, 0.5, 1.0):
            out = apply_chromatic_aberration(f, t, [c], "song")
            self.assertIs(out, f)

    def test_subpixel_shift_not_lost(self) -> None:
        """Regression: integer rounding used to drop magnitudes under 0.5 px."""
        h, w = 16, 16
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
        f[:, :, 1] = 128
        f[:, :, 2] = np.linspace(255, 0, w, dtype=np.uint8)
        c = _make_chromatic(
            "subpx", 0.0, 2.0, shift_px=0.4, jitter=0.0, direction_deg=0.0
        )
        out = apply_chromatic_aberration(f, 0.1, [c], "subpx-song")
        self.assertIsNot(out, f)
        np.testing.assert_array_equal(out[:, :, 1], f[:, :, 1])
        self.assertFalse(np.array_equal(out[:, :, 0], f[:, :, 0]))
        self.assertFalse(np.array_equal(out[:, :, 2], f[:, :, 2]))

    def test_g_channel_unchanged_horizontal_shift(self) -> None:
        # min(H,W)*0.1 must exceed ~0.5 px so round() yields a non-zero shift.
        h, w = 20, 20
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            f[:, x, 0] = min(255, x * 10)
            f[:, x, 1] = min(255, x * 10 + 1)
            f[:, x, 2] = min(255, x * 10 + 2)
        c = _make_chromatic("c1", 0.0, 2.0, shift_px=1.0, jitter=0.0, direction_deg=0.0)
        out = apply_chromatic_aberration(f, 0.1, [c], "hashx")
        np.testing.assert_array_equal(out[:, :, 1], f[:, :, 1])
        y, x = 10, 10
        self.assertEqual(int(out[y, x, 0]), int(f[y, x - 1, 0]))
        self.assertEqual(int(out[y, x, 2]), int(f[y, x + 1, 2]))

    def test_deterministic(self) -> None:
        f = np.random.default_rng(1).integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
        c = _make_chromatic("d", 0.0, 5.0, shift_px=3.0, jitter=0.2, direction_deg=45.0)
        t = 1.2345
        h = "stable"
        a = apply_chromatic_aberration(f, t, [c], h)
        b = apply_chromatic_aberration(f, t, [c], h)
        np.testing.assert_array_equal(a, b)

    def test_non_finite_t_identity(self) -> None:
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        c = _make_chromatic("n", 0.0, 2.0, shift_px=4.0, jitter=0.0, direction_deg=0.0)
        out = apply_chromatic_aberration(f, float("nan"), [c], "h")
        self.assertIs(out, f)

    def test_ignores_non_chromatic(self) -> None:
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[1, 1] = [200, 100, 50]
        chrom = _make_chromatic("c", 0.0, 2.0, shift_px=2.0, jitter=0.0, direction_deg=0.0)
        other = EffectClip(
            id="beam1",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=2.0,
            settings={"strength": 0.5, "color_hex": "#ffffff"},
        )
        a = apply_chromatic_aberration(f, 0.5, [chrom], "hh")
        b = apply_chromatic_aberration(f, 0.5, [chrom, other], "hh")
        np.testing.assert_array_equal(a, b)

    def test_overlapping_shifts_sum(self) -> None:
        h, w = 24, 24
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 0] = np.arange(w, dtype=np.uint8) * 10
        f[:, :, 2] = np.arange(w, dtype=np.uint8) * 5
        f[:, :, 1] = 77
        a = _make_chromatic("a", 0.0, 2.0, shift_px=1.0, jitter=0.0, direction_deg=0.0)
        b = _make_chromatic("b", 0.0, 2.0, shift_px=1.0, jitter=0.0, direction_deg=0.0)
        one = apply_chromatic_aberration(f, 0.5, [a], "sumhash")
        two = apply_chromatic_aberration(f, 0.5, [b], "sumhash")
        both = apply_chromatic_aberration(f, 0.5, [a, b], "sumhash")
        # Same axis: doubling shift should match summing two identical clips.
        dbl = _make_chromatic("d", 0.0, 2.0, shift_px=2.0, jitter=0.0, direction_deg=0.0)
        merged = apply_chromatic_aberration(f, 0.5, [dbl], "sumhash")
        np.testing.assert_array_equal(both, merged)
        np.testing.assert_array_equal(one, two)
        self.assertFalse(np.array_equal(one, both))


if __name__ == "__main__":
    unittest.main()
