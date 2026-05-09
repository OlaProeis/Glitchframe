"""Tests for :mod:`pipeline.pixel_smear`."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.pixel_smear import apply_pixel_smear


def _make_clip(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: object,
) -> EffectClip:
    return EffectClip(
        id=id_,
        kind=EffectKind.PIXEL_SMEAR,
        t_start=t_start,
        duration_s=duration_s,
        settings=dict(settings),
    )


def _gradient_frame(h: int = 16, w: int = 32) -> np.ndarray:
    """Per-column horizontal gradient — easy to spot smears against."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        f[:, x] = np.uint8(min(255, x * 7))
    return f


class TestPixelSmear(unittest.TestCase):
    def test_inactive_window_passthrough(self) -> None:
        c = _make_clip("a", 1.0, 0.5, intensity=1.0, density=1.0)
        f = _gradient_frame()
        out = apply_pixel_smear(f, 0.0, [c], "h")
        self.assertIs(out, f)

    def test_zero_intensity_passthrough(self) -> None:
        c = _make_clip("a", 0.0, 1.0, intensity=0.0, density=1.0)
        f = _gradient_frame()
        out = apply_pixel_smear(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_zero_density_passthrough(self) -> None:
        c = _make_clip("a", 0.0, 1.0, intensity=1.0, density=0.0)
        f = _gradient_frame()
        out = apply_pixel_smear(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_zero_streak_passthrough(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, density=1.0, streak_length_frac=0.0
        )
        f = _gradient_frame()
        out = apply_pixel_smear(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_active_changes_pixels(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, density=0.5, streak_length_frac=0.5
        )
        f = _gradient_frame()
        out = apply_pixel_smear(f, 0.5, [c], "songhash")
        self.assertFalse(np.array_equal(out, f))
        self.assertEqual(out.shape, f.shape)
        self.assertEqual(out.dtype, f.dtype)

    def test_deterministic_for_same_seed(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, density=0.5, streak_length_frac=0.5
        )
        f = _gradient_frame()
        a = apply_pixel_smear(f, 0.5, [c], "songhash")
        b = apply_pixel_smear(f, 0.5, [c], "songhash")
        np.testing.assert_array_equal(a, b)

    def test_different_song_hash_changes_output(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, density=0.5, streak_length_frac=0.5
        )
        f = _gradient_frame()
        a = apply_pixel_smear(f, 0.5, [c], "h1")
        b = apply_pixel_smear(f, 0.5, [c], "h2")
        self.assertFalse(np.array_equal(a, b))

    def test_non_smear_clips_ignored(self) -> None:
        smear = _make_clip(
            "s", 0.0, 1.0, intensity=1.0, density=0.5, streak_length_frac=0.5
        )
        other = EffectClip(
            id="b",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.5},
        )
        f = _gradient_frame()
        a = apply_pixel_smear(f, 0.5, [smear, other], "h")
        b = apply_pixel_smear(f, 0.5, [smear], "h")
        np.testing.assert_array_equal(a, b)

    def test_nonfinite_t_passthrough(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, density=0.5, streak_length_frac=0.5
        )
        f = _gradient_frame()
        out = apply_pixel_smear(f, float("nan"), [c], "h")
        self.assertIs(out, f)

    def test_invalid_shape_raises(self) -> None:
        bad = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_pixel_smear(bad, 0.0, [], "h")


if __name__ == "__main__":
    unittest.main()
