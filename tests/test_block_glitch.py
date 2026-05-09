"""Tests for :mod:`pipeline.block_glitch`."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.block_glitch import apply_block_glitch
from pipeline.effects_timeline import EffectClip, EffectKind


def _make_clip(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: object,
) -> EffectClip:
    return EffectClip(
        id=id_,
        kind=EffectKind.BLOCK_GLITCH,
        t_start=t_start,
        duration_s=duration_s,
        settings=dict(settings),
    )


def _checker_frame(h: int = 64, w: int = 64, block: int = 8) -> np.ndarray:
    """Checkerboard pattern — small displacements are easy to detect."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((y // block) + (x // block)) % 2 == 0:
                f[y : y + block, x : x + block] = 220
    return f


class TestBlockGlitch(unittest.TestCase):
    def test_inactive_window_passthrough(self) -> None:
        c = _make_clip("a", 1.0, 0.5, intensity=1.0)
        f = _checker_frame()
        out = apply_block_glitch(f, 0.0, [c], "h")
        self.assertIs(out, f)

    def test_zero_intensity_passthrough(self) -> None:
        c = _make_clip("a", 0.0, 1.0, intensity=0.0)
        f = _checker_frame()
        out = apply_block_glitch(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_zero_displace_passthrough(self) -> None:
        c = _make_clip("a", 0.0, 1.0, intensity=1.0, displace_frac=0.0)
        f = _checker_frame()
        out = apply_block_glitch(f, 0.5, [c], "h")
        self.assertIs(out, f)

    def test_active_changes_pixels(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, block_size_px=8, displace_frac=1.0
        )
        f = _checker_frame()
        out = apply_block_glitch(f, 0.5, [c], "songhash")
        self.assertFalse(np.array_equal(out, f))
        self.assertEqual(out.shape, f.shape)
        self.assertEqual(out.dtype, f.dtype)

    def test_deterministic_for_same_seed(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, block_size_px=8, displace_frac=1.0
        )
        f = _checker_frame()
        a = apply_block_glitch(f, 0.5, [c], "songhash")
        b = apply_block_glitch(f, 0.5, [c], "songhash")
        np.testing.assert_array_equal(a, b)

    def test_different_song_hash_changes_output(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, block_size_px=8, displace_frac=1.0
        )
        f = _checker_frame()
        a = apply_block_glitch(f, 0.5, [c], "h1")
        b = apply_block_glitch(f, 0.5, [c], "h2")
        self.assertFalse(np.array_equal(a, b))

    def test_non_block_clips_ignored(self) -> None:
        block = _make_clip(
            "s", 0.0, 1.0, intensity=1.0, block_size_px=8, displace_frac=1.0
        )
        other = EffectClip(
            id="b",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.5},
        )
        f = _checker_frame()
        a = apply_block_glitch(f, 0.5, [block, other], "h")
        b = apply_block_glitch(f, 0.5, [block], "h")
        np.testing.assert_array_equal(a, b)

    def test_nonfinite_t_passthrough(self) -> None:
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, block_size_px=8, displace_frac=1.0
        )
        f = _checker_frame()
        out = apply_block_glitch(f, float("nan"), [c], "h")
        self.assertIs(out, f)

    def test_block_size_clamped_to_frame(self) -> None:
        # Requesting an absurdly large block must not raise — it's clamped to
        # ``min(h, w)``.
        c = _make_clip(
            "a", 0.0, 1.0, intensity=1.0, block_size_px=10_000, displace_frac=1.0
        )
        f = _checker_frame(h=16, w=16)
        out = apply_block_glitch(f, 0.5, [c], "h")
        self.assertEqual(out.shape, f.shape)

    def test_invalid_shape_raises(self) -> None:
        bad = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_block_glitch(bad, 0.0, [], "h")


if __name__ == "__main__":
    unittest.main()
