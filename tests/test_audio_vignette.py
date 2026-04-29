"""Unit tests for :mod:`pipeline.audio_vignette` (no GPU)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.audio_vignette import (
    AudioVignetteContext,
    apply_audio_vignette,
    build_audio_vignette_context,
)


class TestBuildContext(unittest.TestCase):
    def test_mask_centre_zero_corners_max(self) -> None:
        ctx = build_audio_vignette_context(640, 360, strength=1.0)
        self.assertEqual(ctx.mask.shape, (360, 640))
        # Centre pixel reads as zero (well below the inner ramp radius).
        self.assertAlmostEqual(float(ctx.mask[180, 320]), 0.0, places=6)
        # Corners are at unit radius and clip to the smoothstep ceiling
        # (~0.97 with the 0.55 / 1.05 ramp).
        self.assertGreater(float(ctx.mask[0, 0]), 0.85)
        self.assertLessEqual(float(ctx.mask[0, 0]), 1.0)

    def test_zero_dimensions_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_audio_vignette_context(0, 100)
        with self.assertRaises(ValueError):
            build_audio_vignette_context(100, 0)


class TestApplyAudioVignette(unittest.TestCase):
    def _flat_frame(self, h: int = 64, w: int = 64, value: int = 200) -> np.ndarray:
        return np.full((h, w, 3), value, dtype=np.uint8)

    def test_none_ctx_is_no_op(self) -> None:
        frame = self._flat_frame()
        out = apply_audio_vignette(frame, {}, None)
        self.assertTrue(np.array_equal(out, frame))

    def test_zero_strength_is_no_op(self) -> None:
        ctx = AudioVignetteContext(
            mask=np.ones((64, 64), dtype=np.float32), strength=0.0
        )
        frame = self._flat_frame()
        out = apply_audio_vignette(frame, {}, ctx)
        self.assertTrue(np.array_equal(out, frame))

    def test_silent_uniforms_darken_corners_only(self) -> None:
        ctx = build_audio_vignette_context(64, 64, strength=1.0)
        frame = self._flat_frame()
        out = apply_audio_vignette(frame, {}, ctx)
        # Centre stays at 200; corners darken by the static base term.
        self.assertEqual(int(out[32, 32, 0]), 200)
        self.assertLess(int(out[0, 0, 0]), 200)
        self.assertLess(int(out[63, 63, 0]), 200)

    def test_loud_uniforms_darken_more(self) -> None:
        ctx = build_audio_vignette_context(64, 64, strength=1.0)
        frame = self._flat_frame()
        silent = apply_audio_vignette(frame, {"bass_hit": 0.0}, ctx)
        loud = apply_audio_vignette(
            frame,
            {"bass_hit": 1.0, "drop_hold": 1.0, "rms": 1.0},
            ctx,
        )
        # Loud uniforms must darken corners strictly more than silent.
        self.assertLess(int(loud[0, 0, 0]), int(silent[0, 0, 0]))

    def test_strength_zero_via_context(self) -> None:
        ctx = build_audio_vignette_context(64, 64, strength=0.0)
        frame = self._flat_frame()
        out = apply_audio_vignette(frame, {"bass_hit": 1.0}, ctx)
        self.assertTrue(np.array_equal(out, frame))

    def test_rejects_wrong_dtype(self) -> None:
        ctx = build_audio_vignette_context(64, 64)
        bad = np.zeros((64, 64, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_audio_vignette(bad, {}, ctx)

    def test_rejects_shape_mismatch(self) -> None:
        ctx = build_audio_vignette_context(64, 64)
        bad = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_audio_vignette(bad, {}, ctx)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
