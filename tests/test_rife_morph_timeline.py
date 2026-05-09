"""Dense RIFE morph timeline structure (centered IFNet sampling, no internal stills).

These tests do not require CUDA or RIFE weights — :func:`load_rife_model` and
the per-pair IFNet inference are stubbed. The goal is to lock in the
timeline-times grid that addresses the perceived stutter at SDXL keyframe
boundaries (see ``docs/technical/rife-morph-background.md``):

* exact start still at ``t = keyframe_times[0]`` (first frame)
* exact end still at ``t = keyframe_times[-1]`` (last frame)
* every internal SDXL keyframe time is **bridged** by two soft IFNet
  predictions equally offset by ``T_seg / (2 * 2**exp)``, so ``t_kf`` itself
  is never an exact still in the dense timeline
* dense spacing is uniform ``T_seg / 2**exp`` within and across segments
"""

from __future__ import annotations

import unittest
from typing import Sequence
from unittest import mock

import numpy as np

import pipeline.rife_runtime as rife_runtime


def _solid_rgb(value: int) -> np.ndarray:
    """Tiny synthetic SDXL-still stand-in (8x8 RGB, distinguishable byte value)."""
    return np.full((8, 8, 3), int(value), dtype=np.uint8)


def _fake_interpolate_pair(
    _model,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    *,
    exp: int,
    device,
    include_endpoints: bool = True,
) -> list[np.ndarray]:
    """Stand-in for ``rife_exp_interpolate_pair`` that returns deterministic
    midpoint blends instead of running IFNet, but preserves the same length
    contract (``2**exp`` frames when ``include_endpoints=False``).
    """
    n = 2**int(exp)
    if include_endpoints:
        # Not exercised by the new timeline builder, but kept for completeness.
        out = [rgb0.copy()]
        for i in range(n - 1):
            ts = (i + 1) * 1.0 / n
            out.append(
                ((1.0 - ts) * rgb0.astype(np.float32) + ts * rgb1.astype(np.float32))
                .astype(np.uint8)
            )
        out.append(rgb1.copy())
        return out
    out: list[np.ndarray] = []
    for i in range(n):
        ts = (i + 0.5) / n
        out.append(
            ((1.0 - ts) * rgb0.astype(np.float32) + ts * rgb1.astype(np.float32))
            .astype(np.uint8)
        )
    return out


class _FakeModel:
    fp16 = False


def _build(
    keyframes: Sequence[np.ndarray],
    keyframe_times: Sequence[float],
    *,
    exp: int,
):
    fake_device = object()  # never reached past the stub
    with mock.patch.object(
        rife_runtime, "load_rife_model", return_value=_FakeModel()
    ), mock.patch.object(
        rife_runtime,
        "rife_exp_interpolate_pair",
        side_effect=_fake_interpolate_pair,
    ), mock.patch("torch.cuda.is_available", return_value=False):
        # ``device.type`` is read by the cudnn-benchmark guard; an object
        # without ``type`` would break that branch, so just stub it.
        class _Dev:
            type = "cpu"

        return rife_runtime.rife_build_morph_timeline(
            keyframes,
            keyframe_times,
            exp=exp,
            device=_Dev(),  # type: ignore[arg-type]
        )


class CenteredMorphTimelineTests(unittest.TestCase):
    def test_two_keyframes_single_segment(self) -> None:
        # Single segment [0, 8] with exp=2 (n=4). Expected dense times:
        #   [exact_start=0, 1.0, 3.0, 5.0, 7.0, exact_end=8.0]
        kf0, kf1 = _solid_rgb(10), _solid_rgb(250)
        frames, times = _build([kf0, kf1], [0.0, 8.0], exp=2)
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(times), 6)
        self.assertEqual(times, [0.0, 1.0, 3.0, 5.0, 7.0, 8.0])
        # Endpoints are the exact source stills, byte-identical.
        np.testing.assert_array_equal(frames[0], kf0)
        np.testing.assert_array_equal(frames[-1], kf1)

    def test_three_keyframes_no_internal_exact_still(self) -> None:
        # Two segments [0, 8] and [8, 16], exp=2 (n=4). Expected dense times:
        #   exact_start=0, seg0=[1, 3, 5, 7], seg1=[9, 11, 13, 15], exact_end=16
        kf0, kf1, kf2 = _solid_rgb(10), _solid_rgb(120), _solid_rgb(250)
        frames, times = _build([kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2)
        self.assertEqual(times, [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 16.0])

        # The internal SDXL keyframe time t=8 is NOT in the timeline at all
        # (the whole point of this fix).
        self.assertNotIn(8.0, times)

        # The two frames flanking the internal keyframe are equidistant
        # IFNet midpoints — neither equals the exact kf1 still.
        idx_left = times.index(7.0)
        idx_right = times.index(9.0)
        self.assertFalse(np.array_equal(frames[idx_left], kf1))
        self.assertFalse(np.array_equal(frames[idx_right], kf1))

        # Boundary spacing equals within-segment spacing → uniform perceived
        # motion straight through the keyframe.
        self.assertAlmostEqual(times[idx_right] - times[idx_left], 2.0)
        # And that's identical to within-segment spacing.
        self.assertAlmostEqual(times[2] - times[1], 2.0)

    def test_uniform_spacing_across_internal_boundaries(self) -> None:
        # Four segments of varying durations to confirm the boundary spacing
        # rule ``T_seg/2**exp`` per segment holds independently per side.
        kfs = [_solid_rgb(v) for v in (5, 50, 100, 150, 200)]
        kts = [0.0, 4.0, 12.0, 16.0, 24.0]  # spans 4, 8, 4, 8
        _frames, times = _build(kfs, kts, exp=2)  # n=4
        # Expected per-segment centered offsets: T_seg/2/n on each side of every kt.
        # Boundary at t=4 (between seg span=4 and seg span=8):
        #   last sample of seg0 at 4 - 4/(2*4) = 3.5; first of seg1 at 4 + 8/(2*4) = 5.0
        self.assertIn(3.5, times)
        self.assertIn(5.0, times)
        self.assertNotIn(4.0, times)
        # Boundary at t=12: last of seg1 at 12 - 8/8 = 11.0; first of seg2 at 12 + 4/8 = 12.5
        self.assertIn(11.0, times)
        self.assertIn(12.5, times)
        self.assertNotIn(12.0, times)

    def test_on_frame_called_once_per_emitted_frame(self) -> None:
        kf0, kf1, kf2 = _solid_rgb(10), _solid_rgb(120), _solid_rgb(250)
        seen: list[tuple[int, float]] = []

        def _cb(idx: int, _arr: np.ndarray, t: float) -> None:
            seen.append((idx, t))

        fake_device = object()
        with mock.patch.object(
            rife_runtime, "load_rife_model", return_value=_FakeModel()
        ), mock.patch.object(
            rife_runtime,
            "rife_exp_interpolate_pair",
            side_effect=_fake_interpolate_pair,
        ), mock.patch("torch.cuda.is_available", return_value=False):
            class _Dev:
                type = "cpu"

            _frames, times = rife_runtime.rife_build_morph_timeline(
                [kf0, kf1, kf2],
                [0.0, 8.0, 16.0],
                exp=2,
                device=_Dev(),  # type: ignore[arg-type]
                on_frame=_cb,
                keep_frames=False,
            )
        # ``keep_frames=False`` still produces the full times grid …
        self.assertEqual(times, [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 16.0])
        # … and ``on_frame`` was invoked once per timeline slot, in order.
        self.assertEqual([t for _, t in seen], times)
        self.assertEqual([i for i, _ in seen], list(range(len(times))))


if __name__ == "__main__":
    unittest.main()
