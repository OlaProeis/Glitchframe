"""Dense RIFE morph timeline structure (inset-warped centered sampling +
velocity-matched IFNet bookends).

These tests do not require CUDA or RIFE weights — :func:`load_rife_model`,
the per-pair IFNet interpolator, and the single-shot IFNet helper used for
the bookends are all stubbed. The goal is to lock in the timeline-times
grid and the bookend behaviour that together address the perceived stutter
at SDXL keyframe boundaries (see ``docs/technical/rife-morph-background.md``):

* a velocity-matched IFNet bookend at ``t = keyframe_times[0]`` (first frame)
  and another at ``t = keyframe_times[-1]`` (last frame) — at IFNet timesteps
  ``s = inset`` and ``s = 1 - inset`` respectively, so the bookend → first
  body sample velocity equals the body's own ``span / T_seg``;
* every internal SDXL keyframe time is **bridged** by two soft IFNet
  predictions equally offset by ``T_seg / (2 * 2**exp)``, so ``t_kf`` itself
  is never an exact still in the dense timeline;
* dense wall-clock spacing is uniform ``T_seg / 2**exp`` within and across
  segments (the IFNet timestep is decoupled from wall-clock by an inset);
* with the default inset > 0 the boundary IFNet samples carry visible flow
  displacement instead of collapsing onto near-keyframe pixels — this is
  what the v3 → v4 → v5 schema bumps lock in.
"""

from __future__ import annotations

import os
import unittest
from typing import Sequence
from unittest import mock

import numpy as np

import pipeline.rife_runtime as rife_runtime


def _solid_rgb(value: int) -> np.ndarray:
    """Tiny synthetic SDXL-still stand-in (8x8 RGB, distinguishable byte value)."""
    return np.full((8, 8, 3), int(value), dtype=np.uint8)


def _lerp_uint8(rgb0: np.ndarray, rgb1: np.ndarray, ts: float) -> np.ndarray:
    """Linear blend used by both fakes — keeps the byte value of every
    output frame a faithful encoding of the IFNet timestep ``ts`` so tests
    can recover the timestep with ``frame[0, 0, 0] / 255``."""
    return (
        (1.0 - ts) * rgb0.astype(np.float32) + ts * rgb1.astype(np.float32)
    ).astype(np.uint8)


def _fake_interpolate_pair(
    _model,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    *,
    exp: int,
    device,
    include_endpoints: bool = True,
    ifnet_timestep_inset: float = 0.0,
) -> list[np.ndarray]:
    """Stand-in for ``rife_exp_interpolate_pair`` that returns deterministic
    midpoint blends instead of running IFNet, but preserves the same length
    contract (``2**exp`` frames when ``include_endpoints=False``) and applies
    the same inset warp on the IFNet timestep so callers can verify it.
    """
    n = 2**int(exp)
    if include_endpoints:
        # Not exercised by the new timeline builder, but kept for completeness.
        out = [rgb0.copy()]
        for i in range(n - 1):
            ts = (i + 1) * 1.0 / n
            out.append(_lerp_uint8(rgb0, rgb1, ts))
        out.append(rgb1.copy())
        return out
    inset = max(0.0, min(0.45, float(ifnet_timestep_inset)))
    span = 1.0 - 2.0 * inset
    return [
        _lerp_uint8(rgb0, rgb1, inset + span * (i + 0.5) / n)
        for i in range(n)
    ]


def _fake_ifnet_at_timestep(
    _model,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    *,
    timestep: float,
    device,
) -> np.ndarray:
    """Stand-in for ``rife_ifnet_at_timestep`` that mirrors the body fake's
    linear-blend convention so a test reading ``frame[0, 0, 0] / 255``
    recovers exactly the requested IFNet timestep ``s``.
    """
    return _lerp_uint8(rgb0, rgb1, float(timestep))


class _FakeModel:
    fp16 = False


def _build(
    keyframes: Sequence[np.ndarray],
    keyframe_times: Sequence[float],
    *,
    exp: int,
    ifnet_timestep_inset: float | None = None,
    on_frame=None,
    keep_frames: bool = True,
):
    with mock.patch.object(
        rife_runtime, "load_rife_model", return_value=_FakeModel()
    ), mock.patch.object(
        rife_runtime,
        "rife_exp_interpolate_pair",
        side_effect=_fake_interpolate_pair,
    ), mock.patch.object(
        rife_runtime,
        "rife_ifnet_at_timestep",
        side_effect=_fake_ifnet_at_timestep,
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
            ifnet_timestep_inset=ifnet_timestep_inset,
            on_frame=on_frame,
            keep_frames=keep_frames,
        )


class CenteredMorphTimelineTests(unittest.TestCase):
    def test_two_keyframes_single_segment(self) -> None:
        # Single segment [0, 8] with exp=2 (n=4). Expected dense times:
        #   [bookend_start=0, body=1.0, 3.0, 5.0, 7.0, bookend_end=8.0]
        # Wall-clock spacing is independent of the IFNet timestep inset —
        # only the per-frame visual *content* depends on the inset.
        kf0, kf1 = _solid_rgb(10), _solid_rgb(250)
        frames, times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.0
        )
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(times), 6)
        self.assertEqual(times, [0.0, 1.0, 3.0, 5.0, 7.0, 8.0])
        # At ``inset = 0`` the bookend IFNet timesteps collapse to ``s = 0``
        # / ``s = 1`` — under the linear-blend fake those reduce to the exact
        # source pixels, byte-identical to the keyframes (with real IFNet
        # they would be visually indistinguishable but not byte-equal).
        np.testing.assert_array_equal(frames[0], kf0)
        np.testing.assert_array_equal(frames[-1], kf1)

    def test_three_keyframes_no_internal_exact_still(self) -> None:
        # Two segments [0, 8] and [8, 16], exp=2 (n=4). Expected dense times:
        #   exact_start=0, seg0=[1, 3, 5, 7], seg1=[9, 11, 13, 15], exact_end=16
        kf0, kf1, kf2 = _solid_rgb(10), _solid_rgb(120), _solid_rgb(250)
        frames, times = _build(
            [kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2, ifnet_timestep_inset=0.0
        )
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
        _frames, times = _build(
            kfs, kts, exp=2, ifnet_timestep_inset=0.0
        )  # n=4
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

        _frames, times = _build(
            [kf0, kf1, kf2],
            [0.0, 8.0, 16.0],
            exp=2,
            ifnet_timestep_inset=0.0,
            on_frame=_cb,
            keep_frames=False,
        )
        # ``keep_frames=False`` still produces the full times grid …
        self.assertEqual(times, [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 16.0])
        # … and ``on_frame`` was invoked once per timeline slot, in order
        # (one bookend at each end + n samples per segment).
        self.assertEqual([t for _, t in seen], times)
        self.assertEqual([i for i, _ in seen], list(range(len(times))))


class IfnetTimestepInsetTests(unittest.TestCase):
    """Lock in the v4 sampling: boundary IFNet samples must carry visible flow."""

    def test_default_inset_pushes_boundary_samples_inward(self) -> None:
        # The fake interpolator linearly blends ``rgb0`` and ``rgb1`` at the
        # IFNet timestep, so the produced byte value directly encodes the
        # effective ``s`` for each emitted frame.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        env_no_override = {
            k: v
            for k, v in os.environ.items()
            if k != "GLITCHFRAME_RIFE_IFNET_INSET"
        }
        with mock.patch.dict(os.environ, env_no_override, clear=True):
            frames, times = _build(
                [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=None
            )  # ``None`` → resolved default (no env override)
        # Strip the bookend exact stills; the remaining 4 frames are the
        # IFNet predictions for this single segment.
        ifnet_frames = frames[1:-1]
        self.assertEqual(len(ifnet_frames), 4)
        recovered_s = [float(f[0, 0, 0]) / 255.0 for f in ifnet_frames]
        default = rife_runtime.DEFAULT_IFNET_TIMESTEP_INSET
        span = 1.0 - 2.0 * default
        expected = [default + span * (j + 0.5) / 4.0 for j in range(4)]
        # Both boundary samples are pushed comfortably away from 0/1, which is
        # exactly what eliminates the perceived "pause on the original still".
        self.assertGreater(recovered_s[0], default - 1e-3)
        self.assertLess(recovered_s[-1], 1.0 - default + 1e-3)
        for got, want in zip(recovered_s, expected):
            self.assertAlmostEqual(got, want, places=2)
        # Wall-clock placement is independent of the inset.
        self.assertEqual(times, [0.0, 1.0, 3.0, 5.0, 7.0, 8.0])

    def test_explicit_zero_inset_recovers_legacy_centered_sampling(self) -> None:
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, _times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.0
        )
        ifnet_frames = frames[1:-1]
        recovered_s = [float(f[0, 0, 0]) / 255.0 for f in ifnet_frames]
        # Plain centered: s_j = (j+0.5)/4 = [0.125, 0.375, 0.625, 0.875]
        for got, want in zip(recovered_s, (0.125, 0.375, 0.625, 0.875)):
            self.assertAlmostEqual(got, want, places=2)

    def test_env_var_override_when_kwarg_omitted(self) -> None:
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "0.25"}, clear=False
        ):
            frames, _times = _build(
                [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=None
            )
        ifnet_frames = frames[1:-1]
        recovered_s = [float(f[0, 0, 0]) / 255.0 for f in ifnet_frames]
        # inset=0.25, n=4 → s_j = 0.25 + 0.5*(j+0.5)/4 = [0.3125, 0.4375, 0.5625, 0.6875]
        for got, want in zip(recovered_s, (0.3125, 0.4375, 0.5625, 0.6875)):
            self.assertAlmostEqual(got, want, places=2)

    def test_env_var_clamped_to_safe_max(self) -> None:
        # Out-of-range overrides are clamped, not raised, so a typo can't
        # accidentally degenerate the interval.
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "9.0"}, clear=False
        ):
            self.assertAlmostEqual(
                rife_runtime._resolve_ifnet_timestep_inset(),
                rife_runtime._IFNET_TIMESTEP_INSET_MAX,
            )
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "-1.0"}, clear=False
        ):
            self.assertAlmostEqual(
                rife_runtime._resolve_ifnet_timestep_inset(),
                rife_runtime._IFNET_TIMESTEP_INSET_MIN,
            )

    def test_env_var_garbage_falls_back_to_default(self) -> None:
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "not-a-number"}, clear=False
        ):
            self.assertAlmostEqual(
                rife_runtime._resolve_ifnet_timestep_inset(),
                rife_runtime.DEFAULT_IFNET_TIMESTEP_INSET,
            )

    def test_explicit_override_wins_over_env_var(self) -> None:
        # An explicit non-None kwarg must beat the env var so callers
        # (notably ``rife_build_morph_timeline``) can always pin a value.
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "0.30"}, clear=False
        ):
            self.assertAlmostEqual(
                rife_runtime._resolve_ifnet_timestep_inset(0.05),
                0.05,
            )


class BookendVelocityTests(unittest.TestCase):
    """Lock in the v5 sampling: bookend IFNet velocity matches body velocity.

    The v4 timeline kept the legacy *exact* SDXL still at ``t = 0`` and
    ``t = duration`` while pulling the body's first/last samples inward by
    the inset, leaving an IFNet jump of ``inset + span/(2n)`` over only
    ``T_seg/(2n)`` wall-clock seconds — about 6× the body's pace at the
    default inset, which the user perceived as a "skip" right before the
    closing still. v5 samples the bookends at ``s = inset`` (start) and
    ``s = 1 - inset`` (end) so the bookend → first body sample velocity
    equals the body's own ``span / T_seg``. These tests pin that contract.
    """

    def test_bookend_at_inset_when_inset_positive(self) -> None:
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, _times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.20
        )
        # ``frame[0, 0, 0] / 255`` recovers the IFNet timestep under the fake.
        self.assertAlmostEqual(float(frames[0][0, 0, 0]) / 255.0, 0.20, places=2)
        self.assertAlmostEqual(float(frames[-1][0, 0, 0]) / 255.0, 0.80, places=2)

    def test_velocity_continuous_through_start_and_end(self) -> None:
        # n=4, T_seg=8, inset=0.20 ⇒
        #   start bookend: s=0.20 at t=0
        #   body: s ∈ {0.275, 0.425, 0.575, 0.725} at t ∈ {1, 3, 5, 7}
        #   end bookend: s=0.80 at t=8
        # Velocity from bookend → first body and from last body → bookend
        # must equal the body's own velocity (span / T_seg).
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.20
        )
        s = [float(f[0, 0, 0]) / 255.0 for f in frames]
        body_velocity = (s[2] - s[1]) / (times[2] - times[1])
        v_open = (s[1] - s[0]) / (times[1] - times[0])
        v_close = (s[-1] - s[-2]) / (times[-1] - times[-2])
        self.assertAlmostEqual(v_open, body_velocity, places=4)
        self.assertAlmostEqual(v_close, body_velocity, places=4)

    def test_velocity_continuous_at_default_inset(self) -> None:
        # Same continuity check at the default inset that ships in production.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=None
        )
        s = [float(f[0, 0, 0]) / 255.0 for f in frames]
        body_velocity = (s[2] - s[1]) / (times[2] - times[1])
        v_open = (s[1] - s[0]) / (times[1] - times[0])
        v_close = (s[-1] - s[-2]) / (times[-1] - times[-2])
        self.assertAlmostEqual(v_open, body_velocity, places=4)
        self.assertAlmostEqual(v_close, body_velocity, places=4)

    def test_end_bookend_uses_last_keyframe_pair(self) -> None:
        # With three keyframes the *end* bookend must be computed from the
        # last pair (kf1, kf2), not the first pair (kf0, kf1) — otherwise
        # the closing image would morph back toward the first half of the
        # song. Verify by giving each keyframe a distinct R-channel value
        # and checking the end bookend lerps between kf1 and kf2.
        kf0 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf0[..., 0] = 0
        kf1 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf1[..., 0] = 100
        kf2 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf2[..., 0] = 200
        frames, _times = _build(
            [kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2, ifnet_timestep_inset=0.20
        )
        # End bookend = lerp(kf1, kf2, s=0.80) = 100*0.20 + 200*0.80 = 180
        self.assertAlmostEqual(float(frames[-1][0, 0, 0]), 180.0, delta=1.0)
        # Start bookend = lerp(kf0, kf1, s=0.20) = 0*0.80 + 100*0.20 = 20
        self.assertAlmostEqual(float(frames[0][0, 0, 0]), 20.0, delta=1.0)

    def test_bookend_at_zero_inset_collapses_to_keyframe(self) -> None:
        # At ``inset = 0`` the bookend timesteps collapse to s = 0 / s = 1,
        # which under the fake's linear blend reproduce the source keyframes
        # byte-exactly — preserving the legacy reproducibility contract.
        kf0, kf1 = _solid_rgb(7), _solid_rgb(241)
        frames, _times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.0
        )
        np.testing.assert_array_equal(frames[0], kf0)
        np.testing.assert_array_equal(frames[-1], kf1)


if __name__ == "__main__":
    unittest.main()
