"""Dense RIFE morph timeline structure (inset-warped centered sampling +
velocity-matched IFNet bookends + cross-pair boundary bridges).

These tests do not require CUDA or RIFE weights — :func:`load_rife_model`,
the per-pair IFNet interpolator, and the single-shot IFNet helper used for
the bookends and the v6 cross-pair bridges are all stubbed. The goal is to
lock in the timeline-times grid and the bookend / bridge behaviour that
together address the perceived stutter at SDXL keyframe boundaries (see
``docs/technical/rife-morph-background.md``):

* a velocity-matched IFNet bookend at ``t = keyframe_times[0]`` (first frame)
  and another at ``t = keyframe_times[-1]`` (last frame) — at IFNet timesteps
  ``s = inset`` and ``s = 1 - inset`` respectively, so the bookend → first
  body sample velocity equals the body's own ``span / T_seg`` (v5);
* a **cross-pair IFNet bridge** sample at every *internal* keyframe time
  ``t_{i+1}``, computed on the cross pair ``(kf_i, kf_{i+2})`` at ``s = 0.5``
  (v6) — its flow goes *across* the shared keyframe rather than into and
  out of it, so the compositor's pixel-space linear blend at the seam no
  longer collapses onto a static near-``kf_{i+1}`` plateau;
* dense wall-clock spacing is uniform ``T_seg / 2**exp`` within a segment;
  cross-pair bridges sit exactly on internal keyframe times (so the
  boundary blend windows are halved to ``T_seg / (2 * 2**exp)`` per side);
* with the default inset > 0 the boundary IFNet samples carry visible flow
  displacement instead of collapsing onto near-keyframe pixels — this is
  what the v3 → v4 → v5 → v6 schema bumps lock in.
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
        # Two segments [0, 8] and [8, 16], exp=2 (n=4). v6 expected dense times:
        #   bookend_start=0, seg0=[1, 3, 5, 7], bridge=8, seg1=[9, 11, 13, 15],
        #   bookend_end=16
        kf0, kf1, kf2 = _solid_rgb(10), _solid_rgb(120), _solid_rgb(250)
        frames, times = _build(
            [kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2, ifnet_timestep_inset=0.0
        )
        self.assertEqual(
            times,
            [0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0, 16.0],
        )

        # The internal SDXL keyframe time t=8 IS in the timeline now (v6
        # cross-pair bridge), but the frame there is *not* an exact kf1
        # still — it's IFNet on the cross pair (kf0, kf2) at s=0.5, which
        # under the linear-blend fake equals (kf0 + kf2)/2 ≠ kf1.
        idx_bridge = times.index(8.0)
        cross_midpoint = ((kf0.astype(np.int32) + kf2.astype(np.int32)) // 2).astype(np.uint8)
        np.testing.assert_array_equal(frames[idx_bridge], cross_midpoint)
        self.assertFalse(np.array_equal(frames[idx_bridge], kf1))

        # The two frames flanking the bridge are equidistant pair-wise IFNet
        # midpoints — neither equals the exact kf1 still.
        idx_left = times.index(7.0)
        idx_right = times.index(9.0)
        self.assertFalse(np.array_equal(frames[idx_left], kf1))
        self.assertFalse(np.array_equal(frames[idx_right], kf1))

        # Boundary windows are halved to T_seg/(2n) per side around the
        # bridge: spacing flank → bridge = bridge → flank = 1.0 (= 2.0 / 2).
        self.assertAlmostEqual(times[idx_bridge] - times[idx_left], 1.0)
        self.assertAlmostEqual(times[idx_right] - times[idx_bridge], 1.0)
        # And the within-segment spacing is unchanged at T_seg/n = 2.0.
        self.assertAlmostEqual(times[2] - times[1], 2.0)

    def test_uniform_spacing_across_internal_boundaries(self) -> None:
        # Four segments of varying durations to confirm the boundary spacing
        # rule ``T_seg/2**exp`` per segment holds independently per side and
        # that v6 cross-pair bridges land exactly on every internal keyframe.
        kfs = [_solid_rgb(v) for v in (5, 50, 100, 150, 200)]
        kts = [0.0, 4.0, 12.0, 16.0, 24.0]  # spans 4, 8, 4, 8
        _frames, times = _build(
            kfs, kts, exp=2, ifnet_timestep_inset=0.0
        )  # n=4
        # Expected per-segment centered offsets: T_seg/2/n on each side of every kt.
        # Boundary at t=4 (between seg span=4 and seg span=8):
        #   last sample of seg0 at 4 - 4/(2*4) = 3.5; bridge at 4.0;
        #   first of seg1 at 4 + 8/(2*4) = 5.0
        self.assertIn(3.5, times)
        self.assertIn(4.0, times)
        self.assertIn(5.0, times)
        # Boundary at t=12: last of seg1 at 12 - 8/8 = 11.0; bridge at 12.0;
        #   first of seg2 at 12 + 4/8 = 12.5
        self.assertIn(11.0, times)
        self.assertIn(12.0, times)
        self.assertIn(12.5, times)
        # Boundary at t=16: bridge sits exactly on the keyframe.
        self.assertIn(16.0, times)
        # All three internal-boundary bridges plus the start (0) and end
        # (24) bookends mean every keyframe time appears in the timeline.
        for kt in kts:
            self.assertIn(kt, times)

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
        # ``keep_frames=False`` still produces the full v6 times grid …
        self.assertEqual(
            times,
            [0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0, 16.0],
        )
        # … and ``on_frame`` was invoked once per timeline slot, in order
        # (start bookend + n body samples + bridge + n body samples + end
        # bookend).
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
        # Pick a value that is *not* the v6 default (0.25) so this test
        # actually verifies the env override path rather than coinciding
        # with the resolved default.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "0.10"}, clear=False
        ):
            frames, _times = _build(
                [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=None
            )
        ifnet_frames = frames[1:-1]
        recovered_s = [float(f[0, 0, 0]) / 255.0 for f in ifnet_frames]
        # inset=0.10, n=4 → s_j = 0.10 + 0.8*(j+0.5)/4 = [0.20, 0.40, 0.60, 0.80]
        for got, want in zip(recovered_s, (0.20, 0.40, 0.60, 0.80)):
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


class CrossPairBridgeTests(unittest.TestCase):
    """Lock in the v6 sampling: cross-pair IFNet bridges at internal keyframes.

    With pair-wise IFNet alone, the last body sample of segment ``i`` and
    the first body sample of segment ``i + 1`` are both visually pulled
    toward the shared keyframe ``kf_{i+1}``. The compositor's pixel-space
    linear blend across the seam therefore crossfades two near-``kf_{i+1}``
    images for the entire ``T_seg/n`` boundary window — perceived by the
    user as the morph "snapping onto a still image" at every keyframe even
    though no exact still is in the timeline (the symptom that survived
    v4's inset and v5's bookend velocity match).

    v6 inserts one extra IFNet inference at every internal keyframe time
    on the **cross pair** ``(kf_i, kf_{i+2})`` at ``s = 0.5``. These tests
    pin (a) that exactly one bridge is emitted per internal boundary, (b)
    that it lands at exactly the keyframe time, and (c) that it is
    computed from the cross pair rather than either flanking pair.
    """

    def test_no_bridge_for_single_segment(self) -> None:
        # Two keyframes ⇒ one segment ⇒ zero internal boundaries ⇒ no
        # bridges. v6 frame count for a single segment is identical to v5.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.20
        )
        # 1 bookend + 4 body + 1 bookend = 6 (no bridge slot in a one-seg run).
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(times), 6)

    def test_one_bridge_per_internal_boundary(self) -> None:
        # 4 keyframes ⇒ 3 segments ⇒ 2 internal boundaries ⇒ 2 bridges.
        # Frame count = 1 + 3*4 + 2 + 1 = 16.
        kfs = [_solid_rgb(v) for v in (0, 80, 160, 240)]
        kts = [0.0, 8.0, 16.0, 24.0]
        frames, times = _build(kfs, kts, exp=2, ifnet_timestep_inset=0.0)
        self.assertEqual(len(frames), 16)
        self.assertEqual(len(times), 16)
        # Both internal keyframe times are present (bridges sit there);
        # the song bookends are also keyframe times.
        for kt in kts:
            self.assertIn(kt, times)

    def test_bridge_uses_cross_pair_not_flanking_pair(self) -> None:
        # Three keyframes with R-channel values chosen so every candidate
        # midpoint is distinct (and ``kf_{i+1}`` is *not* the midpoint of
        # ``kf_i`` and ``kf_{i+2}``):
        #   pair (kf0, kf1) midpoint at s=0.5 → 35   (= flanking-left guess)
        #   pair (kf1, kf2) midpoint at s=0.5 → 135  (= flanking-right guess)
        #   cross (kf0, kf2) midpoint at s=0.5 → 100 (= the actual v6 bridge)
        #   kf1 R=70                                 (would be the bridge if
        #                                             we naively re-used the
        #                                             exact still)
        kf0 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf0[..., 0] = 0
        kf1 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf1[..., 0] = 70
        kf2 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf2[..., 0] = 200
        frames, times = _build(
            [kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2, ifnet_timestep_inset=0.0
        )
        idx_bridge = times.index(8.0)
        bridge_r = float(frames[idx_bridge][0, 0, 0])
        self.assertAlmostEqual(bridge_r, 100.0, delta=1.0)  # cross pair
        # And explicitly *not* any of the alternative interpretations:
        for wrong in (35.0, 135.0, 70.0):
            self.assertGreater(
                abs(bridge_r - wrong),
                5.0,
                msg=f"bridge byte {bridge_r} matched alternative midpoint {wrong}",
            )

    def test_bridge_independent_of_body_inset(self) -> None:
        # Cross-pair bridge always uses ``s = 0.5`` regardless of the body
        # inset (the bridge purpose is purely perceptual smoothing at the
        # seam; tying it to the body inset would defeat that). Verify by
        # baking with two different insets and checking the bridge byte
        # value lands at the cross-pair midpoint in both.
        kf0 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf0[..., 0] = 0
        kf1 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf1[..., 0] = 100
        kf2 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf2[..., 0] = 200
        for inset in (0.0, 0.25, 0.40):
            frames, times = _build(
                [kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2, ifnet_timestep_inset=inset
            )
            idx_bridge = times.index(8.0)
            # Cross (kf0, kf2) midpoint at s=0.5 → (0 + 200)/2 = 100.
            self.assertAlmostEqual(
                float(frames[idx_bridge][0, 0, 0]),
                100.0,
                delta=1.0,
                msg=f"bridge byte mismatch at inset={inset}",
            )

    def test_bridge_at_exact_keyframe_time(self) -> None:
        # Bridge wall-clock placement is exactly ``t = keyframe_times[seg_i + 1]``
        # (the upcoming internal boundary). Verify with non-uniform spacing
        # so the assertion isn't satisfied by a coincidence.
        kfs = [_solid_rgb(v) for v in (0, 80, 160, 240)]
        kts = [0.0, 5.5, 14.0, 24.0]
        _frames, times = _build(kfs, kts, exp=2, ifnet_timestep_inset=0.20)
        # Internal keyframes at 5.5 and 14.0 must both appear in the times
        # grid (one bridge each).
        self.assertIn(5.5, times)
        self.assertIn(14.0, times)
        # The body samples flanking each bridge sit at T_seg/(2n) per side:
        #   seg0 (T=5.5): last body at 5.5 - 5.5/8 = 4.8125
        #   seg1 (T=8.5): first body at 5.5 + 8.5/8 = 6.5625
        self.assertIn(4.8125, times)
        self.assertIn(6.5625, times)


if __name__ == "__main__":
    unittest.main()
