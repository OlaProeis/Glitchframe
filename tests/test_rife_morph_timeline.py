"""Dense RIFE morph timeline structure (inset-warped centered sampling +
velocity-matched IFNet bookends + IFNet-rendered internal-keyframe anchors).

These tests do not require CUDA or RIFE weights — :func:`load_rife_model`,
the per-pair IFNet interpolator, and the single-shot IFNet helper used for
the bookends *and the internal anchors* are all stubbed. The goal is to
lock in the timeline-times grid and the bookend / internal-anchor
behaviour that together address the perceived stutter at SDXL keyframe
boundaries (see ``docs/technical/rife-morph-background.md``):

* a velocity-matched IFNet bookend at ``t = keyframe_times[0]`` (first frame)
  and another at ``t = keyframe_times[-1]`` (last frame) — at IFNet timesteps
  ``s = inset`` and ``s = 1 - inset`` respectively, so the bookend → first
  body sample velocity equals the body's own ``span / T_seg`` (v5);
* an **IFNet render** ``IFNet(kf_i, kf_{i+1}, s=1.0)`` placed at every
  *internal* keyframe time ``t_{i+1}`` (v7.1 / schema v8) — sharing
  IFNet's texture signature with every flanking body sample so the
  compositor's pixel-space linear blend never crosses a VAE↔IFNet hand-
  off at the seam (the texture "blip" v7 introduced when it placed the
  raw SDXL still here on the false assumption that IFNet had an identity
  branch at the endpoints). The post-anchor blend still crosses optical-
  flow pairs but between two IFNet renders both dominated by ``kf_{i+1}``
  content, so the flow residual at the boundary is at its minimum
  visible scale — far smaller than v6's cross-pair ``s=0.5`` bridge
  (which lived in a flow space *neither* flanking sample lived in and
  produced the "clear move of the object" symptom);
* dense wall-clock spacing is uniform ``T_seg / 2**exp`` within a segment;
  internal-keyframe anchors sit exactly on the keyframe times (so the
  boundary blend windows are halved to ``T_seg / (2 * 2**exp)`` per side);
* with the default inset > 0 the boundary IFNet samples carry visible flow
  displacement instead of collapsing onto near-keyframe pixels — this is
  what the v3 → v4 → v5 → v7 → v8 (= v7.1) schema bumps lock in (v6 used
  the same inset-warped body but with a cross-pair bridge; v7 replaced
  the bridge with the raw SDXL still; v7.1 keeps the anchor *position*
  but renders it through IFNet to share texture with its neighbours).
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
    return_ifnet_mock: bool = False,
):
    """Run ``rife_build_morph_timeline`` with all GPU dependencies stubbed.

    When ``return_ifnet_mock`` is set, the third tuple element is the
    :class:`unittest.mock.MagicMock` patched in for ``rife_ifnet_at_timestep``
    — tests can inspect ``call_args_list`` to verify that v7.1 internal
    anchors invoke the IFNet endpoint render rather than emitting a raw
    keyframe copy (the v7 regression: under the linear-blend fake at
    ``s = 1.0`` the IFNet render byte-equals ``kf_{i+1}``, so byte
    comparisons alone can't distinguish the two implementations).
    """
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
    ) as ifnet_mock, mock.patch("torch.cuda.is_available", return_value=False):
        # ``device.type`` is read by the cudnn-benchmark guard; an object
        # without ``type`` would break that branch, so just stub it.
        class _Dev:
            type = "cpu"

        frames, times = rife_runtime.rife_build_morph_timeline(
            keyframes,
            keyframe_times,
            exp=exp,
            device=_Dev(),  # type: ignore[arg-type]
            ifnet_timestep_inset=ifnet_timestep_inset,
            on_frame=on_frame,
            keep_frames=keep_frames,
        )
        if return_ifnet_mock:
            return frames, times, ifnet_mock
        return frames, times


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

    def test_three_keyframes_internal_anchor_at_keyframe_time(self) -> None:
        # Two segments [0, 8] and [8, 16], exp=2 (n=4). v7.1 expected dense
        # times:
        #   bookend_start=0, seg0=[1, 3, 5, 7], anchor=8, seg1=[9, 11, 13, 15],
        #   bookend_end=16
        kf0, kf1, kf2 = _solid_rgb(10), _solid_rgb(120), _solid_rgb(250)
        frames, times = _build(
            [kf0, kf1, kf2], [0.0, 8.0, 16.0], exp=2, ifnet_timestep_inset=0.0
        )
        self.assertEqual(
            times,
            [0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0, 16.0],
        )

        # v7.1: the frame at the internal keyframe time t=8 is
        # ``IFNet(kf0, kf1, s=1.0)`` — IFNet's own render of kf1, *not* a
        # byte-identical copy of the SDXL still kf1. Under the linear-
        # blend fake ``IFNet(kf0, kf1, s=1.0)`` happens to byte-equal kf1
        # (lerp at s=1 collapses to rgb1), so this byte check still
        # passes — but the real contract (anchor produced via
        # ``rife_ifnet_at_timestep``) is verified separately in
        # :class:`InternalKeyframeAnchorTests`. See that class's
        # ``test_anchor_calls_ifnet_at_s_one_on_prior_pair``.
        idx_anchor = times.index(8.0)
        np.testing.assert_array_equal(frames[idx_anchor], kf1)

        # The two frames flanking the anchor are pair-wise IFNet midpoints
        # at s=0.875 (last body of seg0) and s=0.125 (first body of seg1)
        # under the linear-blend fake at inset=0 — *not* the kf1 still.
        idx_left = times.index(7.0)
        idx_right = times.index(9.0)
        self.assertFalse(np.array_equal(frames[idx_left], kf1))
        self.assertFalse(np.array_equal(frames[idx_right], kf1))

        # Boundary windows are halved to T_seg/(2n) per side around the
        # anchor: spacing flank → anchor = anchor → flank = 1.0 (= 2.0 / 2).
        self.assertAlmostEqual(times[idx_anchor] - times[idx_left], 1.0)
        self.assertAlmostEqual(times[idx_right] - times[idx_anchor], 1.0)
        # And the within-segment spacing is unchanged at T_seg/n = 2.0.
        self.assertAlmostEqual(times[2] - times[1], 2.0)

    def test_uniform_spacing_across_internal_boundaries(self) -> None:
        # Four segments of varying durations to confirm the boundary spacing
        # rule ``T_seg/2**exp`` per segment holds independently per side and
        # that v7.1 internal-keyframe anchors land exactly on every internal
        # keyframe time.
        kfs = [_solid_rgb(v) for v in (5, 50, 100, 150, 200)]
        kts = [0.0, 4.0, 12.0, 16.0, 24.0]  # spans 4, 8, 4, 8
        _frames, times = _build(
            kfs, kts, exp=2, ifnet_timestep_inset=0.0
        )  # n=4
        # Expected per-segment centered offsets: T_seg/2/n on each side of every kt.
        # Boundary at t=4 (between seg span=4 and seg span=8):
        #   last sample of seg0 at 4 - 4/(2*4) = 3.5; anchor at 4.0;
        #   first of seg1 at 4 + 8/(2*4) = 5.0
        self.assertIn(3.5, times)
        self.assertIn(4.0, times)
        self.assertIn(5.0, times)
        # Boundary at t=12: last of seg1 at 12 - 8/8 = 11.0; anchor at 12.0;
        #   first of seg2 at 12 + 4/8 = 12.5
        self.assertIn(11.0, times)
        self.assertIn(12.0, times)
        self.assertIn(12.5, times)
        # Boundary at t=16: anchor sits exactly on the keyframe.
        self.assertIn(16.0, times)
        # All three internal-keyframe anchors plus the start (0) and end
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
        # ``keep_frames=False`` still produces the full v7.1 times grid …
        self.assertEqual(
            times,
            [0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0, 16.0],
        )
        # … and ``on_frame`` was invoked once per timeline slot, in order
        # (start bookend + n body samples + internal IFNet anchor + n body
        # samples + end bookend).
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
        # Pick a value that is *not* the v7.1 default (0.12) so this test
        # actually verifies the env override path rather than coinciding
        # with the resolved default.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_RIFE_IFNET_INSET": "0.30"}, clear=False
        ):
            frames, _times = _build(
                [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=None
            )
        ifnet_frames = frames[1:-1]
        recovered_s = [float(f[0, 0, 0]) / 255.0 for f in ifnet_frames]
        # inset=0.30, n=4 → s_j = 0.30 + 0.4*(j+0.5)/4 = [0.35, 0.45, 0.55, 0.65]
        for got, want in zip(recovered_s, (0.35, 0.45, 0.55, 0.65)):
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
        # must equal the body's own velocity (span / T_seg). We compare to
        # ``places=2`` (~1 % tolerance) because the fake interpolator
        # quantises to uint8 via ``.astype(np.uint8)`` truncation — the
        # IFNet-timestep continuity holds exactly in float, but byte-level
        # rounding shaves up to 0.5/255 ≈ 0.002 off each velocity sample.
        # Real IFNet has the same uint8 quantisation step on its output,
        # so the perceptual contract here is "same speed to ~1 %", not
        # "same speed to sub-byte precision".
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=0.20
        )
        s = [float(f[0, 0, 0]) / 255.0 for f in frames]
        body_velocity = (s[2] - s[1]) / (times[2] - times[1])
        v_open = (s[1] - s[0]) / (times[1] - times[0])
        v_close = (s[-1] - s[-2]) / (times[-1] - times[-2])
        self.assertAlmostEqual(v_open, body_velocity, places=2)
        self.assertAlmostEqual(v_close, body_velocity, places=2)

    def test_velocity_continuous_at_default_inset(self) -> None:
        # Same continuity check at the default inset that ships in
        # production. See note above for why this uses ``places=2``.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, times = _build(
            [kf0, kf1], [0.0, 8.0], exp=2, ifnet_timestep_inset=None
        )
        s = [float(f[0, 0, 0]) / 255.0 for f in frames]
        body_velocity = (s[2] - s[1]) / (times[2] - times[1])
        v_open = (s[1] - s[0]) / (times[1] - times[0])
        v_close = (s[-1] - s[-2]) / (times[-1] - times[-2])
        self.assertAlmostEqual(v_open, body_velocity, places=2)
        self.assertAlmostEqual(v_close, body_velocity, places=2)

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


class InternalKeyframeAnchorTests(unittest.TestCase):
    """Lock in the v7.1 sampling: IFNet-rendered anchors at internal boundaries.

    With pair-wise IFNet alone, the last body sample of segment ``i`` and
    the first body sample of segment ``i + 1`` come from *different* IFNet
    pairs but both look visually similar to the shared keyframe ``kf_{i+1}``.
    The flow fields, however, are pair-specific — the cat's eye position
    in ``IFNet(kf_i, kf_{i+1}, s=1-inset)`` is *not* the same pixel as in
    ``IFNet(kf_{i+1}, kf_{i+2}, s=inset)``, even though both frames are
    "mostly cat". The compositor's pixel-space linear blend across that
    pair-mismatched seam therefore translates the dominant content across
    the ``T_seg/n`` boundary window, perceived by the user as a brief
    "skip" of the object's position at every keyframe.

    v6 tried to break the plateau by inserting an IFNet sample on the
    *cross pair* ``(kf_i, kf_{i+2})`` at ``s = 0.5`` at every internal
    keyframe time — but that introduced a *third* pair-mismatched warp
    into the seam (the cross-pair flow is yet another spatial layout)
    and amplified the spatial-jump perception ("clear move of the
    object").

    v7 placed the **exact SDXL keyframe** ``kf_{i+1}`` at every internal
    boundary on the assumption that it was byte-identical to
    ``IFNet(kf_i, kf_{i+1}, s=1.0)`` modulo identity behaviour at the
    endpoints. That assumption is wrong: Practical-RIFE's :class:`IFNet`
    has *no* identity branch at ``s = 0`` or ``s = 1`` — it always runs
    all five flow-refinement blocks with ``timestep`` baked into the
    feature concatenation. The raw VAE still and the IFNet render at
    ``s = 1.0`` therefore differ by a per-network texture signature
    (VAE sharper / more saturated; IFNet smoother). v7 had the
    compositor swap VAE ↔ IFNet texture every internal boundary, which
    read as a 1–2-video-frame texture "blip" on every keyframe.

    v7.1 (schema v8) renders the anchor through IFNet at ``s = 1.0`` on
    the *prior* pair ``(kf_i, kf_{i+1})`` instead of using the raw SDXL
    still. The anchor now shares IFNet's texture signature with every
    flanking body sample. Pre-anchor blend stays entirely within pair
    ``A`` (same flow field, same texture). Post-anchor blend crosses to
    pair ``B`` but between two IFNet renders both dominated by
    ``kf_{i+1}`` content (the anchor is a full warp toward ``kf_{i+1}``;
    the first body of seg_{i+1} is only ``inset`` along the next flow),
    so the flow residual at the boundary is at its minimum visible
    scale — and crucially there is no VAE↔IFNet texture hand-off.

    These tests pin (a) that exactly one anchor is emitted per internal
    boundary, (b) that it lands at exactly the keyframe time, (c) that
    each anchor is produced by an explicit ``rife_ifnet_at_timestep``
    call on the prior pair at ``timestep = 1.0`` (the v7.1 contract that
    distinguishes from a v7-style raw-keyframe copy), and (d) that the
    anchor's IFNet call signature does not depend on the body inset.
    """

    def test_no_anchor_for_single_segment(self) -> None:
        # Two keyframes ⇒ one segment ⇒ zero internal boundaries ⇒ no
        # anchors. Frame count for a single segment is just the two
        # bookends and the body samples.
        kf0, kf1 = _solid_rgb(0), _solid_rgb(255)
        frames, times, ifnet_mock = _build(
            [kf0, kf1],
            [0.0, 8.0],
            exp=2,
            ifnet_timestep_inset=0.20,
            return_ifnet_mock=True,
        )
        # 1 bookend + 4 body + 1 bookend = 6 (no anchor slot in a one-seg run).
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(times), 6)
        # Both IFNet calls in this run must be bookends (s = inset and
        # s = 1 - inset). No s = 1.0 call ⇒ no internal anchor.
        ts_called = [c.kwargs["timestep"] for c in ifnet_mock.call_args_list]
        self.assertEqual(len(ts_called), 2)
        self.assertNotIn(1.0, ts_called)

    def test_one_anchor_per_internal_boundary(self) -> None:
        # 4 keyframes ⇒ 3 segments ⇒ 2 internal boundaries ⇒ 2 anchors.
        # Frame count = 1 + 3*4 + 2 + 1 = 16.
        kfs = [_solid_rgb(v) for v in (0, 80, 160, 240)]
        kts = [0.0, 8.0, 16.0, 24.0]
        frames, times, ifnet_mock = _build(
            kfs, kts, exp=2, ifnet_timestep_inset=0.0, return_ifnet_mock=True
        )
        self.assertEqual(len(frames), 16)
        self.assertEqual(len(times), 16)
        # Both internal keyframe times are present (anchors sit there);
        # the song bookends are also keyframe times.
        for kt in kts:
            self.assertIn(kt, times)
        # Exactly two ``s = 1.0`` calls — one per internal boundary.
        # (The two bookends use ``s = inset`` and ``s = 1 - inset``,
        # which at inset = 0 collapse to ``s = 0`` and ``s = 1``; the
        # latter would *also* be ``1.0``, so isolate the anchor calls
        # by their pair signature: anchors use the *prior* pair, not
        # the *last* pair like the end bookend does.)
        ts_called = [c.kwargs["timestep"] for c in ifnet_mock.call_args_list]
        self.assertEqual(ts_called.count(1.0), 3)  # 2 anchors + 1 end bookend
        self.assertEqual(ts_called.count(0.0), 1)  # 1 start bookend (s = inset = 0)

    def test_anchor_calls_ifnet_at_s_one_on_prior_pair(self) -> None:
        # Three keyframes with distinguishable R-channel values so we can
        # match each IFNet call's pair against the keyframe identities.
        # kf1 R=70 is chosen to differ from every candidate midpoint a
        # regression to v6's cross-pair bridge or to either flanking pair
        # midpoint would land at.
        kf0 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf0[..., 0] = 0
        kf1 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf1[..., 0] = 70
        kf2 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf2[..., 0] = 200
        frames, times, ifnet_mock = _build(
            [kf0, kf1, kf2],
            [0.0, 8.0, 16.0],
            exp=2,
            ifnet_timestep_inset=0.20,
            return_ifnet_mock=True,
        )

        # Locate the IFNet call that produced the internal anchor: it is
        # the one (and only one) with ``timestep == 1.0`` whose pair is
        # ``(kf0, kf1)`` (the *prior* pair). The end bookend's call uses
        # ``timestep = 1 - inset = 0.80``, so timestep alone suffices,
        # but assert the pair too to lock in "prior pair, not last pair".
        anchor_calls = [
            c for c in ifnet_mock.call_args_list if c.kwargs["timestep"] == 1.0
        ]
        self.assertEqual(
            len(anchor_calls),
            1,
            msg="expected exactly one IFNet(s=1.0) call (the internal anchor)",
        )
        anchor_call = anchor_calls[0]
        np.testing.assert_array_equal(anchor_call.args[1], kf0)
        np.testing.assert_array_equal(anchor_call.args[2], kf1)

        # The anchor frame in the timeline (at t = 8.0) is the linear
        # blend's value at s = 1.0 of pair (kf0, kf1), which under the
        # fake byte-equals kf1 — the same byte content v7 emitted, but
        # produced via the IFNet path instead of a raw copy. Real IFNet
        # would render a *slightly* smoother version of kf1 here; the
        # fake collapses to byte-exact rgb1 at s = 1, which is fine for
        # the timeline-structure contract this test pins.
        idx_anchor = times.index(8.0)
        anchor_r = float(frames[idx_anchor][0, 0, 0])
        self.assertAlmostEqual(anchor_r, 70.0, delta=1.0)
        for wrong in (35.0, 135.0, 100.0):  # left mid / right mid / v6 cross-pair
            self.assertGreater(
                abs(anchor_r - wrong),
                5.0,
                msg=f"anchor byte {anchor_r} matched alternative midpoint {wrong}",
            )

    def test_anchor_call_independent_of_body_inset(self) -> None:
        # The v7.1 anchor is always ``rife_ifnet_at_timestep(..., s=1.0)``
        # on the prior pair regardless of body inset (the inset governs
        # body sample timesteps only). Verify by baking at several insets
        # and checking the anchor call signature is identical in each.
        #
        # We filter the IFNet call list by pair identity *and* timestep
        # because at ``inset = 0`` the end bookend collapses to
        # ``timestep = 1 - inset = 1.0`` and would otherwise collide with
        # the anchor call on the same timestep. The end bookend uses the
        # *last* pair ``(kf_{N-1}, kf_N) = (kf1, kf2)``, the anchor for
        # the only internal boundary here uses the *prior* pair
        # ``(kf0, kf1)`` — distinguishable by the first input frame.
        kf0 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf0[..., 0] = 0
        kf1 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf1[..., 0] = 100
        kf2 = np.zeros((4, 4, 3), dtype=np.uint8)
        kf2[..., 0] = 200
        for inset in (0.0, 0.12, 0.25, 0.40):
            _frames, _times, ifnet_mock = _build(
                [kf0, kf1, kf2],
                [0.0, 8.0, 16.0],
                exp=2,
                ifnet_timestep_inset=inset,
                return_ifnet_mock=True,
            )
            anchor_calls = [
                c
                for c in ifnet_mock.call_args_list
                if c.kwargs["timestep"] == 1.0
                and np.array_equal(c.args[1], kf0)
                and np.array_equal(c.args[2], kf1)
            ]
            self.assertEqual(
                len(anchor_calls),
                1,
                msg=(
                    f"missing v7.1 anchor IFNet(s=1.0, pair=(kf0, kf1)) "
                    f"call at inset={inset}"
                ),
            )

    def test_anchor_at_exact_keyframe_time(self) -> None:
        # Anchor wall-clock placement is exactly ``t = keyframe_times[seg_i + 1]``
        # (the upcoming internal boundary). Verify with non-uniform spacing
        # so the assertion isn't satisfied by a coincidence.
        kfs = [_solid_rgb(v) for v in (0, 80, 160, 240)]
        kts = [0.0, 5.5, 14.0, 24.0]
        _frames, times = _build(kfs, kts, exp=2, ifnet_timestep_inset=0.20)
        # Internal keyframes at 5.5 and 14.0 must both appear in the times
        # grid (one anchor each).
        self.assertIn(5.5, times)
        self.assertIn(14.0, times)
        # The body samples flanking each anchor sit at T_seg/(2n) per side:
        #   seg0 (T=5.5): last body at 5.5 - 5.5/8 = 4.8125
        #   seg1 (T=8.5): first body at 5.5 + 8.5/8 = 6.5625
        self.assertIn(4.8125, times)
        self.assertIn(6.5625, times)


if __name__ == "__main__":
    unittest.main()
