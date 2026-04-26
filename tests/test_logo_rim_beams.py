"""Unit tests for :mod:`pipeline.logo_rim_beams`.

Covers:

* :func:`schedule_rim_beams` — deterministic scheduling, 10 s group gating,
  pre-drop snare lead-in detection, and fallback behaviour without a snare
  track.
* :func:`compute_beam_patch` — alpha / premultiplied invariants, beam
  lifecycle (no draw before ``t_start`` or after ``t_start + duration_s``),
  and empty-short-circuit on quiet frames.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pipeline.beat_pulse import PulseTrack
from pipeline.logo_rim_beams import (
    BeamConfig,
    ScheduledBeam,
    compute_beam_patch,
    schedule_rim_beams,
)


def _make_snare_track_with_peaks(
    peak_times: list[float],
    *,
    fps: int = 50,
    duration_sec: float = 10.0,
    peak_strength: float = 0.95,
) -> PulseTrack:
    """Pre-sampled pulse track with narrow local maxima at ``peak_times``."""
    n = int(fps * duration_sec)
    values = np.full(n, 0.05, dtype=np.float32)
    for t_peak in peak_times:
        idx = int(round(t_peak * fps))
        if 0 <= idx < n:
            values[idx] = float(peak_strength)
            if idx - 1 >= 0:
                values[idx - 1] = 0.20
            if idx + 1 < n:
                values[idx + 1] = 0.20
    return PulseTrack(values=values, fps=float(fps))


def _make_impact_track(peak_times: list[float], **kw) -> PulseTrack:
    return _make_snare_track_with_peaks(peak_times, **kw)


class TestScheduleRimBeams(unittest.TestCase):
    def test_drop_with_three_snares_yields_four_beam_group(self) -> None:
        analysis = {"events": {"drops": [{"t": 4.0}]}, "song_hash": "abc"}
        snares = _make_snare_track_with_peaks([2.5, 3.2, 3.7])
        beams = schedule_rim_beams(
            analysis,
            snare_track=snares,
            impact_track=None,
            cfg=BeamConfig(),
            song_hash="abc",
        )
        self.assertEqual(len(beams), 4)
        # Three lead-ins precede the drop; the drop is the final beam.
        self.assertFalse(beams[0].is_drop)
        self.assertTrue(beams[-1].is_drop)
        # Drop beam should be thicker + longer than the lead-ins.
        self.assertGreater(beams[-1].length_px, beams[0].length_px)
        self.assertGreater(beams[-1].thickness_px, beams[0].thickness_px)

    def test_drop_without_snare_track_still_fires_drop_beam(self) -> None:
        analysis = {"events": {"drops": [{"t": 2.0}]}, "song_hash": "h"}
        beams = schedule_rim_beams(
            analysis,
            snare_track=None,
            impact_track=None,
            cfg=BeamConfig(),
            song_hash="h",
        )
        self.assertEqual(len(beams), 1)
        self.assertTrue(beams[0].is_drop)
        self.assertAlmostEqual(beams[0].t_start, 2.0, places=3)

    def test_group_interval_gate_drops_close_second_group(self) -> None:
        # Two drops 5 s apart → second group is filtered out (<10 s gating).
        analysis = {
            "events": {"drops": [{"t": 3.0}, {"t": 8.0}]},
            "song_hash": "gate",
        }
        beams = schedule_rim_beams(
            analysis,
            snare_track=None,
            impact_track=None,
            cfg=BeamConfig(min_group_interval_sec=10.0),
            song_hash="gate",
        )
        # Only one drop beam survives.
        self.assertEqual(len(beams), 1)
        self.assertAlmostEqual(beams[0].t_start, 3.0, places=3)

    def test_group_interval_gate_preserves_well_spaced_drops(self) -> None:
        # 12 s apart → both groups pass.
        analysis = {
            "events": {"drops": [{"t": 3.0}, {"t": 15.0}]},
            "song_hash": "g2",
        }
        beams = schedule_rim_beams(
            analysis,
            snare_track=None,
            impact_track=None,
            cfg=BeamConfig(min_group_interval_sec=10.0),
            song_hash="g2",
        )
        self.assertEqual(len(beams), 2)

    def test_schedule_is_deterministic_per_song_hash(self) -> None:
        analysis = {"events": {"drops": [{"t": 4.0}]}, "song_hash": "det"}
        snares = _make_snare_track_with_peaks([2.5, 3.2, 3.7])
        beams_a = schedule_rim_beams(
            analysis, snare_track=snares, impact_track=None, song_hash="det"
        )
        beams_b = schedule_rim_beams(
            analysis, snare_track=snares, impact_track=None, song_hash="det"
        )
        self.assertEqual(len(beams_a), len(beams_b))
        for a, b in zip(beams_a, beams_b):
            self.assertAlmostEqual(a.angle_rad, b.angle_rad, places=6)
            self.assertAlmostEqual(a.t_start, b.t_start, places=6)
            self.assertEqual(a.color_layer_idx, b.color_layer_idx)

    def test_disabled_config_returns_empty(self) -> None:
        analysis = {"events": {"drops": [{"t": 4.0}]}}
        beams = schedule_rim_beams(
            analysis,
            snare_track=None,
            impact_track=None,
            cfg=BeamConfig(enabled=False),
        )
        self.assertEqual(beams, [])

    def test_standalone_impact_excluded_when_near_drop(self) -> None:
        analysis = {"events": {"drops": [{"t": 3.0}]}, "song_hash": "excl"}
        # Impact at 3.2 s is inside the drop exclusion window → no extra beam.
        impacts = _make_impact_track([3.2])
        beams = schedule_rim_beams(
            analysis,
            snare_track=None,
            impact_track=impacts,
            cfg=BeamConfig(),
            song_hash="excl",
        )
        # Just the single drop beam; impact consumed by the drop group.
        self.assertEqual(len(beams), 1)
        self.assertTrue(beams[0].is_drop)

    def test_schedule_sorted_by_t_start(self) -> None:
        analysis = {
            "events": {"drops": [{"t": 15.0}, {"t": 3.0}, {"t": 30.0}]},
            "song_hash": "sort",
        }
        beams = schedule_rim_beams(
            analysis,
            snare_track=None,
            impact_track=None,
            cfg=BeamConfig(min_group_interval_sec=10.0),
            song_hash="sort",
        )
        times = [b.t_start for b in beams]
        self.assertEqual(times, sorted(times))


class TestComputeBeamPatch(unittest.TestCase):
    def _single_beam(
        self,
        *,
        t_start: float = 0.0,
        duration: float = 0.3,
        angle: float = 0.0,
        length: float = 120.0,
        thickness: float = 6.0,
        is_drop: bool = False,
    ) -> ScheduledBeam:
        return ScheduledBeam(
            t_start=t_start,
            duration_s=duration,
            angle_rad=angle,
            length_px=length,
            thickness_px=thickness,
            intensity=1.0,
            color_layer_idx=0,
            is_drop=is_drop,
        )

    def test_no_beams_returns_none(self) -> None:
        out = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.0,
            scheduled=[],
            rim_rgb=(255, 80, 180),
        )
        self.assertIsNone(out)

    def test_before_t_start_returns_none(self) -> None:
        beam = self._single_beam(t_start=1.0, duration=0.3)
        out = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.5,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        self.assertIsNone(out)

    def test_after_duration_returns_none(self) -> None:
        beam = self._single_beam(t_start=1.0, duration=0.3)
        out = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=2.0,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        self.assertIsNone(out)

    def test_active_beam_produces_nonzero_alpha(self) -> None:
        beam = self._single_beam(t_start=0.0, duration=0.3, angle=0.0, length=150.0)
        # Time just after the linear attack peak.
        out = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.05,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        self.assertIsNotNone(out)
        assert out is not None  # narrow for type-checkers
        self.assertTrue(np.any(out.patch[..., 3] > 0))

    def test_premultiplied_invariant(self) -> None:
        beam = self._single_beam(t_start=0.0, duration=0.3, angle=0.3, length=200.0)
        out = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.06,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        assert out is not None
        a = out.patch[..., 3].astype(np.int32)
        # Premultiplied RGBA: every channel must be <= alpha at each pixel.
        for c in range(3):
            self.assertTrue(
                np.all(out.patch[..., c].astype(np.int32) <= a + 1),
                f"channel {c} exceeds alpha",
            )

    def test_patch_bounds_inside_frame(self) -> None:
        beam = self._single_beam(
            t_start=0.0, duration=0.3, angle=math.pi / 4.0, length=200.0
        )
        out = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.06,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        assert out is not None
        ph, pw = out.patch.shape[:2]
        self.assertGreaterEqual(out.x0, 0)
        self.assertGreaterEqual(out.y0, 0)
        self.assertLessEqual(out.x0 + pw, 960)
        self.assertLessEqual(out.y0 + ph, 540)

    def test_alpha_peaks_during_attack_region(self) -> None:
        # Attack ≈ 0.04 s; sampling at 0.04 s should have stronger alpha than
        # at 0.25 s (decay well underway).
        beam = self._single_beam(t_start=0.0, duration=0.32, length=180.0)
        out_attack = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.04,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        out_decay = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.25,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        assert out_attack is not None and out_decay is not None
        self.assertGreater(
            float(out_attack.patch[..., 3].max()),
            float(out_decay.patch[..., 3].max()),
        )

    def test_core_white_boost_brightens_dark_tint(self) -> None:
        # Regression: without ``core_white_boost`` a dark preset (e.g. deep
        # blue shadow colour) produced an almost-invisible beam against a
        # black background. The boost should inject pure white into the hot
        # axis so even dark tints saturate to a bright core while the halo
        # keeps the original tint.
        beam = self._single_beam(
            t_start=0.0, duration=0.55, angle=0.0, length=800.0, thickness=22.0, is_drop=True
        )
        dim_cfg = BeamConfig(core_white_boost=0.0)
        bright_cfg = BeamConfig()  # defaults include the boost
        dim = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.04,
            scheduled=[beam],
            rim_rgb=(0, 40, 120),  # dark blue
            cfg=dim_cfg,
            logo_radius_px=150.0,
        )
        bright = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.04,
            scheduled=[beam],
            rim_rgb=(0, 40, 120),
            cfg=bright_cfg,
            logo_radius_px=150.0,
        )
        assert dim is not None and bright is not None
        # Red channel is 0 on the tint side, so any non-zero red in the
        # output is proof the white-hot boost lit up the core.
        self.assertLess(int(dim.patch[..., 0].max()), 10)
        self.assertGreater(int(bright.patch[..., 0].max()), 80)
        # Alpha invariant still holds (rgb <= alpha per pixel).
        for c in range(3):
            self.assertTrue(
                np.all(bright.patch[..., c] <= bright.patch[..., 3]),
                msg=f"channel {c} exceeded alpha with core_white_boost",
            )


class TestSustainTimelineShaping(unittest.TestCase):
    def test_strength_increases_during_ramp(self) -> None:
        from pipeline import logo_rim_beams as lrb

        D = 1.4
        self.assertLess(lrb._sustain_strength(0.01, D), lrb._sustain_strength(0.2, D))

    def test_glow_starts_only_after_strength_plateau(self) -> None:
        from pipeline import logo_rim_beams as lrb

        D = 1.2
        _r, _p, _f, t2 = lrb._sustain_knots(D)
        self.assertEqual(lrb._sustain_glow_u(t2 * 0.5, D), 0.0)
        self.assertGreater(lrb._sustain_glow_u(min(t2 + 0.15, D * 0.5), D), 0.0)

    def test_longer_clip_allows_stronger_halo_swell(self) -> None:
        from pipeline import logo_rim_beams as lrb

        # Both samples at full glow, late in the clip, before the end fade.
        w_short, hi_short, br_short = lrb._sustain_halo_scales(0.96, 1.0)
        w_long, hi_long, br_long = lrb._sustain_halo_scales(3.9, 4.0)
        self.assertGreater(w_long, w_short)
        self.assertGreater(hi_long, hi_short)
        self.assertGreater(br_long, br_short)

    def test_blur_footprint_pad_increases_with_sustain_blur(self) -> None:
        from pipeline import logo_rim_beams as lrb

        self.assertLess(
            lrb._gaussian_blur_footprint_pad_px(2.6, max_br=1.0),
            lrb._gaussian_blur_footprint_pad_px(2.6, max_br=2.5),
        )

    def test_sustain_energy_mul_boosts_core_after_plateau(self) -> None:
        from pipeline import logo_rim_beams as lrb

        D = 2.0
        _r, _p, _f, t2 = lrb._sustain_knots(D)
        # Still on strength plateau, before g > 0.
        self.assertAlmostEqual(lrb._sustain_energy_mul(t2 * 0.5, D), 1.0)
        # Mid build phase: global gain > 1 so core + edge brighten, not just halo.
        t_mid = min(t2 + 0.75, D * 0.9)
        self.assertGreater(lrb._sustain_energy_mul(t_mid, D), 1.09)

    def test_sustain_flag_produces_late_bloom_in_patch(self) -> None:
        """Regression: long user BEAM should glow up toward the end of the cue."""
        beam = ScheduledBeam(
            t_start=0.0,
            duration_s=2.0,
            angle_rad=0.0,
            length_px=500.0,
            thickness_px=20.0,
            intensity=1.0,
            color_layer_idx=0,
            is_drop=True,
            sustain_shaping=True,
        )
        out_early = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=0.1,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        out_late = compute_beam_patch(
            (540, 960),
            centroid_xy=(480.0, 270.0),
            t=1.7,
            scheduled=[beam],
            rim_rgb=(255, 80, 180),
        )
        assert out_early is not None and out_late is not None
        # Late time should be in the high-glow phase (bloom → larger foot-print).
        self.assertGreater(
            float(out_late.patch[..., 3].sum()),
            float(out_early.patch[..., 3].sum()),
        )


if __name__ == "__main__":
    unittest.main()
