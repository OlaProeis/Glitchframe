"""Unit tests for :mod:`pipeline.beat_pulse`."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pipeline.beat_pulse import (
    DEFAULT_BASS_BANDS,
    PulseTrack,
    beat_pulse_envelope,
    build_bass_pulse_track,
    build_logo_bass_pulse_track,
    build_rms_impact_pulse_track,
    build_snare_glow_track,
    scale_and_opacity_for_pulse,
)


class TestBeatPulseEnvelope(unittest.TestCase):
    def test_returns_zero_when_no_beats(self) -> None:
        self.assertEqual(beat_pulse_envelope(1.0, []), 0.0)

    def test_returns_zero_before_first_beat(self) -> None:
        # t precedes the earliest beat -> envelope is 0 (no retro-pulse).
        self.assertEqual(beat_pulse_envelope(0.0, [0.5, 1.0, 1.5]), 0.0)

    def test_peaks_to_one_on_exact_beat(self) -> None:
        self.assertAlmostEqual(
            beat_pulse_envelope(1.0, [0.5, 1.0, 1.5]), 1.0, places=6
        )

    def test_decays_exponentially_between_beats(self) -> None:
        beats = [0.0, 10.0]  # long gap so decay has room to breathe
        bpm = 120.0  # period 0.5 s, tau = 0.125 s with default fraction
        v0 = beat_pulse_envelope(0.0, beats, bpm=bpm)
        v_tau = beat_pulse_envelope(0.125, beats, bpm=bpm)
        # At t = tau we expect ~1/e = 0.368
        self.assertAlmostEqual(v0, 1.0, places=6)
        self.assertAlmostEqual(v_tau, math.exp(-1.0), places=2)

    def test_fast_bpm_gives_shorter_tau(self) -> None:
        beats = [0.0, 5.0]
        slow = beat_pulse_envelope(0.1, beats, bpm=60.0)
        fast = beat_pulse_envelope(0.1, beats, bpm=180.0)
        # Faster BPM -> tighter tau -> envelope decays faster -> lower value.
        self.assertGreater(slow, fast)

    def test_missing_bpm_uses_fallback(self) -> None:
        # Should not raise and should give a reasonable value in (0, 1).
        v = beat_pulse_envelope(0.05, [0.0, 1.0], bpm=None)
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1.0)

    def test_ignores_nan_and_negative_beats(self) -> None:
        beats = [float("nan"), -1.0, 0.5]
        # Only the 0.5 beat survives cleaning; t=0.25 precedes it -> 0.
        self.assertEqual(beat_pulse_envelope(0.25, beats), 0.0)
        # t=0.5 hits the survivor -> peak 1.0.
        self.assertAlmostEqual(beat_pulse_envelope(0.5, beats), 1.0, places=6)


class TestScaleAndOpacityForPulse(unittest.TestCase):
    def test_zero_pulse_is_neutral(self) -> None:
        s, o = scale_and_opacity_for_pulse(0.0)
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(o, 1.0)

    def test_peak_pulse_lifts_both_axes(self) -> None:
        s, o = scale_and_opacity_for_pulse(1.0, strength=1.0)
        self.assertGreater(s, 1.0)
        self.assertGreater(o, 1.0)

    def test_strength_two_allows_larger_pulse_than_one(self) -> None:
        s1, o1 = scale_and_opacity_for_pulse(1.0, strength=1.0)
        s2, o2 = scale_and_opacity_for_pulse(1.0, strength=2.0)
        self.assertGreater(s2, s1)
        self.assertGreater(o2, o1)

    def test_zero_strength_disables_effect(self) -> None:
        s, o = scale_and_opacity_for_pulse(1.0, strength=0.0)
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(o, 1.0)

    def test_negative_pulse_clamped(self) -> None:
        s, o = scale_and_opacity_for_pulse(-0.5)
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(o, 1.0)

    def test_pulse_above_one_clamped(self) -> None:
        s1, o1 = scale_and_opacity_for_pulse(1.0)
        s2, o2 = scale_and_opacity_for_pulse(10.0)
        self.assertAlmostEqual(s1, s2)
        self.assertAlmostEqual(o1, o2)


class TestPulseTrack(unittest.TestCase):
    def test_value_at_indexes_by_floor(self) -> None:
        # fps=10 -> samples at t=0.0, 0.1, 0.2, ... floor(0.25 * 10) -> idx 2
        track = PulseTrack(
            values=np.array([0.1, 0.3, 0.7, 0.4, 0.0], dtype=np.float32),
            fps=10.0,
        )
        self.assertAlmostEqual(track.value_at(0.0), 0.1, places=6)
        self.assertAlmostEqual(track.value_at(0.19), 0.3, places=6)
        self.assertAlmostEqual(track.value_at(0.25), 0.7, places=6)

    def test_value_at_before_track_returns_zero(self) -> None:
        track = PulseTrack(
            values=np.array([1.0, 0.5], dtype=np.float32), fps=30.0
        )
        self.assertEqual(track.value_at(-0.01), 0.0)

    def test_value_at_clamps_to_last_sample(self) -> None:
        # Queries past the end fall back to the final stored sample instead of
        # reading out of bounds — that way a compositor over-running the last
        # analyzed frame by a sub-frame doesn't crash.
        track = PulseTrack(
            values=np.array([0.2, 0.9], dtype=np.float32), fps=30.0
        )
        self.assertAlmostEqual(track.value_at(10.0), 0.9, places=6)

    def test_empty_track_returns_zero(self) -> None:
        track = PulseTrack(values=np.zeros((0,), dtype=np.float32), fps=30.0)
        self.assertEqual(track.value_at(0.0), 0.0)


def _synthetic_kick_analysis(
    *,
    fps: float = 30.0,
    duration_sec: float = 4.0,
    num_bands: int = 8,
    kick_times: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5),
    kick_value: float = 0.9,
) -> dict:
    """Build an ``analysis.json``-shaped mapping with bass kicks on cue.

    Bands 0–1 (bass) are zero everywhere except single-frame spikes at
    ``kick_times`` so ``build_bass_pulse_track`` has obvious attacks to
    detect. Upper bands carry sustained energy that must **not** influence
    the bass envelope — that's the whole point of the band slice.
    """
    frames = int(round(fps * duration_sec))
    spectrum = np.zeros((frames, num_bands), dtype=np.float32)
    spectrum[:, 2:] = 0.6  # Sustained mid/high energy — should be ignored.
    for t in kick_times:
        idx = int(round(t * fps))
        if 0 <= idx < frames:
            spectrum[idx, 0] = kick_value
            spectrum[idx, 1] = kick_value * 0.7
    return {
        "spectrum": {
            "num_bands": num_bands,
            "fps": fps,
            "frames": frames,
            "values": spectrum.tolist(),
        }
    }


class TestBuildBassPulseTrack(unittest.TestCase):
    def test_returns_none_without_spectrum(self) -> None:
        self.assertIsNone(build_bass_pulse_track({}))
        self.assertIsNone(build_bass_pulse_track({"spectrum": None}))
        self.assertIsNone(build_bass_pulse_track({"spectrum": {}}))

    def test_returns_none_for_malformed_values(self) -> None:
        self.assertIsNone(
            build_bass_pulse_track(
                {"spectrum": {"values": [], "fps": 30.0}}
            )
        )
        self.assertIsNone(
            build_bass_pulse_track(
                {"spectrum": {"values": [[1.0, 2.0, 3.0]], "fps": 0.0}}
            )
        )

    def test_silent_track_produces_zero_envelope(self) -> None:
        frames = 90
        spec = {
            "spectrum": {
                "num_bands": 8,
                "fps": 30.0,
                "frames": frames,
                "values": np.zeros((frames, 8), dtype=np.float32).tolist(),
            }
        }
        track = build_bass_pulse_track(spec)
        assert track is not None
        self.assertTrue(np.allclose(track.values, 0.0))

    def test_kick_attacks_peak_near_one(self) -> None:
        analysis = _synthetic_kick_analysis()
        track = build_bass_pulse_track(analysis)
        assert track is not None
        # Each kick frame should land at (or very near) the top of the [0, 1]
        # envelope after percentile normalisation — that's the whole visual
        # promise of "bass mode pulses on kicks".
        kick_values = [track.value_at(t) for t in (0.5, 1.5, 2.5, 3.5)]
        for v in kick_values:
            self.assertGreater(v, 0.8)
            self.assertLessEqual(v, 1.0 + 1e-6)

    def test_envelope_decays_between_kicks(self) -> None:
        analysis = _synthetic_kick_analysis()
        track = build_bass_pulse_track(analysis)
        assert track is not None
        # Midway between two kicks (t=1.0, kicks at 0.5 and 1.5) the envelope
        # should have decayed well below the peak. tau≈180ms with a 1s gap
        # puts us far down the exponential tail.
        self.assertLess(track.value_at(1.0), 0.2)

    def test_high_band_only_energy_is_ignored(self) -> None:
        # Sustained energy in upper bands (hats, cymbals, vocals) must not
        # drive the bass envelope — that's exactly the jitter we're
        # eliminating.
        frames = 60
        spectrum = np.zeros((frames, 8), dtype=np.float32)
        spectrum[:, 4:] = 0.8
        analysis = {
            "spectrum": {
                "num_bands": 8,
                "fps": 30.0,
                "frames": frames,
                "values": spectrum.tolist(),
            }
        }
        track = build_bass_pulse_track(analysis)
        assert track is not None
        self.assertTrue(np.allclose(track.values, 0.0))

    def test_sensitivity_scales_output(self) -> None:
        analysis = _synthetic_kick_analysis()
        low = build_bass_pulse_track(analysis, sensitivity=0.5)
        high = build_bass_pulse_track(analysis, sensitivity=2.0)
        assert low is not None and high is not None
        # Same shape, different amplitude. On the sample right after the first
        # kick, the higher sensitivity should produce a larger envelope value
        # (both are clipped to [0, 1]).
        idx = int(round(0.5 * 30.0))
        self.assertGreater(high.values[idx], low.values[idx])

    def test_respects_custom_num_bands(self) -> None:
        # Sanity: if the caller asks for more bands than the spectrum has,
        # we clamp gracefully instead of blowing up on a slice.
        analysis = _synthetic_kick_analysis(num_bands=2)
        track = build_bass_pulse_track(
            analysis, num_bass_bands=DEFAULT_BASS_BANDS + 4
        )
        assert track is not None
        self.assertEqual(track.values.shape[0], int(30.0 * 4.0))


class TestLogoBassPulseTrack(unittest.TestCase):
    def test_narrow_raw_band_gets_full_dynamic_range(self) -> None:
        """Sustain-heavy curves must not collapse visual pulse to a tiny scale delta.

        When ``raw = max(attack, sustain)`` sits in a tight band just below 1.0,
        the compositor's absolute pulse→scale mapping would barely move the logo.
        Post-stretch, kicks should sweep a large fraction of [0, 1].
        """
        fps = 30.0
        frames = 300
        spec = np.zeros((frames, 8), dtype=np.float32)
        # Loud sustained sub (sustain branch pegs high) with sparse kicks.
        spec[:, 0] = 0.88
        spec[:, 1] = 0.75
        for t in (0.4, 1.1, 2.0, 2.8):
            idx = int(round(t * fps))
            if 0 <= idx < frames:
                spec[idx, 0] = 1.0
                spec[idx, 1] = 0.92
        analysis = {
            "spectrum": {
                "num_bands": 8,
                "fps": fps,
                "frames": frames,
                "values": spec.tolist(),
            }
        }
        logo = build_logo_bass_pulse_track(analysis)
        assert logo is not None
        vals = logo.values.astype(np.float64)
        self.assertGreater(float(np.max(vals) - np.min(vals)), 0.45)

    def test_sustained_sub_stays_elevated_vs_attack_only(self) -> None:
        """Flat high bass after a ramp should keep logo envelope up (808 tail)."""
        fps = 30.0
        frames = 120
        spec = np.zeros((frames, 8), dtype=np.float32)
        spec[:20, 0] = np.linspace(0.0, 0.95, 20)
        spec[20:, 0] = 0.92
        spec[:, 1] = spec[:, 0] * 0.85
        analysis = {
            "spectrum": {
                "num_bands": 8,
                "fps": fps,
                "frames": frames,
                "values": spec.tolist(),
            }
        }
        logo = build_logo_bass_pulse_track(analysis)
        plain = build_bass_pulse_track(analysis)
        assert logo is not None and plain is not None
        mid = logo.value_at(2.0)
        # Attack-only curve collapses toward 0 on flat sustained bass.
        self.assertLess(plain.value_at(2.0), 0.15)
        self.assertGreater(mid, 0.35)


class TestRmsImpactPulseTrack(unittest.TestCase):
    def test_returns_none_without_rms(self) -> None:
        self.assertIsNone(build_rms_impact_pulse_track({}))
        self.assertIsNone(build_rms_impact_pulse_track({"rms": {}}))

    def test_drop_creates_pulse(self) -> None:
        fps = 30.0
        n = 300
        rms = np.zeros(n, dtype=np.float32)
        rms[150:] = 0.85
        analysis = {
            "song_hash": "test",
            "fps": fps,
            "rms": {"fps": fps, "frames": n, "values": rms.tolist()},
        }
        track = build_rms_impact_pulse_track(analysis)
        assert track is not None
        # Shortly after the step, impact envelope should rise
        self.assertGreater(track.value_at(5.05), 0.25)
        # Long before the jump, near silence
        self.assertLess(track.value_at(2.0), 0.15)


class TestSnareGlowTrack(unittest.TestCase):
    def test_mid_band_spike_triggers(self) -> None:
        fps = 30.0
        frames = 60
        spec = np.zeros((frames, 8), dtype=np.float32)
        for idx in (15, 16, 17):
            spec[idx, 4] = 0.95
            spec[idx, 5] = 0.85
        analysis = {
            "spectrum": {
                "num_bands": 8,
                "fps": fps,
                "frames": frames,
                "values": spec.tolist(),
            }
        }
        track = build_snare_glow_track(analysis)
        assert track is not None
        self.assertGreater(track.value_at(0.5), 0.25)


if __name__ == "__main__":
    unittest.main()
