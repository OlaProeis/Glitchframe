"""Hermetic tests for the pure-Python samplers in :mod:`pipeline.reactive_shader`.

These exercise the uniform-mapping helpers (onset-strength envelope,
bar-phase, and the extended :func:`uniforms_at_time`) without touching GL —
anything that requires a live moderngl context is covered by app-level
smoke runs, not here.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.reactive_shader import (
    ONSET_ENV_CACHE_KEY,
    _bar_phase_at,
    _interp_onset_strength,
    _normalise_onset_strength,
    uniforms_at_time,
)


class TestNormaliseOnsetStrength(unittest.TestCase):
    def test_flat_series_peaks_near_one(self) -> None:
        # A flat non-zero series has percentile == max, so everything clips
        # to exactly 1.0 after normalisation — the "shouldn't crush on flat
        # input" guarantee called out in the task spec.
        env = _normalise_onset_strength([0.7] * 32)
        self.assertEqual(env.shape, (32,))
        self.assertTrue(np.allclose(env, 1.0))

    def test_single_outlier_does_not_crush_body(self) -> None:
        body = [0.5] * 100
        series = body + [50.0]  # 1% outlier
        env = _normalise_onset_strength(series)
        # 95th percentile sits in the body, so the outlier clamps to 1 and
        # the body lands near 1.0 rather than 0.01 as a naive max-normalise
        # would produce.
        self.assertGreater(float(env[0]), 0.9)
        self.assertEqual(float(env[-1]), 1.0)

    def test_empty_input_returns_empty_array(self) -> None:
        env = _normalise_onset_strength([])
        self.assertEqual(env.shape, (0,))

    def test_zero_series_returns_zeros(self) -> None:
        env = _normalise_onset_strength([0.0] * 8)
        self.assertEqual(env.shape, (8,))
        self.assertTrue(np.all(env == 0.0))


class TestInterpOnsetStrength(unittest.TestCase):
    def _analysis(self, strength: list[float], rate: float) -> dict:
        return {
            "onsets": {
                "strength": strength,
                "frame_rate_hz": rate,
                "hop_length": 512,
                "frames": len(strength),
                "peaks": [],
            }
        }

    def test_missing_block_returns_zero(self) -> None:
        self.assertEqual(_interp_onset_strength({}, 0.5), 0.0)
        self.assertEqual(_interp_onset_strength({"onsets": None}, 0.5), 0.0)

    def test_negative_time_returns_zero(self) -> None:
        analysis = self._analysis([0.1, 0.5, 0.9], 10.0)
        self.assertEqual(_interp_onset_strength(analysis, -0.1), 0.0)

    def test_empty_strength_returns_zero(self) -> None:
        analysis = self._analysis([], 10.0)
        self.assertEqual(_interp_onset_strength(analysis, 0.5), 0.0)

    def test_non_positive_frame_rate_returns_zero(self) -> None:
        analysis = self._analysis([0.1, 0.5], 0.0)
        self.assertEqual(_interp_onset_strength(analysis, 0.0), 0.0)

    def test_linear_interpolation_between_frames(self) -> None:
        # Flat [1.0, 1.0, 1.0] normalises to [1, 1, 1]; a series with one
        # zero and peaks lets us verify the interpolator actually blends.
        analysis = self._analysis([0.0, 1.0, 0.0, 1.0], 10.0)
        at_frame_1 = _interp_onset_strength(analysis, 0.1)
        at_midpoint = _interp_onset_strength(analysis, 0.15)
        self.assertAlmostEqual(at_frame_1, 1.0, places=5)
        self.assertAlmostEqual(at_midpoint, 0.5, places=5)

    def test_accepts_legacy_fps_key(self) -> None:
        # Older callers / fixtures may emit ``fps`` instead of
        # ``frame_rate_hz``; the sampler should tolerate the alias.
        analysis = {
            "onsets": {
                "strength": [0.0, 1.0],
                "fps": 10.0,
            }
        }
        self.assertAlmostEqual(
            _interp_onset_strength(analysis, 0.1), 1.0, places=5
        )

    def test_uses_onset_frame_rate_not_analysis_fps(self) -> None:
        # ``analysis.fps`` is the mel spectrum frame rate (30 fps in
        # production); onset analysis typically runs at ~86 Hz. The
        # sampler must not fall through to ``analysis.fps`` when the
        # onset block carries its own rate.
        analysis = self._analysis([0.0, 1.0, 0.0, 1.0], 10.0)
        analysis["fps"] = 30  # bogus mel fps
        # At t=0.1 the onset-rate interpretation lands on frame 1 → 1.0.
        # A 30 fps misinterpretation would land at frame 3 → also 1.0,
        # so pick a time that distinguishes them instead.
        val = _interp_onset_strength(analysis, 0.2)
        self.assertAlmostEqual(val, 0.0, places=5)

    def test_normalisation_caches_on_analysis_dict(self) -> None:
        analysis = self._analysis([0.0, 1.0, 0.0, 1.0], 10.0)
        _interp_onset_strength(analysis, 0.1)
        self.assertIn(ONSET_ENV_CACHE_KEY, analysis)
        cache = analysis[ONSET_ENV_CACHE_KEY]
        self.assertIsInstance(cache, dict)
        strength_id = id(analysis["onsets"]["strength"])
        self.assertIn(strength_id, cache)
        self.assertIsInstance(cache[strength_id], np.ndarray)

    def test_cache_reuse_does_not_rebuild_array(self) -> None:
        analysis = self._analysis([0.0, 1.0, 0.0, 1.0], 10.0)
        _interp_onset_strength(analysis, 0.1)
        cached = analysis[ONSET_ENV_CACHE_KEY][id(analysis["onsets"]["strength"])]
        _interp_onset_strength(analysis, 0.2)
        # Same array object: no rebuild on the second call.
        self.assertIs(
            analysis[ONSET_ENV_CACHE_KEY][id(analysis["onsets"]["strength"])],
            cached,
        )


class TestBarPhaseAt(unittest.TestCase):
    def test_midpoint_between_downbeats(self) -> None:
        # Task 35 verification case: downbeats at [0, 2, 4, 6], t=1.0 → 0.5.
        self.assertAlmostEqual(_bar_phase_at(1.0, [0.0, 2.0, 4.0, 6.0]), 0.5)

    def test_three_quarters_phase(self) -> None:
        # t=5.5 sits 1.5 s into the [4, 6) bar → 0.75.
        self.assertAlmostEqual(_bar_phase_at(5.5, [0.0, 2.0, 4.0, 6.0]), 0.75)

    def test_downbeat_exact_boundary_is_zero(self) -> None:
        # A beat landing exactly on the next downbeat reports phase 0 of
        # the new bar, not 1.0 of the previous (bisect_right semantics).
        self.assertAlmostEqual(_bar_phase_at(2.0, [0.0, 2.0, 4.0]), 0.0)

    def test_before_first_downbeat_extrapolates(self) -> None:
        # Before the grid, phase winds backwards using the median span.
        # Downbeats at [0, 2, 4] → period 2.0 → t=-0.5 is 0.25 s before the
        # previous synthetic downbeat → phase 0.75.
        self.assertAlmostEqual(
            _bar_phase_at(-0.5, [0.0, 2.0, 4.0]), 0.75, places=5
        )

    def test_after_last_downbeat_extrapolates(self) -> None:
        # Past the grid, phase increments forward using the median span.
        self.assertAlmostEqual(
            _bar_phase_at(5.0, [0.0, 2.0, 4.0]), 0.5, places=5
        )

    def test_empty_downbeats_uses_bpm_fallback(self) -> None:
        # 120 bpm, 4 beats / bar → bar period = 2.0 s → t=0.5 is 0.25 in.
        phase = _bar_phase_at(0.5, [], bpm=120.0)
        self.assertAlmostEqual(phase, 0.25, places=5)
        phase = _bar_phase_at(2.0, [], bpm=120.0)
        self.assertAlmostEqual(phase, 0.0, places=5)

    def test_beats_only_groups_by_beats_per_bar(self) -> None:
        # Beats at [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] → synthetic
        # downbeats every 4th beat → [0.0, 2.0, ...] → t=1.0 ⇒ 0.5.
        beats = [i * 0.5 for i in range(8)]
        self.assertAlmostEqual(
            _bar_phase_at(1.0, [], beats=beats, beats_per_bar=4),
            0.5,
            places=5,
        )

    def test_empty_inputs_with_no_bpm_return_zero(self) -> None:
        self.assertEqual(_bar_phase_at(1.0, []), 0.0)

    def test_invalid_beats_per_bar_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _bar_phase_at(1.0, [0.0, 2.0], beats_per_bar=0)


class TestUniformsAtTimeExtensions(unittest.TestCase):
    def _synthetic_analysis(self, fps: int = 30) -> dict:
        n = fps * 4  # 4 seconds
        # Mirror ``compute_build_tension_series``: ramp up to the drop at
        # t=2.0, then snap to zero. Frames [0, 60] hold the 0→1 build;
        # frames (60, n] hold zero (post-drop release).
        drop_frame = fps * 2
        ramp = np.linspace(0.0, 1.0, drop_frame + 1)
        values = np.concatenate([ramp, np.zeros(n - drop_frame - 1)]).tolist()
        return {
            "fps": fps,
            "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "downbeats": [0.0, 2.0],
            "tempo": {"bpm": 120.0},
            "spectrum": {"fps": fps, "values": [[0.0] * 8] * n},
            "rms": {"fps": fps, "values": [0.3] * n},
            "onsets": {
                "frame_rate_hz": 10.0,
                "strength": [0.0, 0.5, 1.0, 0.5, 0.0] * 8,
                "peaks": [0.2, 1.2],
            },
            "events": {
                "drops": [{"t": 2.0, "confidence": 0.9}],
                "build_tension": {
                    "fps": fps,
                    "frames": n,
                    "values": values,
                },
            },
        }

    def test_returns_onset_env_build_tension_bar_phase_keys(self) -> None:
        analysis = self._synthetic_analysis()
        u = uniforms_at_time(analysis, 1.0)
        for key in (
            "time",
            "beat_phase",
            "bar_phase",
            "rms",
            "onset_pulse",
            "onset_env",
            "build_tension",
            "intensity",
            "band_energies",
        ):
            self.assertIn(key, u)

    def test_bar_phase_matches_downbeat_midpoint(self) -> None:
        analysis = self._synthetic_analysis()
        u = uniforms_at_time(analysis, 1.0)
        self.assertAlmostEqual(float(u["bar_phase"]), 0.5, places=5)

    def test_build_tension_peaks_near_drop(self) -> None:
        # Ramp reaches max one frame before the drop in build-tension
        # semantics; sample just before t=2.0 to verify a non-zero read.
        analysis = self._synthetic_analysis()
        u = uniforms_at_time(analysis, 1.95)
        self.assertGreater(float(u["build_tension"]), 0.9)

    def test_onset_env_non_zero_on_synthetic_peak(self) -> None:
        analysis = self._synthetic_analysis()
        # Frame 2 of the strength array (value 1.0) sits at t = 2 / 10 s.
        u = uniforms_at_time(analysis, 0.2)
        self.assertAlmostEqual(float(u["onset_env"]), 1.0, places=5)

    def test_absent_events_block_returns_zero_without_raising(self) -> None:
        analysis = self._synthetic_analysis()
        del analysis["events"]
        u = uniforms_at_time(analysis, 1.0)
        self.assertEqual(float(u["build_tension"]), 0.0)
        # Other keys still populate from the rest of the bundle.
        self.assertAlmostEqual(float(u["bar_phase"]), 0.5, places=5)

    def test_absent_onsets_block_leaves_onset_env_zero(self) -> None:
        analysis = self._synthetic_analysis()
        del analysis["onsets"]
        u = uniforms_at_time(analysis, 0.2)
        self.assertEqual(float(u["onset_env"]), 0.0)
        self.assertEqual(float(u["onset_pulse"]), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
