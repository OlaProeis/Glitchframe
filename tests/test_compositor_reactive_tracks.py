"""Hermetic tests for the reactive-shader track helpers in ``pipeline.compositor``.

Covers the pure-Python closures that the compositor builds **once per render**
and hands to the per-frame loop:

* :func:`_shader_transient_tracks_for_analysis` — lo / mid / hi band envelopes.
* :func:`_drop_hold_fn_for_analysis` — post-drop exponential afterglow.

The frame-loop path itself requires a live moderngl context and is covered by
app-level smoke runs; the hot-path contract (scalar lookups per frame, zero
dict rebuilds) lives here instead.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.compositor import (
    CompositorConfig,
    _build_beam_render_context,
    _drop_hold_fn_for_analysis,
    _kick_glow_envelope_fn,
    _shader_transient_tracks_for_analysis,
)
from pipeline.musical_events import DEFAULT_DROP_HOLD_DECAY_SEC


def _spectrum_analysis(fps: int = 30, seconds: float = 4.0) -> dict:
    """Minimal analysis with a spiky 8-band spectrum so band transient
    builders have real attacks to rectify."""
    n = int(fps * seconds)
    # Silence baseline; inject a single "kick frame" in the low band and a
    # "hat frame" in the high band so the builders return non-``None`` tracks.
    values = np.zeros((n, 8), dtype=np.float32)
    values[n // 4, 0] = 1.0  # kick → transient_lo
    values[n // 4, 1] = 0.8
    values[n // 2, 4] = 1.0  # snare/body → transient_mid
    values[3 * n // 4, 7] = 1.0  # hat → transient_hi
    return {
        "fps": fps,
        "duration_sec": float(seconds),
        "spectrum": {
            "fps": fps,
            "frames": n,
            "num_bands": 8,
            "values": values.tolist(),
        },
    }


class TestDropHoldFnForAnalysis(unittest.TestCase):
    def test_no_events_block_returns_none(self) -> None:
        self.assertIsNone(
            _drop_hold_fn_for_analysis({}, DEFAULT_DROP_HOLD_DECAY_SEC)
        )

    def test_empty_drops_list_returns_none(self) -> None:
        analysis = {"events": {"drops": []}}
        self.assertIsNone(
            _drop_hold_fn_for_analysis(analysis, DEFAULT_DROP_HOLD_DECAY_SEC)
        )

    def test_non_mapping_drops_filtered_out(self) -> None:
        # Defensive: the JSON loader always yields dicts, but a hand-rolled
        # test fixture could slip a stray float in — the closure must treat
        # that as "no drops" rather than crashing inside ``sample_drop_hold``.
        analysis = {"events": {"drops": [1.0, None, "oops"]}}
        self.assertIsNone(
            _drop_hold_fn_for_analysis(analysis, DEFAULT_DROP_HOLD_DECAY_SEC)
        )

    def test_drop_hold_peaks_at_drop_time(self) -> None:
        analysis = {
            "events": {"drops": [{"t": 2.0, "confidence": 1.0}]}
        }
        fn = _drop_hold_fn_for_analysis(
            analysis, DEFAULT_DROP_HOLD_DECAY_SEC
        )
        self.assertIsNotNone(fn)
        assert fn is not None  # narrow for mypy
        # Before the drop: zero.
        self.assertEqual(fn(0.0), 0.0)
        self.assertEqual(fn(1.99), 0.0)
        # Exactly at the drop: full confidence.
        self.assertAlmostEqual(fn(2.0), 1.0, places=5)

    def test_drop_hold_decays_within_seconds(self) -> None:
        analysis = {
            "events": {"drops": [{"t": 1.0, "confidence": 1.0}]}
        }
        tau = 2.0  # DEFAULT_DROP_HOLD_DECAY_SEC
        fn = _drop_hold_fn_for_analysis(analysis, tau)
        assert fn is not None
        # One time-constant later: 1/e ≈ 0.368.
        self.assertAlmostEqual(fn(1.0 + tau), float(np.exp(-1.0)), places=5)
        # Four time-constants later: roughly 1.8 %.
        self.assertLess(fn(1.0 + 4.0 * tau), 0.05)

    def test_closure_caches_drops_tuple(self) -> None:
        # The helper closes over a tuple snapshot so the compositor can
        # mutate / reuse the analysis dict per render without the per-frame
        # path observing it. Clearing the source list post-build must not
        # zero out the envelope.
        drops = [{"t": 0.5, "confidence": 1.0}]
        analysis = {"events": {"drops": drops}}
        fn = _drop_hold_fn_for_analysis(analysis, 1.0)
        assert fn is not None
        drops.clear()
        self.assertAlmostEqual(fn(0.5), 1.0, places=5)


class TestShaderTransientTracksForAnalysis(unittest.TestCase):
    def test_empty_analysis_returns_three_nones(self) -> None:
        lo, mid, hi = _shader_transient_tracks_for_analysis(
            {}, CompositorConfig()
        )
        self.assertIsNone(lo)
        self.assertIsNone(mid)
        self.assertIsNone(hi)

    def test_spectrum_analysis_builds_three_tracks(self) -> None:
        analysis = _spectrum_analysis()
        cfg = CompositorConfig()
        lo, mid, hi = _shader_transient_tracks_for_analysis(analysis, cfg)
        # All three builders consume the same spectrum, just different band
        # slices, so with a non-trivial spectrum they all produce tracks.
        for track in (lo, mid, hi):
            self.assertIsNotNone(track)
            assert track is not None
            # Scalar contract: ``value_at`` returns a finite number we can
            # feed straight into a shader uniform without further coercion.
            v = track.value_at(1.0)
            self.assertTrue(np.isfinite(v))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_config_decay_knobs_flow_through(self) -> None:
        # Override decay knobs to tiny values so the envelope collapses to
        # near-zero well before the next transient. This proves the config
        # values actually reach the band builders (rather than, say, a
        # default being hard-coded in the helper).
        analysis = _spectrum_analysis()
        cfg = CompositorConfig(
            shader_transient_lo_decay_sec=0.01,
            shader_transient_mid_decay_sec=0.01,
            shader_transient_hi_decay_sec=0.01,
        )
        lo, mid, hi = _shader_transient_tracks_for_analysis(analysis, cfg)
        assert lo is not None and mid is not None and hi is not None
        # Sample well past every injected peak (last peak at t = 3.0 s on a
        # 4 s analysis). With a 10 ms decay the envelopes must have fallen
        # to ~0 by t = 3.9 s.
        self.assertLess(lo.value_at(3.9), 0.05)
        self.assertLess(mid.value_at(3.9), 0.05)
        self.assertLess(hi.value_at(3.9), 0.05)


class TestKickGlowEnvelopeFn(unittest.TestCase):
    """Contract tests for the kick (low-band) contribution to the neon halo."""

    def test_disabled_returns_none(self) -> None:
        analysis = _spectrum_analysis()
        cfg = CompositorConfig(logo_kick_glow=False)
        self.assertIsNone(_kick_glow_envelope_fn(cfg, analysis))

    def test_zero_strength_returns_none(self) -> None:
        # Zero-strength short-circuits without needing an analyser — a
        # disabled contribution shouldn't pay the cost of building a track.
        cfg = CompositorConfig(logo_kick_glow=True, logo_kick_glow_strength=0.0)
        self.assertIsNone(_kick_glow_envelope_fn(cfg, _spectrum_analysis()))

    def test_missing_spectrum_returns_none(self) -> None:
        # Empty analysis → the bass pulse track builder returns ``None`` and
        # the helper must propagate that so the compositor frame loop can
        # skip the lookup entirely.
        cfg = CompositorConfig()
        self.assertIsNone(_kick_glow_envelope_fn(cfg, {}))

    def test_enabled_emits_scalar_envelope_at_kick(self) -> None:
        analysis = _spectrum_analysis()
        cfg = CompositorConfig()
        fn = _kick_glow_envelope_fn(cfg, analysis)
        self.assertIsNotNone(fn)
        assert fn is not None
        # Spectrum fixture puts a low-band peak at n/4 with fps=30, seconds=4
        # → t ≈ 1.0 s. The attack-only bass pulse track peaks on that frame,
        # so the envelope should be strictly positive there.
        self.assertGreater(fn(1.0), 0.0)


def _dummy_logo_rgba(size: int = 48) -> np.ndarray:
    """Opaque square PNG in RGBA shape expected by ``_build_beam_render_context``."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[..., 3] = 255
    return rgba


class TestBuildBeamRenderContext(unittest.TestCase):
    def test_returns_none_when_beams_disabled(self) -> None:
        analysis = {"events": {"drops": [{"t": 2.0}]}, "song_hash": "h"}
        cfg = CompositorConfig(rim_beams_enabled=False)
        ctx = _build_beam_render_context(
            cfg,
            analysis,
            logo_rgba_prepared=_dummy_logo_rgba(),
            resolved_rim_config=None,
        )
        self.assertIsNone(ctx)

    def test_returns_none_when_no_logo(self) -> None:
        analysis = {"events": {"drops": [{"t": 2.0}]}, "song_hash": "h"}
        cfg = CompositorConfig(rim_beams_enabled=True)
        ctx = _build_beam_render_context(
            cfg, analysis, logo_rgba_prepared=None, resolved_rim_config=None
        )
        self.assertIsNone(ctx)

    def test_returns_none_when_schedule_empty(self) -> None:
        # No drops, no impacts → nothing to schedule, short-circuit.
        analysis = {"events": {"drops": []}, "song_hash": "h"}
        cfg = CompositorConfig(rim_beams_enabled=True)
        ctx = _build_beam_render_context(
            cfg,
            analysis,
            logo_rgba_prepared=_dummy_logo_rgba(),
            resolved_rim_config=None,
        )
        self.assertIsNone(ctx)

    def test_populated_context_for_drop(self) -> None:
        analysis = {
            "events": {"drops": [{"t": 2.0}]},
            "song_hash": "hash123",
        }
        cfg = CompositorConfig(rim_beams_enabled=True)
        ctx = _build_beam_render_context(
            cfg,
            analysis,
            logo_rgba_prepared=_dummy_logo_rgba(),
            resolved_rim_config=None,
        )
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertGreater(len(ctx.schedule), 0)
        self.assertEqual(ctx.logo_base_hw, (48, 48))
        self.assertEqual(ctx.song_hash, "hash123")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
