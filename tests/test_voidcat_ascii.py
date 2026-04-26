"""Unit tests for :mod:`pipeline.voidcat_ascii` (no GPU)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.reactive_shader import uniforms_at_time
from pipeline.voidcat_ascii import (
    _beat_ring_strength,
    _best_cat_state,
    build_voidcat_ascii_context,
    render_voidcat_ascii_rgba,
    sanity_check_ascii_layer,
)


class TestVoidcatAscii(unittest.TestCase):
    def _minimal_analysis(self) -> dict:
        return {
            "beats": [0.0, 0.5, 1.0],
            "tempo": {"bpm": 120.0},
            "rms": {"fps": 20.0, "values": [0.2, 0.35, 0.3, 0.4]},
            "spectrum": {
                "fps": 20.0,
                "values": [[0.1] * 8] * 6,
            },
            "onsets": {
                "peaks": [],
                "strength": [0.0] * 20,
                "frame_rate_hz": 10.0,
            },
            "events": {
                "drops": [
                    {"t": 0.5, "confidence": 0.3},
                    {"t": 2.0, "confidence": 0.95},
                ],
            },
            "song_hash": "abc123",
        }

    def test_hero_is_highest_confidence(self) -> None:
        a = self._minimal_analysis()
        ctx = build_voidcat_ascii_context(
            a, ["#000000", "#111111", "#222222", "#333333", "#444444"]
        )
        self.assertAlmostEqual(ctx.hero_drop_t, 2.0)
        self.assertGreater(ctx.hero_confidence, 0.9)
        self.assertEqual(len(ctx.drop_events), 2)

    def test_cat_ambient_always_on_and_roams(self) -> None:
        a = self._minimal_analysis()
        ctx = build_voidcat_ascii_context(a, None)
        u0 = {
            "beat_phase": 0.0,
            "bar_phase": 0.0,
            "bass_hit": 0.0,
        }
        w0, p0, pulse0 = _best_cat_state(0.0, ctx, 64, 36, 20, 44, u0)
        w1, p1, _ = _best_cat_state(22.0, ctx, 64, 36, 20, 44, u0)
        w_pre2, p_pre2, pulse_pre = _best_cat_state(
            1.9, ctx, 64, 36, 20, 44, u0
        )
        # Always on-screen (ambient), not drop-gated.
        self.assertGreater(w0, 0.25)
        self.assertGreater(w1, 0.25)
        self.assertIsNotNone(p0)
        # Drift: cell anchor should change (same umap, different t).
        self.assertNotEqual(p0, p1)
        # Near the big drop, envelope adds juice on top of ambient.
        self.assertGreaterEqual(pulse_pre, pulse0)
        self.assertGreaterEqual(w_pre2, w0)

    def test_center_dimmer_when_full_frame(self) -> None:
        a = self._minimal_analysis()
        ctx = build_voidcat_ascii_context(a, None)
        u = uniforms_at_time(a, 0.1, num_bands=8)
        u["bass_hit"] = 0.0
        u["transient_lo"] = 0.0
        u["transient_mid"] = 0.0
        u["transient_hi"] = 0.0
        u["drop_hold"] = 0.0
        out = render_voidcat_ascii_rgba(256, 144, 0.1, uniforms=u, ctx=ctx)
        self.assertEqual(out.shape, (144, 256, 4))
        # Full frame has glyphs; centre band is softer (lower alpha) than flanks
        mid = out.shape[1] // 2
        a_centre = float(np.mean(out[:, mid - 2 : mid + 2, 3]))
        a_sides = float(
            np.mean(
                np.concatenate(
                    [
                        out[:, 0:20, 3].ravel(),
                        out[:, -20:, 3].ravel(),
                    ]
                )
            )
        )
        self.assertLess(a_centre, a_sides * 0.9)
        self.assertGreater(a_centre, 1.0)  # centre is not empty, just dimmer

    def test_sanity_check_ok(self) -> None:
        r = sanity_check_ascii_layer()
        self.assertTrue(r["ok"], msg=str(r))

    def test_beat_ring_centre_high_when_phase_low(self) -> None:
        """Ring front at small ``beat_phase`` should light the middle of the grid."""
        ncols, nrows = 64, 36
        ci, cj = ncols // 2, nrows // 2
        u_lo = {"beat_phase": 0.05, "bass_hit": 0.0}
        u_hi = {"beat_phase": 0.92, "bass_hit": 0.0}
        s_lo = _beat_ring_strength(ci, cj, ncols, nrows, u_lo)
        s_hi = _beat_ring_strength(ci, cj, ncols, nrows, u_hi)
        self.assertGreater(s_lo, s_hi)
