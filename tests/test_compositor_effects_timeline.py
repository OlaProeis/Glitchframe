"""Hermetic tests for the Task 49 effects-timeline integration in
:mod:`pipeline.compositor`.

These tests exercise the pure-Python helpers that the compositor builds once
per render (and the resulting context objects / pulse functions) without
spinning up a moderngl context. The regression bar we care about here is:

* An **empty** timeline must be a complete no-op — the compositor returns the
  same contexts and scaled envelopes it returns on ``main``.
* A **user BEAM** clip fires **additively** alongside the auto schedule — no
  10 s group gate, and it still fires when the analyser path is disabled.
* A **user LOGO_GLITCH** clip drives an impact envelope even when the
  analyser's RMS peak track is empty.
* ``auto_reactivity_master == 0`` collapses the auto pulse / snare / impact /
  rim envelopes while leaving user-clip contributions untouched.
* The frame-effects pass is order-preserving and short-circuits on inactive
  clips (no allocations).
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.compositor import (
    CompositorConfig,
    _apply_frame_effects,
    _auto_enabled_for,
    _auto_reactivity_master,
    _build_beam_render_context,
    _build_frame_effects_context,
    _combined_glitch_fn,
    _scaled_pulse_fn,
    _user_beam_schedule,
    _user_glitch_envelope_fn,
)
from pipeline.effects_timeline import EffectClip, EffectKind, EffectsTimeline
from pipeline.logo_rim_beams import BeamConfig


def _dummy_logo_rgba(size: int = 48) -> np.ndarray:
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[..., 3] = 255
    return rgba


def _flat_analysis() -> dict:
    """Analysis with no drops + no spectrum so the auto schedule is empty."""
    return {"events": {"drops": []}, "song_hash": "songABC"}


class TestAutoReactivityMaster(unittest.TestCase):
    def test_default_is_identity(self) -> None:
        self.assertEqual(_auto_reactivity_master(CompositorConfig()), 1.0)

    def test_nan_collapses_to_one(self) -> None:
        cfg = CompositorConfig(auto_reactivity_master=float("nan"))
        self.assertEqual(_auto_reactivity_master(cfg), 1.0)

    def test_negative_collapses_to_one(self) -> None:
        cfg = CompositorConfig(auto_reactivity_master=-0.5)
        self.assertEqual(_auto_reactivity_master(cfg), 1.0)

    def test_inrange_passes_through(self) -> None:
        cfg = CompositorConfig(auto_reactivity_master=0.25)
        self.assertEqual(_auto_reactivity_master(cfg), 0.25)


class TestScaledPulseFn(unittest.TestCase):
    def test_none_passthrough(self) -> None:
        self.assertIsNone(_scaled_pulse_fn(None, 0.5))

    def test_unit_passthrough(self) -> None:
        fn = lambda t: 0.5
        self.assertIs(_scaled_pulse_fn(fn, 1.0), fn)

    def test_zero_damps_everything(self) -> None:
        fn = _scaled_pulse_fn(lambda t: 0.9, 0.0)
        assert fn is not None
        self.assertEqual(fn(1.23), 0.0)

    def test_halving(self) -> None:
        fn = _scaled_pulse_fn(lambda t: 0.8, 0.5)
        assert fn is not None
        self.assertAlmostEqual(fn(1.0), 0.4, places=7)


class TestUserGlitchEnvelope(unittest.TestCase):
    def test_no_clips_returns_none(self) -> None:
        self.assertIsNone(_user_glitch_envelope_fn([]))

    def test_clip_active_nonzero_strength(self) -> None:
        c = EffectClip(
            id="g1",
            kind=EffectKind.LOGO_GLITCH,
            t_start=0.5,
            duration_s=0.2,
            settings={"strength": 0.6},
        )
        fn = _user_glitch_envelope_fn([c])
        assert fn is not None
        self.assertEqual(fn(0.4), 0.0)
        self.assertAlmostEqual(fn(0.55), 0.6, places=6)
        self.assertEqual(fn(0.7), 0.0)

    def test_overlap_clamped_to_one(self) -> None:
        c1 = EffectClip(
            id="g1",
            kind=EffectKind.LOGO_GLITCH,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.7},
        )
        c2 = EffectClip(
            id="g2",
            kind=EffectKind.LOGO_GLITCH,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.7},
        )
        fn = _user_glitch_envelope_fn([c1, c2])
        assert fn is not None
        self.assertEqual(fn(0.5), 1.0)

    def test_non_glitch_clips_ignored(self) -> None:
        beam = EffectClip(
            id="b",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=1.0,
            settings={"strength": 0.9},
        )
        self.assertIsNone(_user_glitch_envelope_fn([beam]))


class TestCombinedGlitchFn(unittest.TestCase):
    def test_both_none(self) -> None:
        self.assertIsNone(_combined_glitch_fn(None, None))

    def test_either_passthrough(self) -> None:
        a = lambda t: 0.4
        self.assertIs(_combined_glitch_fn(a, None), a)
        self.assertIs(_combined_glitch_fn(None, a), a)

    def test_sum_clamped(self) -> None:
        a = lambda t: 0.7
        b = lambda t: 0.8
        fn = _combined_glitch_fn(a, b)
        assert fn is not None
        self.assertEqual(fn(0.0), 1.0)

    def test_sum_normal(self) -> None:
        a = lambda t: 0.2
        b = lambda t: 0.3
        fn = _combined_glitch_fn(a, b)
        assert fn is not None
        self.assertAlmostEqual(fn(0.0), 0.5, places=6)


class TestAutoEnabledFor(unittest.TestCase):
    def test_default_true_without_timeline(self) -> None:
        cfg = CompositorConfig()
        self.assertTrue(_auto_enabled_for(cfg, EffectKind.BEAM))

    def test_disabled_kind_respected(self) -> None:
        tl = EffectsTimeline()
        tl.auto_enabled[EffectKind.BEAM] = False
        cfg = CompositorConfig(effects_timeline=tl)
        self.assertFalse(_auto_enabled_for(cfg, EffectKind.BEAM))
        self.assertTrue(_auto_enabled_for(cfg, EffectKind.LOGO_GLITCH))


class TestUserBeamSchedule(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(
            _user_beam_schedule(
                [], beam_cfg=BeamConfig(), song_hash="h", n_color_layers=2
            ),
            [],
        )

    def test_user_beam_converted(self) -> None:
        # Clip duration longer than ``BeamConfig.duration_sec`` is preserved
        # verbatim (no floor needed).
        c = EffectClip(
            id="u1",
            kind=EffectKind.BEAM,
            t_start=1.0,
            duration_s=1.5,
            settings={"strength": 0.75},
        )
        beams = _user_beam_schedule(
            [c], beam_cfg=BeamConfig(), song_hash="h", n_color_layers=3
        )
        self.assertEqual(len(beams), 1)
        b = beams[0]
        self.assertEqual(b.t_start, 1.0)
        self.assertEqual(b.duration_s, 1.5)
        self.assertAlmostEqual(b.intensity, 0.75, places=6)
        self.assertTrue(b.is_drop)
        self.assertTrue(b.sustain_shaping)
        self.assertIn(b.color_layer_idx, (0, 1, 2))

    def test_user_beam_duration_floor(self) -> None:
        # Regression: hand-placed short ticks on the effects timeline used to
        # collapse to a single-frame "point" because the attack alone is 40 ms.
        # The schedule must floor the rendered duration at
        # ``BeamConfig.duration_sec`` so the full envelope + afterglow plays,
        # while keeping the user-chosen ``t_start`` untouched.
        beam_cfg = BeamConfig(duration_sec=0.75)
        c = EffectClip(
            id="uShort",
            kind=EffectKind.BEAM,
            t_start=3.2,
            duration_s=0.05,
            settings={"strength": 1.0},
        )
        beams = _user_beam_schedule(
            [c], beam_cfg=beam_cfg, song_hash="h", n_color_layers=2
        )
        self.assertEqual(len(beams), 1)
        self.assertEqual(beams[0].t_start, 3.2)
        self.assertAlmostEqual(beams[0].duration_s, 0.75, places=6)
        self.assertFalse(beams[0].sustain_shaping)

    def test_zero_strength_dropped(self) -> None:
        c = EffectClip(
            id="u1",
            kind=EffectKind.BEAM,
            t_start=1.0,
            duration_s=0.4,
            settings={"strength": 0.0},
        )
        self.assertEqual(
            _user_beam_schedule(
                [c], beam_cfg=BeamConfig(), song_hash="h", n_color_layers=2
            ),
            [],
        )

    def test_deterministic_across_calls(self) -> None:
        c = EffectClip(
            id="stable",
            kind=EffectKind.BEAM,
            t_start=2.0,
            duration_s=0.5,
            settings={"strength": 0.9, "thickness_px": 14.0},
        )
        a = _user_beam_schedule(
            [c], beam_cfg=BeamConfig(), song_hash="song", n_color_layers=2
        )
        b = _user_beam_schedule(
            [c], beam_cfg=BeamConfig(), song_hash="song", n_color_layers=2
        )
        self.assertEqual(
            (a[0].angle_rad, a[0].color_layer_idx),
            (b[0].angle_rad, b[0].color_layer_idx),
        )


class TestBeamContextWithUserClips(unittest.TestCase):
    def test_user_beam_fires_with_no_auto_schedule(self) -> None:
        # Empty analysis → schedule_rim_beams returns []; adding a user BEAM
        # clip must produce a non-empty beam context so the draw pass runs.
        analysis = _flat_analysis()
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="uBEAM",
                    kind=EffectKind.BEAM,
                    t_start=1.0,
                    duration_s=0.3,
                    settings={"strength": 0.8},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        ctx = _build_beam_render_context(
            cfg,
            analysis,
            logo_rgba_prepared=_dummy_logo_rgba(),
            resolved_rim_config=None,
        )
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(len(ctx.schedule), 1)
        self.assertEqual(ctx.schedule[0].t_start, 1.0)

    def test_user_beam_fires_even_when_auto_disabled(self) -> None:
        analysis = {
            "events": {"drops": [{"t": 2.0, "confidence": 1.0}]},
            "song_hash": "h",
        }
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="uBEAM",
                    kind=EffectKind.BEAM,
                    t_start=0.5,
                    duration_s=0.3,
                    settings={"strength": 1.0},
                )
            ]
        )
        tl.auto_enabled[EffectKind.BEAM] = False
        cfg = CompositorConfig(effects_timeline=tl)
        ctx = _build_beam_render_context(
            cfg,
            analysis,
            logo_rgba_prepared=_dummy_logo_rgba(),
            resolved_rim_config=None,
        )
        self.assertIsNotNone(ctx)
        assert ctx is not None
        # Auto disabled → analyser-driven drop at t=2.0 is NOT on the schedule.
        # Only the single user beam at t=0.5 survives.
        self.assertEqual(len(ctx.schedule), 1)
        self.assertEqual(ctx.schedule[0].t_start, 0.5)

    def test_user_beams_merge_without_group_gate(self) -> None:
        # Three user beams spaced 100 ms apart must all survive the merge —
        # the 10 s group gate inside schedule_rim_beams only applies to the
        # analyser-driven path.
        analysis = _flat_analysis()
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id=f"ub{i}",
                    kind=EffectKind.BEAM,
                    t_start=1.0 + 0.1 * i,
                    duration_s=0.3,
                    settings={"strength": 0.9},
                )
                for i in range(3)
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        ctx = _build_beam_render_context(
            cfg,
            analysis,
            logo_rgba_prepared=_dummy_logo_rgba(),
            resolved_rim_config=None,
        )
        assert ctx is not None
        self.assertEqual(len(ctx.schedule), 3)


class TestBuildFrameEffectsContext(unittest.TestCase):
    def test_no_timeline_returns_none(self) -> None:
        self.assertIsNone(
            _build_frame_effects_context(CompositorConfig(), {})
        )

    def test_empty_timeline_returns_none(self) -> None:
        cfg = CompositorConfig(effects_timeline=EffectsTimeline())
        self.assertIsNone(_build_frame_effects_context(cfg, {}))

    def test_only_beam_clip_returns_none(self) -> None:
        # BEAM is a logo-layer effect, not a post-stack frame effect; a
        # timeline that only contains BEAM / LOGO_GLITCH must leave the
        # frame-pass short-circuited so the output is unchanged.
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="b",
                    kind=EffectKind.BEAM,
                    t_start=0.0,
                    duration_s=1.0,
                    settings={"strength": 1.0},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        self.assertIsNone(_build_frame_effects_context(cfg, {}))

    def test_zoom_clip_builds_context(self) -> None:
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="z",
                    kind=EffectKind.ZOOM_PUNCH,
                    t_start=0.0,
                    duration_s=0.5,
                    settings={"peak_scale": 1.2},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        fx = _build_frame_effects_context(cfg, {"song_hash": "h"})
        self.assertIsNotNone(fx)
        assert fx is not None
        self.assertEqual(len(fx.zoom_clips), 1)
        self.assertEqual(fx.shake_clips, ())
        self.assertEqual(fx.song_hash, "h")


class TestApplyFrameEffects(unittest.TestCase):
    def _frame(self, h: int = 8, w: int = 8) -> np.ndarray:
        # Horizontal gradient; keeps the test easy to reason about when
        # looking at shake / zoom output.
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            f[:, x] = np.uint8(min(255, x * 7))
        return f

    def test_none_ctx_passthrough(self) -> None:
        frame = self._frame()
        out = _apply_frame_effects(frame, 0.0, None)
        self.assertIs(out, frame)

    def test_all_inactive_passthrough(self) -> None:
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="z",
                    kind=EffectKind.ZOOM_PUNCH,
                    t_start=10.0,
                    duration_s=0.2,
                    settings={"peak_scale": 1.2},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        fx = _build_frame_effects_context(cfg, {})
        frame = self._frame()
        out = _apply_frame_effects(frame, 0.0, fx)
        # At t=0 the zoom clip is inactive → identity short-circuit.
        np.testing.assert_array_equal(out, frame)

    def test_color_invert_at_full_mix(self) -> None:
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="i",
                    kind=EffectKind.COLOR_INVERT,
                    t_start=0.0,
                    duration_s=1.0,
                    settings={"mix": 1.0, "intensity": 1.0},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        fx = _build_frame_effects_context(cfg, {})
        frame = self._frame()
        out = _apply_frame_effects(frame, 0.5, fx)
        np.testing.assert_array_equal(out, np.uint8(255) - frame)

    def test_screen_shake_moves_pixels(self) -> None:
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="s",
                    kind=EffectKind.SCREEN_SHAKE,
                    t_start=0.0,
                    duration_s=1.0,
                    settings={"amplitude_px": 30.0, "frequency_hz": 4.0},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        fx = _build_frame_effects_context(cfg, {"song_hash": "hx"})
        frame = self._frame(h=32, w=32)
        found_change = False
        for t in (0.05, 0.1, 0.2, 0.3, 0.4):
            out = _apply_frame_effects(frame, t, fx)
            if not np.array_equal(out, frame):
                found_change = True
                break
        self.assertTrue(found_change, "expected shake to move pixels in-window")


class TestFrameEffectsScanline(unittest.TestCase):
    def test_scanline_clips_build_context_without_warning(self) -> None:
        """SCANLINE_TEAR is indexed like other post-stack kinds; no placeholder log."""
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="sc",
                    kind=EffectKind.SCANLINE_TEAR,
                    t_start=0.0,
                    duration_s=0.4,
                    settings={
                        "intensity": 0.5,
                        "band_count": 3,
                        "band_height_px": 4,
                        "wrap_mode": "wrap",
                    },
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        with self.assertNoLogs("pipeline.compositor", level="WARNING"):
            ctx = _build_frame_effects_context(cfg, {"song_hash": "s"})
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(len(ctx.scanline_clips), 1)

    def test_scanline_tear_runs_in_fixed_order_before_invert(self) -> None:
        """Scanline runs in _apply_frame_effects and can change pixels at active t."""
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="sc",
                    kind=EffectKind.SCANLINE_TEAR,
                    t_start=0.0,
                    duration_s=1.0,
                    settings={
                        "intensity": 1.0,
                        "band_count": 2,
                        "band_height_px": 3,
                        "wrap_mode": "wrap",
                    },
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        fx = _build_frame_effects_context(cfg, {"song_hash": "song"})
        self.assertIsNotNone(fx)
        assert fx is not None
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, :10, :] = 200
        changed = False
        for t in (0.0, 0.05, 0.12, 0.2):
            out = _apply_frame_effects(frame, t, fx)
            if not np.array_equal(out, frame):
                changed = True
                break
        self.assertTrue(changed, "expected scanline tear to move pixels in-window")

    def test_chromatic_clips_build_context_without_warning(self) -> None:
        tl = EffectsTimeline(
            clips=[
                EffectClip(
                    id="ch",
                    kind=EffectKind.CHROMATIC_ABERRATION,
                    t_start=0.0,
                    duration_s=0.4,
                    settings={"shift_px": 2.0, "jitter": 0.0, "direction_deg": 0.0},
                )
            ]
        )
        cfg = CompositorConfig(effects_timeline=tl)
        with self.assertNoLogs("pipeline.compositor", level="WARNING"):
            ctx = _build_frame_effects_context(cfg, {"song_hash": "s"})
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(len(ctx.chromatic_clips), 1)

    def test_fixed_order_zoom_then_invert_not_commutative(self) -> None:
        """Order per compositor: zoom then invert; result differs from each alone."""
        h, w = 8, 8
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            f[:, x] = np.uint8(min(255, x * 7))
        z = EffectClip(
            id="z",
            kind=EffectKind.ZOOM_PUNCH,
            t_start=0.0,
            duration_s=1.0,
            settings={"peak_scale": 1.2, "ease_in_s": 0.0, "ease_out_s": 0.0},
        )
        invc = EffectClip(
            id="i",
            kind=EffectKind.COLOR_INVERT,
            t_start=0.0,
            duration_s=1.0,
            settings={"mix": 1.0, "intensity": 1.0},
        )
        t_both = EffectsTimeline(clips=[z, invc])
        cfg = CompositorConfig(effects_timeline=t_both)
        fx = _build_frame_effects_context(cfg, {})
        self.assertIsNotNone(fx)
        out = _apply_frame_effects(f, 0.1, fx)
        out_inv_only = _apply_frame_effects(
            f,
            0.1,
            _build_frame_effects_context(
                CompositorConfig(effects_timeline=EffectsTimeline(clips=[invc])),
                {},
            ),
        )
        out_zoom_only = _apply_frame_effects(
            f,
            0.1,
            _build_frame_effects_context(
                CompositorConfig(effects_timeline=EffectsTimeline(clips=[z])),
                {},
            ),
        )
        self.assertFalse(np.array_equal(out, f), "zoom+invert should change pixels")
        self.assertFalse(
            np.array_equal(out, out_inv_only),
            "zoom then invert ≠ invert only",
        )
        self.assertFalse(
            np.array_equal(out, out_zoom_only),
            "zoom then invert ≠ zoom only",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
