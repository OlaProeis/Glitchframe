"""Unit tests for :mod:`pipeline.logo_rim_lights`."""

from __future__ import annotations

import colorsys
import dataclasses
import unittest

import numpy as np

from pipeline.beat_pulse import PulseTrack

from pipeline.logo_rim_lights import (
    LogoRimPrep,
    RimAudioModulation,
    RimAudioTuning,
    RimLightConfig,
    RimModulationState,
    advance_rim_audio_modulation,
    compute_logo_rim_light_patch,
    compute_logo_rim_prep,
    rim_base_rgb_from_preset,
    rim_modulation_instant,
)


def _disk_with_horizontal_stroke(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    radius: float,
    stroke_half_thickness: int = 1,
) -> np.ndarray:
    """Black-filled disk (opaque) with a bright horizontal stroke at cy."""
    im = np.zeros((h, w, 4), dtype=np.uint8)
    yx, xx = np.ogrid[0:h, 0:w]
    inside = (xx - cx) ** 2 + (yx - cy) ** 2 <= radius**2
    im[inside, 3] = 255
    im[inside, 0:3] = 0
    y_lo = int(cy) - stroke_half_thickness
    y_hi = int(cy) + stroke_half_thickness + 1
    im[y_lo:y_hi, :, 0:3] = 255
    im[~inside, 3] = 0
    im[~inside, 0:3] = 0
    return im


def _solid_white_square(size: int) -> np.ndarray:
    im = np.zeros((size, size, 4), dtype=np.uint8)
    im[..., 0:3] = 255
    im[..., 3] = 255
    return im


class TestLogoRimPrep(unittest.TestCase):
    def test_fully_transparent_centroid_centered(self) -> None:
        im = np.zeros((24, 32, 4), dtype=np.uint8)
        p = compute_logo_rim_prep(im)
        self.assertIsInstance(p, LogoRimPrep)
        self.assertEqual(p.centroid_xy, (16.0, 12.0))
        self.assertEqual(p.line_confidence, 0.0)
        self.assertFalse(p.use_line_features)
        self.assertEqual(p.line_mask.shape, (24, 32))

    def test_stroke_on_dark_disc_enables_line_features(self) -> None:
        im = _disk_with_horizontal_stroke(48, 48, cx=24.0, cy=24.0, radius=20.0)
        p = compute_logo_rim_prep(im, min_line_confidence=0.05)
        self.assertTrue(
            p.use_line_features, "bright sparse stroke on dark fill should be line-like"
        )
        self.assertGreater(p.line_confidence, 0.08)
        self.assertGreater(np.max(p.line_mask), 0.2)

    def test_solid_white_square_disables_line_features(self) -> None:
        p = compute_logo_rim_prep(_solid_white_square(32), min_line_confidence=0.01)
        self.assertFalse(
            p.use_line_features, "uniform bright fill should fall back to halo logic"
        )
        self.assertEqual(float(np.max(p.line_mask)), 0.0)

    def test_centroid_matches_com(self) -> None:
        im = np.zeros((16, 16, 4), dtype=np.uint8)
        im[4:8, 5:9, 0:3] = 255
        im[4:8, 5:9, 3] = 255
        a = im[..., 3].astype(np.float64) / 255.0
        sy, sx = 16, 16
        ys, xs = np.indices((sy, sx))
        ax, ay = float((xs * a).sum() / a.sum()), float((ys * a).sum() / a.sum())
        p = compute_logo_rim_prep(im)
        self.assertAlmostEqual(p.centroid_xy[0], ax, places=4)
        self.assertAlmostEqual(p.centroid_xy[1], ay, places=4)

    def test_rejects_non_uint8_rgba(self) -> None:
        imf = np.zeros((8, 8, 4), dtype=np.float32)
        with self.assertRaises(TypeError):
            compute_logo_rim_prep(imf)

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            compute_logo_rim_prep(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_determinism_synthetic(self) -> None:
        im = _disk_with_horizontal_stroke(32, 32, cx=16.0, cy=16.0, radius=12.0)
        a = compute_logo_rim_prep(im, min_line_confidence=0.05)
        b = compute_logo_rim_prep(im, min_line_confidence=0.05)
        np.testing.assert_array_equal(a.line_mask, b.line_mask)
        np.testing.assert_array_equal(a.alpha_f, b.alpha_f)
        self.assertEqual(a.centroid_xy, b.centroid_xy)
        self.assertEqual(a.use_line_features, b.use_line_features)


def _solid_disk(h: int, w: int, *, cx: float, cy: float, radius: float) -> np.ndarray:
    """Fully opaque white-filled disk on transparent background."""
    im = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.ogrid[0:h, 0:w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    im[inside, 3] = 255
    im[inside, 0:3] = 255
    return im


class TestRimLightPatch(unittest.TestCase):
    """Task 25: traveling-wave rim render via ``compute_logo_rim_light_patch``."""

    def setUp(self) -> None:
        self.h = self.w = 48
        self.cx = self.cy = 24.0
        self.radius = 18.0
        self.im = _solid_disk(self.h, self.w, cx=self.cx, cy=self.cy, radius=self.radius)
        self.prep = compute_logo_rim_prep(self.im)

    def test_patch_shape_and_dtype(self) -> None:
        cfg = RimLightConfig(pad_px=16)
        patch, pad = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg)
        self.assertEqual(pad, 16)
        self.assertEqual(patch.dtype, np.uint8)
        self.assertEqual(patch.shape, (self.h + 32, self.w + 32, 4))
        self.assertTrue(np.all(np.isfinite(patch)))

    def test_patch_is_premultiplied(self) -> None:
        cfg = RimLightConfig(rim_rgb=(255, 180, 220))
        patch, _ = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg)
        a = patch[..., 3].astype(np.int32)
        for ch in range(3):
            self.assertTrue(
                bool(np.all(patch[..., ch].astype(np.int32) <= a + 1)),
                f"channel {ch} exceeds alpha (not premultiplied)",
            )

    def test_phase_changes_over_time(self) -> None:
        cfg = RimLightConfig(phase_hz=0.5, waves=3, pad_px=16)
        a, _ = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg)
        b, _ = compute_logo_rim_light_patch(self.prep, t=1.0, config=cfg)
        delta = np.abs(a.astype(np.int32) - b.astype(np.int32)).mean()
        self.assertGreater(
            delta, 1.0, "travelling wave should measurably change the patch over 1 second"
        )

    def test_phase_zero_hz_is_static(self) -> None:
        cfg = RimLightConfig(phase_hz=0.0, waves=3)
        a, _ = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg)
        b, _ = compute_logo_rim_light_patch(self.prep, t=2.75, config=cfg)
        np.testing.assert_array_equal(a, b)

    def test_inward_mix_changes_interior(self) -> None:
        # Isolate inward term: disable halo/line/angular and compare across interior.
        base = RimLightConfig(
            halo_boost=0.0,
            line_boost=0.0,
            phase_hz=0.0,
            wave_floor=1.0,
            pad_px=16,
            inward_depth_px=12.0,
            blur_px=0.0,
        )
        no_inward = compute_logo_rim_light_patch(
            self.prep, t=0.0, config=dataclasses.replace(base, inward_mix=0.0)
        )[0]
        full_inward = compute_logo_rim_light_patch(
            self.prep, t=0.0, config=dataclasses.replace(base, inward_mix=1.0)
        )[0]
        pad = base.pad_px
        yy, xx = np.ogrid[0 : self.h, 0 : self.w]
        interior = (xx - self.cx) ** 2 + (yy - self.cy) ** 2 <= (self.radius - 3) ** 2
        interior_pad = np.zeros((self.h + 2 * pad, self.w + 2 * pad), dtype=bool)
        interior_pad[pad : pad + self.h, pad : pad + self.w] = interior
        self.assertEqual(int(no_inward[..., 3][interior_pad].sum()), 0)
        self.assertGreater(float(full_inward[..., 3][interior_pad].mean()), 10.0)

    def test_empty_alpha_returns_zero_patch(self) -> None:
        empty = np.zeros((32, 24, 4), dtype=np.uint8)
        prep = compute_logo_rim_prep(empty)
        cfg = RimLightConfig(pad_px=8)
        patch, pad = compute_logo_rim_light_patch(prep, t=0.0, config=cfg)
        self.assertEqual(pad, 8)
        self.assertEqual(patch.shape, (32 + 16, 24 + 16, 4))
        self.assertEqual(int(patch.max()), 0)

    def test_opacity_and_intensity_scale_alpha(self) -> None:
        cfg_full = RimLightConfig(pad_px=16, opacity_pct=100.0, intensity=1.0)
        cfg_half = RimLightConfig(pad_px=16, opacity_pct=50.0, intensity=1.0)
        cfg_off = RimLightConfig(pad_px=16, opacity_pct=0.0, intensity=1.0)
        a = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg_full)[0]
        b = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg_half)[0]
        c = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg_off)[0]
        self.assertGreater(int(a[..., 3].sum()), int(b[..., 3].sum()))
        self.assertEqual(int(c[..., 3].max()), 0)

    def test_determinism(self) -> None:
        cfg = RimLightConfig(phase_hz=0.3, waves=4)
        a = compute_logo_rim_light_patch(self.prep, t=0.37, config=cfg)[0]
        b = compute_logo_rim_light_patch(self.prep, t=0.37, config=cfg)[0]
        np.testing.assert_array_equal(a, b)

    def test_line_features_off_degrades_to_halo(self) -> None:
        # Solid white fill → ``use_line_features`` is False; line term must be
        # zero yet halo should still drive a visible glow.
        cfg = RimLightConfig(line_boost=5.0, halo_boost=1.0, phase_hz=0.0)
        patch, _ = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg)
        self.assertFalse(self.prep.use_line_features)
        self.assertGreater(int(patch[..., 3].max()), 40)

    def test_line_boost_no_op_when_line_features_disabled(self) -> None:
        """Mask term is ignored when prep falls back to halo-only (task 24/25)."""
        prep = compute_logo_rim_prep(_solid_white_square(32))
        self.assertFalse(prep.use_line_features)
        cfg_lo = RimLightConfig(phase_hz=0.0, line_boost=0.0, pad_px=8)
        cfg_hi = dataclasses.replace(cfg_lo, line_boost=50.0)
        a, _ = compute_logo_rim_light_patch(prep, t=0.0, config=cfg_lo)
        b, _ = compute_logo_rim_light_patch(prep, t=0.0, config=cfg_hi)
        np.testing.assert_array_equal(a, b)

    def test_line_boost_matters_when_line_features_enabled(self) -> None:
        """Stroke prep: radial base can be line-only; boosting line_mask changes output."""
        im = _disk_with_horizontal_stroke(40, 40, cx=20.0, cy=20.0, radius=16.0)
        prep = compute_logo_rim_prep(im, min_line_confidence=0.05)
        self.assertTrue(prep.use_line_features)
        base = RimLightConfig(
            halo_boost=0.0,
            inward_mix=0.0,
            phase_hz=0.0,
            wave_floor=1.0,
            line_boost=0.0,
            blur_px=1.0,
            pad_px=8,
        )
        off, _ = compute_logo_rim_light_patch(prep, t=0.0, config=base)
        on, _ = compute_logo_rim_light_patch(
            prep, t=0.0, config=dataclasses.replace(base, line_boost=4.0)
        )
        self.assertEqual(int(off[..., 3].max()), 0)
        self.assertGreater(int(on[..., 3].max()), 30)

    def test_phase_offset_changes_spatial_pattern(self) -> None:
        """Static wave (``phase_hz=0``): ``phase_offset`` rotates lobes in space."""
        cfg0 = RimLightConfig(
            phase_hz=0.0, phase_offset=0.0, waves=3, pad_px=16
        )
        cfg_pi = dataclasses.replace(cfg0, phase_offset=float(np.pi))
        a, _ = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg0)
        b, _ = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg_pi)
        self.assertGreater(
            float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean()),
            1.0,
            "π phase offset should visibly shift the angular pattern",
        )

    def test_determinism_with_audio_modulation(self) -> None:
        m = rim_modulation_instant(0.65, 0.4)
        cfg = RimLightConfig(phase_hz=0.18, pad_px=8)
        a, _ = compute_logo_rim_light_patch(
            self.prep, t=0.41, config=cfg, audio_mod=m
        )
        b, _ = compute_logo_rim_light_patch(
            self.prep, t=0.41, config=cfg, audio_mod=m
        )
        np.testing.assert_array_equal(a, b)


def _max_abs_diff_patches(
    a: np.ndarray, b: np.ndarray, channels: int = 4
) -> float:
    d = (a.astype(np.int32) - b.astype(np.int32)).reshape(-1, channels)
    return float(np.max(np.sum(np.abs(d), axis=1)))


def _dominant_hue_saturated_pixel(patch: np.ndarray) -> float:
    """HSV hue at the brightest-alpha pixel (avoids desaturated / dark points)."""
    a = patch[:, :, 3]
    m = a > 16
    if not np.any(m):
        return 0.0
    yx = np.unravel_index(int(np.argmax(a * m)), a.shape)
    r, g, b = (int(patch[yx[0], yx[1], c]) for c in range(3))
    return float(colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[0])


class TestRimMulticolor(unittest.TestCase):
    """Task 26: multi-layer hue, determinism, drift, frame continuity."""

    def test_rim_base_rgb_from_preset(self) -> None:
        a = rim_base_rgb_from_preset("#aabbcc", "#112233")
        b = rim_base_rgb_from_preset("#aabbcc", "#112233")
        self.assertEqual(a, b)
        from pipeline.logo_composite import resolve_logo_glow_rgb

        self.assertEqual(a, resolve_logo_glow_rgb("#aabbcc", "#112233"))

    def test_multicolor_determinism_song_hash(self) -> None:
        im = _disk_with_horizontal_stroke(40, 40, cx=20.0, cy=20.0, radius=16.0)
        prep = compute_logo_rim_prep(im, min_line_confidence=0.05)
        cfg = RimLightConfig(
            rim_color_layers=3,
            color_spread_rad=2.0 * np.pi / 3.0,
            song_hash="test-track-1",
            hue_drift_per_sec=0.0,
            phase_hz=0.2,
            pad_px=12,
        )
        t = 0.31
        a, _ = compute_logo_rim_light_patch(prep, t=t, config=cfg)
        b, _ = compute_logo_rim_light_patch(prep, t=t, config=cfg)
        np.testing.assert_array_equal(a, b)
        c, _ = compute_logo_rim_light_patch(
            prep,
            t=t,
            config=dataclasses.replace(cfg, song_hash="other-song"),
        )
        self.assertFalse(np.array_equal(a, c))

    def test_hue_drift_monotonic_small_step(self) -> None:
        im = _disk_with_horizontal_stroke(48, 48, cx=24.0, cy=24.0, radius=18.0)
        prep = compute_logo_rim_prep(im, min_line_confidence=0.05)
        cfg = RimLightConfig(
            rim_color_layers=2,
            phase_hz=0.0,
            wave_floor=1.0,
            hue_drift_per_sec=0.25,
            song_hash=42,
            pad_px=16,
        )
        p0, _ = compute_logo_rim_light_patch(prep, t=0.0, config=cfg)
        p1, _ = compute_logo_rim_light_patch(prep, t=0.1, config=cfg)
        h0 = _dominant_hue_saturated_pixel(p0)
        h1 = _dominant_hue_saturated_pixel(p1)
        delta = (h1 - h0) % 1.0
        self.assertGreater(delta, 0.01, "expected hue to advance with ``hue_drift``")
        self.assertLess(
            abs(delta - 0.1 * 0.25),
            0.2,
            "expected hue shift ~0.1 s × 0.25 Hz of the hue ring (pixel sampling is approximate)",
        )

    def test_sequential_frame_delta_bounded(self) -> None:
        im = _disk_with_horizontal_stroke(36, 36, cx=18.0, cy=18.0, radius=12.0)
        prep = compute_logo_rim_prep(im, min_line_confidence=0.05)
        cfg = RimLightConfig(
            rim_color_layers=3,
            phase_hz=0.35,
            hue_drift_per_sec=0.1,
            song_hash=9001,
            pad_px=10,
        )
        prev, _ = compute_logo_rim_light_patch(prep, t=0.0, config=cfg)
        m = 0.0
        for s in range(1, 15):
            nxt, _ = compute_logo_rim_light_patch(
                prep, t=0.002 * s, config=cfg
            )
            m = max(m, _max_abs_diff_patches(nxt, prev))
            prev = nxt
        self.assertLess(
            m, 100.0, "tight t-steps should not jump per pixel (no harsh banding)"
        )

    def test_multicolor_premultiplied_roughly(self) -> None:
        im = _disk_with_horizontal_stroke(32, 32, cx=16.0, cy=16.0, radius=12.0)
        prep = compute_logo_rim_prep(im, min_line_confidence=0.05)
        cfg = RimLightConfig(rim_color_layers=2, phase_hz=0.1, pad_px=8)
        patch, _ = compute_logo_rim_light_patch(prep, t=0.2, config=cfg)
        a = patch[..., 3].astype(np.int32)
        for ch in range(3):
            self.assertTrue(bool(np.all(patch[..., ch].astype(np.int32) <= a + 2)))


class TestRimAudioModulationTask27(unittest.TestCase):
    """Task 27: snare / bass scalers and absolute-time ``t`` on rim patch."""

    def setUp(self) -> None:
        self.h = self.w = 48
        self.im = _solid_disk(self.h, self.w, cx=24.0, cy=24.0, radius=18.0)
        self.prep = compute_logo_rim_prep(self.im)

    def test_instant_envelope_in_declared_ranges(self) -> None:
        tune = RimAudioTuning(
            global_strength=1.0,
            glow_snare_max_delta=0.36,
            phase_snare_max_rad=0.55,
            inward_bass_max_delta=0.12,
        )
        m_lo = rim_modulation_instant(0.0, 0.0, tuning=tune)
        m_hi = rim_modulation_instant(1.0, 1.0, tuning=tune)
        self.assertEqual(m_lo.glow_strength_mul, 1.0)
        self.assertAlmostEqual(
            m_hi.glow_strength_mul, 1.0 + tune.glow_snare_max_delta, places=5
        )
        self.assertAlmostEqual(
            m_hi.phase_offset_rad, tune.phase_snare_max_rad, places=5
        )
        ib = tune.inward_bass_max_delta
        self.assertAlmostEqual(m_lo.inward_strength_mul, 1.0 - ib, places=5)
        self.assertAlmostEqual(m_hi.inward_strength_mul, 1.0 + ib, places=5)

    def test_snare_0_vs_1_changes_patch(self) -> None:
        """Measurable premult difference when only snare-glow analog differs (mid bass)."""
        mid_b = 0.5
        m0 = rim_modulation_instant(0.0, mid_b)
        m1 = rim_modulation_instant(1.0, mid_b)
        cfg = RimLightConfig(phase_hz=0.2, pad_px=8)
        a, _ = compute_logo_rim_light_patch(
            self.prep, t=0.11, config=cfg, audio_mod=m0
        )
        b, _ = compute_logo_rim_light_patch(
            self.prep, t=0.11, config=cfg, audio_mod=m1
        )
        diff = float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())
        self.assertGreater(
            diff, 0.2, "snare 0 vs 1 should change rim pixels, not a no-op"
        )

    def test_advance_stays_in_bounds_mock_tracks(self) -> None:
        """Constant / spike / sinusoidal track samples with modulation in range."""
        fps = 30.0
        n = 300
        dt = 1.0 / fps
        tuning = RimAudioTuning(
            global_strength=1.0,
            glow_snare_max_delta=0.36,
            phase_snare_max_rad=0.55,
            inward_bass_max_delta=0.12,
        )
        ib = float(tuning.inward_bass_max_delta)

        def run_track(values: np.ndarray) -> None:
            tr = PulseTrack(values=values.astype(np.float32), fps=fps)
            st = RimModulationState()
            for i in range(n):
                s = tr.value_at((i + 0.5) / fps)
                bass_val = 0.5
                m = advance_rim_audio_modulation(
                    st,
                    snare_env=s,
                    bass_env=bass_val,
                    dt_sec=dt,
                    tuning=tuning,
                )
                self.assertGreaterEqual(m.glow_strength_mul, 0.6)
                self.assertLessEqual(m.glow_strength_mul, 1.6)
                self.assertGreaterEqual(m.inward_strength_mul, 1.0 - ib - 0.01)
                self.assertLessEqual(m.inward_strength_mul, 1.0 + ib + 0.01)
                self.assertGreaterEqual(
                    m.phase_offset_rad, -tuning.phase_snare_max_rad * 1.01
                )
                self.assertLessEqual(
                    m.phase_offset_rad, tuning.phase_snare_max_rad * 1.01
                )

        run_track(np.full(n, 0.3, dtype=np.float32))
        sp = np.zeros(n, dtype=np.float32)
        sp[10] = 1.0
        run_track(sp)
        sines = 0.5 + 0.5 * np.sin(2.0 * np.pi * 2.0 * np.arange(n) / fps)
        run_track(sines.astype(np.float32))

    def test_audio_mod_none_matches_no_kwarg(self) -> None:
        cfg = RimLightConfig(phase_hz=0.0, pad_px=8)
        a, pa = compute_logo_rim_light_patch(self.prep, t=0.0, config=cfg)
        b, pb = compute_logo_rim_light_patch(
            self.prep, t=0.0, config=cfg, audio_mod=None
        )
        self.assertEqual(pa, pb)
        np.testing.assert_array_equal(a, b)

    def test_audio_mod_extreme_values_clamped_no_nan(self) -> None:
        """``glow_strength_mul`` / ``inward_strength_mul`` are clamped inside the patch."""
        baseline = compute_logo_rim_light_patch(
            self.prep,
            t=0.07,
            config=RimLightConfig(phase_hz=0.1, pad_px=8),
            audio_mod=RimAudioModulation(
                glow_strength_mul=1.0,
                phase_offset_rad=0.0,
                inward_strength_mul=1.0,
            ),
        )[0]
        hi = compute_logo_rim_light_patch(
            self.prep,
            t=0.07,
            config=RimLightConfig(phase_hz=0.1, pad_px=8),
            audio_mod=RimAudioModulation(
                glow_strength_mul=50.0,
                phase_offset_rad=0.0,
                inward_strength_mul=50.0,
            ),
        )[0]
        lo = compute_logo_rim_light_patch(
            self.prep,
            t=0.07,
            config=RimLightConfig(phase_hz=0.1, pad_px=8),
            audio_mod=RimAudioModulation(
                glow_strength_mul=-5.0,
                phase_offset_rad=0.0,
                inward_strength_mul=-5.0,
            ),
        )[0]
        self.assertTrue(np.all(np.isfinite(hi)))
        self.assertTrue(np.all(np.isfinite(lo)))
        self.assertGreater(int(hi[..., 3].sum()), int(baseline[..., 3].sum()))
        self.assertLess(int(lo[..., 3].sum()), int(baseline[..., 3].sum()))


class TestCompositorRimModulationStepper(unittest.TestCase):
    def test_reactive_flag_off_yields_no_stepper(self) -> None:
        from pipeline.compositor import CompositorConfig, _create_rim_modulation_stepper

        self.assertIsNone(
            _create_rim_modulation_stepper(
                CompositorConfig(logo_rim_audio_reactive=False), {}
            )
        )
