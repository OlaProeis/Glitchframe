"""Regression and rim wiring for :mod:`pipeline.logo_composite` (task 28)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.logo_composite import (
    LogoGlowMode,
    _rgb_glitch_logo_rgba,
    build_classic_neon_glow_patch,
    composite_logo_onto_frame,
    prepare_logo_rgba,
)
from pipeline.compositor import CompositorConfig, _effective_rim_light_config
from pipeline.logo_rim_lights import RimLightConfig, compute_logo_rim_prep


def _small_disc_rgba(h: int = 48, w: int = 48) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    r = np.hypot(yy - cy, xx - cx)
    a = np.clip(18.0 - r, 0.0, 1.0)
    a_u8 = (a * 255.0).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = a_u8
    rgba[:, :, :3] = 240
    return rgba


class TestLogoCompositeRim(unittest.TestCase):
    def test_effective_rim_config_respects_enabled_flag(self) -> None:
        analysis = {"song_hash": "testhash"}
        self.assertIsNone(
            _effective_rim_light_config(CompositorConfig(logo_rim_enabled=False), analysis)
        )
        cfg_on = CompositorConfig(
            logo_rim_enabled=True,
            base_color="#FFFFFF",
            shadow_color="#112233",
        )
        rc = _effective_rim_light_config(cfg_on, analysis)
        self.assertIsNotNone(rc)
        assert rc is not None
        self.assertEqual(rc.song_hash, "testhash")

    def test_default_rim_off_matches_classic_only(self) -> None:
        frame = np.full((90, 120, 3), 30, dtype=np.uint8)
        logo = _small_disc_rgba()
        kwargs = dict(
            glow_amount=0.55,
            glow_rgb=(200, 80, 255),
            glow_blur_radius=4.0,
            glow_pad_px=20,
            t_sec=0.25,
        )
        baseline = composite_logo_onto_frame(
            frame.copy(), logo, "center", 100.0, **kwargs
        )
        explicit_off = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            rim_light_config=None,
            **kwargs,
        )
        np.testing.assert_array_equal(baseline, explicit_off)

    def test_rim_on_changes_frame_stats(self) -> None:
        frame = np.full((120, 160, 3), 22, dtype=np.uint8)
        logo = _small_disc_rgba(56, 56)
        base = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            88.0,
            glow_amount=0.0,
            t_sec=0.1,
        )
        rim_cfg = RimLightConfig(
            intensity=1.0,
            opacity_pct=100.0,
            phase_hz=0.0,
            pad_px=12,
            halo_spread_px=10.0,
            blur_px=2.0,
        )
        lit = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            88.0,
            glow_amount=0.0,
            t_sec=0.1,
            rim_light_config=rim_cfg,
        )
        diff = np.abs(lit.astype(np.int16) - base.astype(np.int16)).sum()
        self.assertGreater(
            diff,
            800,
            "rim patch should alter pixels meaningfully vs rim disabled",
        )

    def test_classic_mode_ignores_rim(self) -> None:
        frame = np.full((100, 130, 3), 40, dtype=np.uint8)
        logo = _small_disc_rgba()
        rim_cfg = RimLightConfig(
            intensity=1.0,
            opacity_pct=100.0,
            phase_hz=0.0,
            pad_px=10,
        )
        only_classic = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            glow_amount=0.4,
            glow_rgb=(255, 100, 200),
            rim_light_config=rim_cfg,
            t_sec=0.0,
            logo_glow_mode=LogoGlowMode.CLASSIC,
        )
        no_rim = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            glow_amount=0.4,
            glow_rgb=(255, 100, 200),
            t_sec=0.0,
        )
        np.testing.assert_array_equal(only_classic, no_rim)

    def test_rim_only_skips_neon(self) -> None:
        frame = np.full((100, 130, 3), 35, dtype=np.uint8)
        logo = _small_disc_rgba()
        rim_cfg = RimLightConfig(
            intensity=1.0,
            opacity_pct=100.0,
            phase_hz=0.0,
            pad_px=10,
        )
        rim_only = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            glow_amount=0.9,
            glow_rgb=(255, 0, 255),
            rim_light_config=rim_cfg,
            t_sec=0.0,
            logo_glow_mode=LogoGlowMode.RIM_ONLY,
        )
        rim_alone = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            glow_amount=0.0,
            rim_light_config=rim_cfg,
            t_sec=0.0,
        )
        np.testing.assert_array_equal(rim_only, rim_alone)

    def test_cached_prep_matches_inline(self) -> None:
        frame = np.full((96, 96, 3), 15, dtype=np.uint8)
        logo = _small_disc_rgba(40, 40)
        prep = compute_logo_rim_prep(logo)
        cfg = RimLightConfig(phase_hz=0.0, pad_px=8, intensity=0.9)
        a = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            glow_amount=0.0,
            rim_light_config=cfg,
            t_sec=0.05,
            logo_rim_prep=prep,
        )
        b = composite_logo_onto_frame(
            frame.copy(),
            logo,
            "center",
            100.0,
            glow_amount=0.0,
            rim_light_config=cfg,
            t_sec=0.05,
        )
        np.testing.assert_array_equal(a, b)

    def test_build_classic_neon_alias(self) -> None:
        logo = _small_disc_rgba(32, 32)
        a = build_classic_neon_glow_patch(
            logo,
            glow_rgb=(255, 0, 0),
            amount=0.8,
            blur_radius=3.0,
            pad=8,
            opacity_pct=100.0,
        )
        self.assertIsNotNone(a)
        patch, pad = a
        self.assertEqual(patch.shape[2], 4)
        self.assertGreater(int(np.max(patch[:, :, 3])), 10)


class TestGlitchTiltStability(unittest.TestCase):
    """Regression: glitch tilt direction must be stable per-impact.

    Prior to this fix, the tilt was seeded from ``seed`` (which changed every
    frame via ``glitch_seed_for_time``), producing a random ±25° flip on each
    of the 5--8 frames of a single 0.2 s impact -- the logo visibly thrashed.
    The contract is now: one **stable** ``tilt_seed`` per impact event makes
    every frame of that event tilt in the same direction so the impact reads
    as a crisp hit, not shake noise.
    """

    def _asymmetric_logo(self) -> np.ndarray:
        # An asymmetric mark (bright top-left quadrant only) so the alpha
        # centre-of-mass moves predictably under ± rotation -- that lets us
        # distinguish a left-tilt from a right-tilt without eyeballing it.
        logo = np.zeros((48, 48, 4), dtype=np.uint8)
        logo[4:20, 4:20, :3] = 230
        logo[4:20, 4:20, 3] = 255
        return logo

    @staticmethod
    def _alpha_center_x(rgba: np.ndarray) -> float:
        a = rgba[:, :, 3].astype(np.float64)
        if a.sum() <= 0:
            return float(rgba.shape[1]) * 0.5
        xs = np.arange(rgba.shape[1], dtype=np.float64)
        return float((a.sum(axis=0) * xs).sum() / a.sum())

    def test_same_tilt_seed_yields_same_tilt_direction(self) -> None:
        logo = self._asymmetric_logo()
        # Same ``tilt_seed`` must mean the same sign regardless of the
        # per-frame RGB/tear seed -- i.e. every frame of one glitch event
        # tilts the same way.
        a = _rgb_glitch_logo_rgba(logo, 0.9, seed=111, tilt_seed=42)
        b = _rgb_glitch_logo_rgba(logo, 0.9, seed=999, tilt_seed=42)
        self.assertEqual(a.shape, b.shape)
        # Centres of mass align (same rotation direction). Sub-pixel interp
        # and different tear bytes give a small tolerance.
        self.assertAlmostEqual(
            self._alpha_center_x(a), self._alpha_center_x(b), delta=1.0
        )

    def test_opposite_tilt_seeds_flip_direction(self) -> None:
        logo = self._asymmetric_logo()
        even = _rgb_glitch_logo_rgba(logo, 0.9, seed=111, tilt_seed=2)
        odd = _rgb_glitch_logo_rgba(logo, 0.9, seed=111, tilt_seed=3)
        self.assertEqual(even.shape, odd.shape)
        # Opposite directions => alpha distribution rotated the opposite
        # way. For an off-centre mass, the (cx, cy) offsets swap in a
        # predictable pattern (one picks up the rotation, the other picks
        # up its mirror). The two outputs must therefore differ materially
        # in their alpha layout -- a strict "alpha equal" would prove the
        # direction flag had no effect.
        self.assertFalse(np.array_equal(even[:, :, 3], odd[:, :, 3]))
        # And their alpha centre-of-mass x must differ by a real amount.
        self.assertGreater(
            abs(self._alpha_center_x(even) - self._alpha_center_x(odd)),
            2.0,
        )

    def test_zero_amount_is_noop(self) -> None:
        logo = self._asymmetric_logo()
        out = _rgb_glitch_logo_rgba(logo, 0.0, seed=1, tilt_seed=1)
        np.testing.assert_array_equal(out, logo)


class TestPrepareLogoRgbaMaxSizePct(unittest.TestCase):
    """``prepare_logo_rgba(max_size_pct=...)`` caps the logo relative to the
    shorter frame edge so the size slider behaves consistently across
    resolutions (720p / 1080p / 4K)."""

    @staticmethod
    def _square_rgba(side: int) -> np.ndarray:
        rgba = np.zeros((side, side, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[..., :3] = 128
        return rgba

    def test_none_preserves_legacy_behaviour(self) -> None:
        logo = self._square_rgba(200)
        out = prepare_logo_rgba(logo, frame_h=1080, frame_w=1920, max_size_pct=None)
        self.assertEqual(out.shape, (200, 200, 4))

    def test_caps_large_logo_to_pct_of_short_edge(self) -> None:
        # 2000 px square logo on 1080p: 30 % of 1080 = 324 px.
        logo = self._square_rgba(2000)
        out = prepare_logo_rgba(logo, frame_h=1080, frame_w=1920, max_size_pct=30.0)
        self.assertEqual(out.shape[0], 324)
        self.assertEqual(out.shape[1], 324)

    def test_same_pct_same_relative_size_on_4k(self) -> None:
        # Identical logo on 4K should land at 30 % of 2160 = 648 px; the
        # visual fraction of the frame matches 1080p within 1 px rounding.
        logo = self._square_rgba(4000)
        out_1080 = prepare_logo_rgba(logo, 1080, 1920, max_size_pct=30.0)
        out_4k = prepare_logo_rgba(logo, 2160, 3840, max_size_pct=30.0)
        self.assertEqual(out_1080.shape[0], 324)
        self.assertEqual(out_4k.shape[0], 648)

    def test_small_logo_not_upscaled(self) -> None:
        # 30 % of 1080 is 324 px; a 100 px logo must NOT be upscaled.
        logo = self._square_rgba(100)
        out = prepare_logo_rgba(logo, 1080, 1920, max_size_pct=30.0)
        self.assertEqual(out.shape[:2], (100, 100))

    def test_aspect_ratio_preserved_for_rectangular(self) -> None:
        # 1600 x 800 logo → longest edge 1600 → capped at 324 on 1080p.
        # Aspect = 2:1 must be kept within 1 px rounding tolerance.
        logo = np.zeros((800, 1600, 4), dtype=np.uint8)
        logo[..., 3] = 255
        out = prepare_logo_rgba(logo, 1080, 1920, max_size_pct=30.0)
        self.assertEqual(out.shape[1], 324)
        self.assertAlmostEqual(out.shape[0], 162, delta=1)

    def test_nonpositive_pct_falls_back_to_frame_fit(self) -> None:
        # Oversized logo with the cap disabled should still be shrunk to fit
        # inside the frame (legacy guard).
        logo = self._square_rgba(3000)
        out = prepare_logo_rgba(logo, 1080, 1920, max_size_pct=0.0)
        self.assertEqual(out.shape[:2], (1080, 1080))


if __name__ == "__main__":
    unittest.main()
