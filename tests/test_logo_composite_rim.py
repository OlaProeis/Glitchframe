"""Regression and rim wiring for :mod:`pipeline.logo_composite` (task 28)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.logo_composite import (
    LogoGlowMode,
    build_classic_neon_glow_patch,
    composite_logo_onto_frame,
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


if __name__ == "__main__":
    unittest.main()
