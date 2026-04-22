"""Task 29: OrchestratorInputs rim fields map to CompositorConfig kwargs."""

from __future__ import annotations

import math
import unittest
from dataclasses import asdict, replace

from orchestrator import OrchestratorInputs, resolve_logo_rim_compositor_fields
from pipeline.logo_composite import LogoGlowMode


class TestOrchestratorLogoRimInputs(unittest.TestCase):
    def test_default_off_disables_rim(self) -> None:
        inp = OrchestratorInputs(song_hash="abc")
        d = resolve_logo_rim_compositor_fields(inp)
        self.assertFalse(d["logo_rim_enabled"])
        self.assertIsNone(d["logo_rim_light_config"])
        self.assertEqual(d["logo_glow_mode"], LogoGlowMode.AUTO)
        self.assertFalse(d["logo_rim_audio_reactive"])

    def test_classic_mode_forces_classic_glow(self) -> None:
        inp = OrchestratorInputs(song_hash="abc", logo_rim_mode="classic")
        d = resolve_logo_rim_compositor_fields(inp)
        self.assertFalse(d["logo_rim_enabled"])
        self.assertEqual(d["logo_glow_mode"], LogoGlowMode.CLASSIC)

    def test_rim_mode_builds_config_ccw(self) -> None:
        inp = OrchestratorInputs(
            song_hash="abc",
            logo_rim_mode="rim",
            logo_rim_travel_speed=0.4,
            logo_rim_color_spread_deg=90.0,
            logo_rim_inward_mix=0.35,
            logo_rim_direction="ccw",
            logo_rim_audio_reactive=True,
            logo_rim_sync_snare=False,
            logo_rim_sync_bass=True,
            logo_rim_mod_strength=1.5,
        )
        d = resolve_logo_rim_compositor_fields(inp)
        self.assertTrue(d["logo_rim_enabled"])
        self.assertEqual(d["logo_glow_mode"], LogoGlowMode.AUTO)
        cfg = d["logo_rim_light_config"]
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertAlmostEqual(cfg.phase_hz, -0.4)
        self.assertAlmostEqual(cfg.color_spread_rad, 90.0 * math.pi / 180.0, places=5)
        self.assertEqual(cfg.rim_color_layers, 2)
        self.assertAlmostEqual(cfg.inward_mix, 0.35)
        self.assertTrue(d["logo_rim_audio_reactive"])
        self.assertFalse(d["logo_rim_sync_snare"])
        self.assertTrue(d["logo_rim_sync_bass"])
        self.assertAlmostEqual(d["logo_rim_mod_strength"], 1.5)

    def test_zero_spread_single_layer(self) -> None:
        inp = OrchestratorInputs(
            song_hash="x",
            logo_rim_mode="rim",
            logo_rim_color_spread_deg=0.0,
        )
        d = resolve_logo_rim_compositor_fields(inp)
        cfg = d["logo_rim_light_config"]
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.rim_color_layers, 1)

    def test_asdict_round_trip_replace(self) -> None:
        base = OrchestratorInputs(song_hash="h")
        clone = replace(
            base,
            logo_rim_mode="rim",
            logo_rim_travel_speed=0.5,
            logo_rim_mod_strength=0.75,
        )
        roundtrip = OrchestratorInputs(**asdict(clone))
        self.assertEqual(asdict(clone), asdict(roundtrip))


if __name__ == "__main__":
    unittest.main()
