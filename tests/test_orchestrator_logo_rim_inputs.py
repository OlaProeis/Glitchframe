"""Task 29: OrchestratorInputs rim fields map to CompositorConfig kwargs."""

from __future__ import annotations

import math
import unittest
from dataclasses import asdict, replace

from orchestrator import OrchestratorInputs, resolve_logo_rim_compositor_fields
from pipeline.logo_composite import LogoGlowMode


class TestOrchestratorLogoRimInputs(unittest.TestCase):
    def test_explicit_off_disables_rim(self) -> None:
        inp = OrchestratorInputs(
            song_hash="abc",
            logo_rim_mode="off",
            logo_rim_audio_reactive=False,
        )
        d = resolve_logo_rim_compositor_fields(inp)
        self.assertFalse(d["logo_rim_enabled"])
        self.assertIsNone(d["logo_rim_light_config"])
        self.assertEqual(d["logo_glow_mode"], LogoGlowMode.AUTO)
        self.assertFalse(d["logo_rim_audio_reactive"])

    def test_dataclass_defaults_traveling_rim_and_audio_reactive(self) -> None:
        inp = OrchestratorInputs(song_hash="abc")
        d = resolve_logo_rim_compositor_fields(inp)
        self.assertTrue(d["logo_rim_enabled"])
        self.assertIsNotNone(d["logo_rim_light_config"])
        self.assertTrue(d["logo_rim_audio_reactive"])

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

    def test_rim_enabled_applies_default_brightness_halo_and_comet_shape(self) -> None:
        """Rim mode with dataclass defaults gives a punchy, visibly moving comet."""
        inp = OrchestratorInputs(song_hash="abc", logo_rim_mode="rim")
        d = resolve_logo_rim_compositor_fields(inp)
        cfg = d["logo_rim_light_config"]
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertAlmostEqual(cfg.intensity, 3.0)
        self.assertAlmostEqual(cfg.halo_boost, 3.0)
        self.assertAlmostEqual(cfg.halo_spread_px, 22.0)
        self.assertEqual(cfg.waves, 1)
        self.assertAlmostEqual(cfg.wave_sharpness, 4.0)

    def test_brightness_above_one_lifts_halo_boost(self) -> None:
        inp = OrchestratorInputs(
            song_hash="x",
            logo_rim_mode="rim",
            logo_rim_brightness=2.5,
        )
        cfg = resolve_logo_rim_compositor_fields(inp)["logo_rim_light_config"]
        assert cfg is not None
        self.assertAlmostEqual(cfg.intensity, 2.5)
        self.assertAlmostEqual(cfg.halo_boost, 2.5)

    def test_brightness_below_one_clamps_halo_boost_floor(self) -> None:
        """Dim emissive still leaves the halo weight at engine default (1.0)."""
        inp = OrchestratorInputs(
            song_hash="x",
            logo_rim_mode="rim",
            logo_rim_brightness=0.4,
        )
        cfg = resolve_logo_rim_compositor_fields(inp)["logo_rim_light_config"]
        assert cfg is not None
        self.assertAlmostEqual(cfg.intensity, 0.4)
        self.assertAlmostEqual(cfg.halo_boost, 1.0)

    def test_halo_spread_px_is_clamped(self) -> None:
        too_small = OrchestratorInputs(
            song_hash="x", logo_rim_mode="rim", logo_rim_halo_spread_px=0.5
        )
        too_big = OrchestratorInputs(
            song_hash="x", logo_rim_mode="rim", logo_rim_halo_spread_px=9999.0
        )
        cfg_small = resolve_logo_rim_compositor_fields(too_small)["logo_rim_light_config"]
        cfg_big = resolve_logo_rim_compositor_fields(too_big)["logo_rim_light_config"]
        assert cfg_small is not None and cfg_big is not None
        self.assertAlmostEqual(cfg_small.halo_spread_px, 4.0)
        self.assertAlmostEqual(cfg_big.halo_spread_px, 64.0)

    def test_wave_shape_presets_map_to_expected_tuples(self) -> None:
        expected = {
            "comet": (1, 4.0),
            "twin": (2, 4.0),
            "lobes": (2, 2.0),
            "ring": (3, 1.5),
        }
        for key, (waves, sharp) in expected.items():
            inp = OrchestratorInputs(
                song_hash="x", logo_rim_mode="rim", logo_rim_wave_shape=key
            )
            cfg = resolve_logo_rim_compositor_fields(inp)["logo_rim_light_config"]
            assert cfg is not None
            self.assertEqual(cfg.waves, waves, f"waves mismatch for {key}")
            self.assertAlmostEqual(
                cfg.wave_sharpness, sharp, msg=f"sharpness mismatch for {key}"
            )

    def test_wave_shape_unknown_falls_back_to_comet(self) -> None:
        inp = OrchestratorInputs(
            song_hash="x",
            logo_rim_mode="rim",
            logo_rim_wave_shape="nonexistent-preset",
        )
        cfg = resolve_logo_rim_compositor_fields(inp)["logo_rim_light_config"]
        assert cfg is not None
        self.assertEqual(cfg.waves, 1)
        self.assertAlmostEqual(cfg.wave_sharpness, 4.0)

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
