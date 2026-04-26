"""Sanity tests for the ``nebula_flow`` pilot shader + ``cosmic-flow`` preset.

Covers the three things the allowlist + preset validator contract promises
without needing a live GL context:

1. ``nebula_flow`` is in the :data:`BUILTIN_SHADERS` allowlist.
2. ``assets/shaders/nebula_flow.frag`` resolves through the same helper
   the GL-backed :class:`ReactiveShader` uses.
3. ``presets/cosmic-flow.yaml`` loads through :func:`load_preset_registry`
   without raising and points at ``nebula_flow``.

A GPU smoke render is intentionally out of scope here — ``compileall`` and
this unittest run hermetically; the live-GLSL link test is manual and
documented in ``docs/technical/reactive-shader-layer.md``.
"""

from __future__ import annotations

import unittest

from config import PRESETS_DIR, load_preset_registry
from pipeline.builtin_shaders import BUILTIN_SHADERS
from pipeline.reactive_shader import resolve_builtin_shader_stem


class TestNebulaFlowAllowlist(unittest.TestCase):
    def test_nebula_flow_in_builtin_shaders(self) -> None:
        self.assertIn("nebula_flow", BUILTIN_SHADERS)

    def test_nebula_drift_still_in_allowlist(self) -> None:
        # nebula_flow is an *addition* — the A/B peer must not disappear.
        self.assertIn("nebula_drift", BUILTIN_SHADERS)

    def test_nebula_flow_frag_resolves(self) -> None:
        # Raises ValueError / FileNotFoundError if allowlisted but missing.
        self.assertEqual(resolve_builtin_shader_stem("nebula_flow"), "nebula_flow")


class TestTunnelFlightShader(unittest.TestCase):
    def test_tunnel_flight_in_builtin_shaders(self) -> None:
        self.assertIn("tunnel_flight", BUILTIN_SHADERS)

    def test_voidcat_laser_in_builtin_shaders(self) -> None:
        self.assertIn("voidcat_laser", BUILTIN_SHADERS)
        self.assertEqual(resolve_builtin_shader_stem("voidcat_laser"), "voidcat_laser")

    def test_void_ascii_bg_in_builtin_shaders(self) -> None:
        self.assertIn("void_ascii_bg", BUILTIN_SHADERS)

    def test_spectral_milkdrop_in_builtin_shaders(self) -> None:
        self.assertIn("spectral_milkdrop", BUILTIN_SHADERS)
        self.assertEqual(resolve_builtin_shader_stem("void_ascii_bg"), "void_ascii_bg")

    def test_tunnel_flight_frag_resolves(self) -> None:
        self.assertEqual(resolve_builtin_shader_stem("tunnel_flight"), "tunnel_flight")


class TestCyberTunnelPreset(unittest.TestCase):
    def test_cyber_tunnel_loads_and_targets_tunnel_flight(self) -> None:
        registry = load_preset_registry()
        self.assertIn("cyber-tunnel", registry)
        self.assertEqual(registry["cyber-tunnel"]["shader"], "tunnel_flight")

    def test_cyber_tunnel_yaml_file_exists(self) -> None:
        self.assertTrue((PRESETS_DIR / "cyber-tunnel.yaml").is_file())


class TestCosmicFlowPreset(unittest.TestCase):
    def test_cosmic_flow_loads_and_targets_nebula_flow(self) -> None:
        registry = load_preset_registry()
        self.assertIn("cosmic-flow", registry)
        self.assertEqual(registry["cosmic-flow"]["shader"], "nebula_flow")

    def test_cosmic_flow_palette(self) -> None:
        expected = [
            "#120458",
            "#3A0CA3",
            "#7B2CBF",
            "#4CC9F0",
            "#F72585",
        ]
        registry = load_preset_registry()
        self.assertIn("cosmic-flow", registry)
        self.assertEqual(registry["cosmic-flow"]["colors"], expected)

    def test_cosmic_flow_yaml_file_exists(self) -> None:
        self.assertTrue((PRESETS_DIR / "cosmic-flow.yaml").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
