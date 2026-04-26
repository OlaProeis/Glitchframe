"""Orchestrator wires song-cache effects timeline + merged auto-reactivity master."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orchestrator import OrchestratorInputs, _effects_compositor_config
from pipeline.effects_timeline import (
    EFFECTS_TIMELINE_JSON,
    SCHEMA_VERSION,
    EffectKind,
)


class TestEffectsCompositorConfig(unittest.TestCase):
    def test_disabled_matches_legacy(self) -> None:
        inp = OrchestratorInputs(
            song_hash="notused",
            effects_timeline_enabled=False,
            auto_reactivity_master=0.25,
        )
        tl, master = _effects_compositor_config(inp, Path("/no/read"))
        self.assertIsNone(tl)
        self.assertEqual(master, 1.0)

    def test_enabled_missing_file_is_default_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            inp = OrchestratorInputs(
                song_hash="notused",
                effects_timeline_enabled=True,
                auto_reactivity_master=0.8,
            )
            tl, master = _effects_compositor_config(inp, cache)
        self.assertIsNotNone(tl)
        assert tl is not None
        self.assertEqual(tl.clips, [])
        self.assertAlmostEqual(master, 0.8 * 1.0)

    def test_enabled_multiplies_disk_and_input_master(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "auto_reactivity_master": 0.5,
            "auto_enabled": {k.name: True for k in EffectKind},
            "clips": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / EFFECTS_TIMELINE_JSON).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            inp = OrchestratorInputs(
                song_hash="notused",
                effects_timeline_enabled=True,
                auto_reactivity_master=2.0,
            )
            tl, master = _effects_compositor_config(inp, cache)
        self.assertIsNotNone(tl)
        self.assertAlmostEqual(master, 1.0)

    def test_negative_input_clamps_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            inp = OrchestratorInputs(
                song_hash="notused",
                effects_timeline_enabled=True,
                auto_reactivity_master=-3.0,
            )
            _tl, master = _effects_compositor_config(inp, cache)
        self.assertAlmostEqual(master, 0.0)


if __name__ == "__main__":
    unittest.main()
