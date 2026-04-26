"""Tests for :mod:`pipeline.effects_timeline`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.effects_timeline import (
    EFFECTS_TIMELINE_JSON,
    EffectClip,
    EffectKind,
    EffectsTimeline,
    load,
    save,
    validate_effects_timeline,
    validate_settings_for_kind,
)


class TestEffectsTimeline(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            t0 = EffectsTimeline(
                clips=[
                    EffectClip(
                        id="a1",
                        kind=EffectKind.BEAM,
                        t_start=0.0,
                        duration_s=0.5,
                        settings={"strength": 0.7, "color_hex": "#ff00ff"},
                        auto_source=False,
                    )
                ],
                auto_reactivity_master=0.5,
            )
            path = save(tmp_path, t0)
            self.assertEqual(path, tmp_path / EFFECTS_TIMELINE_JSON)
            t1 = load(tmp_path)
            self.assertEqual(len(t1.clips), 1)
            c = t1.clips[0]
            self.assertEqual(c.id, "a1")
            self.assertIs(c.kind, EffectKind.BEAM)
            self.assertEqual(c.t_start, 0.0)
            self.assertEqual(c.duration_s, 0.5)
            self.assertEqual(
                c.settings, {"strength": 0.7, "color_hex": "#ff00ff"}
            )
            self.assertIs(c.auto_source, False)
            self.assertEqual(t1.auto_reactivity_master, 0.5)
            self.assertIs(t1.auto_enabled[EffectKind.BEAM], True)

    def test_unknown_settings_key_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown settings keys"):
            EffectClip(
                id="x",
                kind=EffectKind.SCREEN_SHAKE,
                t_start=0.0,
                duration_s=0.1,
                settings={"not_a_thing": 1.0},
            )

    def test_validate_settings_rejects_non_scalar(self) -> None:
        bad: dict[str, object] = {"strength": [1, 2]}
        with self.assertRaisesRegex(TypeError, "JSON-scalar"):
            validate_settings_for_kind(EffectKind.BEAM, bad)

    def test_load_missing_file_returns_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = load(Path(tmp))
            self.assertEqual(t.clips, [])
            self.assertEqual(t.auto_reactivity_master, 1.0)
            self.assertTrue(
                all(t.auto_enabled[k] is True for k in EffectKind)
            )

    def test_stale_json_tmp_does_not_corrupt_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = save(
                root,
                EffectsTimeline(
                    clips=[
                        EffectClip(
                            id="1",
                            kind=EffectKind.ZOOM_PUNCH,
                            t_start=0.0,
                            duration_s=0.1,
                            settings={},
                        )
                    ]
                ),
            )
            tmp_path = out.with_suffix(out.suffix + ".tmp")
            tmp_path.write_text("{NOT JSON", encoding="utf-8")
            t2 = load(root)
            self.assertEqual(len(t2.clips), 1)
            self.assertEqual(t2.clips[0].id, "1")

    def test_save_overwrites_broken_stale_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / EFFECTS_TIMELINE_JSON
            bad_tmp = out.with_suffix(out.suffix + ".tmp")
            bad_tmp.write_text("GARBAGE", encoding="utf-8")
            save(
                tmp_path,
                EffectsTimeline(
                    clips=[
                        EffectClip(
                            id="1",
                            kind=EffectKind.COLOR_INVERT,
                            t_start=0.0,
                            duration_s=0.1,
                            settings={"intensity": 0.2},
                        )
                    ]
                ),
            )
            with out.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(len(data["clips"]), 1)
            self.assertFalse(bad_tmp.is_file())

    def test_load_invalid_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / EFFECTS_TIMELINE_JSON
            p.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                load(tmp)

    def test_load_unsupported_schema_version_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / EFFECTS_TIMELINE_JSON
            p.write_text(
                '{"schema_version": 99, "auto_reactivity_master": 1.0, '
                '"auto_enabled": {}, "clips": []}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load(tmp)

    def test_validate_timeline_rejects_high_master(self) -> None:
        t = EffectsTimeline()
        t.auto_reactivity_master = 2.01
        with self.assertRaisesRegex(ValueError, "auto_reactivity_master"):
            validate_effects_timeline(t)

    def test_round_trip_multiple_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t0 = EffectsTimeline(
                clips=[
                    EffectClip(
                        id="s1",
                        kind=EffectKind.SCREEN_SHAKE,
                        t_start=0.0,
                        duration_s=0.1,
                        settings={"amplitude_px": 4.0, "frequency_hz": 2.0},
                    ),
                    EffectClip(
                        id="z1",
                        kind=EffectKind.ZOOM_PUNCH,
                        t_start=1.0,
                        duration_s=0.2,
                        settings={"peak_scale": 1.1},
                    ),
                ],
                auto_reactivity_master=1.25,
            )
            t0.auto_enabled[EffectKind.BEAM] = False
            save(Path(tmp), t0)
            t1 = load(Path(tmp))
            self.assertEqual(len(t1.clips), 2)
            self.assertIs(t1.auto_enabled[EffectKind.BEAM], False)
            self.assertEqual(t1.auto_reactivity_master, 1.25)
