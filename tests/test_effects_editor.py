"""Unit tests for :mod:`pipeline.effects_editor` backend handlers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.effects_editor import (
    DEDUPE_TOL_S,
    bake_auto_schedule,
    build_editor_html,
    build_ghost_events,
    load_editor_state,
    save_edited_timeline,
)
from pipeline.effects_timeline import (
    EFFECTS_TIMELINE_JSON,
    EffectClip,
    EffectKind,
    EffectsTimeline,
    load,
    save,
)


def _write_sine_wav(path: Path, seconds: float = 3.0, sr: int = 44_100) -> None:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(str(path), y.astype(np.float32), sr)


def _write_minimal_analysis(path: Path, song_hash: str) -> None:
    fps = 30.0
    n = 300
    rms_vals = (np.ones(n, dtype=np.float32) * 0.1).tolist()
    rms_vals[180] = 0.9
    spec_arr = np.ones((n, 8), dtype=np.float32) * 0.1
    # Inject a pair of low-band (kick) spikes and a cluster of high-band
    # (hat) spikes so the low/high transient peak-pickers have something to
    # snap to. Spacing respects the ghost-event minimum spacings in
    # :mod:`pipeline.effects_editor`.
    for i in (90, 140):
        spec_arr[i, 0] = 1.0
        spec_arr[i, 1] = 0.8
    for i in (60, 75, 90, 105, 120):
        spec_arr[i, 7] = 1.0
        spec_arr[i, 6] = 0.7
    spec = spec_arr.tolist()
    data = {
        "schema_version": 2,
        "song_hash": song_hash,
        "fps": fps,
        "rms": {"fps": fps, "values": rms_vals},
        "spectrum": {"fps": fps, "values": spec},
        "events": {
            "drops": [{"t": 5.0, "confidence": 0.9}],
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _new_cache(
    parent: Path, song_hash: str, *, with_analysis: bool = True, with_wav: bool = True
) -> Path:
    c = parent / song_hash
    c.mkdir(parents=True)
    if with_wav:
        _write_sine_wav(c / "analysis_mono.wav", seconds=10.0, sr=44_100)
    if with_analysis:
        _write_minimal_analysis(c / ANALYSIS_JSON_NAME, song_hash)
    return c


def _valid_payload(overrides: dict | None = None) -> dict:
    t = EffectsTimeline(
        clips=[
            EffectClip(
                id="c1",
                kind=EffectKind.ZOOM_PUNCH,
                t_start=1.0,
                duration_s=0.2,
                settings={},
            )
        ],
        auto_reactivity_master=1.0,
    )
    from pipeline.effects_timeline import _timeline_to_dict  # local import

    d = _timeline_to_dict(t)
    if overrides:
        d.update(overrides)
    return d


class TestEffectsEditor(unittest.TestCase):
    def test_load_editor_state_includes_ghosts_and_peaks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            c = _new_cache(root, "abc123de")
            st = load_editor_state(c)
            self.assertEqual(st["song_hash"], "abc123de")
            self.assertIn("peaks", st)
            self.assertGreater(len(st["peaks"]), 10)
            self.assertIn("duration", st)
            self.assertGreater(st["duration"], 1.0)
            self.assertIn("ghost_events", st)
            self.assertIn("clips", st)
            self.assertIn("auto_enabled", st)
            gk = [e for e in st["ghost_events"] if e.get("source") == "drop"]
            self.assertTrue(any(e.get("kind") == "ZOOM_PUNCH" for e in gk))
            self.assertIsInstance(build_ghost_events({}, song_hash="x"), list)

    def test_load_missing_cache_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_editor_state("/no/such/directory/xyz12345")

    def test_save_rejects_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "h1")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                save_edited_timeline(c, "{ not json")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                save_edited_timeline(c, "[]")
            with self.assertRaisesRegex(ValueError, "missing a 'clips'"):
                save_edited_timeline(c, '{"schema_version": 1, "auto_enabled": {}}')

    def test_save_rejects_wrong_song_hash_in_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "correcthash0")
            p = _valid_payload()
            p["song_hash"] = "wrong"
            with self.assertRaisesRegex(ValueError, "does not match cache"):
                save_edited_timeline(c, p)

    def test_save_rejects_song_hash_from_dir_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "realhash0")
            p = _valid_payload()
            p["song_hash"] = "realhash0"
            with self.assertRaisesRegex(ValueError, "song_hash_from_dir"):
                save_edited_timeline(
                    c, p, "otherhash0"
                )

    def test_save_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "roundtrip")
            p = _valid_payload()
            p["song_hash"] = "roundtrip"
            out = save_edited_timeline(c, p)
            self.assertEqual(out, c / EFFECTS_TIMELINE_JSON)
            t1 = load(c)
            self.assertEqual(len(t1.clips), 1)
            self.assertIs(t1.clips[0].kind, EffectKind.ZOOM_PUNCH)

    def test_save_omits_song_hash_in_payload(self) -> None:
        """save_edited_timeline only rejects song_hash when it disagrees; omitted is OK."""
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "nohash1")
            p = _valid_payload()
            p.pop("song_hash", None)
            save_edited_timeline(c, p)
            t1 = load(c)
            self.assertEqual(len(t1.clips), 1)

    def test_save_strips_ui_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "stripui0")
            p = _valid_payload()
            p["song_hash"] = "stripui0"
            p["peaks"] = [[0.0, 0.1]]
            p["ghost_events"] = [{"kind": "BEAM", "t": 0.0, "source": "x"}]
            save_edited_timeline(c, p)
            t1 = load(c)
            self.assertEqual(len(t1.clips), 1)

    def test_bake_dedupes_zoom_against_existing_clip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "bakededupe")
            et = EffectsTimeline(
                clips=[
                    EffectClip(
                        id="user",
                        kind=EffectKind.ZOOM_PUNCH,
                        t_start=5.0,
                        duration_s=0.3,
                        settings={},
                        auto_source=False,
                    )
                ],
            )
            # Only bake ZOOM (same time as analysis drop) — turn off other kinds
            et.auto_enabled[EffectKind.BEAM] = False
            et.auto_enabled[EffectKind.LOGO_GLITCH] = False
            et.auto_enabled[EffectKind.SCREEN_SHAKE] = False
            et.auto_enabled[EffectKind.CHROMATIC_ABERRATION] = False
            et.auto_enabled[EffectKind.ZOOM_PUNCH] = True
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            self.assertEqual(len(t2.clips), 1)
            self.assertEqual(t2.clips[0].id, "user")

    def test_bake_adds_zoom_when_no_collision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "bakeadd")
            et = EffectsTimeline(clips=[])
            et.auto_enabled[EffectKind.BEAM] = False
            et.auto_enabled[EffectKind.LOGO_GLITCH] = False
            et.auto_enabled[EffectKind.SCREEN_SHAKE] = False
            et.auto_enabled[EffectKind.CHROMATIC_ABERRATION] = False
            et.auto_enabled[EffectKind.ZOOM_PUNCH] = True
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            self.assertEqual(len(t2.clips), 1)
            self.assertIs(t2.clips[0].kind, EffectKind.ZOOM_PUNCH)
            self.assertTrue(t2.clips[0].auto_source)
            self.assertAlmostEqual(t2.clips[0].t_start, 5.0, places=3)

    def test_bake_adds_glitch_from_rms_impact_peaks(self) -> None:
        """Minimal analysis with a high RMS peak produces at least one baked glitch clip."""
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "bakegli")
            et = EffectsTimeline(clips=[])
            et.auto_enabled[EffectKind.BEAM] = False
            et.auto_enabled[EffectKind.ZOOM_PUNCH] = False
            et.auto_enabled[EffectKind.SCREEN_SHAKE] = False
            et.auto_enabled[EffectKind.CHROMATIC_ABERRATION] = False
            et.auto_enabled[EffectKind.LOGO_GLITCH] = True
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            glitches = [x for x in t2.clips if x.kind is EffectKind.LOGO_GLITCH]
            self.assertGreaterEqual(len(glitches), 1, "expected a baked LOGO_GLITCH")
            self.assertTrue(all(x.auto_source for x in glitches))

    def test_bake_adds_shake_from_kick_peaks(self) -> None:
        """Low-band (kick) peaks bake into SCREEN_SHAKE clips when enabled."""
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "bakeshake")
            et = EffectsTimeline(clips=[])
            for k in EffectKind:
                et.auto_enabled[k] = False
            et.auto_enabled[EffectKind.SCREEN_SHAKE] = True
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            shakes = [x for x in t2.clips if x.kind is EffectKind.SCREEN_SHAKE]
            self.assertGreaterEqual(
                len(shakes), 1, "expected at least one baked SCREEN_SHAKE"
            )
            for clip in shakes:
                self.assertTrue(clip.auto_source)
                self.assertIn("amplitude_px", clip.settings)
                self.assertIn("frequency_hz", clip.settings)
                self.assertGreater(float(clip.settings["amplitude_px"]), 0.0)

    def test_bake_adds_chroma_from_hat_peaks(self) -> None:
        """High-band (hat) peaks bake into CHROMATIC_ABERRATION clips when enabled."""
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "bakechroma")
            et = EffectsTimeline(clips=[])
            for k in EffectKind:
                et.auto_enabled[k] = False
            et.auto_enabled[EffectKind.CHROMATIC_ABERRATION] = True
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            chromas = [
                x for x in t2.clips if x.kind is EffectKind.CHROMATIC_ABERRATION
            ]
            self.assertGreaterEqual(
                len(chromas), 1, "expected at least one baked CHROMATIC_ABERRATION"
            )
            for clip in chromas:
                self.assertTrue(clip.auto_source)
                self.assertIn("shift_px", clip.settings)
                self.assertGreater(float(clip.settings["shift_px"]), 0.0)

    def test_bake_auto_disabled_skips_new_kinds(self) -> None:
        """Disabling auto for SCREEN_SHAKE / CHROMATIC_ABERRATION keeps them out."""
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "bakeoffnew")
            et = EffectsTimeline(clips=[])
            for k in EffectKind:
                et.auto_enabled[k] = False
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            self.assertEqual(
                [x.kind for x in t2.clips if x.auto_source],
                [],
                "no auto clips should be baked when every kind is off",
            )

    def test_ghost_events_include_kick_and_hat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "ghostnew0")
            st = load_editor_state(c)
            kinds = {e.get("kind") for e in st["ghost_events"]}
            self.assertIn("SCREEN_SHAKE", kinds)
            self.assertIn("CHROMATIC_ABERRATION", kinds)

    def test_dedupe_threshold_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "edgecase")
            # Existing clip 5.0, drop 5.0+15ms = still within 20ms? 0.015 < 0.02 → dedupe
            et = EffectsTimeline(
                clips=[
                    EffectClip(
                        id="edge",
                        kind=EffectKind.ZOOM_PUNCH,
                        t_start=5.0 - DEDUPE_TOL_S * 0.5,
                        duration_s=0.2,
                        settings={},
                    )
                ],
            )
            et.auto_enabled[EffectKind.BEAM] = False
            et.auto_enabled[EffectKind.LOGO_GLITCH] = False
            et.auto_enabled[EffectKind.SCREEN_SHAKE] = False
            et.auto_enabled[EffectKind.CHROMATIC_ABERRATION] = False
            et.auto_enabled[EffectKind.ZOOM_PUNCH] = True
            save(c, et)
            bake_auto_schedule(c)
            t2 = load(c)
            self.assertEqual(len(t2.clips), 1)

    def test_bake_requires_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "noana", with_analysis=False)
            save(
                c,
                EffectsTimeline(),
            )
            with self.assertRaises(FileNotFoundError):
                bake_auto_schedule(c)

    def test_build_editor_html_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "uihash01")
            st = load_editor_state(c)
            html = build_editor_html(
                st, audio_url="#", container_id="eff-timeline"
            )
            self.assertIsInstance(html, str)
            self.assertGreater(len(html), 1000)
            # Container id + inert script tag (the pattern that lets the JS
            # re-run on every Gradio innerHTML replace).
            self.assertIn('id="eff-timeline"', html)
            self.assertIn('type="text/plain"', html)
            # Default state var name is exposed on window so the Save
            # handler can grab it.
            self.assertIn("_glitchframe_effects_state", html)
            # All seven EffectKind rows + toolbar buttons must be present —
            # the UI contract is "seven rows regardless of renderer status".
            for kind in (
                "BEAM",
                "LOGO_GLITCH",
                "SCREEN_SHAKE",
                "COLOR_INVERT",
                "CHROMATIC_ABERRATION",
                "SCANLINE_TEAR",
                "ZOOM_PUNCH",
            ):
                self.assertIn(f'data-mv-fx-row="{kind}"', html)
                self.assertIn(f'data-mv-fx-add="{kind}"', html)
                self.assertIn(f'data-mv-fx-auto="{kind}"', html)
            # Master reactivity slider is the top-bar knob required by the PRD.
            self.assertIn("data-mv-fx-master", html)

    def test_build_editor_html_custom_state_var(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = _new_cache(Path(td), "uihash02")
            st = load_editor_state(c)
            html = build_editor_html(
                st,
                audio_url="#",
                container_id="x",
                state_js_var="my_custom_var",
            )
            self.assertIn("my_custom_var", html)

    def test_load_without_wav_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = Path(td) / "nowav"
            c.mkdir()
            p = c / ANALYSIS_JSON_NAME
            p.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_editor_state(c)


if __name__ == "__main__":
    unittest.main()
