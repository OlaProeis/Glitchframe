"""Unit tests for the visual lyrics-timeline editor backend.

These focus on the pure-Python helpers that don't need a live browser:
peaks computation, JSON round-tripping with the ``manually_edited`` flag,
revert behaviour, and HTML smoke-checks.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.lyrics_editor import (
    build_editor_html,
    compute_peaks,
    load_editor_state,
    revert_manual_edits,
    save_edited_alignment,
)


def _write_sine_wav(path: Path, seconds: float = 2.0, sr: int = 16000) -> None:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(path), y.astype(np.float32), sr)


def _write_aligned_json(
    path: Path,
    song_hash: str,
    manual: bool = False,
    *,
    include_whisper: bool = False,
) -> None:
    payload = {
        "schema_version": 3,
        "song_hash": song_hash,
        "model": "large-v3",
        "language": "en",
        "vocals_wav": "vocals.wav",
        "lyrics_sha256": "deadbeef",
        "manually_edited": manual,
        "lines": ["hello world"],
        "words": [
            {"word": "hello", "line_idx": 0, "t_start": 0.10, "t_end": 0.40, "score": 0.92},
            {"word": "world", "line_idx": 0, "t_start": 0.50, "t_end": 0.80, "score": 0.75},
        ],
    }
    if include_whisper:
        payload["whisper_words"] = [
            {"word": "hello", "t_start": 0.12, "t_end": 0.39, "score": 0.9},
            {"word": "whirled", "t_start": 0.51, "t_end": 0.82, "score": 0.6},
        ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestComputePeaks(unittest.TestCase):
    def test_returns_requested_width_for_long_signal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "v.wav"
            _write_sine_wav(wav, seconds=2.0, sr=16000)
            peaks, sr, dur = compute_peaks(wav, target_width=400)
            self.assertEqual(sr, 16000)
            self.assertAlmostEqual(dur, 2.0, places=1)
            self.assertEqual(len(peaks), 400)
            # (min, max) sanity check: min <= max.
            for mn, mx in peaks:
                self.assertLessEqual(mn, mx + 1e-6)
            # Normalised into roughly [-1, 1].
            abs_max = max(abs(mn) for mn, _ in peaks)
            self.assertLessEqual(abs_max, 1.01)

    def test_short_signal_caps_width(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "v.wav"
            _write_sine_wav(wav, seconds=0.05, sr=16000)
            peaks, _sr, _dur = compute_peaks(wav, target_width=1000)
            # 0.05 s * 16000 = 800 samples; with min 4 samples/bucket → ≤200.
            self.assertLessEqual(len(peaks), 200)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            compute_peaks(Path("/this/path/does/not/exist.wav"))


class TestLoadEditorState(unittest.TestCase):
    def test_round_trip_state_from_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "abc123"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=1.0, sr=16000)
            _write_aligned_json(cache / "lyrics.aligned.json", "abc123")
            state = load_editor_state(cache, target_peak_width=200)
            self.assertEqual(state.song_hash, "abc123")
            self.assertEqual(len(state.words), 2)
            self.assertEqual(state.words[0]["word"], "hello")
            self.assertAlmostEqual(state.words[0]["score"], 0.92, places=6)
            self.assertFalse(state.manually_edited)
            self.assertEqual(state.lines, ["hello world"])

    def test_missing_aligned_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "x"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav")
            with self.assertRaises(FileNotFoundError):
                load_editor_state(cache)

    def test_missing_vocals_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "x"
            cache.mkdir()
            _write_aligned_json(cache / "lyrics.aligned.json", "x")
            with self.assertRaises(FileNotFoundError):
                load_editor_state(cache)

    def test_whisper_words_are_loaded_when_present(self) -> None:
        """Caches produced by the current aligner include ``whisper_words``
        — the CTC-aligned per-word timings for *what whisper heard*, used
        by the editor to show ghost text above each user word so the
        user can align manually without listening to the track."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=1.0, sr=16000)
            _write_aligned_json(
                cache / "lyrics.aligned.json", "h", include_whisper=True
            )
            state = load_editor_state(cache, target_peak_width=80)
            self.assertEqual(len(state.whisper_words), 2)
            self.assertEqual(state.whisper_words[0]["word"], "hello")
            self.assertAlmostEqual(state.whisper_words[0]["t_start"], 0.12)

    def test_whisper_words_missing_key_degrades_to_empty_list(self) -> None:
        """Older pre-whisper-words caches must still load cleanly — the
        editor just shows zero ghost labels until the user re-aligns."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=1.0, sr=16000)
            _write_aligned_json(cache / "lyrics.aligned.json", "h")
            state = load_editor_state(cache, target_peak_width=80)
            self.assertEqual(state.whisper_words, [])


class TestSaveEditedAlignment(unittest.TestCase):
    def test_save_writes_manually_edited_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav")
            _write_aligned_json(cache / "lyrics.aligned.json", "h")
            edited = json.dumps(
                {
                    "song_hash": "h",
                    "lines": ["hello world"],
                    "words": [
                        {"word": "hello", "line_idx": 0, "t_start": 0.2, "t_end": 0.5, "score": 0.9},
                        {"word": "world", "line_idx": 0, "t_start": 0.6, "t_end": 0.9},
                    ],
                }
            )
            path = save_edited_alignment(cache, edited, lyrics_text_snapshot="hello world")
            self.assertTrue(path.is_file())
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertTrue(data["manually_edited"])
            self.assertEqual(data["schema_version"], 3)
            self.assertEqual(len(data["words"]), 2)
            self.assertAlmostEqual(data["words"][0]["t_start"], 0.2, places=6)
            # Word 2 had no score → score key must be absent from the saved entry.
            self.assertNotIn("score", data["words"][1])
            # manual_lyrics_sha256 is populated from the snapshot.
            self.assertIn("manual_lyrics_sha256", data)
            self.assertTrue(data["manual_lyrics_sha256"])

    def test_save_preserves_whisper_words_from_previous_cache(self) -> None:
        """The editor never mutates whisper's transcription — it's
        display-only data. But a naive save that rewrites the JSON from
        the payload alone would silently drop ``whisper_words`` on the
        first manual edit, which would make the ghost-text overlay
        vanish after the user corrects even one word. Guard that."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav")
            _write_aligned_json(
                cache / "lyrics.aligned.json", "h", include_whisper=True
            )
            edited = json.dumps(
                {
                    "song_hash": "h",
                    "lines": ["hello world"],
                    "words": [
                        {"word": "hello", "line_idx": 0, "t_start": 0.2, "t_end": 0.5},
                        {"word": "world", "line_idx": 0, "t_start": 0.6, "t_end": 0.9},
                    ],
                }
            )
            path = save_edited_alignment(cache, edited)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("whisper_words", data)
            self.assertEqual(len(data["whisper_words"]), 2)
            self.assertEqual(data["whisper_words"][1]["word"], "whirled")

    def test_save_rejects_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            with self.assertRaises(ValueError):
                save_edited_alignment(cache, "not json")

    def test_save_rejects_missing_words(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            with self.assertRaises(ValueError):
                save_edited_alignment(cache, json.dumps({"lines": []}))

    def test_save_clamps_inverted_times(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav")
            _write_aligned_json(cache / "lyrics.aligned.json", "h")
            edited = json.dumps(
                {
                    "lines": ["x"],
                    "words": [{"word": "x", "line_idx": 0, "t_start": 2.0, "t_end": 1.5}],
                }
            )
            path = save_edited_alignment(cache, edited)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["words"][0]["t_start"], 2.0)
            self.assertEqual(data["words"][0]["t_end"], 2.0)


class TestRevertManualEdits(unittest.TestCase):
    def test_revert_clears_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_aligned_json(cache / "lyrics.aligned.json", "h", manual=True)
            path = revert_manual_edits(cache)
            self.assertIsNotNone(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertFalse(data["manually_edited"])
            self.assertNotIn("manual_lyrics_sha256", data)

    def test_revert_noop_when_no_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(revert_manual_edits(Path(td)))

    def test_revert_noop_when_already_auto(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_aligned_json(cache / "lyrics.aligned.json", "h", manual=False)
            path = revert_manual_edits(cache)
            self.assertIsNotNone(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertFalse(data["manually_edited"])


class TestBuildEditorHtml(unittest.TestCase):
    def test_html_contains_payload_and_audio_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=0.5)
            _write_aligned_json(cache / "lyrics.aligned.json", "h")
            state = load_editor_state(cache, target_peak_width=80)
            html = build_editor_html(
                state,
                audio_url="/file=/tmp/vocals.wav",
                audio_element_id="test_audio",
                container_id="test_root",
            )
            self.assertIn("test_root", html)
            self.assertIn("test_audio", html)
            # Payload must be a JSON blob inlined into the script.
            self.assertIn("\"song_hash\": \"h\"", html)
            # Basic UI widgets are present.
            self.assertIn("data-mv-waveform", html)
            self.assertIn("data-mv-words", html)
            # The editor embeds its own <audio> element pointed at the
            # Gradio /file= proxy so play/pause work without fishing
            # around in WaveSurfer's internals.
            self.assertIn("<audio", html)
            self.assertIn("/file=/tmp/vocals.wav", html)
            self.assertIn('id="test_audio"', html)

    def test_html_uses_img_onerror_bootstrap(self) -> None:
        # Gradio's ``gr.HTML`` updates via innerHTML, which silently drops
        # any ``<script>`` tags. The editor therefore parks its code in a
        # ``<script type="text/plain">`` and bootstraps execution via an
        # ``<img onerror>``. Regressing to a plain ``<script>`` tag would
        # reintroduce the "no words appear on Load timeline" bug, so lock
        # the shape in here.
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=0.5)
            _write_aligned_json(cache / "lyrics.aligned.json", "h")
            state = load_editor_state(cache, target_peak_width=60)
            html = build_editor_html(
                state, audio_url="/file=/tmp/vocals.wav", container_id="abc"
            )
            self.assertIn('<script type="text/plain" id="mv_editor_code_abc">', html)
            self.assertIn("onerror=", html)
            self.assertIn("new Function", html)
            # There must be exactly one ``</script>`` in the output — the
            # one that closes our inert ``text/plain`` tag. If any ``</``
            # inside the code blob slipped through unescaped, we'd close
            # the tag early and the count would be >1, silently breaking
            # bootstrap.
            self.assertEqual(html.count("</script>"), 1)

    def test_html_contains_multiselect_machinery(self) -> None:
        """Multi-select + group-drag + rubber-band is user-visible
        behaviour that also has no other unit-test handle, so we lock
        the shape of the JS blob here. If one of these token strings
        disappears from the compiled editor code, the selection UX is
        almost certainly broken — regression-catch."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=0.5)
            _write_aligned_json(cache / "lyrics.aligned.json", "h")
            state = load_editor_state(cache, target_peak_width=60)
            html = build_editor_html(state, audio_url="/file=/tmp/vocals.wav")
            self.assertIn("const selected = new Set()", html)
            self.assertIn("updateSelectionStyling", html)
            self.assertIn("clearSelection", html)
            self.assertIn('bandEl.className = "mv-band"', html)
            self.assertIn("onBandMove", html)
            self.assertIn("onBandUp", html)
            self.assertIn("drag.indices", html)
            self.assertIn("ev.shiftKey || ev.ctrlKey || ev.metaKey", html)
            self.assertIn("Escape", html)
            # Help text surfaces the new shortcuts so users know they exist.
            self.assertIn("rubber-band", html)

    def test_html_contains_whisper_overlay_and_drag_guide(self) -> None:
        """The two UX affordances added to help manual alignment on
        tracks where automatic alignment struggles: the faint whisper-
        transcription ghost layer over the waveform, and the vertical
        drag-guide line that appears at the primary word's start edge
        while dragging. If either DOM hook disappears from the build,
        the user loses a major alignment aid."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=0.5)
            _write_aligned_json(
                cache / "lyrics.aligned.json", "h", include_whisper=True
            )
            state = load_editor_state(cache, target_peak_width=60)
            html = build_editor_html(state, audio_url="/file=/tmp/vocals.wav")
            self.assertIn("data-mv-whisper", html)
            self.assertIn("data-mv-drag-guide", html)
            self.assertIn("mv-whisper-word", html)
            self.assertIn("renderWhisperWords", html)
            self.assertIn("showDragGuide", html)
            self.assertIn("hideDragGuide", html)
            # The payload must carry whisper_words + scores all the way
            # through. Opacity scaling in JS needs the score to exist
            # on each payload entry so low-confidence labels fade out.
            self.assertIn("\"whisper_words\":", html)
            self.assertIn("\"whirled\"", html)
            self.assertIn("\"score\":", html)
            self.assertIn("el.style.opacity", html)


class TestWhisperScorePassThrough(unittest.TestCase):
    """The editor needs the CTC score for each whisper-transcribed word
    so it can scale the label's opacity — low-confidence placements are
    faded out so the user can tell at a glance which ghost labels to
    trust. ``load_editor_state`` must therefore preserve the ``score``
    field when parsing older / partial cache files too."""

    def test_score_is_loaded_and_passed_to_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=0.5)
            _write_aligned_json(
                cache / "lyrics.aligned.json", "h", include_whisper=True
            )
            state = load_editor_state(cache, target_peak_width=40)
            # whirled was written with score=0.6 in _write_aligned_json.
            self.assertEqual(len(state.whisper_words), 2)
            self.assertAlmostEqual(state.whisper_words[1]["score"], 0.6)
            html = build_editor_html(state, audio_url="/file=/tmp/vocals.wav")
            # score value should make it into the JSON payload verbatim.
            self.assertIn("0.6", html)

    def test_missing_score_degrades_to_none(self) -> None:
        """Older caches may lack the score field on individual whisper
        words; the editor has to accept that without crashing, just
        rendering them at the floor opacity."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "h"
            cache.mkdir()
            _write_sine_wav(cache / "vocals.wav", seconds=0.5)
            # Minimal aligned.json with scoreless whisper_words.
            payload = {
                "schema_version": 3,
                "song_hash": "h",
                "model": "m",
                "language": "en",
                "vocals_wav": "vocals.wav",
                "lyrics_sha256": "deadbeef",
                "manually_edited": False,
                "lines": [],
                "words": [],
                "whisper_words": [
                    {"word": "one", "t_start": 0.0, "t_end": 0.2},
                ],
            }
            (cache / "lyrics.aligned.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            state = load_editor_state(cache, target_peak_width=40)
            self.assertEqual(len(state.whisper_words), 1)
            self.assertIsNone(state.whisper_words[0]["score"])


if __name__ == "__main__":
    unittest.main()
