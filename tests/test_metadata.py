"""Tests for ``pipeline.metadata`` and ``orchestrator.write_run_metadata``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator import OrchestratorInputs, write_run_metadata
from pipeline.lyrics_aligner import AlignmentResult, AlignedWord
from pipeline.metadata import (
    METADATA_TXT_NAME,
    chapter_lines_from_analysis,
    compose_metadata_txt,
    format_chapter_timestamp,
    parse_metadata_txt,
    write_metadata_txt,
)


class TestMetadataHelpers(unittest.TestCase):
    def test_format_chapter_timestamp(self) -> None:
        self.assertEqual(format_chapter_timestamp(0), "0:00")
        self.assertEqual(format_chapter_timestamp(65.4), "1:05")
        self.assertEqual(format_chapter_timestamp(3600), "1:00:00")
        self.assertEqual(format_chapter_timestamp(3661), "1:01:01")

    def test_chapter_lines_match_segment_starts(self) -> None:
        analysis = {
            "segments": [
                {"t_start": 0.0, "t_end": 16.2, "label": 0},
                {"t_start": 16.2, "t_end": 60.0, "label": 1},
                {"t_start": 60.0, "t_end": 120.0, "label": 2},
            ]
        }
        lines = chapter_lines_from_analysis(analysis)
        self.assertEqual(
            lines,
            [
                "0:00 Intro",
                "0:16 Verse 1",
                "1:00 Chorus",
            ],
        )


class TestMetadataFile(unittest.TestCase):
    def test_parse_roundtrip(self) -> None:
        analysis = {
            "segments": [{"t_start": 0.0, "t_end": 10.0, "label": 0}],
            "tempo": {"bpm": 90.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_metadata_txt(
                tmp,
                song_metadata={
                    "artist": "Test Artist",
                    "title": "Test Song",
                    "genre": "Electronic",
                },
                analysis=analysis,
                preset_id="neon-synthwave",
                preset={"shader": "wave", "typo_style": "bold"},
                lyrics_lines=["Line one", "Line two"],
            )
            self.assertEqual(path.name, METADATA_TXT_NAME)
            parsed = parse_metadata_txt(path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["version"], 1)
            self.assertIn("Test Artist", parsed["title"])
            self.assertIn("Test Song", parsed["title"])
            self.assertIn("[Official Visualizer]", parsed["title"])
            self.assertEqual(parsed["chapters"], ["0:00 Intro"])
            self.assertIn("Line one", parsed["description"])
            self.assertIn("Chapters:", parsed["description"])
            self.assertIn("music visualizer", [t.lower() for t in parsed["tags"]])

    def test_compose_preserves_description_newlines(self) -> None:
        body = "A\n\nB"
        text = compose_metadata_txt(
            youtube_title="T",
            description=body,
            chapter_lines=["0:00 Intro"],
            tags=["x"],
        )
        parsed = parse_metadata_txt(text)
        self.assertIn("A\n\nB", parsed["description"])


class TestWriteRunMetadata(unittest.TestCase):
    def test_orchestrator_writes_same_chapters(self) -> None:
        analysis = {
            "segments": [
                {"t_start": 0.0, "t_end": 5.0, "label": 0},
                {"t_start": 5.0, "t_end": 12.5, "label": 1},
            ],
            "tempo": {"bpm": 100.0},
        }
        align = AlignmentResult(
            song_hash="ab",
            cache_dir=Path("."),
            aligned_json=Path("lyrics.aligned.json"),
            language="en",
            model="large-v3",
            lines=["la la"],
            words=[
                AlignedWord("la", 0, 0.0, 0.5),
                AlignedWord("la", 0, 0.5, 1.0),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_run_metadata(
                tmp,
                inputs=OrchestratorInputs(
                    metadata={"artist": "A", "title": "B"},
                    presets={"id": "minimal-mono"},
                    lyrics_text="ignored when alignment set",
                ),
                analysis_doc=analysis,
                alignment=align,
            )
            parsed = parse_metadata_txt(
                (Path(tmp) / METADATA_TXT_NAME).read_text(encoding="utf-8")
            )
        self.assertEqual(parsed["chapters"], ["0:00 Intro", "0:05 Verse 1"])
        self.assertIn("la la", parsed["description"])


if __name__ == "__main__":
    unittest.main()
