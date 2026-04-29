"""Tests for SubRip export from aligned lyrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.kinetic_typography import AlignedWord
from pipeline.srt_export import (
    build_srt_document,
    export_aligned_json_to_srt,
    format_srt_timestamp,
)


class TestSrtExport(unittest.TestCase):
    def test_format_srt_timestamp_zero(self) -> None:
        self.assertEqual(format_srt_timestamp(0), "00:00:00,000")

    def test_format_srt_timestamp_rounding(self) -> None:
        self.assertEqual(format_srt_timestamp(61.380665649414055), "00:01:01,381")

    def test_build_srt_document_two_words(self) -> None:
        words = [
            AlignedWord(word="Hello", line_idx=0, t_start=1.0, t_end=2.0),
            AlignedWord(word="World", line_idx=0, t_start=2.5, t_end=3.0),
        ]
        doc = build_srt_document(words)
        self.assertIn("1\n", doc)
        self.assertIn("00:00:01,000 --> 00:00:02,000\n", doc)
        self.assertIn("Hello\n", doc)
        self.assertIn("2\n", doc)
        self.assertIn("00:00:02,500 --> 00:00:03,000\n", doc)
        self.assertIn("World\n", doc)

    def test_build_srt_document_skips_blank_words(self) -> None:
        words = [
            AlignedWord(word="A", line_idx=0, t_start=0.0, t_end=0.5),
            AlignedWord(word="   ", line_idx=0, t_start=0.5, t_end=1.0),
            AlignedWord(word="B", line_idx=0, t_start=1.0, t_end=1.5),
        ]
        doc = build_srt_document(words)
        self.assertIn("1\n", doc)
        self.assertIn("2\n", doc)
        self.assertNotIn("3\n", doc)
        self.assertIn("A", doc)
        self.assertIn("B", doc)

    def test_export_aligned_json_to_srt_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            aligned = cache / "lyrics.aligned.json"
            aligned.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "lines": ["Hi"],
                        "words": [
                            {
                                "word": "Hi",
                                "line_idx": 0,
                                "t_start": 0.0,
                                "t_end": 0.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = export_aligned_json_to_srt(aligned)
            self.assertEqual(out.name, "lyrics.aligned.srt")
            self.assertTrue(out.is_file())
            text = out.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:00,500", text)
            self.assertIn("Hi", text)


if __name__ == "__main__":
    unittest.main()
