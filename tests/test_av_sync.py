"""Tests for ``pipeline.av_sync.ffprobe_av_sync``."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.av_sync import (
    DEFAULT_AV_SYNC_TOLERANCE_MS,
    AvSyncReport,
    ffprobe_av_sync,
)


def _touch(tmp: Path) -> Path:
    p = tmp / "fake.mp4"
    p.write_bytes(b"\x00")
    return p


class TestFfprobeAvSync(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ffprobe_av_sync(Path("nope.mp4"))

    def test_negative_tolerance_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = _touch(Path(tmp))
            with self.assertRaises(ValueError):
                ffprobe_av_sync(p, tolerance_ms=-1)

    def test_ffprobe_missing_returns_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = _touch(Path(tmp))
            with mock.patch("pipeline.av_sync.resolve_ffprobe", return_value=None):
                report = ffprobe_av_sync(p)
        self.assertFalse(report.ffprobe_available)
        self.assertFalse(report.ok)
        self.assertIsNone(report.drift_ms)
        self.assertIn("ffprobe", report.message.lower())

    def _run_with_fake_ffprobe(
        self, stdout_doc: dict, returncode: int = 0
    ) -> AvSyncReport:
        stdout = json.dumps(stdout_doc)
        fake = mock.Mock()
        fake.returncode = returncode
        fake.stdout = stdout
        fake.stderr = ""
        with tempfile.TemporaryDirectory() as tmp:
            p = _touch(Path(tmp))
            with mock.patch(
                "pipeline.av_sync.resolve_ffprobe", return_value="ffprobe"
            ), mock.patch("pipeline.av_sync.subprocess.run", return_value=fake):
                return ffprobe_av_sync(p, tolerance_ms=50.0)

    def test_pass_within_tolerance(self) -> None:
        report = self._run_with_fake_ffprobe(
            {
                "streams": [
                    {"codec_type": "video", "duration": "10.000"},
                    {"codec_type": "audio", "duration": "10.020"},
                ]
            }
        )
        self.assertTrue(report.ffprobe_available)
        self.assertTrue(report.ok)
        self.assertIsNotNone(report.drift_ms)
        self.assertAlmostEqual(report.drift_ms or 0.0, 20.0, places=1)
        self.assertIn("PASS", report.message)

    def test_fail_above_tolerance(self) -> None:
        report = self._run_with_fake_ffprobe(
            {
                "streams": [
                    {"codec_type": "video", "duration": "10.000"},
                    {"codec_type": "audio", "duration": "10.500"},
                ]
            }
        )
        self.assertTrue(report.ffprobe_available)
        self.assertFalse(report.ok)
        self.assertAlmostEqual(report.drift_ms or 0.0, 500.0, places=1)
        self.assertIn("FAIL", report.message)

    def test_missing_stream_reports_failure(self) -> None:
        report = self._run_with_fake_ffprobe(
            {"streams": [{"codec_type": "video", "duration": "10.0"}]}
        )
        self.assertFalse(report.ok)
        self.assertIsNotNone(report.video_duration_sec)
        self.assertIsNone(report.audio_duration_sec)

    def test_non_zero_return_code_reports_failure(self) -> None:
        report = self._run_with_fake_ffprobe({}, returncode=1)
        self.assertFalse(report.ok)
        self.assertIsNone(report.drift_ms)

    def test_default_tolerance_exported(self) -> None:
        self.assertEqual(DEFAULT_AV_SYNC_TOLERANCE_MS, 50.0)


if __name__ == "__main__":
    unittest.main()
