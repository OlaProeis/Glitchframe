"""Tests for ``pipeline.ffmpeg_tools`` binary resolution."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from pipeline import ffmpeg_tools


class TestResolveFfmpeg(unittest.TestCase):
    def setUp(self) -> None:
        ffmpeg_tools.clear_cache()

    def tearDown(self) -> None:
        ffmpeg_tools.clear_cache()

    def test_env_override_wins(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"MUSICVIDS_FFMPEG": __file__},  # a real file, good enough for existence
            clear=False,
        ):
            self.assertEqual(ffmpeg_tools.resolve_ffmpeg(), __file__)

    def test_env_override_missing_file_falls_through_to_path(self) -> None:
        missing = str(Path(__file__).parent / "definitely-not-a-file.xyz")
        with mock.patch.dict(
            os.environ, {"MUSICVIDS_FFMPEG": missing}, clear=False
        ), mock.patch(
            "pipeline.ffmpeg_tools.shutil.which", return_value="/fake/path/ffmpeg"
        ):
            self.assertEqual(ffmpeg_tools.resolve_ffmpeg(), "/fake/path/ffmpeg")

    def test_path_lookup_used_when_no_env_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("MUSICVIDS_FFMPEG", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ):
                self.assertEqual(ffmpeg_tools.resolve_ffmpeg(), "/usr/bin/ffmpeg")

    def test_falls_back_to_well_known_dirs_when_not_on_path(self) -> None:
        # Fake a windows-style candidate dir containing ffmpeg.exe (or posix name)
        # so we exercise the fallback without caring about the host OS.
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("MUSICVIDS_FFMPEG", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which", return_value=None
            ), mock.patch(
                "pipeline.ffmpeg_tools._candidate_dirs",
                return_value=[Path(__file__).parent],
            ), mock.patch(
                "pathlib.Path.is_file", return_value=True
            ):
                result = ffmpeg_tools.resolve_ffmpeg()
        self.assertIsNotNone(result)
        self.assertIn("ffmpeg", result or "")

    def test_require_raises_when_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("MUSICVIDS_FFMPEG", None)
            env.pop("MUSICVIDS_FFPROBE", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which", return_value=None
            ), mock.patch(
                "pipeline.ffmpeg_tools._candidate_dirs", return_value=[]
            ):
                with self.assertRaises(RuntimeError) as cm:
                    ffmpeg_tools.require_ffmpeg()
                self.assertIn("MUSICVIDS_FFMPEG", str(cm.exception))

    def test_result_is_cached(self) -> None:
        with mock.patch(
            "pipeline.ffmpeg_tools.shutil.which", return_value="/first/ffmpeg"
        ) as which_mock:
            ffmpeg_tools.resolve_ffmpeg()
            ffmpeg_tools.resolve_ffmpeg()
            ffmpeg_tools.resolve_ffmpeg()
        self.assertEqual(which_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
