"""Tests for ``pipeline.renderer._build_ffmpeg_cmd`` audio trimming args."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.renderer import _build_ffmpeg_cmd


def _idx(cmd: list[str], flag: str) -> int:
    return cmd.index(flag)


class TestBuildFfmpegCmd(unittest.TestCase):
    def _base_kwargs(self) -> dict:
        return dict(
            width=1920,
            height=1080,
            fps=30,
            audio_path=Path("song.wav"),
            output_mp4=Path("out.mp4"),
            video_codec="libx264",
        )

    def test_no_trim_has_no_ss_or_t_audio_args(self) -> None:
        cmd = _build_ffmpeg_cmd(**self._base_kwargs())
        # audio -i must appear once, and before it there should be no -ss / -t
        audio_idx = cmd.index("song.wav")
        self.assertEqual(cmd[audio_idx - 1], "-i")
        # no -ss anywhere (raw video input doesn't use it either)
        self.assertNotIn("-ss", cmd)
        # no -t anywhere
        self.assertNotIn("-t", cmd)

    def test_audio_start_adds_ss_before_audio_input(self) -> None:
        cmd = _build_ffmpeg_cmd(audio_start_sec=12.5, **self._base_kwargs())
        ss_idx = _idx(cmd, "-ss")
        i_audio = cmd.index("song.wav")
        # The audio -i appears after -ss; the rawvideo -i is the pipe "-".
        self.assertLess(ss_idx, i_audio)
        self.assertEqual(cmd[ss_idx + 1], "12.500000")

    def test_duration_adds_t(self) -> None:
        cmd = _build_ffmpeg_cmd(audio_duration_sec=10.0, **self._base_kwargs())
        t_idx = _idx(cmd, "-t")
        self.assertEqual(cmd[t_idx + 1], "10.000000")

    def test_negative_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            _build_ffmpeg_cmd(audio_start_sec=-0.1, **self._base_kwargs())

    def test_non_positive_duration_raises(self) -> None:
        with self.assertRaises(ValueError):
            _build_ffmpeg_cmd(audio_duration_sec=0.0, **self._base_kwargs())

    def test_includes_mp4_browser_color_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GLITCHFRAME_FFMPEG_COLOR_METADATA": "",
                "GLITCHFRAME_FFMPEG_COLOR_METADATA_ARGS": "",
            },
        ):
            cmd = _build_ffmpeg_cmd(**self._base_kwargs())
        self.assertIn("-color_primaries:v", cmd)
        i = cmd.index("-color_primaries:v")
        self.assertEqual(cmd[i + 1], "bt709")
        self.assertIn("-color_trc:v", cmd)
        j = cmd.index("-color_trc:v")
        self.assertEqual(cmd[j + 1], "iec61966-2-1")
        self.assertIn("-pix_fmt:v", cmd)
        self.assertEqual(cmd[cmd.index("-pix_fmt:v") + 1], "yuv420p")

    def test_color_metadata_can_be_disabled_by_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GLITCHFRAME_FFMPEG_COLOR_METADATA": "off",
                "GLITCHFRAME_FFMPEG_COLOR_METADATA_ARGS": "",
            },
        ):
            cmd = _build_ffmpeg_cmd(**self._base_kwargs())
        self.assertNotIn("-color_primaries:v", cmd)
        self.assertNotIn("-color_trc:v", cmd)


if __name__ == "__main__":
    unittest.main()
