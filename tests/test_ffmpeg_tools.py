"""Tests for ``pipeline.ffmpeg_tools`` binary resolution + diagnostics."""

from __future__ import annotations

import logging
import os
import subprocess
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
            {"GLITCHFRAME_FFMPEG": __file__},  # a real file, good enough for existence
            clear=False,
        ):
            self.assertEqual(ffmpeg_tools.resolve_ffmpeg(), __file__)

    def test_glitchframe_ffmpeg_wins_over_legacy_musicvids(self) -> None:
        other = str(Path(__file__).parent / "other-ffmpeg.exe")
        with mock.patch.dict(
            os.environ,
            {
                "GLITCHFRAME_FFMPEG": __file__,
                "MUSICVIDS_FFMPEG": other,
            },
            clear=False,
        ):
            self.assertEqual(ffmpeg_tools.resolve_ffmpeg(), __file__)

    def test_env_override_missing_file_falls_through_to_path(self) -> None:
        missing = str(Path(__file__).parent / "definitely-not-a-file.xyz")
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_FFMPEG": missing}, clear=False
        ), mock.patch(
            "pipeline.ffmpeg_tools.shutil.which", return_value="/fake/path/ffmpeg"
        ):
            self.assertEqual(ffmpeg_tools.resolve_ffmpeg(), "/fake/path/ffmpeg")

    def test_path_lookup_used_when_no_env_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("GLITCHFRAME_FFMPEG", None)
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
            env.pop("GLITCHFRAME_FFMPEG", None)
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
            env.pop("GLITCHFRAME_FFMPEG", None)
            env.pop("MUSICVIDS_FFMPEG", None)
            env.pop("GLITCHFRAME_FFPROBE", None)
            env.pop("MUSICVIDS_FFPROBE", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which", return_value=None
            ), mock.patch(
                "pipeline.ffmpeg_tools._candidate_dirs", return_value=[]
            ):
                with self.assertRaises(RuntimeError) as cm:
                    ffmpeg_tools.require_ffmpeg()
                self.assertIn("GLITCHFRAME_FFMPEG", str(cm.exception))

    def test_result_is_cached(self) -> None:
        with mock.patch(
            "pipeline.ffmpeg_tools.shutil.which", return_value="/first/ffmpeg"
        ) as which_mock:
            ffmpeg_tools.resolve_ffmpeg()
            ffmpeg_tools.resolve_ffmpeg()
            ffmpeg_tools.resolve_ffmpeg()
        self.assertEqual(which_mock.call_count, 1)


class TestResolveLogging(unittest.TestCase):
    """``_resolve`` must log the resolved path AND the source (env / PATH /
    well-known) so a Pinokio log shows which ffmpeg the process picked.

    The "via PATH" case used to be silent — only the well-known-location
    branch logged. That made conda env activation shadowing the user's
    PATH ffmpeg invisible from the log alone (the actual root cause when a
    fork user reports "NVENC unavailable").
    """

    def setUp(self) -> None:
        ffmpeg_tools.clear_cache()

    def tearDown(self) -> None:
        ffmpeg_tools.clear_cache()

    def test_logs_when_resolved_via_path(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("GLITCHFRAME_FFMPEG", None)
            env.pop("MUSICVIDS_FFMPEG", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ):
                with self.assertLogs(
                    "pipeline.ffmpeg_tools", level="INFO"
                ) as ctx:
                    ffmpeg_tools.resolve_ffmpeg()
        joined = "\n".join(ctx.output)
        self.assertIn("/usr/bin/ffmpeg", joined)
        self.assertIn("via PATH", joined)

    def test_logs_when_resolved_via_env_override(self) -> None:
        with mock.patch.dict(
            os.environ, {"GLITCHFRAME_FFMPEG": __file__}, clear=False
        ):
            with self.assertLogs("pipeline.ffmpeg_tools", level="INFO") as ctx:
                ffmpeg_tools.resolve_ffmpeg()
        joined = "\n".join(ctx.output)
        self.assertIn(__file__, joined)
        self.assertIn("env override", joined)

    def test_warns_when_not_found_anywhere(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("GLITCHFRAME_FFMPEG", None)
            env.pop("MUSICVIDS_FFMPEG", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which", return_value=None
            ), mock.patch(
                "pipeline.ffmpeg_tools._candidate_dirs", return_value=[]
            ):
                with self.assertLogs(
                    "pipeline.ffmpeg_tools", level="WARNING"
                ) as ctx:
                    self.assertIsNone(ffmpeg_tools.resolve_ffmpeg())
        self.assertIn("not found", "\n".join(ctx.output))

    def test_path_logging_runs_only_once_due_to_cache(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("GLITCHFRAME_FFMPEG", None)
            env.pop("MUSICVIDS_FFMPEG", None)
            with mock.patch(
                "pipeline.ffmpeg_tools.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ):
                with self.assertLogs(
                    "pipeline.ffmpeg_tools", level="INFO"
                ) as ctx:
                    ffmpeg_tools.resolve_ffmpeg()
                    ffmpeg_tools.resolve_ffmpeg()
                    ffmpeg_tools.resolve_ffmpeg()
        # Exactly one resolution log even though we called three times.
        path_logs = [ln for ln in ctx.output if "Resolved ffmpeg" in ln]
        self.assertEqual(len(path_logs), 1, ctx.output)


class TestProbeEncoderLogging(unittest.TestCase):
    """``_probe_encoder`` must capture the actual NVENC/encoder error from
    stderr (not just ffmpeg's generic last-line wrapper).

    Regression guard for the Pinokio NVENC mystery: when h264_nvenc fails,
    we used to log just ``Error while opening encoder for output stream
    ... maybe incorrect parameters such as bit_rate, rate, width or height``
    which is ffmpeg's wrapper. The actual cause (``Cannot load
    nvEncodeAPI64.dll`` / ``Driver does not support the required nvenc
    API version`` / ``OpenEncodeSession failed``) prints earlier in the
    stream and got thrown away.
    """

    def setUp(self) -> None:
        ffmpeg_tools.clear_cache()

    def tearDown(self) -> None:
        ffmpeg_tools.clear_cache()

    def _fake_completed(
        self,
        returncode: int,
        stderr_text: str,
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=b"",
            stderr=stderr_text.encode("utf-8"),
        )

    def test_logs_full_stderr_tail_on_failure(self) -> None:
        # Simulate the genuine NVENC failure pattern: real error first,
        # generic wrapper last. The old code kept only the last line.
        nvenc_stderr = (
            "[h264_nvenc @ 000001abcdef0000] Loaded Nvenc version 12.1\n"
            "[h264_nvenc @ 000001abcdef0000] Cannot load nvEncodeAPI64.dll\n"
            "[h264_nvenc @ 000001abcdef0000] No capable devices found\n"
            "[vost#0:0 @ 000001abcdef0000] Error while opening encoder -- "
            "maybe incorrect parameters such as bit_rate, rate, width or height\n"
        )
        with mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg", return_value="/fake/ffmpeg"
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run",
            return_value=self._fake_completed(1, nvenc_stderr),
        ):
            with self.assertLogs(
                "pipeline.ffmpeg_tools", level="INFO"
            ) as ctx:
                ok = ffmpeg_tools._probe_encoder("h264_nvenc")
        self.assertFalse(ok)
        joined = "\n".join(ctx.output)
        # The REAL error must be in the log, not just the wrapper.
        self.assertIn("Cannot load nvEncodeAPI64.dll", joined)
        # And the ffmpeg path is logged so the user sees which binary failed.
        self.assertIn("/fake/ffmpeg", joined)

    def test_uses_loglevel_info_so_nvenc_diagnostics_are_visible(self) -> None:
        """Probe must run ffmpeg at info level — at error level NVENC's
        actual diagnostic messages are suppressed by ffmpeg itself."""
        captured: dict[str, list[str]] = {}

        def _fake_run(cmd: list[str], **kwargs):
            captured["cmd"] = list(cmd)
            return self._fake_completed(0, "")

        with mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg", return_value="/fake/ffmpeg"
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run", side_effect=_fake_run
        ):
            ffmpeg_tools._probe_encoder("h264_nvenc")

        self.assertIn("-loglevel", captured["cmd"])
        idx = captured["cmd"].index("-loglevel")
        self.assertEqual(
            captured["cmd"][idx + 1],
            "info",
            "Probe must use -loglevel info; -loglevel error masks the real "
            "NVENC error and made the original Pinokio bug undebuggable",
        )


class TestLogFfmpegDiagnostics(unittest.TestCase):
    """Startup diagnostic must surface the resolved ffmpeg path AND its
    configure-line NVIDIA flags so a Pinokio log alone tells us whether
    NVENC could ever work with this binary."""

    def setUp(self) -> None:
        ffmpeg_tools.clear_cache()

    def tearDown(self) -> None:
        ffmpeg_tools.clear_cache()

    def test_logs_path_and_banner_when_ffmpeg_is_found(self) -> None:
        version_output = (
            b"ffmpeg version 8.1-full_build-www.gyan.dev Copyright "
            b"(c) 2000-2026 the FFmpeg developers\n"
            b"  built with gcc 14.2.0 (Rev1, Built by MSYS2)\n"
            b"  configuration: --enable-gpl --enable-cuda-llvm "
            b"--enable-cuvid --enable-nvdec --enable-nvenc\n"
        )
        with mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg",
            return_value="/winget/ffmpeg.exe",
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=version_output, stderr=b""
            ),
        ):
            with self.assertLogs(
                "pipeline.ffmpeg_tools", level="INFO"
            ) as ctx:
                ffmpeg_tools.log_ffmpeg_diagnostics()
        joined = "\n".join(ctx.output)
        self.assertIn("/winget/ffmpeg.exe", joined)
        self.assertIn("ffmpeg version 8.1", joined)
        # NVIDIA flags must be surfaced for the encoder probe to make sense.
        self.assertIn("--enable-nvenc", joined)

    def test_warns_when_ffmpeg_lacks_nvidia_support(self) -> None:
        version_output = (
            b"ffmpeg version 5.0 Copyright (c) 2000-2022 the FFmpeg developers\n"
            b"  configuration: --enable-gpl --enable-libx264\n"
        )
        with mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg",
            return_value="/conda/ffmpeg",
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=version_output, stderr=b""
            ),
        ):
            with self.assertLogs(
                "pipeline.ffmpeg_tools", level="WARNING"
            ) as ctx:
                ffmpeg_tools.log_ffmpeg_diagnostics()
        self.assertIn("nvenc", "\n".join(ctx.output).lower())

    def test_warns_when_ffmpeg_not_found(self) -> None:
        with mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg", return_value=None
        ):
            with self.assertLogs(
                "pipeline.ffmpeg_tools", level="WARNING"
            ) as ctx:
                ffmpeg_tools.log_ffmpeg_diagnostics()
        self.assertIn("not found", "\n".join(ctx.output))

    def test_diagnostics_never_raise(self) -> None:
        """A best-effort diagnostic must NEVER raise — startup logging that
        crashes the app would be worse than no diagnostic at all."""
        with mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg",
            return_value="/some/ffmpeg",
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run",
            side_effect=OSError("boom"),
        ):
            # Should NOT raise.
            ffmpeg_tools.log_ffmpeg_diagnostics()


if __name__ == "__main__":
    unittest.main()
