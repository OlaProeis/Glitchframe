"""Regression tests for the compositor's request-thread progress fix.

The Pinokio bug being guarded here: the compositor used to call
``progress(...)`` from its daemon producer thread. Gradio's ``gr.Progress``
silently drops cross-thread updates, so the UI bar froze on the
orchestrator's outer label ("Compositing video (frames + encode) …") for
the entire render even though terminal-side INFO logs (which ARE
thread-safe) showed steady frame progress.

The fix: producer writes to a shared :class:`_CompositorStats` object, the
encoder-feed loop on the request thread polls that object every
``_PROGRESS_TICK_SEC`` and forwards to ``progress``. These tests exercise
the ``progress_pair()`` formatter and the stats object's behavior in both
warmup and steady-state phases without spinning up moderngl, ffmpeg, or
threads — the lock-free correctness comes from the GIL, not from anything
test-able with stubs.

The other Pinokio-relevant guard here is the codec-capable ffmpeg picker:
when one ffmpeg binary lacks NVENC support we should sweep all candidates
and pick a working one, not silently fall back to libx264.
"""

from __future__ import annotations

import subprocess
import time
import unittest
from unittest import mock

from pipeline import ffmpeg_tools
from pipeline.compositor import (
    _CompositorStats,
    _format_eta_compositor,
    _PROGRESS_TICK_SEC,
)


class TestFormatEtaCompositor(unittest.TestCase):
    """``_format_eta_compositor`` is duplicated from ``app._format_eta`` so the
    compositor can be imported in CLI / test contexts without Gradio. We
    therefore lock the output format here (the UI parses it visually)."""

    def test_subsecond(self) -> None:
        self.assertEqual(_format_eta_compositor(0.4), "<1s")

    def test_seconds_only(self) -> None:
        self.assertEqual(_format_eta_compositor(42.7), "43s")

    def test_minutes_seconds(self) -> None:
        self.assertEqual(_format_eta_compositor(125.0), "2m05s")

    def test_hours(self) -> None:
        # 1h 23m 45s -- this is the "render takes 90 minutes on libx264" case.
        self.assertEqual(_format_eta_compositor(3600 + 23 * 60 + 45), "1h23m45s")

    def test_negative_clamped(self) -> None:
        self.assertEqual(_format_eta_compositor(-5.0), "<1s")


class TestCompositorStatsProgressPair(unittest.TestCase):
    """``_CompositorStats.progress_pair()`` builds the description shown in
    the Gradio progress bar. These tests pin the format so a UI redesign
    can't accidentally break the regex any external monitor / log scraper
    might rely on."""

    def test_warmup_shows_phase_when_no_frames_encoded(self) -> None:
        # Producer hasn't pushed a frame yet -- consumer poll happens but
        # the message must still be informative ("warming up", "preparing
        # typography", ...) instead of a stale "Compositing video..." label.
        stats = _CompositorStats(
            total_frames=4923,
            started_at=time.monotonic(),
            phase="warming up",
            layer_label="layers=BG+TYPO+LOGO",
        )
        p, msg = stats.progress_pair()
        self.assertEqual(p, 0.0)
        self.assertIn("0/4923", msg)
        self.assertIn("warming up", msg)
        self.assertIn("layers=BG+TYPO+LOGO", msg)

    def test_steady_state_shows_frames_fps_and_eta(self) -> None:
        # 1843 frames encoded in 25 minutes wall time -> ~1.23 fps avg.
        # ETA for the remaining 3080 frames at 1.23 fps -> ~2502s -> ~41m42s.
        # We pin the textual format the UI shows so a "fps" label change
        # doesn't silently slip through.
        started = time.monotonic() - (25.0 * 60.0)
        stats = _CompositorStats(
            total_frames=4923,
            started_at=started,
            frames_encoded=1843,
            phase="encoding",
            layer_label="layers=BG+TYPO",
        )
        p, msg = stats.progress_pair()
        self.assertAlmostEqual(p, 1843 / 4923, places=4)
        self.assertIn("Compositing 1843/4923", msg)
        self.assertIn("fps", msg)
        self.assertIn("ETA", msg)
        self.assertIn("layers=BG+TYPO", msg)

    def test_progress_pair_is_safe_when_started_at_now(self) -> None:
        """Right after ``stats.started_at = time.monotonic()`` the elapsed
        is ~0; we must not divide by zero or report nonsense fps."""
        stats = _CompositorStats(
            total_frames=100,
            started_at=time.monotonic(),
            frames_encoded=1,
            phase="encoding",
        )
        # Should not raise; resulting fps is finite (very large is fine,
        # the next tick re-measures).
        p, msg = stats.progress_pair()
        self.assertGreater(p, 0.0)
        self.assertIn("Compositing 1/100", msg)

    def test_progress_pair_reads_a_consistent_snapshot(self) -> None:
        """The producer can mutate frames_encoded between two reads inside
        progress_pair(); we should at least not crash. (No torn reads under
        the GIL for ints, but the test guards the general invariant.)"""
        stats = _CompositorStats(
            total_frames=1000,
            started_at=time.monotonic() - 10.0,
            frames_encoded=500,
        )
        for _ in range(50):
            stats.frames_encoded += 1
            p, msg = stats.progress_pair()
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            self.assertIn(f"/{stats.total_frames}", msg)


class TestProgressTickConstant(unittest.TestCase):
    """The progress poll cadence is a sensitive knob: too fast spams Gradio's
    websocket queue (visible UI lag at 1080p NVENC ~30fps), too slow makes
    the bar feel frozen during long encodes. Lock the value here so a future
    "let's just bump it" change becomes visible in code review."""

    def test_progress_tick_is_a_quarter_second(self) -> None:
        # 4 progress callbacks per second is the sweet spot we landed on
        # while debugging the Pinokio "stuck at 40%" UI bug.
        self.assertEqual(_PROGRESS_TICK_SEC, 0.25)


class TestCodecCapableFfmpegPicker(unittest.TestCase):
    """Pinokio NVENC fallback regression:

    On Pinokio Windows the conda env's bundled ``ffmpeg`` (e.g. from
    ``imageio-ffmpeg`` or a transitive dep) sometimes lacks NVENC support
    while the user's PATH ffmpeg has it. The first-match-wins behaviour
    silently fell back to ``libx264`` (~5-10× slower at 1080p). The fix is
    to sweep every discovered candidate and prefer a codec-capable one;
    these tests guard that behaviour without needing two actual ffmpegs.
    """

    def setUp(self) -> None:
        ffmpeg_tools.clear_cache()

    def tearDown(self) -> None:
        ffmpeg_tools.clear_cache()

    def _completed(
        self, returncode: int, stderr_text: str = ""
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=b"",
            stderr=stderr_text.encode("utf-8"),
        )

    def test_picks_second_candidate_when_first_lacks_codec(self) -> None:
        """First-match ffmpeg fails NVENC; second one succeeds. Picker must
        return the second one and promote it into the resolver cache so
        encode actually uses it (otherwise we'd probe the working binary
        but encode through the broken one)."""
        first = ffmpeg_tools.FfmpegCandidate(
            path="C:/pinokio/env/Library/bin/ffmpeg.exe",
            source="active env (Library/bin)",
        )
        second = ffmpeg_tools.FfmpegCandidate(
            path="C:/winget/ffmpeg.exe",
            source="PATH",
        )
        responses = {
            first.path: self._completed(
                1,
                "[h264_nvenc] Cannot load nvEncodeAPI64.dll\n"
                "[vost#0:0] Error while opening encoder",
            ),
            second.path: self._completed(0),
        }

        def _fake_run(cmd, **kwargs):
            return responses[cmd[0]]

        with mock.patch(
            "pipeline.ffmpeg_tools._iter_candidates",
            return_value=iter([first, second]),
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run", side_effect=_fake_run
        ):
            picked = ffmpeg_tools._pick_codec_capable_ffmpeg("h264_nvenc")
        self.assertEqual(picked, second.path)
        # Resolver cache must be promoted to the working binary; otherwise
        # the eventual encode would fail with the same NVENC error.
        self.assertEqual(ffmpeg_tools._cache.get("ffmpeg"), second.path)

    def test_returns_none_when_no_candidate_supports_codec(self) -> None:
        """Every candidate fails -> picker reports None and select_video_codec
        falls back to libx264 (covered by the warning log)."""
        only = ffmpeg_tools.FfmpegCandidate(
            path="/usr/bin/ffmpeg", source="PATH"
        )
        with mock.patch(
            "pipeline.ffmpeg_tools._iter_candidates",
            return_value=iter([only]),
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run",
            return_value=self._completed(
                1, "[h264_nvenc] No capable devices found"
            ),
        ):
            self.assertIsNone(
                ffmpeg_tools._pick_codec_capable_ffmpeg("h264_nvenc")
            )

    def test_returns_none_when_no_candidates_discovered(self) -> None:
        """No ffmpeg anywhere on the system."""
        with mock.patch(
            "pipeline.ffmpeg_tools._iter_candidates",
            return_value=iter([]),
        ):
            self.assertIsNone(
                ffmpeg_tools._pick_codec_capable_ffmpeg("h264_nvenc")
            )

    def test_select_video_codec_uses_picker_after_first_probe_fails(
        self,
    ) -> None:
        """End-to-end: select_video_codec -> _probe_encoder fails ->
        _pick_codec_capable_ffmpeg succeeds -> we return the default codec
        and the resolver cache points at the working binary."""
        first = ffmpeg_tools.FfmpegCandidate(
            path="/conda/ffmpeg", source="active env (bin)"
        )
        second = ffmpeg_tools.FfmpegCandidate(path="/winget/ffmpeg", source="PATH")
        responses = {
            first.path: self._completed(
                1, "[h264_nvenc] Cannot load nvEncodeAPI64.dll"
            ),
            second.path: self._completed(0),
        }

        def _fake_run(cmd, **kwargs):
            return responses[cmd[0]]

        with mock.patch(
            "pipeline.ffmpeg_tools._iter_candidates",
            return_value=iter([first, second]),
        ), mock.patch(
            "pipeline.ffmpeg_tools.resolve_ffmpeg",
            return_value=first.path,
        ), mock.patch(
            "pipeline.ffmpeg_tools.subprocess.run", side_effect=_fake_run
        ):
            chosen = ffmpeg_tools.select_video_codec()
        self.assertEqual(chosen, ffmpeg_tools.DEFAULT_VIDEO_CODEC)
        self.assertEqual(ffmpeg_tools._cache.get("ffmpeg"), second.path)


class TestActiveEnvBindirPriority(unittest.TestCase):
    """The active venv/conda env's bin directory is preferred over a generic
    PATH lookup so Pinokio's conda-installed ffmpeg is picked deterministically
    regardless of whether the conda activation script has fully populated
    PATH yet (it usually has, but DLL search paths sometimes lag)."""

    def setUp(self) -> None:
        ffmpeg_tools.clear_cache()

    def tearDown(self) -> None:
        ffmpeg_tools.clear_cache()

    def test_returns_none_when_not_in_a_venv(self) -> None:
        """sys.prefix == sys.base_prefix means we're using the system
        Python directly; treating its bin/ as a special-priority slot
        would shadow the user's chosen ffmpeg, so we deliberately skip."""
        with mock.patch("pipeline.ffmpeg_tools.sys") as fake_sys:
            fake_sys.prefix = "/usr"
            fake_sys.base_prefix = "/usr"
            fake_sys.platform = "linux"
            self.assertIsNone(ffmpeg_tools._active_env_bindir())


if __name__ == "__main__":
    unittest.main()
