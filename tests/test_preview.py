"""Tests for ``pipeline.preview.pick_loudest_window_start``."""

from __future__ import annotations

import unittest

from pipeline.preview import (
    DEFAULT_PREVIEW_WINDOW_SEC,
    pick_loudest_window_start,
)


class TestPickLoudestWindowStart(unittest.TestCase):
    def test_rms_peak_centred_near_end(self) -> None:
        # 30 fps envelope, 20 seconds long; the loudest 1-second block sits at
        # seconds 15 so the 2-second window should start close to there.
        fps = 30
        duration = 20.0
        values = [0.1] * int(duration * fps)
        peak_start = int(15 * fps)
        for i in range(peak_start, peak_start + fps):
            values[i] = 1.0
        analysis = {
            "duration_sec": duration,
            "fps": fps,
            "rms": {"fps": fps, "frames": len(values), "values": values},
        }
        start = pick_loudest_window_start(analysis, window_sec=2.0)
        self.assertGreaterEqual(start, 14.0)
        self.assertLessEqual(start, 15.5)

    def test_short_track_returns_zero(self) -> None:
        fps = 30
        values = [0.5] * 60  # 2 seconds of data
        analysis = {
            "duration_sec": 2.0,
            "fps": fps,
            "rms": {"fps": fps, "frames": len(values), "values": values},
        }
        self.assertEqual(
            pick_loudest_window_start(analysis, window_sec=10.0),
            0.0,
        )

    def test_missing_rms_returns_zero(self) -> None:
        self.assertEqual(
            pick_loudest_window_start({}, window_sec=DEFAULT_PREVIEW_WINDOW_SEC),
            0.0,
        )
        self.assertEqual(
            pick_loudest_window_start({"rms": {"values": []}}, window_sec=5.0),
            0.0,
        )

    def test_start_clamped_to_duration_minus_window(self) -> None:
        fps = 10
        duration = 12.0
        # Loudness biased to the very end so argmax would overshoot.
        values = [0.1] * int(duration * fps)
        for i in range(int(10.5 * fps), int(duration * fps)):
            values[i] = 1.0
        analysis = {
            "duration_sec": duration,
            "fps": fps,
            "rms": {"fps": fps, "frames": len(values), "values": values},
        }
        start = pick_loudest_window_start(analysis, window_sec=5.0)
        self.assertLessEqual(start, duration - 5.0 + 1e-6)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            pick_loudest_window_start({}, window_sec=0.0)
        with self.assertRaises(ValueError):
            pick_loudest_window_start({}, window_sec=-1.0)


if __name__ == "__main__":
    unittest.main()
