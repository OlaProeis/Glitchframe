"""Thumbnail sampling time helpers."""

from __future__ import annotations

import unittest

from pipeline.compositor import CompositorConfig
from pipeline.effects_timeline import EffectClip, EffectKind, EffectsTimeline
from pipeline.thumbnail import pick_thumbnail_time, resolve_thumbnail_sample_time


class TestThumbnailSampleTime(unittest.TestCase):
    def test_pick_thumbnail_time_clips_to_duration(self) -> None:
        analysis = {"duration_sec": 5.0, "rms": {"fps": 10, "values": [1.0, 9.0, 1.0]}}
        t = pick_thumbnail_time(analysis)
        self.assertLessEqual(t, 5.0 - 1e-3)

    def test_resolve_skips_opening_fade_in(self) -> None:
        analysis = {"duration_sec": 120.0, "rms": {"fps": 10, "values": [9.0] + [0.1] * 50}}
        t0 = pick_thumbnail_time(analysis)
        fade_clip = EffectClip(
            id="f",
            kind=EffectKind.FADE,
            t_start=0.0,
            duration_s=4.0,
            settings={"direction_mode": "in", "peak_alpha": 1.0},
            auto_source=False,
        )
        tl = EffectsTimeline(clips=[fade_clip])
        cfg = CompositorConfig(effects_timeline=tl, fps=30)
        t = resolve_thumbnail_sample_time(analysis, cfg)
        self.assertGreaterEqual(t, 4.0 - 0.15)
        self.assertGreater(t, t0 - 0.01)


if __name__ == "__main__":
    unittest.main()
