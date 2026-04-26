"""Tests for :mod:`pipeline.zoom_punch`."""

from __future__ import annotations

import unittest

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.zoom_punch import zoom_scale


def _make_zoom(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: float,
) -> EffectClip:
    s = {k: v for k, v in settings.items()}
    return EffectClip(
        id=id_,
        kind=EffectKind.ZOOM_PUNCH,
        t_start=t_start,
        duration_s=duration_s,
        settings=s,
    )


class TestZoomPunch(unittest.TestCase):
    def test_one_outside_and_at_end(self) -> None:
        c = _make_zoom("a", 1.0, 1.0, peak_scale=1.2)
        self.assertEqual(zoom_scale(0.0, [c]), 1.0)
        self.assertEqual(zoom_scale(0.99, [c]), 1.0)
        self.assertEqual(zoom_scale(2.0, [c]), 1.0)

    def test_peak_reached_mid_hold(self) -> None:
        c = _make_zoom(
            "b",
            0.0,
            2.0,
            peak_scale=1.25,
            ease_in_s=0.1,
            ease_out_s=0.1,
            width_frac=1.0,
        )
        self.assertAlmostEqual(zoom_scale(0.5, [c]), 1.25, places=6)

    def test_ease_in_starts_at_identity(self) -> None:
        c = _make_zoom(
            "c",
            0.0,
            2.0,
            peak_scale=1.2,
            ease_in_s=0.2,
            ease_out_s=0.2,
            width_frac=1.0,
        )
        self.assertEqual(zoom_scale(0.0, [c]), 1.0)

    def test_peak_at_or_below_one_is_no_zoom(self) -> None:
        c = _make_zoom("d", 0.0, 1.0, peak_scale=1.0, width_frac=1.0)
        self.assertEqual(zoom_scale(0.5, [c]), 1.0)

    def test_width_frac_limits_punch_prefix(self) -> None:
        c = _make_zoom(
            "e",
            0.0,
            2.0,
            peak_scale=1.3,
            ease_in_s=0.0,
            ease_out_s=0.0,
            width_frac=0.25,
        )
        self.assertAlmostEqual(zoom_scale(0.1, [c]), 1.3, places=6)
        self.assertEqual(zoom_scale(0.6, [c]), 1.0)

    def test_overlapping_uses_max(self) -> None:
        t = 0.5
        a = _make_zoom("a", 0.0, 2.0, peak_scale=1.1, ease_in_s=0.0, ease_out_s=0.0)
        b = _make_zoom("b", 0.0, 2.0, peak_scale=1.3, ease_in_s=0.0, ease_out_s=0.0)
        self.assertAlmostEqual(zoom_scale(t, [a, b]), 1.3, places=6)

    def test_non_zoom_ignored(self) -> None:
        z = _make_zoom("z", 0.0, 2.0, peak_scale=1.2, ease_in_s=0.0, ease_out_s=0.0)
        other = EffectClip(
            id="beam1",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=2.0,
            settings={"strength": 0.5, "color_hex": "#ffffff"},
        )
        t = 0.5
        self.assertEqual(zoom_scale(t, [z, other]), zoom_scale(t, [z]))

    def test_nonfinite_t_returns_identity(self) -> None:
        c = _make_zoom("n", 0.0, 1.0, peak_scale=1.5, ease_in_s=0.0, ease_out_s=0.0)
        self.assertEqual(zoom_scale(float("nan"), [c]), 1.0)
        self.assertEqual(zoom_scale(float("inf"), [c]), 1.0)

    def test_empty_clips(self) -> None:
        self.assertEqual(zoom_scale(0.0, []), 1.0)

    def test_defaults_give_peak_above_one_inside(self) -> None:
        c = EffectClip(
            id="def",
            kind=EffectKind.ZOOM_PUNCH,
            t_start=0.0,
            duration_s=2.0,
            settings={},
        )
        mid = zoom_scale(0.5, [c])
        self.assertGreater(mid, 1.0)
        self.assertAlmostEqual(mid, 1.12, places=6)


if __name__ == "__main__":
    unittest.main()
