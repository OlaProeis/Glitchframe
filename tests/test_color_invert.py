"""Tests for :mod:`pipeline.color_invert`."""

from __future__ import annotations

import unittest

from pipeline.color_invert import invert_mix
from pipeline.effects_timeline import EffectClip, EffectKind


def _make_invert(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: float,
) -> EffectClip:
    s = {k: v for k, v in settings.items()}
    return EffectClip(
        id=id_,
        kind=EffectKind.COLOR_INVERT,
        t_start=t_start,
        duration_s=duration_s,
        settings=s,
    )


class TestColorInvertMix(unittest.TestCase):
    def test_zero_outside_and_at_end(self) -> None:
        c = _make_invert("a", 1.0, 1.0, mix=1.0, intensity=1.0)
        self.assertEqual(invert_mix(0.0, [c]), 0.0)
        self.assertEqual(invert_mix(0.99, [c]), 0.0)
        self.assertEqual(invert_mix(2.0, [c]), 0.0)
        # Half-open: end boundary is outside
        self.assertEqual(invert_mix(2.0, [_make_invert("b", 1.0, 1.0)]), 0.0)

    def test_one_inside_with_default_settings(self) -> None:
        c = EffectClip(
            id="c",
            kind=EffectKind.COLOR_INVERT,
            t_start=0.0,
            duration_s=2.0,
            settings={},
        )
        self.assertEqual(invert_mix(0.0, [c]), 1.0)
        self.assertEqual(invert_mix(1.0, [c]), 1.0)
        self.assertEqual(invert_mix(1.99, [c]), 1.0)

    def test_mix_intensity_product(self) -> None:
        c = _make_invert("d", 0.0, 1.0, mix=0.5, intensity=1.0)
        self.assertEqual(invert_mix(0.0, [c]), 0.5)
        c2 = _make_invert("e", 0.0, 1.0, mix=0.5, intensity=0.5)
        self.assertEqual(invert_mix(0.0, [c2]), 0.25)

    def test_clamps_invalid_scalars_to_defaults_then_01(self) -> None:
        c = EffectClip(
            id="f",
            kind=EffectKind.COLOR_INVERT,
            t_start=0.0,
            duration_s=1.0,
            settings={"mix": "nope", "intensity": "bad"},
        )
        # Falls back to 1.0 each → 1.0
        self.assertEqual(invert_mix(0.5, [c]), 1.0)

    def test_clamps_mix_and_intensity_to_01(self) -> None:
        c = _make_invert("g", 0.0, 1.0, mix=2.0, intensity=0.3)
        self.assertEqual(invert_mix(0.0, [c]), 0.3)
        c2 = _make_invert("h", 0.0, 1.0, mix=-1.0, intensity=0.5)
        self.assertEqual(invert_mix(0.0, [c2]), 0.0)

    def test_overlapping_capped_sum(self) -> None:
        t = 0.5
        a = _make_invert("a", 0.0, 2.0, mix=0.4, intensity=1.0)
        b = _make_invert("b", 0.0, 2.0, mix=0.4, intensity=1.0)
        self.assertEqual(invert_mix(t, [a, b]), 0.8)
        a2 = _make_invert("a2", 0.0, 2.0, mix=0.6, intensity=1.0)
        b2 = _make_invert("b2", 0.0, 2.0, mix=0.6, intensity=1.0)
        self.assertEqual(invert_mix(t, [a2, b2]), 1.0)

    def test_non_invert_ignored(self) -> None:
        inv = _make_invert("i", 0.0, 2.0, mix=1.0, intensity=1.0)
        other = EffectClip(
            id="beam1",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=2.0,
            settings={"strength": 0.5, "color_hex": "#ffffff"},
        )
        t = 0.5
        self.assertEqual(invert_mix(t, [inv, other]), invert_mix(t, [inv]))

    def test_nonfinite_t_returns_zero(self) -> None:
        c = _make_invert("j", 0.0, 1.0, mix=1.0, intensity=1.0)
        self.assertEqual(invert_mix(float("nan"), [c]), 0.0)
        self.assertEqual(invert_mix(float("inf"), [c]), 0.0)

    def test_empty_clips(self) -> None:
        self.assertEqual(invert_mix(0.0, []), 0.0)

    def test_no_color_invert_clips(self) -> None:
        other = EffectClip(
            id="b2",
            kind=EffectKind.SCREEN_SHAKE,
            t_start=0.0,
            duration_s=2.0,
            settings={"amplitude_px": 4.0, "frequency_hz": 2.0},
        )
        self.assertEqual(invert_mix(0.5, [other]), 0.0)


if __name__ == "__main__":
    unittest.main()
