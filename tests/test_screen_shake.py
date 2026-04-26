"""Tests for :mod:`pipeline.screen_shake`."""

from __future__ import annotations

import unittest

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.screen_shake import shake_offset


def _make_shake(
    id_: str,
    t_start: float,
    duration_s: float,
    **settings: float,
) -> EffectClip:
    s = {k: v for k, v in settings.items()}
    return EffectClip(
        id=id_,
        kind=EffectKind.SCREEN_SHAKE,
        t_start=t_start,
        duration_s=duration_s,
        settings=s,
    )


class TestScreenShake(unittest.TestCase):
    def test_zero_outside_and_at_end(self) -> None:
        c = _make_shake("a", 1.0, 1.0, amplitude_px=10.0, frequency_hz=2.0)
        clips = [c]
        h = "songabc"
        self.assertEqual(shake_offset(0.99, clips, h), (0.0, 0.0))
        self.assertEqual(shake_offset(2.0, clips, h), (0.0, 0.0))

    def test_nonzero_inside_somewhere(self) -> None:
        c = _make_shake("a", 0.0, 2.0, amplitude_px=20.0, frequency_hz=3.0)
        clips = [c]
        h = "hash1"
        found = 0.0
        t = 0.0
        while t < 2.0 and found < 1e-6:
            dx, dy = shake_offset(t, clips, h)
            found = dx * dx + dy * dy
            t += 0.02
        self.assertGreater(
            found, 1e-6, "expected some non-trivial offset inside the clip"
        )

    def test_deterministic(self) -> None:
        c = _make_shake("x", 0.0, 5.0, amplitude_px=8.0, frequency_hz=1.0)
        clips = [c]
        h = "stable_hash"
        t = 1.2345
        a = shake_offset(t, clips, h)
        b = shake_offset(t, clips, h)
        self.assertEqual(a, b)

    def test_respects_zero_amplitude(self) -> None:
        c = _make_shake("z", 0.0, 10.0, amplitude_px=0.0, frequency_hz=5.0)
        for t in (0.0, 0.5, 1.0, 1.5):
            self.assertEqual(shake_offset(t, [c], "h"), (0.0, 0.0))

    def test_larger_amplitude_increases_magnitude(self) -> None:
        t = 0.55
        h = "hh"
        lo = _make_shake("c", 0.0, 2.0, amplitude_px=2.0, frequency_hz=4.0)
        hi = _make_shake("c", 0.0, 2.0, amplitude_px=20.0, frequency_hz=4.0)
        ax, ay = shake_offset(t, [lo], h)
        bx, by = shake_offset(t, [hi], h)
        mag_lo = (ax * ax + ay * ay) ** 0.5
        mag_hi = (bx * bx + by * by) ** 0.5
        self.assertGreater(mag_hi, mag_lo * 1.5)

    def test_overlapping_sums(self) -> None:
        t = 1.0
        h = "sumhash"
        a = _make_shake("a", 0.0, 2.0, amplitude_px=3.0, frequency_hz=2.0)
        b = _make_shake("b", 0.0, 2.0, amplitude_px=3.0, frequency_hz=2.0)
        combined = shake_offset(t, [a, b], h)
        a_only = shake_offset(t, [a], h)
        b_only = shake_offset(t, [b], h)
        self.assertAlmostEqual(
            combined[0], a_only[0] + b_only[0], places=5
        )
        self.assertAlmostEqual(
            combined[1], a_only[1] + b_only[1], places=5
        )

    def test_ignores_non_shake(self) -> None:
        c = _make_shake("s", 0.0, 2.0, amplitude_px=5.0, frequency_hz=2.0)
        other = EffectClip(
            id="beam1",
            kind=EffectKind.BEAM,
            t_start=0.0,
            duration_s=2.0,
            settings={"strength": 0.5, "color_hex": "#ffffff"},
        )
        t = 0.5
        self.assertEqual(shake_offset(t, [c, other], "h"), shake_offset(t, [c], "h"))


if __name__ == "__main__":
    unittest.main()
