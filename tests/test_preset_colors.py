"""Unit tests for :mod:`pipeline.preset_colors`."""

from __future__ import annotations

import unittest

from pipeline.preset_colors import resolve_text_colors


# Real preset palettes (dark -> bright) so the tests pin the actual
# user-visible behaviour. Copied from ``presets/*.yaml`` — keep in sync if a
# preset's ``colors`` list is re-ordered.
_COSMIC = [
    "#120458",  # dark navy
    "#3A0CA3",
    "#7B2CBF",
    "#4CC9F0",
    "#F72585",  # hot pink
]
_NEON_SYNTHWAVE = [
    "#1A0A2E",
    "#16213E",
    "#00F5FF",  # cyan
    "#FF2EE6",  # magenta
    "#E8F7FF",  # near white
]
_MINIMAL_MONO = [
    "#0D0D0D",  # near black
    "#F5F5F0",
    "#8A8A8A",
    "#FFFFFF",  # white
]
_LOFI_WARM = [
    "#5C4033",
    "#8B5A2B",
    "#D4A574",
    "#E8B86D",
    "#FFF8E7",  # cream
]


class TestResolveTextColors(unittest.TestCase):
    def test_empty_palette_returns_defaults(self) -> None:
        self.assertEqual(resolve_text_colors(None), ("#FFFFFF", None))
        self.assertEqual(resolve_text_colors([]), ("#FFFFFF", None))

    def test_custom_defaults_propagate(self) -> None:
        self.assertEqual(
            resolve_text_colors(
                None, default_fill="#AAAAAA", default_glow="#111111"
            ),
            ("#AAAAAA", "#111111"),
        )

    def test_single_color_palette_uses_default_glow(self) -> None:
        fill, glow = resolve_text_colors(["#abcdef"], default_glow="#000000")
        self.assertEqual(fill, "#ABCDEF")
        self.assertEqual(glow, "#000000")

    def test_cosmic_picks_bright_and_saturated_glow(self) -> None:
        fill, glow = resolve_text_colors(_COSMIC)
        # Fill must be one of the bright end slots (cyan or hot pink), never
        # one of the dark navy/purple entries that previously made the title
        # unreadable.
        self.assertIn(fill, {"#4CC9F0", "#F72585"})
        # Glow must exist and be distinctly not the fill.
        self.assertIsNotNone(glow)
        self.assertNotEqual(glow, fill)
        # Glow should be saturated (the point of this fix — cosmic's #120458
        # / #3A0CA3 glow on a dark sky is exactly the screenshot bug).
        self.assertIn(glow, {"#F72585", "#7B2CBF", "#4CC9F0", "#3A0CA3"})

    def test_neon_synthwave_fill_is_bright_and_glow_is_neon(self) -> None:
        fill, glow = resolve_text_colors(_NEON_SYNTHWAVE)
        # Brightest entry is the near-white.
        self.assertEqual(fill, "#E8F7FF")
        # Glow should be one of the two saturated neons, not the near-black
        # base tones.
        self.assertIn(glow, {"#00F5FF", "#FF2EE6"})

    def test_minimal_mono_still_produces_usable_pair(self) -> None:
        fill, glow = resolve_text_colors(_MINIMAL_MONO)
        self.assertEqual(fill, "#FFFFFF")
        # Low-chroma palette: every candidate has chroma 0, so the glow
        # falls through to the luma tie-break and picks a darker grey.
        # White fill + dark grey halo is the classic Swiss-design stroke,
        # so any entry that isn't pure white is acceptable here.
        self.assertIn(glow, {"#0D0D0D", "#8A8A8A", "#F5F5F0"})
        self.assertNotEqual(glow, fill)

    def test_lofi_warm_picks_cream_fill(self) -> None:
        fill, glow = resolve_text_colors(_LOFI_WARM)
        self.assertEqual(fill, "#FFF8E7")
        # Glow should be one of the warm mid-tones, not the darkest brown.
        self.assertIn(glow, {"#D4A574", "#E8B86D", "#8B5A2B"})

    def test_glow_has_meaningful_luma_gap(self) -> None:
        """The glow should be visibly darker than the fill so the halo reads."""
        for palette in (_COSMIC, _NEON_SYNTHWAVE, _LOFI_WARM):
            fill, glow = resolve_text_colors(palette)
            self.assertIsNotNone(glow)

            def _luma(h: str) -> float:
                r = int(h[1:3], 16)
                g = int(h[3:5], 16)
                b = int(h[5:7], 16)
                return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

            self.assertLess(
                _luma(glow),  # type: ignore[arg-type]
                _luma(fill),
                f"glow ({glow}) not darker than fill ({fill}) for palette {palette}",
            )

    def test_hex_values_are_normalised_to_uppercase(self) -> None:
        fill, glow = resolve_text_colors(["#0d0d0d", "#ffffff"])
        self.assertEqual(fill, "#FFFFFF")
        self.assertEqual(glow, "#0D0D0D")

    def test_invalid_hex_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_text_colors(["not-a-color"])
        with self.assertRaises(ValueError):
            resolve_text_colors(["#12345"])  # too short


if __name__ == "__main__":
    unittest.main()
