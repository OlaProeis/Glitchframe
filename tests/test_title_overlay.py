"""Unit tests for :mod:`pipeline.title_overlay`."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.title_overlay import (
    TITLE_POSITIONS,
    format_title_text,
    normalize_title_position,
    normalize_title_size,
    render_title_rgba,
)


class TestNormalizers(unittest.TestCase):
    def test_position_aliases_roundtrip_to_canonical(self) -> None:
        self.assertEqual(normalize_title_position("Top-center"), "top-center")
        self.assertEqual(normalize_title_position("TOP"), "top-center")
        self.assertEqual(normalize_title_position("bottom right"), "bottom-right")
        self.assertEqual(normalize_title_position("center"), "center")
        self.assertEqual(normalize_title_position("middle"), "center")

    def test_unknown_position_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_title_position("diagonal")

    def test_size_normalizer_case_insensitive(self) -> None:
        self.assertEqual(normalize_title_size("Medium"), "medium")
        self.assertEqual(normalize_title_size("LARGE"), "large")

    def test_unknown_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_title_size("extra-large")

    def test_all_documented_positions_accepted(self) -> None:
        for canonical in TITLE_POSITIONS:
            self.assertEqual(normalize_title_position(canonical), canonical)


class TestFormatTitleText(unittest.TestCase):
    def test_both_present_uses_hyphen(self) -> None:
        self.assertEqual(format_title_text("Daft Punk", "Around the World"),
                         "Daft Punk - Around the World")

    def test_only_title_returns_title(self) -> None:
        self.assertEqual(format_title_text("", "Solo"), "Solo")
        self.assertEqual(format_title_text(None, "Solo"), "Solo")

    def test_only_artist_returns_artist(self) -> None:
        self.assertEqual(format_title_text("Artist", ""), "Artist")

    def test_blank_returns_none(self) -> None:
        self.assertIsNone(format_title_text(None, None))
        self.assertIsNone(format_title_text("  ", "\t"))


class TestRenderTitleRgba(unittest.TestCase):
    def test_blank_text_returns_none(self) -> None:
        out = render_title_rgba(
            "   ", width=320, height=120, font_path=None, font_size=40.0
        )
        self.assertIsNone(out)

    def test_returns_expected_shape_and_dtype(self) -> None:
        out = render_title_rgba(
            "Hello World",
            width=320,
            height=120,
            font_path=None,
            font_size=40.0,
            size="small",
            position="top-center",
        )
        assert out is not None  # for type narrowing
        self.assertEqual(out.shape, (120, 320, 4))
        self.assertEqual(out.dtype, np.uint8)

    def test_text_actually_rasterised(self) -> None:
        out = render_title_rgba(
            "A",
            width=320,
            height=120,
            font_path=None,
            font_size=40.0,
            fill_hex="#FFFFFF",
            size="large",
        )
        assert out is not None
        # Some pixels in the alpha channel should be non-zero (text was drawn).
        self.assertGreater(int(out[..., 3].max()), 0)

    def test_top_position_draws_above_center(self) -> None:
        top = render_title_rgba(
            "X",
            width=320,
            height=240,
            font_path=None,
            font_size=40.0,
            position="top-center",
        )
        bot = render_title_rgba(
            "X",
            width=320,
            height=240,
            font_path=None,
            font_size=40.0,
            position="bottom-center",
        )
        assert top is not None and bot is not None
        # Compare vertical centroid of the alpha mask: top variant's text
        # mass should sit well above the bottom variant's.
        def _centroid_y(arr: np.ndarray) -> float:
            a = arr[..., 3].astype(np.float64)
            total = a.sum()
            if total <= 0:
                return 0.0
            ys = np.arange(a.shape[0])
            return float((a.sum(axis=1) * ys).sum() / total)

        self.assertLess(_centroid_y(top), _centroid_y(bot))


if __name__ == "__main__":
    unittest.main()
