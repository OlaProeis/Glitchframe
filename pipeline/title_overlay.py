"""
Static per-frame title card: render ``Artist - Title`` once via Skia into a
premultiplied RGBA layer and alpha-blend it over every video frame.

Unlike :mod:`pipeline.thumbnail` (which rasterises the line directly onto a
single 1080p still for YouTube covers), this module produces a pre-rendered
RGBA layer sized to the *video* resolution. The compositor reuses the same
layer for every frame in the render window, so the per-frame cost is just an
``np.clip + alpha blend`` — negligible compared to the reactive shader or
kinetic typography.

Positions use a 9-point grid (``top-left`` through ``bottom-right``); size is
picked from a small/medium/large preset that scales relative to the
compositor's base :attr:`~pipeline.compositor.CompositorConfig.font_size` so
the card works at 720p, 1080p and 4K without re-tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import skia

from pipeline.kinetic_typography import KineticTypographyLayer

LOGGER = logging.getLogger(__name__)


TITLE_POSITIONS: tuple[str, ...] = (
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)

_POSITION_ALIASES: dict[str, tuple[str, ...]] = {
    "top-left": ("top-left", "topleft", "tl"),
    "top-center": ("top-center", "top", "topcenter", "tc"),
    "top-right": ("top-right", "topright", "tr"),
    "middle-left": ("middle-left", "left", "middleleft", "ml"),
    "center": ("center", "centre", "middle", "mc"),
    "middle-right": ("middle-right", "right", "middleright", "mr"),
    "bottom-left": ("bottom-left", "bottomleft", "bl"),
    "bottom-center": ("bottom-center", "bottom", "bottomcenter", "bc"),
    "bottom-right": ("bottom-right", "bottomright", "br"),
}

# Size presets map to a font-size multiplier (relative to the compositor
# ``font_size``). The numbers are picked so "medium" is unobtrusive but
# readable, "small" is a corner credit, and "large" is a proper title card.
TITLE_SIZE_MULTIPLIERS: Mapping[str, float] = {
    "small": 0.55,
    "medium": 0.80,
    "large": 1.15,
}
DEFAULT_TITLE_SIZE = "small"
DEFAULT_TITLE_POSITION = "bottom-left"

# Horizontal / vertical padding from the frame edges, as a fraction of frame
# dimensions. 4% keeps the text off the bleed area without looking floating.
_EDGE_PAD_X = 0.04
_EDGE_PAD_Y = 0.05


def normalize_title_position(label: str) -> str:
    """Map free-form UI labels to one of :data:`TITLE_POSITIONS`."""
    key = label.strip().lower().replace(" ", "-")
    for canonical, aliases in _POSITION_ALIASES.items():
        if key == canonical or key in aliases:
            return canonical
    raise ValueError(
        f"Unknown title position: {label!r}; expected one of {TITLE_POSITIONS}"
    )


def normalize_title_size(label: str) -> str:
    """Map ``small`` / ``medium`` / ``large`` (any case) to a canonical key."""
    key = label.strip().lower()
    if key not in TITLE_SIZE_MULTIPLIERS:
        raise ValueError(
            f"Unknown title size: {label!r}; expected one of "
            f"{tuple(TITLE_SIZE_MULTIPLIERS)}"
        )
    return key


def format_title_text(artist: str | None, title: str | None) -> str | None:
    """Compose the rendered line from metadata, or ``None`` when both blank."""
    a = (artist or "").strip()
    t = (title or "").strip()
    if a and t:
        return f"{a} - {t}"
    return t or a or None


def _parse_hex_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.strip()
    if not s.startswith("#") or len(s) != 7:
        raise ValueError(f"Expected #RRGGBB hex color, got {hex_str!r}")
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return r, g, b


def _argb_color(rgb: tuple[int, int, int], alpha: float) -> int:
    a = int(round(max(0.0, min(1.0, alpha)) * 255.0))
    r, g, b = rgb
    return (a << 24) | (r << 16) | (g << 8) | b


def _configure_font_for_display(font: "skia.Font") -> None:
    """Tune a :class:`skia.Font` for crisp display-size text.

    Skia defaults to ``kNormal`` hinting, which is designed for tiny UI text
    (12-16 px) and distorts glyph outlines at the larger sizes we render
    here — that's the main source of the "bad edges" on the burned-in card.
    For display-size type the best combination is no hinting plus subpixel
    positioning, which lets Skia's greyscale anti-aliasing produce smooth,
    evenly-weighted strokes.
    """
    try:
        font.setEdging(skia.Font.Edging.kAntiAlias)
    except AttributeError:  # pragma: no cover - older skia-python
        pass
    try:
        font.setSubpixel(True)
    except AttributeError:  # pragma: no cover - older skia-python
        pass
    try:
        font.setHinting(skia.FontHinting.kNone)
    except AttributeError:  # pragma: no cover - older skia-python
        pass


def _text_anchor(
    position: str, frame_w: int, frame_h: int, text_w: float, text_h: float
) -> tuple[float, float]:
    """Return ``(x_baseline_start, y_baseline)`` for the given grid cell."""
    pad_x = _EDGE_PAD_X * float(frame_w)
    pad_y = _EDGE_PAD_Y * float(frame_h)

    if position.startswith("top-"):
        # Baseline sits ``text_h`` below the top padding so ascenders fit.
        y = pad_y + text_h
    elif position.startswith("bottom-"):
        y = float(frame_h) - pad_y
    else:
        y = float(frame_h) * 0.5 + text_h * 0.35

    if position.endswith("-left"):
        x = pad_x
    elif position.endswith("-right"):
        x = float(frame_w) - pad_x - text_w
    else:
        x = (float(frame_w) - text_w) * 0.5
    return x, y


def render_title_rgba(
    text: str,
    *,
    width: int,
    height: int,
    font_path: Path | str | None,
    font_size: float,
    size: str = DEFAULT_TITLE_SIZE,
    position: str = DEFAULT_TITLE_POSITION,
    fill_hex: str = "#FFFFFF",
    shadow_hex: str | None = None,
    alpha: float = 1.0,
) -> np.ndarray | None:
    """Rasterise ``text`` into a ``(height, width, 4)`` premultiplied RGBA layer.

    Returns ``None`` when ``text`` is blank, so callers can skip the composite
    path without branching on an empty array.
    """
    line = " ".join((text or "").split())
    if not line:
        return None
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid title layer size: {width}x{height}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")

    size_key = normalize_title_size(size)
    position_key = normalize_title_position(position)

    # The compositor's ``font_size`` is tuned for full-screen kinetic
    # typography; the title card should be smaller so it doesn't dominate.
    base_px = max(14.0, float(font_size) * TITLE_SIZE_MULTIPLIERS[size_key])
    max_w = float(width) * (1.0 - 2.0 * _EDGE_PAD_X)

    typeface = KineticTypographyLayer._load_typeface(font_path)

    # Iteratively shrink until the rendered line fits. This mirrors
    # ``pipeline.thumbnail`` so behaviour is consistent between the static
    # YouTube thumbnail and the burned-in card.
    size_px = base_px
    font: skia.Font | None = None
    while size_px >= 10.0:
        font = skia.Font(typeface, size_px)
        _configure_font_for_display(font)
        tw = float(font.measureText(line))
        if tw <= max_w:
            break
        size_px *= 0.92
    if font is None:  # pragma: no cover - only possible with empty line
        font = skia.Font(typeface, 10.0)
        _configure_font_for_display(font)

    text_w = float(font.measureText(line))
    metrics = font.getMetrics()
    text_h = float(metrics.fDescent - metrics.fAscent)

    pixels: np.ndarray = np.zeros((height, width, 4), dtype=np.uint8)
    surface = skia.Surface(pixels)

    fill_rgb = _parse_hex_rgb(fill_hex)
    shadow_rgb = _parse_hex_rgb(shadow_hex) if shadow_hex else (0, 0, 0)

    x0, y_base = _text_anchor(position_key, width, height, text_w, text_h)

    # --- Layered bloom recipe -----------------------------------------------
    # Three stacked Gaussian passes approximate a soft neon / display glow.
    # Each uses ``kOuter_BlurStyle`` so the blur only bleeds *outside* the
    # glyph — the fill stays crisp instead of being washed out by the wide
    # pass. Wide-to-narrow sigmas with decreasing alpha give a smooth
    # falloff (bright edge, soft halo, distant wash) rather than the hard
    # "stamped blob" a single-pass normal blur produces.
    #
    # Followed by a thin outline for edge contrast on similar-luminance
    # backgrounds, and finally the crisp fill on top.
    wide_sigma = max(1.6, size_px * 0.22)
    mid_sigma = max(1.0, size_px * 0.11)
    tight_sigma = max(0.6, size_px * 0.045)

    has_shadow = shadow_hex is not None
    wide_alpha = (0.18 if has_shadow else 0.14) * alpha
    mid_alpha = (0.30 if has_shadow else 0.24) * alpha
    tight_alpha = (0.55 if has_shadow else 0.42) * alpha

    stroke_width = max(0.5, size_px * 0.010)
    stroke_alpha = 0.26 * alpha

    # ``kOuter_BlurStyle`` / ``MakeBlur`` aren't part of every skia-python
    # build — detect at runtime so CI / older wheels still get the old
    # single-pass halo rather than a hard crash.
    outer_style = getattr(skia, "kOuter_BlurStyle", None)
    normal_style = getattr(skia, "kNormal_BlurStyle", None)

    def _blur_paint(rgb: tuple[int, int, int], sigma: float, a: float) -> skia.Paint:
        paint = skia.Paint(AntiAlias=True, Color=_argb_color(rgb, a))
        style = outer_style if outer_style is not None else normal_style
        if style is not None:
            paint.setMaskFilter(skia.MaskFilter.MakeBlur(style, sigma))
        return paint

    with surface as canvas:
        for sigma, halo_alpha in (
            (wide_sigma, wide_alpha),
            (mid_sigma, mid_alpha),
            (tight_sigma, tight_alpha),
        ):
            canvas.drawString(
                line, x0, y_base, font, _blur_paint(shadow_rgb, sigma, halo_alpha)
            )

        stroke_paint = skia.Paint(
            AntiAlias=True,
            Color=_argb_color(shadow_rgb, stroke_alpha),
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=stroke_width,
            StrokeJoin=skia.Paint.kRound_Join,
        )
        canvas.drawString(line, x0, y_base, font, stroke_paint)

        main_paint = skia.Paint(
            AntiAlias=True, Color=_argb_color(fill_rgb, alpha)
        )
        canvas.drawString(line, x0, y_base, font, main_paint)

    # skia-python surfaces over an np.ndarray use BGRA premultiplied by
    # default; swap B/R channels so the result is valid RGBA for
    # :func:`pipeline.reactive_shader.composite_premultiplied_rgba_over_rgb`.
    rgba = np.empty_like(pixels)
    rgba[..., 0] = pixels[..., 2]
    rgba[..., 1] = pixels[..., 1]
    rgba[..., 2] = pixels[..., 0]
    rgba[..., 3] = pixels[..., 3]
    return rgba


__all__ = [
    "DEFAULT_TITLE_POSITION",
    "DEFAULT_TITLE_SIZE",
    "TITLE_POSITIONS",
    "TITLE_SIZE_MULTIPLIERS",
    "format_title_text",
    "normalize_title_position",
    "normalize_title_size",
    "render_title_rgba",
]
