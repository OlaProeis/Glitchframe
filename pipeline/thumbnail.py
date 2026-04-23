"""
YouTube-style thumbnail: sample one frame (chorus / loudest RMS window), overlay
a static title line with Skia using preset colors, save ``1920×1080`` PNG.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import skia
from PIL import Image

from pipeline.background import BackgroundSource
from pipeline.compositor import CompositorConfig, render_single_frame
from pipeline.kinetic_typography import DEFAULT_BASELINE_RATIO, KineticTypographyLayer
from pipeline.preset_colors import resolve_text_colors
from pipeline.reactive_shader import composite_premultiplied_rgba_over_rgb

LOGGER = logging.getLogger(__name__)

THUMB_WIDTH = 1920
THUMB_HEIGHT = 1080


def _parse_hex_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.strip()
    if not s.startswith("#") or len(s) != 7:
        raise ValueError(f"Expected #RRGGBB hex color, got {hex_str!r}")
    try:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color {hex_str!r}: {exc}") from exc
    return r, g, b


def _argb_color(rgb: tuple[int, int, int], alpha: float) -> int:
    a = int(round(max(0.0, min(1.0, alpha)) * 255.0))
    r, g, b = rgb
    return (a << 24) | (r << 16) | (g << 8) | b


def _swizzle_bgra_to_rgba(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) uint8 pixels, got shape {pixels.shape}")
    out = np.empty_like(pixels)
    out[..., 0] = pixels[..., 2]
    out[..., 1] = pixels[..., 1]
    out[..., 2] = pixels[..., 0]
    out[..., 3] = pixels[..., 3]
    return out


def pick_thumbnail_time(analysis: Mapping[str, Any]) -> float:
    """
    Prefer the first downbeat on or after the second structural segment's
    start (proxy for chorus); otherwise the loudest 1 s-smoothed RMS window.
    """
    fps = int(analysis.get("fps") or 30)
    duration = float(analysis.get("duration_sec") or 0.0)

    segments = analysis.get("segments")
    t: float
    if isinstance(segments, list) and len(segments) >= 2:
        seg = segments[1]
        if isinstance(seg, dict) and "t_start" in seg:
            t_chorus = float(seg["t_start"])
            dbs = analysis.get("downbeats") or []
            if isinstance(dbs, list) and dbs:
                times = sorted(float(x) for x in dbs)
                chosen = next((db for db in times if db >= t_chorus - 1e-6), None)
                t = float(chosen) if chosen is not None else float(times[0])
            else:
                t = t_chorus
        else:
            t = _time_from_rms_peak_smoothed(analysis, fps)
    else:
        t = _time_from_rms_peak_smoothed(analysis, fps)

    if duration > 0:
        t = max(0.0, min(t, duration - 1e-3))
    return t


def _time_from_rms_peak_smoothed(analysis: Mapping[str, Any], default_fps: int) -> float:
    rms_block = analysis.get("rms") or {}
    values = rms_block.get("values")
    rfps = int(rms_block.get("fps") or analysis.get("fps") or default_fps or 30)
    if not isinstance(values, list) or not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    win = max(1, int(round(rfps * 1.0)))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    smoothed = np.convolve(arr, kernel, mode="same")
    idx = int(np.argmax(smoothed))
    return (idx + 0.5) / float(rfps)


def _resolve_title_colors(
    palette: Sequence[str] | None,
    cfg: CompositorConfig,
) -> tuple[str, str | None]:
    """Pick a readable ``(fill, glow)`` pair for the thumbnail line.

    Prefers :func:`pipeline.preset_colors.resolve_text_colors` when a preset
    palette is available (dark -> bright convention, text wants the bright
    end). Falls back to ``cfg.base_color`` / ``cfg.shadow_color`` so callers
    that pre-resolved the pair keep working.
    """
    if palette:
        return resolve_text_colors(
            palette,
            default_fill=cfg.base_color,
            default_glow=cfg.shadow_color,
        )
    return cfg.base_color, cfg.shadow_color


def _overlay_title_line_skia(
    rgb: np.ndarray,
    text: str,
    *,
    font_path: Path | str | None,
    font_size: float,
    fill_hex: str,
    shadow_hex: str | None,
    baseline_y_ratio: float = DEFAULT_BASELINE_RATIO,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    line = " ".join(text.split())
    if not line:
        return rgb

    pixels: np.ndarray = np.zeros((h, w, 4), dtype=np.uint8)
    surface = skia.Surface(pixels)
    typeface = KineticTypographyLayer._load_typeface(font_path)
    max_w = float(w) * 0.92
    size_px = float(max(12.0, font_size * 1.12))
    font: skia.Font | None = None
    while size_px >= 12.0:
        font = skia.Font(typeface, size_px)
        try:
            font.setEdging(skia.Font.Edging.kAntiAlias)
        except AttributeError:  # pragma: no cover
            pass
        tw = float(font.measureText(line))
        if tw <= max_w:
            break
        size_px *= 0.9

    if font is None:  # pragma: no cover - loop always assigns when line non-empty
        font = skia.Font(typeface, 12.0)

    fill_rgb = _parse_hex_rgb(fill_hex)
    shadow_rgb = _parse_hex_rgb(shadow_hex) if shadow_hex else (0, 0, 0)
    tw = float(font.measureText(line))
    x0 = (float(w) - tw) * 0.5
    y_base = baseline_y_ratio * float(h)

    # Layered bloom — matches :mod:`pipeline.title_overlay` so the static
    # thumbnail and the burned-in card read as the same typographic style.
    # Wide-to-narrow ``kOuter_BlurStyle`` passes fall off smoothly around
    # the glyph edges without bleeding into the fill. ``setMaskFilter`` is
    # guarded so older skia-python wheels that don't expose ``MakeBlur`` /
    # ``kOuter_BlurStyle`` still render a plain fill instead of crashing.
    # Match :mod:`pipeline.title_overlay` halos (smaller, softer bloom).
    wide_sigma = max(1.2, size_px * 0.14)
    mid_sigma = max(0.75, size_px * 0.07)
    tight_sigma = max(0.45, size_px * 0.03)
    has_shadow = shadow_hex is not None
    wide_alpha = 0.10 if has_shadow else 0.08
    mid_alpha = 0.18 if has_shadow else 0.14
    tight_alpha = 0.36 if has_shadow else 0.28

    outer_style = getattr(skia, "kOuter_BlurStyle", None)
    normal_style = getattr(skia, "kNormal_BlurStyle", None)

    def _blur_paint(sigma: float, a: float) -> skia.Paint:
        paint = skia.Paint(AntiAlias=True, Color=_argb_color(shadow_rgb, a))
        style = outer_style if outer_style is not None else normal_style
        if style is not None:
            paint.setMaskFilter(skia.MaskFilter.MakeBlur(style, sigma))
        return paint

    stroke_width = max(0.5, size_px * 0.008)
    stroke_paint = skia.Paint(
        AntiAlias=True,
        Color=_argb_color(shadow_rgb, 0.18),
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=stroke_width,
        StrokeJoin=skia.Paint.kRound_Join,
    )

    with surface as canvas:
        for sigma, halo_alpha in (
            (wide_sigma, wide_alpha),
            (mid_sigma, mid_alpha),
            (tight_sigma, tight_alpha),
        ):
            canvas.drawString(line, x0, y_base, font, _blur_paint(sigma, halo_alpha))

        canvas.drawString(line, x0, y_base, font, stroke_paint)

        main_paint = skia.Paint(
            AntiAlias=True,
            Color=_argb_color(fill_rgb, 1.0),
        )
        canvas.drawString(line, x0, y_base, font, main_paint)

    rgba = _swizzle_bgra_to_rgba(pixels)
    return composite_premultiplied_rgba_over_rgb(rgba, rgb)


def save_thumbnail_png(
    path: Path | str,
    *,
    line: str,
    analysis: Mapping[str, Any],
    background: BackgroundSource,
    config: CompositorConfig,
    palette: Sequence[str] | None = None,
) -> Path:
    """
    Render one frame at :func:`pick_thumbnail_time`, overlay ``line``, resize
    to 1920×1080, and write a PNG.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = pick_thumbnail_time(analysis)
    LOGGER.debug("Thumbnail sampling time: %.3fs", t)

    # Omit the per-frame burned title: ``save_thumbnail_png`` draws its own
    # centered line; keeping ``title_text`` would duplicate artist/title.
    frame_cfg = replace(config, title_text=None)
    frame = render_single_frame(
        t,
        background=background,
        analysis=analysis,
        aligned_words=None,
        config=frame_cfg,
    )

    fill_hex, shadow_hex = _resolve_title_colors(palette, config)
    title_face = config.title_font_path or config.font_path
    composed = _overlay_title_line_skia(
        frame,
        line.strip(),
        font_path=title_face,
        font_size=config.font_size,
        fill_hex=fill_hex,
        shadow_hex=shadow_hex,
    )

    if composed.shape[1] != THUMB_WIDTH or composed.shape[0] != THUMB_HEIGHT:
        pil = Image.fromarray(composed, mode="RGB")
        pil = pil.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.Resampling.LANCZOS)
        composed = np.asarray(pil, dtype=np.uint8)

    Image.fromarray(composed, mode="RGB").save(out, format="PNG", optimize=True)
    return out


__all__ = [
    "THUMB_HEIGHT",
    "THUMB_WIDTH",
    "pick_thumbnail_time",
    "save_thumbnail_png",
]
