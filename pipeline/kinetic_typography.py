"""
Kinetic typography layer rendered with :mod:`skia-python`.

Consumes :file:`cache/<song_hash>/lyrics.aligned.json` (produced by
:mod:`pipeline.lyrics_aligner`) and renders a single transparent RGBA frame
per ``t`` seconds, suitable for compositing on top of the reactive shader
pass.

Motion presets are keyed by the ``typo_style`` string used in
``presets/*.yaml`` (``pop-in``, ``slide``, ``flicker``, ``scale-pulse``,
``beat-shake``). Each preset is a pure function that takes per-word timing
state (+ optional reactive uniforms from
:func:`pipeline.reactive_shader.uniforms_at_time`) and returns
``(alpha, scale, dx, dy)``.

Layout is a simple lower-third: the currently-active line is centered
horizontally around canvas width / 2 at ``baseline_y_ratio * height``. The
previous line fades out over ``line_fade_seconds`` after its last word ends
so consecutive lines cross-fade gracefully. All GL-less; Skia's raster
backend is used so the module works on headless hosts.

All failures raise with enough context to debug (missing font file, corrupt
aligned JSON, unknown motion preset, Skia typeface load failure, etc.) — no
silent fallbacks that would ship a blank typography pass.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import skia

LOGGER = logging.getLogger(__name__)

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FONT_SIZE = 72
# Generic Skia lower-third anchor (e.g. thumbnail title line).
DEFAULT_BASELINE_RATIO = 0.75
# Full compositor path: sit lyrics lower so they center better between a
# center-placed logo and the bottom Artist — Title imprint.
DEFAULT_KINETIC_BASELINE_RATIO = 0.82
DEFAULT_LINE_FADE_SECONDS = 0.4
DEFAULT_INTRO_SECONDS = 0.18
DEFAULT_OUTRO_SECONDS = 0.25
DEFAULT_WORD_SPACING_PX = 18.0
DEFAULT_MOTION = "pop-in"

SUPPORTED_MOTIONS: tuple[str, ...] = (
    "pop-in",
    "slide",
    "flicker",
    "scale-pulse",
    "beat-shake",
)


# ---------------------------------------------------------------------------
# Aligned-word loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedWord:
    """One user-supplied word with its WhisperX-borrowed timings."""

    word: str
    line_idx: int
    t_start: float
    t_end: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedWord":
        try:
            word = str(raw["word"])
            line_idx = int(raw["line_idx"])
            t_start = float(raw["t_start"])
            t_end = float(raw["t_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid aligned-word entry {raw!r}: {exc}"
            ) from exc
        if t_end < t_start:
            t_end = t_start
        return cls(word=word, line_idx=line_idx, t_start=t_start, t_end=t_end)


def load_aligned_words(
    aligned_json: Path | str,
) -> tuple[list[str], list[AlignedWord]]:
    """
    Read ``lyrics.aligned.json`` and return ``(lines, words)``.

    ``lines`` is the original pasted-line list (without blanks) and ``words``
    are in source order with ``line_idx`` pointing into ``lines``.
    """
    path = Path(aligned_json)
    if not path.is_file():
        raise FileNotFoundError(f"Aligned lyrics not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Aligned lyrics is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Aligned lyrics root must be a mapping; got {type(data).__name__}"
        )
    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("Aligned lyrics is missing a 'lines' list")
    lines = [str(line) for line in raw_lines]
    raw_words = data.get("words")
    if not isinstance(raw_words, list):
        raise ValueError("Aligned lyrics is missing a 'words' list")
    words = [AlignedWord.from_dict(w) for w in raw_words if isinstance(w, dict)]
    return lines, words


# ---------------------------------------------------------------------------
# Motion presets (pure functions over per-word timing + reactive uniforms)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordMotion:
    """Per-word motion state for the compositor to apply around layout pos."""

    alpha: float
    scale: float
    dx: float
    dy: float


@dataclass(frozen=True)
class _WordState:
    """Input to a motion function for a single word at time ``t``."""

    t: float
    t_start: float
    t_end: float
    intro_seconds: float
    outro_seconds: float
    word_index: int  # Stable-ish seed for per-word variation (flicker/shake).


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _envelope_alpha(state: _WordState) -> float:
    """
    Base alpha envelope shared by most motions: 0 before start, smooth ramp
    during ``intro``, 1 while active, linear fade during ``outro``, 0 after.
    """
    t = state.t
    if t < state.t_start - state.intro_seconds:
        return 0.0
    if t < state.t_start:
        return _smoothstep((t - (state.t_start - state.intro_seconds)) / max(1e-6, state.intro_seconds))
    if t <= state.t_end:
        return 1.0
    if t < state.t_end + state.outro_seconds:
        return 1.0 - _smoothstep((t - state.t_end) / max(1e-6, state.outro_seconds))
    return 0.0


MotionFn = Callable[[_WordState, Mapping[str, Any]], WordMotion]


def _motion_pop_in(state: _WordState, _u: Mapping[str, Any]) -> WordMotion:
    alpha = _envelope_alpha(state)
    # Overshoot scale during intro: 0.2 → 1.15 → 1.0.
    if state.t < state.t_start - state.intro_seconds:
        scale = 0.2
    elif state.t < state.t_start:
        p = _smoothstep(
            (state.t - (state.t_start - state.intro_seconds)) / max(1e-6, state.intro_seconds)
        )
        scale = 0.2 + (1.15 - 0.2) * p
    elif state.t < state.t_start + state.intro_seconds:
        # Quick settle from overshoot to 1.0 after the word hits t_start.
        p = _smoothstep((state.t - state.t_start) / max(1e-6, state.intro_seconds))
        scale = 1.15 + (1.0 - 1.15) * p
    else:
        scale = 1.0
    return WordMotion(alpha=alpha, scale=scale, dx=0.0, dy=0.0)


def _motion_slide(state: _WordState, _u: Mapping[str, Any]) -> WordMotion:
    alpha = _envelope_alpha(state)
    # Slide up from +48 px below during intro; drift up 24 px during outro.
    if state.t < state.t_start:
        span = max(1e-6, state.intro_seconds)
        p = _smoothstep((state.t - (state.t_start - span)) / span)
        dy = (1.0 - p) * 48.0
    elif state.t > state.t_end:
        span = max(1e-6, state.outro_seconds)
        p = _smoothstep((state.t - state.t_end) / span)
        dy = -p * 24.0
    else:
        dy = 0.0
    return WordMotion(alpha=alpha, scale=1.0, dx=0.0, dy=dy)


def _motion_flicker(state: _WordState, _u: Mapping[str, Any]) -> WordMotion:
    base = _envelope_alpha(state)
    # Deterministic pseudo-random flicker per word + time — no RNG state.
    phase = (state.t * 17.0) + float(state.word_index) * 1.37
    jitter = 0.5 + 0.5 * math.sin(phase) * math.cos(phase * 0.73 + 1.1)
    alpha = base * max(0.0, min(1.0, 0.55 + 0.45 * jitter))
    return WordMotion(alpha=alpha, scale=1.0, dx=0.0, dy=0.0)


def _motion_scale_pulse(state: _WordState, u: Mapping[str, Any]) -> WordMotion:
    alpha = _envelope_alpha(state)
    rms = float(u.get("rms", 0.0)) if u else 0.0
    onset = float(u.get("onset_pulse", 0.0)) if u else 0.0
    bh = float(u.get("bass_hit", 0.0)) if u else 0.0
    rms_n = max(0.0, min(1.0, rms))
    rms_g = rms_n * rms_n
    scale = (
        1.0
        + 0.09 * rms_g
        + 0.05 * max(0.0, min(1.0, onset))
        + 0.09 * max(0.0, min(1.0, bh))
    )
    if scale < 0.8:
        scale = 0.8
    return WordMotion(alpha=alpha, scale=scale, dx=0.0, dy=0.0)


def _motion_beat_shake(state: _WordState, u: Mapping[str, Any]) -> WordMotion:
    alpha = _envelope_alpha(state)
    phase = float(u.get("beat_phase", 0.0)) if u else 0.0
    intensity = float(u.get("intensity", 1.0)) if u else 1.0
    # Amplitude tapers as beat phase moves away from beat hit (phase -> 1).
    amp = 5.0 * max(0.0, 1.0 - phase) * max(0.0, min(1.0, intensity))
    seed = float(state.word_index) * 0.97
    dx = amp * math.sin(phase * math.tau + seed)
    dy = amp * 0.5 * math.cos(phase * math.tau * 2.0 + seed)
    return WordMotion(alpha=alpha, scale=1.0, dx=dx, dy=dy)


MOTION_PRESETS: dict[str, MotionFn] = {
    "pop-in": _motion_pop_in,
    "slide": _motion_slide,
    "flicker": _motion_flicker,
    "scale-pulse": _motion_scale_pulse,
    "beat-shake": _motion_beat_shake,
}


def get_motion_preset(name: str) -> MotionFn:
    """Return the motion function for ``name`` or raise ``KeyError``."""
    if name not in MOTION_PRESETS:
        raise KeyError(
            f"Unknown motion preset {name!r}; expected one of {sorted(MOTION_PRESETS)}"
        )
    return MOTION_PRESETS[name]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


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


def _argb_with_alpha(rgb: tuple[int, int, int], alpha: float) -> int:
    a = int(round(max(0.0, min(1.0, alpha)) * 255.0))
    r, g, b = rgb
    return (a << 24) | (r << 16) | (g << 8) | b


# ---------------------------------------------------------------------------
# Line grouping helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LineTiming:
    line_idx: int
    t_start: float
    t_end: float
    words: tuple[AlignedWord, ...]
    word_global_indices: tuple[int, ...]  # Stable seed for shake/flicker.


def _group_words_by_line(words: Sequence[AlignedWord]) -> list[_LineTiming]:
    """
    Group ``words`` into per-line timing blocks, preserving source order.
    Keeps each word's global index so motion functions can seed deterministic
    per-word variation.
    """
    buckets: dict[int, list[tuple[int, AlignedWord]]] = {}
    order: list[int] = []
    for gi, w in enumerate(words):
        if w.line_idx not in buckets:
            buckets[w.line_idx] = []
            order.append(w.line_idx)
        buckets[w.line_idx].append((gi, w))

    out: list[_LineTiming] = []
    for line_idx in order:
        entries = buckets[line_idx]
        entry_words = tuple(w for _, w in entries)
        entry_indices = tuple(i for i, _ in entries)
        t_start = min(w.t_start for w in entry_words)
        t_end = max(w.t_end for w in entry_words)
        out.append(
            _LineTiming(
                line_idx=line_idx,
                t_start=t_start,
                t_end=t_end,
                words=entry_words,
                word_global_indices=entry_indices,
            )
        )
    # Sorting by t_start keeps chronological order even if pasted lyrics have
    # been re-ordered by the aligner (shouldn't happen, but cheap safety).
    out.sort(key=lambda ln: (ln.t_start, ln.line_idx))
    return out


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LineLayout:
    """Precomputed horizontal layout for the words of one line."""

    x_positions: tuple[float, ...]  # Per-word anchor X (center of the word).
    y_baseline: float
    total_width: float
    word_widths: tuple[float, ...]


class KineticTypographyLayer:
    """
    Per-word kinetic-typography renderer for the compositor.

    Parameters
    ----------
    aligned_words:
        Word list from :func:`load_aligned_words` (or equivalent), in source
        order with per-word ``t_start`` / ``t_end`` / ``line_idx``.
    motion:
        Motion-preset name matching a preset's ``typo_style`` string (one of
        :data:`SUPPORTED_MOTIONS`).
    font_path:
        Optional path to a ``.ttf`` / ``.otf`` display font under
        :file:`assets/fonts/`. When ``None`` the Skia default typeface is used
        so the layer still renders in headless / CI environments.
    width, height:
        Output resolution in pixels (defaults to 1920×1080).
    font_size:
        Font size in pixels.
    base_color:
        Active-line text color as ``#RRGGBB``. Defaults to white.
    shadow_color:
        Optional ``#RRGGBB`` drop-shadow color. ``None`` disables the shadow.
    baseline_y_ratio:
        Vertical placement of the line baseline as a fraction of ``height``.
        Default ``0.75`` is a comfortable lower-third.
    word_spacing_px:
        Horizontal gap (pixels) inserted between words when laying out a line.
    line_fade_seconds:
        Duration of the outgoing-line cross-fade after its last word ends.
    intro_seconds, outro_seconds:
        Per-word envelope lead-in and fade-out durations.

    Use as a context manager (or call :meth:`close`) to release the Skia
    surface deterministically.
    """

    def __init__(
        self,
        aligned_words: Sequence[AlignedWord],
        *,
        motion: str = DEFAULT_MOTION,
        font_path: Path | str | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        font_size: float = DEFAULT_FONT_SIZE,
        base_color: str = "#FFFFFF",
        shadow_color: str | None = None,
        baseline_y_ratio: float = DEFAULT_BASELINE_RATIO,
        word_spacing_px: float = DEFAULT_WORD_SPACING_PX,
        line_fade_seconds: float = DEFAULT_LINE_FADE_SECONDS,
        intro_seconds: float = DEFAULT_INTRO_SECONDS,
        outro_seconds: float = DEFAULT_OUTRO_SECONDS,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution: {width}x{height}")
        if font_size <= 0:
            raise ValueError(f"font_size must be positive, got {font_size}")
        if not 0.0 < baseline_y_ratio < 1.0:
            raise ValueError(
                f"baseline_y_ratio must be in (0, 1), got {baseline_y_ratio}"
            )
        if motion not in MOTION_PRESETS:
            raise ValueError(
                f"Unknown motion preset {motion!r}; expected one of "
                f"{sorted(MOTION_PRESETS)}"
            )

        self._width = int(width)
        self._height = int(height)
        self._font_size = float(font_size)
        self._motion_name = motion
        self._motion_fn = MOTION_PRESETS[motion]
        self._base_rgb = _parse_hex_rgb(base_color)
        self._shadow_rgb: tuple[int, int, int] | None = (
            _parse_hex_rgb(shadow_color) if shadow_color else None
        )
        self._word_spacing_px = float(word_spacing_px)
        self._baseline_y = baseline_y_ratio * float(self._height)
        self._line_fade_seconds = float(max(0.0, line_fade_seconds))
        self._intro_seconds = float(max(1e-3, intro_seconds))
        self._outro_seconds = float(max(1e-3, outro_seconds))

        self._typeface = self._load_typeface(font_path)
        self._font = skia.Font(self._typeface, self._font_size)
        # Display-size tuning: kAntiAlias edging + subpixel positioning +
        # disabled hinting. Skia's default ``kNormal`` hinting is optimised
        # for 12-16 px UI text and quantises glyph outlines onto the pixel
        # grid at the sizes kinetic typography renders (60-200 px), which
        # shows up as stair-stepped diagonals / curves on letters like
        # ``n``, ``e``, ``v``. Turning hinting off and enabling subpixel
        # positioning lets Skia's greyscale anti-aliasing produce smooth
        # evenly-weighted edges -- the same recipe used by the title
        # overlay (:mod:`pipeline.title_overlay`).
        try:
            self._font.setEdging(skia.Font.Edging.kAntiAlias)
        except AttributeError:  # pragma: no cover - very old skia-python
            pass
        try:
            self._font.setSubpixel(True)
        except AttributeError:  # pragma: no cover - older skia-python
            pass
        try:
            self._font.setHinting(skia.FontHinting.kNone)
        except AttributeError:  # pragma: no cover - older skia-python
            pass

        self._lines = _group_words_by_line(list(aligned_words))
        self._line_layouts: list[_LineLayout] = [
            self._layout_line(line) for line in self._lines
        ]

        # Pre-allocate the RGBA backing buffer; reused each frame to avoid
        # per-frame allocations in the hot path.
        self._pixels: np.ndarray = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8
        )
        self._surface: skia.Surface | None = skia.Surface(self._pixels)
        self._closed = False

    # -- font loading -------------------------------------------------------

    @staticmethod
    def _system_ui_sans_typeface() -> skia.Typeface:
        """Prefer a modern system sans when no bundled font path is set."""
        style = skia.FontStyle.Normal()
        for family in (
            "Segoe UI Variable",
            "Segoe UI",
            "Inter",
            "Helvetica Neue",
            "Arial",
        ):
            try:
                tf = skia.Typeface(family, style)
            except Exception:  # noqa: BLE001
                tf = None
            if tf is not None:
                return tf
        return skia.Typeface("")

    @staticmethod
    def _load_typeface(font_path: Path | str | None) -> skia.Typeface:
        if font_path is None:
            try:
                from config import default_ui_font_path

                bundled = default_ui_font_path()
            except Exception:  # noqa: BLE001
                bundled = None
            if bundled is not None:
                p = Path(bundled)
                if p.is_file():
                    return KineticTypographyLayer._load_typeface(p)
            return KineticTypographyLayer._system_ui_sans_typeface()

        path = Path(font_path)
        if not path.is_file():
            raise FileNotFoundError(f"Font file not found: {path}")

        # Prefer the classic :meth:`Typeface.MakeFromFile`; if the current
        # skia-python build only exposes :class:`FontMgr` helpers, fall back
        # to them so upgrades don't silently break font loading.
        maker = getattr(skia.Typeface, "MakeFromFile", None)
        if maker is not None:
            try:
                tf = maker(str(path))
            except Exception as exc:  # noqa: BLE001
                tf = None
                LOGGER.debug("Typeface.MakeFromFile(%s) raised: %s", path, exc)
            if tf is not None:
                return tf

        fm_factory = getattr(skia.FontMgr, "OneFontMgr", None) or getattr(
            skia.FontMgr, "New_Custom_Empty", None
        )
        if fm_factory is not None:
            try:
                fm = fm_factory(str(path))
                family_count = fm.countFamilies()
                if family_count > 0:
                    style_set = fm.createStyleSet(0)
                    if style_set is not None and style_set.count() > 0:
                        tf = style_set.createTypeface(0)
                        if tf is not None:
                            return tf
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load font via FontMgr from {path}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Could not load typeface from {path}; skia-python exposes "
            "neither Typeface.MakeFromFile nor FontMgr.OneFontMgr"
        )

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def motion(self) -> str:
        return self._motion_name

    @property
    def lines(self) -> tuple[_LineTiming, ...]:
        return tuple(self._lines)

    # -- layout -------------------------------------------------------------

    def _measure_word(self, text: str) -> float:
        return float(self._font.measureText(text))

    def _layout_line(self, line: _LineTiming) -> _LineLayout:
        if not line.words:
            return _LineLayout(
                x_positions=(),
                y_baseline=self._baseline_y,
                total_width=0.0,
                word_widths=(),
            )
        widths = tuple(self._measure_word(w.word) for w in line.words)
        total = sum(widths) + self._word_spacing_px * max(0, len(widths) - 1)
        cursor = (float(self._width) - total) * 0.5
        centers: list[float] = []
        for w, width in zip(line.words, widths):
            centers.append(cursor + width * 0.5)
            cursor += width + self._word_spacing_px
        return _LineLayout(
            x_positions=tuple(centers),
            y_baseline=self._baseline_y,
            total_width=total,
            word_widths=widths,
        )

    # -- time → visible lines ----------------------------------------------

    def _visible_line_indices(self, t: float) -> list[tuple[int, float]]:
        """
        Return ``(line_index, line_alpha_multiplier)`` pairs for every line
        that contributes to frame ``t``. The current line multiplies at 1.0;
        the previous line fades out over ``line_fade_seconds`` after its
        last word ends.
        """
        if not self._lines:
            return []

        # Current line: the latest line whose ``t_start - intro`` <= t.
        current: int | None = None
        for i, line in enumerate(self._lines):
            if t >= line.t_start - self._intro_seconds:
                current = i
            else:
                break

        visible: list[tuple[int, float]] = []

        # Previous line fading out.
        if current is not None and current - 1 >= 0:
            prev = self._lines[current - 1]
            dt = t - prev.t_end
            if 0.0 <= dt < self._line_fade_seconds:
                mult = 1.0 - _smoothstep(dt / max(1e-6, self._line_fade_seconds))
                if mult > 0.0:
                    visible.append((current - 1, mult))
            # If the new line hasn't actually started yet but we're already
            # past the end of the previous one, keep showing the old one at
            # 1.0 (no blank flash between lines).
            elif dt < 0.0:
                visible.append((current - 1, 1.0))

        # Current line (or the first upcoming line if nothing is "current").
        if current is not None:
            cur_line = self._lines[current]
            # Don't draw the current line until its first word starts
            # fading in — otherwise long gaps would show stale text.
            if t >= cur_line.t_start - self._intro_seconds:
                if t <= cur_line.t_end + self._outro_seconds + self._line_fade_seconds:
                    visible.append((current, 1.0))
        else:
            # Still before the first line starts: show nothing.
            pass

        return visible

    # -- rendering ----------------------------------------------------------

    def render_frame(
        self, t: float, uniforms: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        """
        Render a single RGBA frame for time ``t`` seconds.

        Returns an ``(H, W, 4) uint8`` array in top-left origin. The array is
        owned by this instance and reused across calls; copy it if the
        caller needs to keep the frame beyond the next ``render_frame``.
        """
        if self._closed or self._surface is None:
            raise RuntimeError("KineticTypographyLayer has been closed")
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")

        u = dict(uniforms) if uniforms else {}

        # Reset backing buffer to fully transparent.
        self._pixels.fill(0)

        with self._surface as canvas:
            for line_index, line_mult in self._visible_line_indices(float(t)):
                line = self._lines[line_index]
                layout = self._line_layouts[line_index]
                self._draw_line(canvas, line, layout, float(t), u, line_mult)

        # Return the buffer directly; Skia writes BGRA on little-endian hosts
        # under the kN32 color type, so normalize to RGBA for downstream
        # consumers (ffmpeg / numpy compositing) that expect channel-order.
        return _swizzle_to_rgba(self._pixels)

    def _draw_line(
        self,
        canvas: skia.Canvas,
        line: _LineTiming,
        layout: _LineLayout,
        t: float,
        u: Mapping[str, Any],
        line_mult: float,
    ) -> None:
        if not line.words or not layout.x_positions:
            return

        y_baseline = layout.y_baseline
        for word_pos, word_width, word, word_index in zip(
            layout.x_positions, layout.word_widths, line.words, line.word_global_indices
        ):
            state = _WordState(
                t=t,
                t_start=word.t_start,
                t_end=word.t_end,
                intro_seconds=self._intro_seconds,
                outro_seconds=self._outro_seconds,
                word_index=word_index,
            )
            motion = self._motion_fn(state, u)
            alpha = max(0.0, min(1.0, motion.alpha * line_mult))
            if alpha <= 1e-3:
                continue

            canvas.save()
            try:
                cx = word_pos + motion.dx
                cy = y_baseline + motion.dy
                canvas.translate(cx, cy)
                if motion.scale != 1.0:
                    canvas.scale(motion.scale, motion.scale)

                # drawString baseline is at (0, 0); offset by half-width so the
                # transform above (translate/scale) pivots around the glyph
                # centre, not its left edge.
                x0 = -word_width * 0.5

                # No halo / glow / stroke — every variant (tight single
                # pass, wide multi-pass, soft whisper, kStroke outline)
                # drew as a coloured rim or blob on saturated preset
                # palettes. Ship a clean fill and let
                # :func:`pipeline.preset_colors.resolve_text_colors`
                # guarantee fill-vs-background contrast. ``shadow_color``
                # is accepted at the API boundary but currently unused in
                # the render path.
                main_paint = skia.Paint(
                    AntiAlias=True,
                    Color=_argb_with_alpha(self._base_rgb, alpha),
                )
                canvas.drawString(word.word, x0, 0.0, self._font, main_paint)
            finally:
                canvas.restore()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the Skia surface. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        self._surface = None

    def __enter__(self) -> "KineticTypographyLayer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pixel swizzling (Skia kN32 native order → RGBA)
# ---------------------------------------------------------------------------


def _swizzle_to_rgba(pixels: np.ndarray) -> np.ndarray:
    """
    Reorder Skia's native ``kN32_SkColorType`` pixels to RGBA.

    ``skia.Surface(numpy_array)`` writes pixels in the platform's native
    32-bit layout, which on little-endian hosts is BGRA in memory. The
    compositor / ffmpeg stage expects explicit RGBA, so we swap channels
    before returning. The returned array is a copy.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(
            f"Expected (H, W, 4) uint8 pixels, got shape {pixels.shape}"
        )
    # Detect native byte order; on every platform we support this is
    # little-endian, so BGRA → RGBA via a simple channel swap.
    out = np.empty_like(pixels)
    out[..., 0] = pixels[..., 2]
    out[..., 1] = pixels[..., 1]
    out[..., 2] = pixels[..., 0]
    out[..., 3] = pixels[..., 3]
    return out


__all__: Sequence[str] = [
    "AlignedWord",
    "DEFAULT_BASELINE_RATIO",
    "DEFAULT_KINETIC_BASELINE_RATIO",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_HEIGHT",
    "DEFAULT_INTRO_SECONDS",
    "DEFAULT_LINE_FADE_SECONDS",
    "DEFAULT_MOTION",
    "DEFAULT_OUTRO_SECONDS",
    "DEFAULT_WIDTH",
    "DEFAULT_WORD_SPACING_PX",
    "KineticTypographyLayer",
    "MOTION_PRESETS",
    "SUPPORTED_MOTIONS",
    "WordMotion",
    "get_motion_preset",
    "load_aligned_words",
]
