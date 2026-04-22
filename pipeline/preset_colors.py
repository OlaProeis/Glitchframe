"""Resolve preset palettes to *text-legible* fill / glow colors.

Preset palettes in ``presets/*.yaml`` are ordered dark → bright because they
feed the shader pass as ``u_palette[5]``, where slot 0 is a base/background
tone and slots 3-4 are the brightest highlight colors. The title card and
kinetic-typography layers need the *opposite* — a bright fill that reads over
any background, plus a saturated mid-tone glow that frames it.

Historically the orchestrator fed the first two palette entries straight into
``CompositorConfig.base_color`` / ``shadow_color``, which meant dark-theme
presets (cosmic-flow, neon-synthwave, organic-liquid, glitch-vhs)
burned the title in the *darkest* color over the darkest backgrounds —
effectively invisible. This module centralises the "pick a readable pair"
decision so the orchestrator, thumbnail renderer, and any future consumer
all share the same rule.

The heuristic:

* **Fill** is the palette entry with the highest perceptual luma (Rec. 709).
  Ties are broken by picking the later slot (presets tend to place their
  cleanest highlight last).
* **Glow** is chosen from the remaining entries by picking the one with the
  highest chroma (HSV saturation × value) whose luma is meaningfully below
  the fill — so neon-synthwave's ``#FF2EE6`` magenta wins over its
  ``#E8F7FF`` near-white. Falls back to the next-brightest slot when every
  remaining entry is low-saturation (minimal-mono case).
* Returns ``("#FFFFFF", None)`` on an empty palette so callers don't need to
  branch.
"""

from __future__ import annotations

import re
from typing import Sequence

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

# Minimum luma gap (0..1) between fill and glow — prevents the resolver from
# picking two near-identical bright colors on cosmic-style palettes.
_MIN_GLOW_LUMA_GAP = 0.08


def _parse_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.strip()
    if not _HEX_RE.fullmatch(s):
        raise ValueError(f"Expected #RRGGBB hex color, got {hex_str!r}")
    return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)


def _luma(rgb: tuple[int, int, int]) -> float:
    """Rec. 709 relative luma in 0..1."""
    r, g, b = rgb
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _chroma(rgb: tuple[int, int, int]) -> float:
    """HSV chroma proxy: (max - min) / 255 — favours saturated mid-tones."""
    r, g, b = rgb
    return (max(r, g, b) - min(r, g, b)) / 255.0


def _normalize(hex_str: str) -> str:
    return "#" + hex_str.strip().lstrip("#").upper()


def resolve_text_colors(
    palette: Sequence[str] | None,
    *,
    default_fill: str = "#FFFFFF",
    default_glow: str | None = None,
) -> tuple[str, str | None]:
    """Return ``(fill_hex, glow_hex)`` for display text over the preset.

    Parameters
    ----------
    palette:
        The preset's ``colors`` list (``#RRGGBB`` entries, dark→bright by
        convention). ``None`` / empty returns the supplied defaults.
    default_fill, default_glow:
        Returned verbatim when ``palette`` is empty. ``default_glow=None``
        disables the shadow pass in downstream renderers.

    Notes
    -----
    Validates every entry; malformed hex raises :class:`ValueError` rather
    than silently falling back to white, matching the project's "no silent
    failures" convention.
    """
    if not palette:
        return default_fill, default_glow

    entries: list[tuple[int, tuple[int, int, int]]] = []
    for i, c in enumerate(palette):
        if not isinstance(c, str):
            raise ValueError(f"palette[{i}] must be a hex string, got {c!r}")
        entries.append((i, _parse_rgb(c)))

    # --- Fill = highest luma (later-slot wins ties, matching palette convention).
    fill_idx, fill_rgb = max(entries, key=lambda e: (_luma(e[1]), e[0]))

    if len(entries) == 1:
        fill_hex = _normalize(palette[fill_idx])
        return fill_hex, default_glow

    fill_luma = _luma(fill_rgb)

    # --- Glow = most saturated mid-tone, clearly darker than the fill.
    candidates = [e for e in entries if e[0] != fill_idx]
    darker = [e for e in candidates if _luma(e[1]) <= fill_luma - _MIN_GLOW_LUMA_GAP]
    pool = darker if darker else candidates

    # Prefer chroma; tie-break by lower luma so we get a punchy mid-tone
    # instead of another near-white.
    glow_idx, _glow_rgb = max(pool, key=lambda e: (_chroma(e[1]), -_luma(e[1]), e[0]))

    return _normalize(palette[fill_idx]), _normalize(palette[glow_idx])


__all__ = ["resolve_text_colors"]
