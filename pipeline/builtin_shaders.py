"""Bundled reactive fragment shader stems (under ``assets/shaders/``).

Listed here so :mod:`config` and the UI can validate overrides without importing
OpenGL-backed :mod:`pipeline.reactive_shader`.

The sentinel stem ``none`` skips the reactive GL pass entirely (clean SDXL /
Ken Burns composites only).

The remaining stems are curated for readability over photographic backgrounds —
no heavy fog washes. See :mod:`pipeline.visual_style` for example prompts /
palettes per stem.
"""

from __future__ import annotations

from typing import Sequence

BUILTIN_SHADERS: tuple[str, ...] = (
    "none",
    "void_ascii_bg",
    "spectral_milkdrop",
    "tunnel_flight",
    "synth_grid",
)

__all__: Sequence[str] = ["BUILTIN_SHADERS"]
