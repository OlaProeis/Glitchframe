"""Bundled reactive fragment shader stems (under ``assets/shaders/``).

Defined here so :mod:`config` and the UI can validate preset YAML without
importing OpenGL-backed :mod:`pipeline.reactive_shader`.
"""

from __future__ import annotations

from typing import Sequence

BUILTIN_SHADERS: tuple[str, ...] = (
    "spectrum_bars",
    "particles",
    "geometry_pulse",
    "nebula_drift",
    "liquid_chrome",
    "vhs_tracking",
    "synth_grid",
    "paper_grain",
)

__all__: Sequence[str] = ["BUILTIN_SHADERS"]
