"""Bundled reactive fragment shader stems (under ``assets/shaders/``).

Defined here so :mod:`config` and the UI can validate preset YAML without
importing OpenGL-backed :mod:`pipeline.reactive_shader`.

Nine preset YAML files each pin one stem (``cosmic-flow`` → ``nebula_flow``,
etc.). Additional stems remain in the dropdown for manual experiments and
legacy paths: ``spectrum_bars`` (also :data:`ReactiveShader` default stem),
``particles``, ``nebula_drift`` (A/B sibling of ``nebula_flow``), and
``voidcat_laser`` (non-ASCII voidcat motif).
"""

from __future__ import annotations

from typing import Sequence

BUILTIN_SHADERS: tuple[str, ...] = (
    "spectrum_bars",
    "particles",
    "geometry_pulse",
    "nebula_drift",
    "nebula_flow",
    "liquid_chrome",
    "vhs_tracking",
    "synth_grid",
    "tunnel_flight",
    "paper_grain",
    "voidcat_laser",
    "void_ascii_bg",
    "spectral_milkdrop",
)

__all__: Sequence[str] = ["BUILTIN_SHADERS"]
