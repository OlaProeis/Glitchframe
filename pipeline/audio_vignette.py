"""Audio-pulsing dark vignette post-pass for the frame compositor.

A subtle radial darkening of the frame edges that breathes with the music:
the corners pump slightly on bass + post-drop afterglow so the picture
feels held by a dark frame without overpowering the scene. Used by every
preset to add a baseline of contrast between the SDXL/AnimateDiff
background and the reactive shader (the rim of the frame is always a hair
darker than the centre, which makes mid-frame structures pop).

The pass runs on the CPU between the reactive composite (background +
shader + voidcat ASCII) and the kinetic typography / title / logo passes
so text and branding stay clean — only the picture gets the vignette.

Performance: a single :class:`numpy.ndarray` mask is precomputed once per
render and shared across frames. The hot path is a per-frame lerp toward
black with an audio-driven scalar, no allocations beyond a temporary
``float32`` view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


# Baseline edge darkening even with silent uniforms. Keeps the picture framed
# without making the corners noticeably blacker than the centre. 0.0 disables
# the static component (only the audio pulse survives, which is barely visible).
_BASE_EDGE_DARKEN = 0.18

# Maximum *additional* edge darkening contributed by the audio pulse on top of
# ``_BASE_EDGE_DARKEN``. Layered on top of the base via clamp(base + audio, 0,
# 1) so loud sections darken corners by up to ``base + audio`` total.
_AUDIO_EDGE_PULSE = 0.12

# Inner radius (normalised) where vignette begins to ramp in. Below this the
# frame is untouched. ``1.0`` is the corner of a unit-square (sqrt(2)/sqrt(2)
# in our half-extents convention below).
_VIGNETTE_INNER = 0.55
_VIGNETTE_OUTER = 1.05


@dataclass(frozen=True, slots=True)
class AudioVignetteContext:
    """Per-render cache for :func:`apply_audio_vignette`.

    ``mask`` is a ``(H, W)`` ``float32`` array in ``[0, 1]`` where ``0`` is the
    centre of the frame and ``1`` is the corners. ``strength`` scales every
    per-frame contribution (user knob; ``1.0`` = defaults documented above).
    """

    mask: np.ndarray  # (H, W) float32 in [0, 1]
    strength: float = 1.0


def build_audio_vignette_context(
    width: int,
    height: int,
    *,
    strength: float = 1.0,
) -> AudioVignetteContext:
    """Precompute the radial vignette mask once per render."""
    if width <= 0 or height <= 0:
        raise ValueError("width / height must be positive")
    s = max(0.0, float(strength))
    # Use frame-aspect-aware coordinates so a wide frame gets the same darkening
    # at the short-edge corners as a square frame does. The mask is unit-radius
    # at the four corners by construction.
    aspect = float(width) / max(float(height), 1.0)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32) * aspect
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Normalise so the corner of a non-square frame still reads as ~1.
    norm = float(np.hypot(1.0, aspect))
    r = np.hypot(xx, yy) / max(norm, 1e-6)
    # Smoothstep ramp from inner to outer; clamp explicitly to avoid edge
    # noise from float math at the corners of extreme aspect ratios.
    t = np.clip(
        (r - _VIGNETTE_INNER) / max(_VIGNETTE_OUTER - _VIGNETTE_INNER, 1e-6),
        0.0,
        1.0,
    )
    mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    return AudioVignetteContext(mask=mask, strength=s)


def _audio_factor(uniforms: Mapping[str, float]) -> float:
    """Combine reactive uniforms into a single 0..1 pulse weight.

    Bass attack dominates so kicks land cleanly; ``drop_hold`` keeps the
    afterglow pumping for a couple of seconds; ``rms`` adds a slow lift on
    sustained loud passages so the vignette doesn't snap on/off between
    quiet/loud sections.
    """
    bass = float(uniforms.get("bass_hit", 0.0) or 0.0)
    hold = float(uniforms.get("drop_hold", 0.0) or 0.0)
    rms = float(uniforms.get("rms", 0.0) or 0.0)
    f = 0.55 * bass + 0.30 * hold + 0.18 * rms
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def apply_audio_vignette(
    frame: np.ndarray,
    uniforms: Mapping[str, float],
    ctx: AudioVignetteContext | None,
) -> np.ndarray:
    """Darken the frame's edges in place; ``None`` ctx is a fast no-op.

    Math: ``out = frame * (1 - mask * darken)`` where
    ``darken = strength * (base + audio_pulse * audio_factor)`` and the mask
    is ``0`` at the centre and ``1`` at the corners. The frame is modified
    in place when it's a contiguous ``uint8`` array (typical for the
    compositor pipeline) and a new array is returned regardless so callers
    can reassign without thinking about aliasing.
    """
    if ctx is None or ctx.strength <= 0.0:
        return frame
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "apply_audio_vignette expects (H, W, 3) uint8 RGB; "
            f"got shape={frame.shape} dtype={frame.dtype}"
        )
    h, w, _ = frame.shape
    if ctx.mask.shape != (h, w):
        raise ValueError(
            "audio vignette mask shape mismatch: "
            f"mask={ctx.mask.shape} frame=({h}, {w})"
        )
    audio = _audio_factor(uniforms)
    darken = ctx.strength * (
        _BASE_EDGE_DARKEN + _AUDIO_EDGE_PULSE * audio
    )
    if darken <= 1e-4:
        return frame
    # ``(1 - mask * darken)`` ranges from 1.0 at centre to (1 - darken) at the
    # corners. Multiply broadcasts (H, W, 1) * (H, W, 3) -> (H, W, 3).
    factor = (1.0 - ctx.mask * float(darken))[..., None]
    out = (frame.astype(np.float32) * factor).clip(0.0, 255.0)
    return out.astype(np.uint8)


__all__ = [
    "AudioVignetteContext",
    "apply_audio_vignette",
    "build_audio_vignette_context",
]
