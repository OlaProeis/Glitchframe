"""
Per-frame colour-invert mix from :class:`EffectKind.COLOR_INVERT` timeline clips.

Each active clip contributes a **scalar in [0, 1]**, the lerp weight toward the
inverted frame (e.g. ``out = lerp(frame, 255 - frame, mix)`` in uint8 space).

Overlapping clips: per-clip weights are **summed**, then the result is **clamped
to [0, 1]**. (Unlike :mod:`pipeline.screen_shake`, which sums vectors, here a
capped sum keeps the final mix in range.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind

_DEFAULT_MIX = 1.0
_DEFAULT_INTENSITY = 1.0


def _float_setting(settings: dict[str, object], key: str, default: float) -> float:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _active_color_invert_clips(
    t: float, clips: Sequence[EffectClip]
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.COLOR_INVERT:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _clip_contribution(clip: EffectClip) -> float:
    s = clip.settings
    m = _clamp01(_float_setting(s, "mix", _DEFAULT_MIX))
    i = _clamp01(_float_setting(s, "intensity", _DEFAULT_INTENSITY))
    return m * i


def invert_mix(t: float, clips: Sequence[EffectClip]) -> float:
    """
    Interpolate active ``COLOR_INVERT`` clips at time ``t`` and return a mix
    factor in **[0, 1]** for lerping a frame toward its colour inverse.

    A clip is **active** iff ``t_start <= t < t_start + duration_s`` (same
    half-open window as :func:`pipeline.screen_shake.shake_offset`). Other
    effect kinds are ignored.

    For each active clip, the contribution is ``mix * intensity`` (each
    read from :attr:`~pipeline.effects_timeline.EffectClip.settings` and
    clamped to **[0, 1]**; missing or invalid values use defaults **1.0**).
    Overlapping contributions are **added**, then the total is **clamped** to
    **[0, 1]**.
    """
    if not math.isfinite(t):
        return 0.0
    total = 0.0
    for clip in _active_color_invert_clips(t, clips):
        total += _clip_contribution(clip)
    return _clamp01(total)


def apply_invert_mix(frame: np.ndarray, mix: float) -> np.ndarray:
    """
    Blend ``frame`` (H, W, 3 uint8 RGB) toward ``255 - frame`` by ``mix`` in
    ``[0, 1]``. ``mix == 0`` returns ``frame`` unchanged; ``mix == 1`` returns
    the fully inverted frame. The result is a fresh uint8 array (never an
    alias of the input).
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"apply_invert_mix expects (H, W, 3) uint8, got shape={frame.shape} dtype={frame.dtype}"
        )
    m = _clamp01(float(mix))
    if m <= 1e-4:
        return frame
    if m >= 1.0 - 1e-4:
        return (np.uint8(255) - frame).astype(np.uint8, copy=False)
    f32 = frame.astype(np.float32)
    out = f32 * (1.0 - m) + (255.0 - f32) * m
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
