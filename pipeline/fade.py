"""
Per-frame fade-to-black overlay from :class:`EffectKind.FADE` timeline clips.

A FADE clip is a single timeline lane that **fades to or from black** over the
clip's duration. ``direction_mode`` chooses between:

- ``"in"`` — the screen starts black at ``t_start`` and reveals to clear at
  ``t_start + duration_s``. Use to bring the picture *in* from black.
- ``"out"`` — the screen starts clear at ``t_start`` and ends fully black at
  ``t_start + duration_s``. Use to fade the picture *out* to black.

The longer the clip, the longer the fade — that is the entire UX contract: the
user simply drags the clip's edges in the editor to dial the fade length, and
flips ``direction_mode`` in the gear panel.

Overlapping clips combine via **max** of their alpha contributions and the
result is clamped to ``[0, 1]`` (so two stacked fades never push past full
black). The pass is a no-op outside any active clip.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind

_DEFAULT_PEAK_ALPHA = 1.0
_DEFAULT_DIRECTION = "in"
_VALID_DIRECTIONS = frozenset({"in", "out"})
_DEFAULT_EASE = "smoothstep"
_VALID_EASES = frozenset({"smoothstep", "linear"})


def _float_setting(settings: dict[str, object], key: str, default: float) -> float:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _str_setting(
    settings: dict[str, object], key: str, default: str, allow: frozenset[str]
) -> str:
    v = settings.get(key, default)
    if not isinstance(v, str):
        return default
    s = v.strip().lower()
    if s in allow:
        return s
    return default


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _smoothstep01(x: float) -> float:
    """Hermite smoothstep for ``x`` in ``[0, 1]`` (values outside are clamped)."""
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def _active_fade_clips(t: float, clips: Sequence[EffectClip]) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.FADE:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _clip_alpha(clip: EffectClip, t: float) -> float:
    """Black-overlay alpha contribution for one FADE clip at time ``t``.

    ``alpha == 0`` is fully transparent (frame untouched); ``alpha == 1`` is
    fully black. Fade-in starts at ``peak_alpha`` and ramps down; fade-out
    starts at ``0`` and ramps up to ``peak_alpha``.
    """
    d = float(clip.duration_s)
    if not (math.isfinite(d) and d > 0.0):
        return 0.0
    rel = (t - float(clip.t_start)) / d
    if not math.isfinite(rel):
        return 0.0
    rel = _clamp01(rel)

    s = clip.settings
    direction = _str_setting(s, "direction_mode", _DEFAULT_DIRECTION, _VALID_DIRECTIONS)
    peak = _clamp01(_float_setting(s, "peak_alpha", _DEFAULT_PEAK_ALPHA))
    ease = _str_setting(s, "ease_mode", _DEFAULT_EASE, _VALID_EASES)

    if direction == "in":
        # Black at the start, clear at the end (1 → 0 progression).
        progress = 1.0 - rel
    else:
        # Clear at the start, black at the end (0 → 1 progression).
        progress = rel

    if ease == "smoothstep":
        progress = _smoothstep01(progress)
    return _clamp01(progress) * peak


def fade_alpha(t: float, clips: Sequence[EffectClip]) -> float:
    """Return the **black-overlay alpha** for active ``FADE`` clips at time ``t``.

    Activity uses the same half-open window as the rest of the post-stack
    renderers (``t_start <= t < t_start + duration_s``). Other effect kinds
    are ignored.

    Combination: per-clip alphas are merged with **max** (then clamped to
    ``[0, 1]``). Non-finite ``t`` returns ``0.0`` (no fade applied).
    """
    if not math.isfinite(t):
        return 0.0
    best = 0.0
    for c in _active_fade_clips(t, clips):
        a = _clip_alpha(c, t)
        if a > best:
            best = a
    return _clamp01(best)


def apply_fade(frame: np.ndarray, alpha: float) -> np.ndarray:
    """Blend ``frame`` (H, W, 3 uint8 RGB) toward black by ``alpha`` in ``[0, 1]``.

    ``alpha <= 0`` returns ``frame`` unchanged (same array — never mutated).
    ``alpha >= 1`` returns a fresh black frame. Values in between scale the
    frame by ``(1 - alpha)``.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            "apply_fade expects (H, W, 3) uint8, "
            f"got shape={frame.shape} dtype={frame.dtype}"
        )
    a = _clamp01(float(alpha))
    if a <= 1e-4:
        return frame
    if a >= 1.0 - 1e-4:
        return np.zeros_like(frame)
    keep = 1.0 - a
    out = frame.astype(np.float32) * keep
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


__all__ = ["apply_fade", "fade_alpha"]
