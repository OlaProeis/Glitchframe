"""
Per-frame zoom scale from :class:`EffectKind.ZOOM_PUNCH` timeline clips.

Each active clip contributes a **scale factor** ``>= 1.0`` suitable for a
whole-frame bilinear resample plus center crop in the compositor (values
``> 1.0`` enlarge the source before crop, producing a punch-in).

Overlapping clips: per-clip scales are combined with **max** (identity is
``1.0``; taking the maximum avoids compounding multiple punches into extreme
zoom).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind

_DEFAULT_PEAK_SCALE = 1.12
_DEFAULT_EASE_IN_S = 0.08
_DEFAULT_EASE_OUT_S = 0.12
_DEFAULT_WIDTH_FRAC = 1.0
_EPS = 1e-12


def _float_setting(settings: dict[str, object], key: str, default: float) -> float:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _smoothstep01(x: float) -> float:
    """Hermite smoothstep for ``x`` in ``[0, 1]`` (values outside are clamped)."""
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def _active_zoom_clips(t: float, clips: Sequence[EffectClip]) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.ZOOM_PUNCH:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _clip_zoom_scale(clip: EffectClip, t: float) -> float:
    """Scale ``>= 1.0`` from one clip at time ``t`` (``t`` must lie in the clip)."""
    t0 = float(clip.t_start)
    d = float(clip.duration_s)
    if not (math.isfinite(d) and d > 0.0):
        return 1.0
    t_rel = t - t0
    if not math.isfinite(t_rel):
        return 1.0

    s = clip.settings
    peak = _float_setting(s, "peak_scale", _DEFAULT_PEAK_SCALE)
    if not math.isfinite(peak) or peak <= 1.0:
        return 1.0

    ease_in = max(0.0, _float_setting(s, "ease_in_s", _DEFAULT_EASE_IN_S))
    ease_out = max(0.0, _float_setting(s, "ease_out_s", _DEFAULT_EASE_OUT_S))
    if not math.isfinite(ease_in):
        ease_in = 0.0
    if not math.isfinite(ease_out):
        ease_out = 0.0

    wf = _float_setting(s, "width_frac", _DEFAULT_WIDTH_FRAC)
    if not math.isfinite(wf) or wf <= 0.0:
        wf = _DEFAULT_WIDTH_FRAC
    wf = min(1.0, max(wf, _EPS))

    w = wf * d
    w = min(w, d)
    if w <= _EPS:
        return 1.0

    if t_rel < 0.0 or t_rel >= w:
        return 1.0

    ramp = ease_in + ease_out
    if ramp > w and ramp > _EPS:
        scale_t = w / ramp
        ease_in *= scale_t
        ease_out *= scale_t
    hold = max(0.0, w - ease_in - ease_out)

    if t_rel < ease_in:
        if ease_in <= _EPS:
            env = 1.0
        else:
            env = _smoothstep01(t_rel / ease_in)
    elif t_rel < ease_in + hold:
        env = 1.0
    else:
        t_out = t_rel - (ease_in + hold)
        if ease_out <= _EPS:
            env = 0.0
        else:
            env = 1.0 - _smoothstep01(t_out / ease_out)

    return 1.0 + (peak - 1.0) * env


def zoom_scale(t: float, clips: Sequence[EffectClip]) -> float:
    """
    Return a **scale factor** for active ``ZOOM_PUNCH`` clips at time ``t``.

    A clip is **timeline-active** iff ``t_start <= t < t_start + duration_s``
    (same half-open window as :func:`pipeline.screen_shake.shake_offset`).
    Within the clip, the punch envelope runs over the first ``width_frac *
    duration_s`` seconds (clamped to the clip length); outside that prefix the
    scale is ``1.0`` even though the clip is still active.

    Settings (``EFFECT_SETTINGS_KEYS[ZOOM_PUNCH]``):

    - **peak_scale** — maximum scale; missing/invalid → **1.12**; ``<= 1`` → no zoom.
    - **ease_in_s** / **ease_out_s** — smoothstep ease durations (compressed if
      they exceed the punch window).
    - **width_frac** — fraction ``(0, 1]`` of clip duration for the punch window;
      invalid/``<= 0`` → **1.0**.

    Easing uses a Hermite **smoothstep** (``C^1``), which matches common GPU
    bilinear sampling assumptions (smooth velocity, no discontinuity).

    Overlapping punches: **maximum** of per-clip scales (each ``>= 1.0``).

    Non-finite ``t`` returns **1.0** (identity scale).
    """
    if not math.isfinite(t):
        return 1.0
    best = 1.0
    for clip in _active_zoom_clips(t, clips):
        best = max(best, _clip_zoom_scale(clip, t))
    return best


def apply_zoom_scale(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Return a center-punched version of ``frame`` (H, W, 3 uint8 RGB) scaled by
    ``scale >= 1.0``. Values ``<= 1`` are identity (the input array is returned
    unchanged). The enlarged frame is bilinear-resampled with Pillow and then
    center-cropped back to the input resolution.
    """
    if not math.isfinite(scale) or scale <= 1.0 + 1e-9:
        return frame
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"apply_zoom_scale expects (H, W, 3) uint8, got shape={frame.shape} dtype={frame.dtype}"
        )
    h, w = int(frame.shape[0]), int(frame.shape[1])
    new_h = max(h, int(round(h * float(scale))))
    new_w = max(w, int(round(w * float(scale))))
    from PIL import Image  # lazy: only paid when a ZOOM_PUNCH clip actually fires

    img = Image.fromarray(frame, mode="RGB").resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    arr = np.asarray(img, dtype=np.uint8)
    y0 = max(0, (arr.shape[0] - h) // 2)
    x0 = max(0, (arr.shape[1] - w) // 2)
    return np.ascontiguousarray(arr[y0 : y0 + h, x0 : x0 + w])
