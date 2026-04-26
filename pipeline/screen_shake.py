"""
Per-frame screen shake from :class:`EffectKind.SCREEN_SHAKE` timeline clips.

Overlapping clips: horizontal and vertical components are **summed** (each
clip contributes a vector at time ``t``; combined offset is the sum of those
vectors). This keeps the rule simple and deterministic.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.logo_composite import glitch_seed_for_time

_DEFAULT_FREQ_HZ = 4.0
_LCG_MUL = 1103515245
_LCG_ADD = 12345
_LCG_MASK = 0x7FFFFFFF


def _active_shake_clips(
    t: float, clips: Sequence[EffectClip]
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.SCREEN_SHAKE:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _float_setting(settings: dict[str, object], key: str, default: float) -> float:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _lcg_u01_u01(seed: int) -> tuple[float, float]:
    """Two values in (0, 1] from a 32-bit seed (LCG, deterministic)."""
    x = seed & 0xFFFFFFFF
    x = (_LCG_MUL * x + _LCG_ADD) & _LCG_MASK
    u1 = (x + 1) / (_LCG_MASK + 1)
    x = (_LCG_MUL * x + _LCG_ADD) & _LCG_MASK
    u2 = (x + 1) / (_LCG_MASK + 1)
    return (u1, u2)


def _clip_offset(
    clip: EffectClip, t: float, song_hash: str
) -> tuple[float, float]:
    s = clip.settings
    amp = _float_setting(s, "amplitude_px", 0.0)
    if not math.isfinite(amp) or amp == 0.0:
        return (0.0, 0.0)

    raw_freq = _float_setting(s, "frequency_hz", _DEFAULT_FREQ_HZ)
    if not math.isfinite(raw_freq) or raw_freq <= 0.0:
        freq = _DEFAULT_FREQ_HZ
    else:
        freq = raw_freq

    tau = t - float(clip.t_start)
    if not math.isfinite(tau):
        return (0.0, 0.0)

    base = glitch_seed_for_time(song_hash, t)
    salt = zlib.adler32(clip.id.encode("utf-8", errors="ignore")) & 0xFFFFFFFF
    mix = (base ^ salt) & 0xFFFFFFFF
    u1, u2 = _lcg_u01_u01(mix)
    p = 2.0 * math.pi * freq * tau
    dx = amp * math.sin(p + 2.0 * math.pi * u1)
    dy = amp * math.sin(p * 1.371 + 2.0 * math.pi * u2)
    return (dx, dy)


def shake_offset(
    t: float, clips: Sequence[EffectClip], song_hash: str
) -> tuple[float, float]:
    """
    Return pixel offset ``(dx, dy)`` for active ``SCREEN_SHAKE`` clips at
    time ``t``.

    Only clips with ``[t_start, t_start + duration_s)`` containing ``t`` are
    used. The random-looking motion is deterministic from ``song_hash`` and
    ``t`` (same pattern as :func:`glitch_seed_for_time` plus per-clip salt).
    Settings ``amplitude_px`` and ``frequency_hz`` scale the motion; missing
    or invalid values fall back to 0 amplitude and/or a default frequency.
    """
    dx_total = 0.0
    dy_total = 0.0
    for clip in _active_shake_clips(t, clips):
        dx, dy = _clip_offset(clip, t, song_hash)
        dx_total += dx
        dy_total += dy
    return (dx_total, dy_total)


def apply_shake_offset(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Return ``frame`` (H, W, 3 uint8 RGB) shifted by ``(dx, dy)`` pixels with
    the vacated border filled black. Sub-pixel offsets are rounded to the
    nearest integer. Offsets that exceed the frame size collapse to the
    original frame (nothing survives to copy).
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"apply_shake_offset expects (H, W, 3) uint8, got shape={frame.shape} dtype={frame.dtype}"
        )
    idx = int(round(float(dx)))
    idy = int(round(float(dy)))
    if idx == 0 and idy == 0:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if abs(idx) >= w or abs(idy) >= h:
        return np.zeros_like(frame)
    out = np.zeros_like(frame)
    # Source row range
    if idy >= 0:
        sy0, dy0, hh = 0, idy, h - idy
    else:
        sy0, dy0, hh = -idy, 0, h - (-idy)
    if idx >= 0:
        sx0, dx0, ww = 0, idx, w - idx
    else:
        sx0, dx0, ww = -idx, 0, w - (-idx)
    out[dy0 : dy0 + hh, dx0 : dx0 + ww] = frame[sy0 : sy0 + hh, sx0 : sx0 + ww]
    return out
