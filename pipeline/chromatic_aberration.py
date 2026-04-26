"""
Full-frame chromatic aberration from :class:`EffectKind.CHROMATIC_ABERRATION` clips.

Splits RGB, shifts R and B in opposite directions along a per-clip axis, leaves G
fixed. Shifts are deterministic from ``song_hash``, ``clip.id``, and ``t``
(aligned with :func:`pipeline.logo_composite.glitch_seed_for_time`).
"""

from __future__ import annotations

import hashlib
import math
import zlib
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from pipeline.effects_timeline import EffectClip, EffectKind
from pipeline.logo_composite import glitch_seed_for_time

_LCG_MUL = 1103515245
_LCG_ADD = 12345
_LCG_MASK = 0x7FFFFFFF


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


def _active_chromatic_clips(
    t: float, clips: Sequence[EffectClip]
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.CHROMATIC_ABERRATION:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _lcg_u11(seed: int) -> float:
    """Single value in [-1, 1] from 32-bit seed (LCG, deterministic)."""
    x = seed & 0xFFFFFFFF
    x = (_LCG_MUL * x + _LCG_ADD) & _LCG_MASK
    u = (x + 1) / (_LCG_MASK + 1)
    return 2.0 * u - 1.0


def _direction_rad(clip: EffectClip, song_hash: str) -> float:
    s = clip.settings
    raw = s.get("direction_deg")
    if raw is not None:
        try:
            deg = float(raw)
            if math.isfinite(deg):
                return math.radians(deg)
        except (TypeError, ValueError):
            pass
    seed_src = (
        f"{song_hash or ''}::chromatic_dir::{clip.id}".encode("utf-8", errors="ignore")
    )
    rng_seed = int.from_bytes(
        hashlib.sha256(seed_src).digest()[:4], "little", signed=False
    ) & 0x7FFFFFFF
    rng = np.random.default_rng(rng_seed)
    return float(rng.uniform(0.0, 2.0 * math.pi))


def _clip_shift_vector(
    clip: EffectClip, t: float, song_hash: str
) -> tuple[float, float]:
    """Return (vx, vy) float offset for the R channel (B uses the negation)."""
    s = clip.settings
    shift_px = _float_setting(s, "shift_px", 0.0)
    if not math.isfinite(shift_px) or abs(shift_px) < 1e-9:
        return (0.0, 0.0)

    jitter = _clamp01(_float_setting(s, "jitter", 0.0))
    base = glitch_seed_for_time(song_hash, t)
    salt = zlib.adler32(clip.id.encode("utf-8", errors="ignore")) & 0xFFFFFFFF
    mix = (base ^ salt) & 0xFFFFFFFF
    wobble = _lcg_u11(mix)
    mag = abs(shift_px) * (1.0 + jitter * wobble)
    if not math.isfinite(mag) or mag < 1e-9:
        return (0.0, 0.0)

    theta = _direction_rad(clip, song_hash)
    ux = math.cos(theta)
    uy = math.sin(theta)
    return (mag * ux, mag * uy)


def _clamp_vec(vx: float, vy: float, cap: float) -> tuple[float, float]:
    if cap <= 0.0 or not (math.isfinite(vx) and math.isfinite(vy)):
        return (0.0, 0.0)
    L = math.hypot(vx, vy)
    if L <= 1e-9:
        return (0.0, 0.0)
    if L > cap:
        s = cap / L
        return (vx * s, vy * s)
    return (vx, vy)


def _shift_plane(plane: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift one channel by fractional (dx, dy) in pixel space; empty edges are black.

    Uses linear interpolation so sub-pixel shifts (and small ``shift_px`` after
    jitter) still produce visible full-frame fringing, not just on sharp edges.
    """
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return np.asarray(plane, dtype=np.uint8, copy=True)
    arr = np.asarray(plane, dtype=np.float32)
    # ``ndimage.shift``: (shift_along_rows, shift_along_cols) = (dy, dx)
    out = ndimage.shift(
        arr,
        shift=(float(dy), float(dx)),
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def apply_chromatic_aberration(
    frame: np.ndarray,
    t: float,
    clips: Sequence[EffectClip],
    song_hash: str,
) -> np.ndarray:
    """
    Apply active ``CHROMATIC_ABERRATION`` clips at time ``t``.

    Returns the input ``frame`` unchanged (same array) when no clip is active
    or the net shift rounds to zero — never mutates ``frame`` in place.

    R and B are shifted in opposite directions along each clip's axis; G is
    unchanged. Combined shift magnitude is capped to ``min(H, W) * 0.1`` pixels.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            "apply_chromatic_aberration expects (H, W, 3) uint8, "
            f"got shape={frame.shape} dtype={frame.dtype}"
        )
    if not math.isfinite(float(t)):
        return frame

    active = _active_chromatic_clips(t, clips)
    if not active:
        return frame

    h, w = int(frame.shape[0]), int(frame.shape[1])
    cap = float(min(h, w)) * 0.1

    vx = 0.0
    vy = 0.0
    for clip in active:
        cx, cy = _clip_shift_vector(clip, t, song_hash)
        vx += cx
        vy += cy

    vx, vy = _clamp_vec(vx, vy, cap)
    if math.hypot(vx, vy) < 1e-6:
        return frame

    dx_b, dy_b = -vx, -vy

    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]
    r_out = _shift_plane(r, vx, vy)
    b_out = _shift_plane(b, dx_b, dy_b)
    return np.stack((r_out, g, b_out), axis=2)


__all__ = ["apply_chromatic_aberration"]
