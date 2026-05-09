"""
Per-frame horizontal pixel-smear ("datamosh") from
:class:`EffectKind.PIXEL_SMEAR` clips.

For each active clip the renderer picks a pseudo-random subset of rows
(controlled by ``density``) and, for each picked row, samples one source
column and copies that single pixel across a horizontal streak (controlled by
``streak_length_frac``). The streak direction (left vs right) is also
randomised so smears scatter in both directions like authentic glitch art.

Determinism: per-frame seed derives from ``song_hash``, ``clip.id``, and
``round(t * 1000)`` so re-renders of the same cache produce identical pixels.
Multiple active clips compose by running each clip's smear pass sequentially
on the running output (later clips see earlier ones' smears).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind

_DEFAULT_INTENSITY = 0.6
_DEFAULT_DENSITY = 0.18
_DEFAULT_STREAK_LENGTH_FRAC = 0.45


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


def _frame_seed(song_hash: str, clip_id: str, t: float) -> int:
    if not math.isfinite(float(t)):
        return 0
    ms = int(round(float(t) * 1000.0))
    payload = f"{song_hash or ''}{clip_id}{ms}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _active_pixel_smear_clips(
    t: float, clips: Sequence[EffectClip]
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.PIXEL_SMEAR:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _apply_one_clip(
    out: np.ndarray, clip: EffectClip, t: float, song_hash: str
) -> np.ndarray:
    s = clip.settings
    intensity = _clamp01(_float_setting(s, "intensity", _DEFAULT_INTENSITY))
    if intensity <= 0.0:
        return out
    density = _clamp01(_float_setting(s, "density", _DEFAULT_DENSITY))
    if density <= 0.0:
        return out
    streak_frac = _clamp01(
        _float_setting(s, "streak_length_frac", _DEFAULT_STREAK_LENGTH_FRAC)
    )
    if streak_frac <= 0.0:
        return out

    h = int(out.shape[0])
    w = int(out.shape[1])
    if h <= 0 or w <= 0:
        return out

    seed = _frame_seed(song_hash, clip.id, t)
    rng = np.random.Generator(np.random.PCG64(seed & (2**64 - 1)))

    # Random row sample (without replacement); how many to take is the density
    # fraction of total rows. Cap at ``h`` so a density of 1.0 still works.
    n_rows = int(round(density * h))
    if n_rows <= 0:
        return out
    n_rows = min(n_rows, h)
    rows = rng.choice(h, size=n_rows, replace=False)

    max_streak_px = max(1, int(round(streak_frac * w)))

    buf: np.ndarray | None = None
    for y in rows.tolist():
        # Source column sampled inside the visible row.
        sx = int(rng.integers(0, w))
        # Streak extends a random number of pixels up to ``max_streak_px``.
        run = int(rng.integers(1, max_streak_px + 1))
        # Direction: 0 → streak rightward starting at sx; 1 → leftward.
        direction = int(rng.integers(0, 2))
        if direction == 0:
            x0 = sx
            x1 = min(w, sx + run)
        else:
            x0 = max(0, sx - run + 1)
            x1 = sx + 1
        if x1 - x0 <= 1:
            continue
        if buf is None:
            buf = out.copy()
        src_pixel = buf[y, sx, :]
        if intensity >= 1.0 - 1e-4:
            buf[y, x0:x1, :] = src_pixel
        else:
            # Lerp the streak pixels toward the source colour by ``intensity``.
            dst = buf[y, x0:x1, :].astype(np.float32)
            src = src_pixel.astype(np.float32)
            lerped = dst * (1.0 - intensity) + src * intensity
            buf[y, x0:x1, :] = np.clip(lerped, 0.0, 255.0).astype(np.uint8)
    if buf is None:
        return out
    return buf


def apply_pixel_smear(
    frame: np.ndarray,
    t: float,
    clips: Sequence[EffectClip],
    song_hash: str,
) -> np.ndarray:
    """Apply active ``PIXEL_SMEAR`` clips at time ``t``.

    Returns the input ``frame`` unchanged (same object) when no clip is
    active, ``t`` is non-finite, or every active clip's settings collapse to
    zero contribution. Never mutates ``frame`` in place.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            "apply_pixel_smear expects (H, W, 3) uint8, "
            f"got shape={frame.shape} dtype={frame.dtype}"
        )
    if not math.isfinite(float(t)):
        return frame

    active = _active_pixel_smear_clips(t, clips)
    if not active:
        return frame

    out: np.ndarray = frame
    for clip in active:
        out = _apply_one_clip(out, clip, t, song_hash)
    return out


__all__ = ["apply_pixel_smear"]
