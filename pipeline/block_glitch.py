"""
Per-frame macroblock displacement ("JPEG corruption") from
:class:`EffectKind.BLOCK_GLITCH` clips.

Splits the frame into a grid of square blocks of size ``block_size_px`` and
randomly displaces a fraction of those blocks by per-axis offsets bounded by
``displace_frac * block_size_px``. The visual reads as discrete rectangular
chunks of the frame jumping around — distinct from the band-level sliding of
``SCANLINE_TEAR`` (which only shifts horizontal slices by N pixels) and the
sub-pixel R/B drift of ``CHROMATIC_ABERRATION``.

Determinism: per-frame seed is derived from ``song_hash``, ``clip.id``, and
``round(t * 1000)``. Multiple active clips compose by running each clip's
displacement pass on the running output (later clips see earlier glitches).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind

_DEFAULT_INTENSITY = 0.35
_DEFAULT_BLOCK_SIZE_PX = 32
_DEFAULT_DISPLACE_FRAC = 0.6


def _float_setting(settings: dict[str, object], key: str, default: float) -> float:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int_setting(settings: dict[str, object], key: str, default: int) -> int:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return int(round(float(v)))
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


def _active_block_glitch_clips(
    t: float, clips: Sequence[EffectClip]
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.BLOCK_GLITCH:
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
    block_size = _int_setting(s, "block_size_px", _DEFAULT_BLOCK_SIZE_PX)
    if block_size < 2:
        block_size = 2
    displace_frac = _float_setting(s, "displace_frac", _DEFAULT_DISPLACE_FRAC)
    if not math.isfinite(displace_frac) or displace_frac <= 0.0:
        return out

    h = int(out.shape[0])
    w = int(out.shape[1])
    if h <= 0 or w <= 0:
        return out

    block_size = max(2, min(block_size, min(h, w)))
    rows = max(1, h // block_size)
    cols = max(1, w // block_size)
    total_blocks = rows * cols
    n_pick = int(round(intensity * total_blocks))
    if n_pick <= 0:
        return out
    n_pick = min(n_pick, total_blocks)

    seed = _frame_seed(song_hash, clip.id, t)
    rng = np.random.Generator(np.random.PCG64(seed & (2**64 - 1)))
    flat_idx = rng.choice(total_blocks, size=n_pick, replace=False)

    max_dx = max(1, int(round(displace_frac * block_size)))

    # ``buf`` stays as the source of truth so picked-block reads come from
    # the un-displaced frame; writes land on a fresh copy so neighbouring
    # blocks don't cascade-displace into each other.
    src = out
    buf = out.copy()
    for idx in flat_idx.tolist():
        br = idx // cols
        bc = idx % cols
        sy0 = br * block_size
        sx0 = bc * block_size
        sy1 = min(h, sy0 + block_size)
        sx1 = min(w, sx0 + block_size)
        bh = sy1 - sy0
        bw = sx1 - sx0
        if bh <= 0 or bw <= 0:
            continue
        dx = int(rng.integers(-max_dx, max_dx + 1))
        dy = int(rng.integers(-max_dx, max_dx + 1))
        if dx == 0 and dy == 0:
            continue
        dy0 = sy0 + dy
        dx0 = sx0 + dx
        dy1 = dy0 + bh
        dx1 = dx0 + bw
        # Clamp the destination patch to the frame; trim the source to match
        # so partial off-screen displacements still look correct.
        if dy0 < 0:
            sy0 += -dy0
            dy0 = 0
        if dx0 < 0:
            sx0 += -dx0
            dx0 = 0
        if dy1 > h:
            sy1 -= dy1 - h
            dy1 = h
        if dx1 > w:
            sx1 -= dx1 - w
            dx1 = w
        if dy1 - dy0 <= 0 or dx1 - dx0 <= 0:
            continue
        buf[dy0:dy1, dx0:dx1, :] = src[sy0:sy1, sx0:sx1, :]
    return buf


def apply_block_glitch(
    frame: np.ndarray,
    t: float,
    clips: Sequence[EffectClip],
    song_hash: str,
) -> np.ndarray:
    """Apply active ``BLOCK_GLITCH`` clips at time ``t``.

    Returns the input ``frame`` unchanged (same object) when no clip is
    active, ``t`` is non-finite, or every active clip's settings collapse to
    zero contribution. Never mutates ``frame`` in place.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            "apply_block_glitch expects (H, W, 3) uint8, "
            f"got shape={frame.shape} dtype={frame.dtype}"
        )
    if not math.isfinite(float(t)):
        return frame

    active = _active_block_glitch_clips(t, clips)
    if not active:
        return frame

    out: np.ndarray = frame
    for clip in active:
        out = _apply_one_clip(out, clip, t, song_hash)
    return out


__all__ = ["apply_block_glitch"]
