"""
Per-frame macroblock displacement ("JPEG corruption") from
:class:`EffectKind.BLOCK_GLITCH` clips.

Splits the frame into a grid of square blocks of size ``block_size_px`` and
displaces a fraction of those blocks. Roughly half the time, blocks are
chosen as **full macroblock rows** (strip coherence) with a shared horizontal
and vertical shift per row — the look of broken motion vectors — otherwise
scattered blocks stay chaotic. **Source slip** reads horizontal slices from a
misaligned offset inside the frame, producing tearing inside tiles.

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

_DEFAULT_INTENSITY = 0.38
_DEFAULT_BLOCK_SIZE_PX = 28
_DEFAULT_DISPLACE_FRAC = 0.85


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


def _pick_block_indices(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    total_blocks: int,
    n_pick: int,
) -> tuple[np.ndarray, bool]:
    """Return ``n_pick`` flat indices and whether picks are strip-coherent (full MB rows).

    When strip-coherent, every block in a given macro-row shares the same random
    motion vector — closer to broken MPEG predictions than scattered tiles.
    """
    strip_mode = bool(rng.random() < 0.52) and rows >= 2
    if strip_mode:
        min_rows = max(1, min(rows, n_pick // max(cols // 2, 1)))
        n_strip_rows = min(rows, max(min_rows, int(round(min_rows * rng.uniform(0.75, 1.25)))))
        br_pick = rng.choice(rows, size=n_strip_rows, replace=False)
        candidates = np.array(
            [br * cols + bc for br in br_pick.tolist() for bc in range(cols)],
            dtype=np.int64,
        )
        if candidates.size <= n_pick:
            flat_idx = candidates
            if flat_idx.size < n_pick:
                pool = np.arange(total_blocks, dtype=np.int64)
                mask = np.ones(total_blocks, dtype=bool)
                mask[flat_idx] = False
                rest = pool[mask]
                need = n_pick - flat_idx.size
                extra = rng.choice(rest, size=min(need, rest.size), replace=False)
                flat_idx = np.concatenate([flat_idx, extra])
        else:
            flat_idx = rng.choice(candidates, size=n_pick, replace=False)
        return flat_idx, True

    return rng.choice(total_blocks, size=n_pick, replace=False), False


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

    flat_idx, row_coherent = _pick_block_indices(rng, rows, cols, total_blocks, n_pick)

    max_dx = max(1, int(round(displace_frac * block_size)))

    src = out
    buf = out.copy()

    dx_cache: dict[int, int] = {}
    dy_cache: dict[int, int] = {}

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

        if row_coherent:
            if br not in dx_cache:
                dx_cache[br] = int(rng.integers(-max_dx, max_dx + 1))
            if br not in dy_cache:
                dy_cache[br] = int(rng.integers(-max_dx, max_dx + 1))
            dx = dx_cache[br]
            dy = dy_cache[br]
        else:
            dx = int(rng.integers(-max_dx, max_dx + 1))
            dy = int(rng.integers(-max_dx, max_dx + 1))

        if dx == 0 and dy == 0:
            continue

        slip_max = max(1, min(max_dx, max(1, bw // 2)))
        slip_roll = float(rng.random())
        slip_x = int(rng.integers(-slip_max, slip_max + 1)) if slip_roll < 0.72 else 0

        dy0 = sy0 + dy
        dx0 = sx0 + dx
        dy1 = dy0 + bh
        dx1 = dx0 + bw

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

        bw_src = sx1 - sx0
        read_lo = sx0 + slip_x
        read_lo = max(0, min(read_lo, w - bw_src))
        read_hi = read_lo + bw_src

        buf[dy0:dy1, dx0:dx1, :] = src[sy0:sy1, read_lo:read_hi, :]

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
