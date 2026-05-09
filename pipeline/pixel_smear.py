"""
Per-frame horizontal pixel-smear ("datamosh") from
:class:`EffectKind.PIXEL_SMEAR` clips.

Rows are clustered into **horizontal bands** (contiguous scanline ranges) so
the read reads like corrupted slices rather than scattered one-line noise.
Within each band, each affected row gets one streak biased toward **long**
runs. The streak content is usually a **tiled micro-segment** of the row or a
**shifted copy** of a contiguous slice — authentic MPEG-style horizontal
corruption — with a smaller chance of the legacy single-column streak.

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

_DEFAULT_INTENSITY = 0.65
_DEFAULT_DENSITY = 0.16
_DEFAULT_STREAK_LENGTH_FRAC = 0.62


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


def _bands_covering_row_budget(h: int, target_rows: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return ``(y0, band_height)`` tuples whose heights sum to at least ``target_rows``."""
    if h <= 0 or target_rows <= 0:
        return []
    bands: list[tuple[int, int]] = []
    covered = 0
    max_iters = max(target_rows * 6, 24)
    it = 0
    while covered < target_rows and it < max_iters:
        it += 1
        remaining_need = target_rows - covered
        max_bh = min(h, max(1, remaining_need))
        min_bh = min(max_bh, max(2, min(remaining_need, max(3, h // 8 + 1))))
        if min_bh > max_bh:
            min_bh = max_bh
        if min_bh < max_bh:
            bh = int(rng.integers(min_bh, max_bh + 1))
        else:
            bh = min_bh
        bh = min(bh, h)
        y0 = int(rng.integers(0, max(1, h - bh + 1)))
        bands.append((y0, bh))
        covered += bh
    return bands


def _lerp_rows(dst: np.ndarray, src_patch: np.ndarray, intensity: float) -> None:
    """Write ``src_patch`` into ``dst`` row slice with intensity blend (``dst`` same shape)."""
    if intensity >= 1.0 - 1e-4:
        dst[:] = src_patch
        return
    a = float(intensity)
    d = dst.astype(np.float32)
    s = src_patch.astype(np.float32)
    dst[:] = np.clip(d * (1.0 - a) + s * a, 0.0, 255.0).astype(np.uint8)


def _smear_single_row(
    src_row: np.ndarray,
    buf_row: np.ndarray,
    rng: np.random.Generator,
    w: int,
    intensity: float,
    max_streak_px: int,
) -> None:
    sx = int(rng.integers(0, w))
    # Bias toward long streaks (avoids "tiny dashes" everywhere).
    streak_roll = float(rng.random())
    streak_mult = 0.52 + 0.48 * streak_roll * streak_roll
    run = max(8, int(round(max_streak_px * streak_mult)))
    run = min(run, w)

    direction = int(rng.integers(0, 2))
    if direction == 0:
        x0, x1 = sx, min(w, sx + run)
    else:
        x0, x1 = max(0, sx - run + 1), sx + 1

    span = x1 - x0
    if span <= 2:
        return

    mode_roll = float(rng.random())

    if mode_roll < 0.58:
        # Tiled segment: repeat a short slice across the streak (chunky datamosh).
        seg_hi = min(span, max(4, w // 10), 56)
        lo_seg = max(2, min(6, max(2, span // 4)))
        seg_len = int(rng.integers(lo_seg, seg_hi + 1)) if lo_seg < seg_hi else min(lo_seg, seg_hi)
        seg_len = max(2, min(seg_len, span))
        s_max = max(0, w - seg_len)
        center = max(0, min(sx, w - 1))
        lo_pick = max(0, center - seg_len)
        hi_pick = min(center + 1, s_max + 1)
        if lo_pick > hi_pick:
            lo_pick = hi_pick
        s0 = int(rng.integers(lo_pick, hi_pick + 1)) if lo_pick < hi_pick else lo_pick
        s0 = max(0, min(s0, s_max))
        chunk = src_row[s0 : s0 + seg_len].astype(np.float32)
        reps = (span + seg_len - 1) // seg_len
        tiled = np.tile(chunk, (reps, 1))[:span]
        patch = np.clip(tiled, 0.0, 255.0).astype(np.uint8)
        _lerp_rows(buf_row[x0:x1], patch, intensity)

    elif mode_roll < 0.88:
        # Shift-copy: contiguous slice read with horizontal slip (slipped macroblock row).
        slip = int(
            rng.integers(-max(1, span // 2), max(1, span // 2) + 1)
        )
        xs = np.arange(x0, x1, dtype=np.int32) + np.int32(slip)
        xs = np.clip(xs, 0, w - 1)
        patch = src_row[xs]
        _lerp_rows(buf_row[x0:x1], patch, intensity)

    else:
        # Classic single-column streak.
        src_pixel = src_row[sx]
        if intensity >= 1.0 - 1e-4:
            buf_row[x0:x1] = src_pixel
        else:
            dst = buf_row[x0:x1].astype(np.float32)
            src = src_pixel.astype(np.float32)
            buf_row[x0:x1] = np.clip(dst * (1.0 - intensity) + src * intensity, 0.0, 255.0).astype(
                np.uint8
            )


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

    target_rows = max(1, int(round(density * h)))
    bands = _bands_covering_row_budget(h, target_rows, rng)
    if not bands:
        return out

    max_streak_px = max(8, int(round(streak_frac * w)))

    src = out
    buf: np.ndarray | None = None

    for y0, bh in bands:
        y1 = min(h, y0 + bh)
        for y in range(y0, y1):
            if buf is None:
                buf = out.copy()
            _smear_single_row(src[y], buf[y], rng, w, intensity, max_streak_px)

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
