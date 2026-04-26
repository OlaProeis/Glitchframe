"""
Horizontal row-offset distortion from :class:`EffectKind.SCANLINE_TEAR` clips.

Each active clip samples several horizontal bands at independent y positions and
applies a per-band integer horizontal shift to all RGB channels. Shifts are
bounded to at most ``|intensity| * width * 0.25`` pixels. Later clips in the
timeline are applied to the result of earlier clips.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from pipeline.effects_timeline import EffectClip, EffectKind


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _int_setting(settings: dict[str, object], key: str, default: int) -> int:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _float_setting(settings: dict[str, object], key: str, default: float) -> float:
    v = settings.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _wrap_mode(settings: dict[str, object]) -> str:
    raw = settings.get("wrap_mode", "wrap")
    if not isinstance(raw, str):
        return "wrap"
    s = raw.strip().lower()
    if s in ("wrap", "clamp", "black"):
        return s
    return "wrap"


def _frame_seed(song_hash: str, clip_id: str, t: float) -> int:
    """Stable 64-bit seed: ``sha256`` over ``song_hash + clip_id + round(t*1000)``."""
    if not math.isfinite(float(t)):
        return 0
    ms = int(round(float(t) * 1000.0))
    payload = f"{song_hash or ''}{clip_id}{ms}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _active_scanline_clips(
    t: float, clips: Sequence[EffectClip]
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for c in clips:
        if c.kind is not EffectKind.SCANLINE_TEAR:
            continue
        t0 = float(c.t_start)
        t1 = t0 + float(c.duration_s)
        if t0 <= t < t1:
            out.append(c)
    return out


def _max_shift_px(intensity: float, w: int) -> int:
    cap = _clamp01(intensity) * float(w) * 0.25
    if not math.isfinite(cap) or cap < 0.0:
        return 0
    m = int(round(cap))
    return max(0, m)


def _band_count(clip: EffectClip, seed: int) -> int:
    n = _int_setting(clip.settings, "band_count", 0)
    if n >= 1:
        return n
    # 3..6 from seed nibble
    return 3 + (seed & 0x3)


def _default_band_height(h: int, n_bands: int) -> int:
    if n_bands < 1:
        n_bands = 1
    return max(1, h // (n_bands + 1))


def _shift_bgr_band(band: np.ndarray, dx: int, mode: str) -> np.ndarray:
    """Shift a (H, W, 3) uint8 band horizontally; vacated / edges per *mode*."""
    if band.ndim != 3 or band.shape[2] != 3 or band.dtype != np.uint8:
        raise ValueError("band must be (H, W, 3) uint8")
    if dx == 0:
        return np.array(band, copy=True, dtype=np.uint8, order="C")
    h, w, _ = int(band.shape[0]), int(band.shape[1]), 3
    if w <= 0 or h <= 0:
        return np.array(band, copy=True, dtype=np.uint8)
    dxi = int(dx)
    if mode == "wrap":
        return np.roll(band, dxi, axis=1)
    if mode == "black":
        out = np.zeros_like(band)
        if dxi == 0:
            return out
        if dxi > 0:
            out[:, dxi:, :] = band[:, : w - dxi, :]
        else:
            s = -dxi
            out[:, : w - s, :] = band[:, s:, :]
        return out
    if mode == "clamp":
        idx = np.arange(w, dtype=np.int32) - np.int32(dxi)
        idx = np.clip(idx, 0, w - 1)
        return np.asarray(band[:, idx, :], dtype=np.uint8, order="C")
    return np.roll(band, dxi, axis=1)


def _apply_one_clip(
    out: np.ndarray, clip: EffectClip, t: float, song_hash: str
) -> np.ndarray:
    seed = _frame_seed(song_hash, clip.id, t)
    rng = np.random.Generator(np.random.PCG64(seed & (2**64 - 1)))
    s = clip.settings
    w_set = s.get("band_height_px")
    intensity = _float_setting(s, "intensity", 0.0)
    max_dx = _max_shift_px(intensity, int(out.shape[1]))
    if max_dx < 1 or _clamp01(intensity) <= 0.0:
        return out

    h = int(out.shape[0])
    w = int(out.shape[1])
    n_bands = _band_count(clip, seed)
    n_bands = max(1, n_bands)

    if w_set is not None and w_set != "":
        try:
            h_band = int(round(float(w_set)))
        except (TypeError, ValueError):
            h_band = _default_band_height(h, n_bands)
        h_band = max(1, min(h_band, h))
    else:
        h_band = _default_band_height(h, n_bands)

    mode = _wrap_mode(s)
    w_buf: np.ndarray | None = None
    for _ in range(n_bands):
        y0 = int(rng.integers(0, max(1, h - h_band + 1)))
        y1 = y0 + h_band
        dx = int(rng.integers(-max_dx, max_dx + 1))
        if dx == 0:
            continue
        if w_buf is None:
            w_buf = out.copy()
        w_buf[y0:y1, :, :] = _shift_bgr_band(w_buf[y0:y1, :, :], dx, mode)
    if w_buf is None:
        return out
    return w_buf


def apply_scanline_tear(
    frame: np.ndarray,
    t: float,
    clips: Sequence[EffectClip],
    song_hash: str,
) -> np.ndarray:
    """
    Apply active ``SCANLINE_TEAR`` clips at time ``t``.

    Returns the input ``frame`` unchanged (same object) when no clip is active,
    when ``t`` is non-finite, or when no band applies a non-zero shift — never
    mutates ``frame`` in place.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            "apply_scanline_tear expects (H, W, 3) uint8, "
            f"got shape={frame.shape} dtype={frame.dtype}"
        )
    if not math.isfinite(float(t)):
        return frame

    active = _active_scanline_clips(t, clips)
    if not active:
        return frame

    out: np.ndarray = frame
    for clip in active:
        nxt = _apply_one_clip(out, clip, t, song_hash)
        out = nxt
    return out


__all__ = ["apply_scanline_tear"]
