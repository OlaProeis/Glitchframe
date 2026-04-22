"""
Preview window selection: pick the loudest ~10 s slice of an ``analysis.json``.

Used by :func:`orchestrator.orchestrate_preview_10s` so a short preview focuses
on the most energetic section of the track (proxy for the chorus). Reuses the
smoothed RMS envelope already computed by :mod:`pipeline.audio_analyzer` so no
extra audio decoding is required.

The selection is deterministic: the starting frame is the argmax of a
box-filter convolution over ``rms.values`` of length ``round(window_sec * fps)``.
Tracks shorter than ``window_sec`` fall back to ``t_start = 0.0``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

DEFAULT_PREVIEW_WINDOW_SEC = 10.0


def pick_loudest_window_start(
    analysis: Mapping[str, Any],
    *,
    window_sec: float = DEFAULT_PREVIEW_WINDOW_SEC,
) -> float:
    """
    Return the start time (seconds) of the loudest ``window_sec`` slice.

    Falls back to ``0.0`` when ``analysis['rms']`` is missing/empty, the RMS
    fps is unusable, or the track is shorter than ``window_sec``. The result
    is clamped to ``[0, max(0, duration - window_sec)]`` when
    ``analysis['duration_sec']`` is known.

    Raises
    ------
    ValueError
        If ``window_sec`` is not positive.
    """
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec!r}")

    duration = float(analysis.get("duration_sec") or 0.0)
    rms_block = analysis.get("rms") or {}
    values = rms_block.get("values")
    rfps = float(rms_block.get("fps") or analysis.get("fps") or 30)

    if not isinstance(values, list) or not values or rfps <= 0:
        return 0.0

    arr = np.asarray(values, dtype=np.float64)
    window_frames = max(1, int(round(window_sec * rfps)))
    if arr.size <= window_frames:
        return 0.0

    kernel = np.ones(window_frames, dtype=np.float64)
    energy = np.convolve(arr, kernel, mode="valid")
    start_idx = int(np.argmax(energy))
    t_start = float(start_idx) / rfps

    if duration > 0:
        t_start = min(t_start, max(0.0, duration - window_sec))
    return max(0.0, float(t_start))


__all__ = [
    "DEFAULT_PREVIEW_WINDOW_SEC",
    "pick_loudest_window_start",
]
