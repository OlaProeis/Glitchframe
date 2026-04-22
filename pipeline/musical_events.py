"""Song-level musical events: drops and build-up tension.

Given the ``analysis.json`` bundle produced by :mod:`pipeline.audio_analyzer`,
compute per-song structural events that reactive shaders and overlays can key
off:

``drops``
    Discrete ``{t, confidence}`` events where overall level **and** bass energy
    step up sharply, typically at (or just after) a segment boundary. Think
    "the drop" in electronic music, the first chorus on a pop track, or the
    first full-band entry in a rock song.

``build_tension``
    A per-frame envelope in ``[0, 1]`` that rises over the seconds leading up
    to each drop and snaps back to ``0`` immediately after it fires. Shaders
    can use it to **dampen** motion before a drop (the "stop shaking just
    before impact" effect) and then unleash on ``drop_hold``.

The detection is fully deterministic (no ML), runs on data already present in
``analysis.json`` (RMS, 8-band log-mel spectrum, segment boundaries), and is
called exactly once at analysis time so every subsequent render samples the
same events.

Output is persisted under the new ``events`` key of ``analysis.json`` (schema
v2); see :func:`build_events_block`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tunables (exposed as module constants so callers / tests can override)
# ---------------------------------------------------------------------------

DEFAULT_RMS_SMOOTH_SEC = 0.30
"""Moving-average window applied to RMS before differencing."""

DEFAULT_LAG_SEC = 0.60
"""Look-back horizon for "rising" RMS / bass jumps (seconds).

Longer than the onset / snare windows in :mod:`pipeline.beat_pulse` because a
drop typically unfolds over hundreds of ms of crescendo, not a single transient.
"""

DEFAULT_MIN_DROP_INTERVAL_SEC = 5.0
"""Minimum spacing between reported drop events.

Prevents double-counting: a single real drop usually spreads its step response
over a second or two and would otherwise peak-pick twice.
"""

DEFAULT_SEGMENT_SNAP_SEC = 1.0
"""Half-width of the window around each candidate where we look for an
existing structural segment boundary to confirm the drop."""

DEFAULT_SCORE_PERCENTILE = 99.0
"""Percentile used to normalise the raw drop-score envelope."""

DEFAULT_SCORE_FLOOR = 0.25
"""Absolute floor on the normalised score (``[0, 1]``) below which peaks are
discarded even if they are local maxima."""

DEFAULT_SEGMENT_OVERRIDE_SCORE = 0.55
"""Normalised score above which a candidate is kept even without a nearby
segment boundary (very strong jumps are drops regardless of segmentation)."""

DEFAULT_MAX_DROPS = 12
"""Safety cap on reported drops per song so pathological inputs can't blow up
downstream consumers."""

DEFAULT_BUILD_WINDOW_SEC = 6.0
"""Length of the pre-drop build-up tension ramp (seconds)."""

DEFAULT_DROP_HOLD_DECAY_SEC = 2.0
"""Exponential decay constant for the post-drop hold envelope. Kept long
enough that a whole 8-bar phrase gets the "drop afterglow" treatment."""

DEFAULT_BASS_BANDS = 2
"""Number of low-frequency log-mel bands averaged into the bass step signal."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_float_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32, copy=False)
    k = np.ones(int(win), dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32, copy=False), k, mode="same").astype(
        np.float32, copy=False
    )


def _positive_lag_step(x: np.ndarray, lag: int) -> np.ndarray:
    """Half-wave rectified ``x[i] - x[i - lag]``; zeros for ``i < lag``."""
    out = np.zeros_like(x, dtype=np.float32)
    if lag <= 0 or x.size <= lag:
        return out
    out[lag:] = np.clip(x[lag:] - x[:-lag], 0.0, None)
    return out


def _smoothstep(t: float) -> float:
    """Standard Hermite smoothstep clamped to ``[0, 1]``."""
    u = max(0.0, min(1.0, float(t)))
    return u * u * (3.0 - 2.0 * u)


def _segment_boundaries(analysis: Mapping[str, Any]) -> list[float]:
    segments = analysis.get("segments")
    if not isinstance(segments, list):
        return []
    boundaries: set[float] = set()
    for seg in segments:
        if not isinstance(seg, Mapping):
            continue
        for key in ("t_start", "t_end"):
            v = seg.get(key)
            try:
                boundaries.add(float(v))
            except (TypeError, ValueError):
                continue
    return sorted(boundaries)


def _has_nearby_boundary(
    t: float, boundaries: Sequence[float], window_sec: float
) -> bool:
    if not boundaries:
        return False
    idx = bisect_right(boundaries, float(t))
    candidates: list[float] = []
    if idx > 0:
        candidates.append(boundaries[idx - 1])
    if idx < len(boundaries):
        candidates.append(boundaries[idx])
    return any(abs(float(t) - float(b)) <= float(window_sec) for b in candidates)


def _rms_series(analysis: Mapping[str, Any]) -> tuple[np.ndarray, float] | None:
    rms = analysis.get("rms")
    if not isinstance(rms, Mapping):
        return None
    arr = _as_float_array(rms.get("values"))
    if arr is None or arr.ndim != 1:
        return None
    fps_raw = rms.get("fps") or analysis.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fps) or fps <= 0.0:
        return None
    return arr, fps


def _bass_series(
    analysis: Mapping[str, Any], num_bass_bands: int = DEFAULT_BASS_BANDS
) -> np.ndarray | None:
    spec = analysis.get("spectrum")
    if not isinstance(spec, Mapping):
        return None
    arr = _as_float_array(spec.get("values"))
    if arr is None or arr.ndim != 2 or arr.shape[1] == 0:
        return None
    bands = max(1, min(int(num_bass_bands), int(arr.shape[1])))
    return arr[:, :bands].mean(axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Drop detection
# ---------------------------------------------------------------------------


def detect_drops(
    analysis: Mapping[str, Any],
    *,
    smooth_sec: float = DEFAULT_RMS_SMOOTH_SEC,
    lag_sec: float = DEFAULT_LAG_SEC,
    min_interval_sec: float = DEFAULT_MIN_DROP_INTERVAL_SEC,
    segment_snap_sec: float = DEFAULT_SEGMENT_SNAP_SEC,
    score_percentile: float = DEFAULT_SCORE_PERCENTILE,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    segment_override_score: float = DEFAULT_SEGMENT_OVERRIDE_SCORE,
    max_drops: int = DEFAULT_MAX_DROPS,
) -> list[dict[str, float]]:
    """Return a list of drop events sorted by time.

    Each event is ``{"t": float_seconds, "confidence": float in [0, 1]}``.
    An empty list is returned for any malformed input (missing RMS, missing
    spectrum, zero-length series) so callers can treat drop detection as a
    best-effort enrichment.
    """
    rms_block = _rms_series(analysis)
    if rms_block is None:
        return []
    rms, fps = rms_block

    bass = _bass_series(analysis)
    # Allow a spectrum-less cache to still produce (weaker) drop candidates
    # by falling back to RMS alone; in that case the bass-step multiplier
    # collapses to 1.0 and we rely on RMS magnitude only.
    if bass is not None and bass.shape[0] > rms.shape[0]:
        bass = bass[: rms.shape[0]]
    if bass is not None and bass.shape[0] < rms.shape[0]:
        rms = rms[: bass.shape[0]]

    n = int(rms.shape[0])
    if n < 8:
        return []

    smooth_win = max(1, int(round(float(smooth_sec) * fps)))
    lag = max(1, int(round(float(lag_sec) * fps)))
    if lag >= n:
        return []

    rms_s = _moving_average(rms, smooth_win)
    d_rms = _positive_lag_step(rms_s, lag)

    if bass is not None:
        bass_s = _moving_average(bass, smooth_win)
        d_bass = _positive_lag_step(bass_s, lag)
        score = d_rms * d_bass
    else:
        score = d_rms

    # Normalise to ``[0, 1]`` using a high percentile so one outlier doesn't
    # squash every other candidate.
    p = float(np.clip(score_percentile, 50.0, 99.9))
    peak = float(np.percentile(score, p)) if score.size else 0.0
    peak = max(peak, float(score.max()) * 0.5, 1e-6)
    norm = np.clip(score / peak, 0.0, 1.0).astype(np.float32, copy=False)

    # Peak-pick: indices where ``norm`` is a strict local max and above the
    # floor. Using a neighbour window proportional to the min-interval keeps
    # the detector from reporting shoulder samples of the same event.
    neighbour_win = max(1, int(round(float(min_interval_sec) * fps * 0.5)))
    candidates: list[tuple[float, float]] = []
    last_i = -10**9
    for i in range(1, n - 1):
        v = float(norm[i])
        if v < float(score_floor):
            continue
        if v < float(norm[i - 1]) or v < float(norm[i + 1]):
            continue
        lo = max(0, i - neighbour_win)
        hi = min(n, i + neighbour_win + 1)
        if v < float(norm[lo:hi].max()):
            continue
        # Enforce the hard minimum spacing across already-accepted peaks.
        if i - last_i < int(round(float(min_interval_sec) * fps)):
            # Replace previous if this is stronger.
            if candidates and v > candidates[-1][1]:
                candidates[-1] = (float(i) / fps, v)
                last_i = i
            continue
        candidates.append((float(i) / fps, v))
        last_i = i

    if not candidates:
        return []

    boundaries = _segment_boundaries(analysis)

    confirmed: list[dict[str, float]] = []
    for t, v in candidates:
        near_boundary = _has_nearby_boundary(t, boundaries, segment_snap_sec)
        if not near_boundary and v < float(segment_override_score):
            continue
        confirmed.append({"t": float(t), "confidence": float(v)})

    if not confirmed:
        return []

    # Keep the strongest ``max_drops`` while preserving time order.
    if len(confirmed) > int(max_drops):
        strongest_idx = sorted(
            range(len(confirmed)),
            key=lambda k: confirmed[k]["confidence"],
            reverse=True,
        )[: int(max_drops)]
        strongest_idx.sort()
        confirmed = [confirmed[k] for k in strongest_idx]

    return confirmed


# ---------------------------------------------------------------------------
# Build-up tension envelope (per-frame)
# ---------------------------------------------------------------------------


def compute_build_tension_series(
    drops: Sequence[Mapping[str, Any]],
    *,
    duration_sec: float,
    fps: int,
    build_window_sec: float = DEFAULT_BUILD_WINDOW_SEC,
) -> np.ndarray:
    """Rendered ``[0, 1]`` tension values, one per frame at ``fps``.

    For each drop event at ``t_d`` with confidence ``c`` this function paints
    a smoothstep ramp rising from ``0`` at ``t_d - build_window_sec`` to ``c``
    just before ``t_d``. Immediately after ``t_d`` the value snaps back to
    ``0`` so the "release" is up to :func:`sample_drop_hold` (a different
    envelope with a different time scale).
    """
    n = max(0, int(math.ceil(float(duration_sec) * float(fps))))
    if n == 0 or not drops:
        return np.zeros(n, dtype=np.float32)

    out = np.zeros(n, dtype=np.float32)
    W = max(1e-3, float(build_window_sec))

    for drop in drops:
        try:
            t_d = float(drop["t"])
            c = float(drop.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if c <= 0.0 or not math.isfinite(t_d) or not math.isfinite(c):
            continue

        start_f = max(0, int(math.floor((t_d - W) * fps)))
        end_f = min(n - 1, int(math.floor(t_d * fps)))
        if end_f < start_f:
            continue

        for i in range(start_f, end_f + 1):
            t = float(i) / float(fps)
            x = (t - (t_d - W)) / W
            val = _smoothstep(x) * c
            if val > out[i]:
                out[i] = val

    return out


def sample_drop_hold(
    t: float,
    drops: Sequence[Mapping[str, Any]],
    *,
    decay_sec: float = DEFAULT_DROP_HOLD_DECAY_SEC,
) -> float:
    """Post-drop exponential afterglow sampled at time ``t``.

    Returns ``c * exp(-(t - t_d) / decay_sec)`` for the most recent drop
    before ``t``, or ``0`` when no drop has fired yet. Useful as a shader
    uniform for "bloom + camera kick for a couple of bars after the drop".
    """
    if not drops or not math.isfinite(float(t)):
        return 0.0
    tau = max(1e-3, float(decay_sec))

    best: float = 0.0
    for drop in drops:
        try:
            t_d = float(drop["t"])
            c = float(drop.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if t_d > float(t):
            break
        dt = float(t) - t_d
        v = c * math.exp(-dt / tau)
        if v > best:
            best = v
    return float(best)


# ---------------------------------------------------------------------------
# Analysis JSON integration
# ---------------------------------------------------------------------------


def build_events_block(
    analysis: Mapping[str, Any],
    *,
    build_window_sec: float = DEFAULT_BUILD_WINDOW_SEC,
    drop_hold_decay_sec: float = DEFAULT_DROP_HOLD_DECAY_SEC,
) -> dict[str, Any]:
    """Return the serialisable ``events`` dict for ``analysis.json``.

    Fields:

    * ``drops`` — list of ``{t, confidence}``.
    * ``build_tension`` — ``{fps, values}``, same shape convention as the
      existing ``rms`` / ``spectrum`` blocks so shader consumers can reuse
      :func:`pipeline.reactive_shader._interp_scalar_series`.
    * ``build_window_sec`` and ``drop_hold_decay_sec`` — record the tuning
      used to render the series so shaders / UI can scale consistently.
    """
    fps_raw = analysis.get("fps")
    try:
        fps = int(fps_raw) if fps_raw is not None else 0
    except (TypeError, ValueError):
        fps = 0
    duration_raw = analysis.get("duration_sec")
    try:
        duration = float(duration_raw) if duration_raw is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    drops = detect_drops(analysis)

    if fps <= 0 or duration <= 0.0:
        values: list[float] = []
    else:
        series = compute_build_tension_series(
            drops,
            duration_sec=duration,
            fps=fps,
            build_window_sec=build_window_sec,
        )
        values = [float(v) for v in series.tolist()]

    return {
        "drops": drops,
        "build_tension": {
            "fps": int(fps),
            "frames": len(values),
            "values": values,
        },
        "build_window_sec": float(build_window_sec),
        "drop_hold_decay_sec": float(drop_hold_decay_sec),
    }


__all__ = [
    "DEFAULT_BASS_BANDS",
    "DEFAULT_BUILD_WINDOW_SEC",
    "DEFAULT_DROP_HOLD_DECAY_SEC",
    "DEFAULT_LAG_SEC",
    "DEFAULT_MAX_DROPS",
    "DEFAULT_MIN_DROP_INTERVAL_SEC",
    "DEFAULT_RMS_SMOOTH_SEC",
    "DEFAULT_SCORE_FLOOR",
    "DEFAULT_SCORE_PERCENTILE",
    "DEFAULT_SEGMENT_OVERRIDE_SCORE",
    "DEFAULT_SEGMENT_SNAP_SEC",
    "build_events_block",
    "compute_build_tension_series",
    "detect_drops",
    "sample_drop_hold",
]
