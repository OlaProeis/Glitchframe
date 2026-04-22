"""
Audio-reactive "pulse" envelopes for the logo / title card / future overlays.

Two envelope shapes are supported so the same compositor hook can drive
different aesthetics:

``beats`` (``beat_pulse_envelope``)
    A piecewise exponential keyed off the analyzer's beat grid. At each beat
    time it snaps to ``1.0`` and decays with a BPM-aware ``tau``. This is the
    original, rhythm-locked behaviour and mirrors every subdivision reported
    by the beat tracker.

``bass`` (``build_bass_pulse_track`` → :class:`PulseTrack`)
    A per-frame envelope derived from the low-frequency bands of the cached
    log-mel spectrum in ``analysis.json``. A half-wave rectified first
    difference isolates bass *attacks* (kicks / sub drops) and a fast-attack
    / slow-decay one-pole filter shapes them into clean pulses. This keys
    the visual on actual low-end energy, not on every hi-hat or snare the
    beat tracker happens to land on.

Both shapes feed :func:`scale_and_opacity_for_pulse` which maps ``[0, 1]`` to
a ``(scale, opacity_multiplier)`` tuple so every consumer blends the same
way.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_PULSE_TAU_FRACTION = 0.25
"""Decay time constant as a fraction of the beat period (60 / BPM)."""

DEFAULT_FALLBACK_BPM = 120.0
"""Assumed BPM when the analyzer didn't provide one (e.g. percussion-free music)."""

DEFAULT_BASS_BANDS = 2
"""Number of low-frequency log-mel bands averaged into the bass envelope.

With the analyzer's default ``fmin=40 Hz`` / ``fmax≈16 kHz`` over 8 mel bands
the first two cover roughly sub-bass and bass (~40–300 Hz), which is what a
listener perceives as ``kick / low end``.
"""

DEFAULT_BASS_DECAY_SEC = 0.18
"""One-pole decay time constant for the bass envelope (seconds).

~180 ms gives a visible but short tail on each kick: sharp enough to read as
a beat, slow enough that fast hats / subdivisions don't retrigger visually.
"""

DEFAULT_BASS_NORM_PERCENTILE = 90.0
"""Percentile used to normalise the raw bass envelope before clipping.

Using the 90th percentile (rather than the max) prevents a single dominant
transient from squashing every other kick to near zero while still mapping
typical kicks to ~1.0 on energetic tracks.
"""


def _coerce_beats(beats: Iterable[float]) -> list[float]:
    """Return a sorted, float list of beat times (drops NaN / negative)."""
    out: list[float] = []
    for b in beats:
        try:
            v = float(b)
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or v < 0.0:
            continue
        out.append(v)
    out.sort()
    return out


def _tau_from_bpm(bpm: float | None, fraction: float) -> float:
    """Decay ``tau`` in seconds so the envelope falls to ``1/e`` at ``fraction``
    of the beat period. Clamped to a sensible range so very slow / very fast
    BPM values from a noisy analyzer still produce watchable pulses.
    """
    b = float(bpm) if bpm and bpm > 1e-3 else DEFAULT_FALLBACK_BPM
    period = 60.0 / b
    tau = max(0.04, min(0.40, period * float(fraction)))
    return tau


def beat_pulse_envelope(
    t: float,
    beats: Sequence[float],
    *,
    bpm: float | None = None,
    tau_fraction: float = DEFAULT_PULSE_TAU_FRACTION,
) -> float:
    """Return the pulse amount in ``[0, 1]`` for absolute time ``t``.

    ``beats`` must be in seconds (as produced by
    :func:`pipeline.audio_analyzer.analyze_song`). Finds the most recent beat
    ``t_b <= t`` and returns ``exp(-(t - t_b) / tau)``. Returns ``0.0`` when
    ``t`` precedes the first beat or when ``beats`` is empty.
    """
    sorted_beats = _coerce_beats(beats)
    if not sorted_beats:
        return 0.0
    idx = bisect_right(sorted_beats, float(t)) - 1
    if idx < 0:
        return 0.0
    dt = float(t) - sorted_beats[idx]
    if dt < 0.0:
        return 0.0
    tau = _tau_from_bpm(bpm, tau_fraction)
    return math.exp(-dt / tau)


@dataclass(frozen=True)
class PulseTrack:
    """Pre-sampled pulse envelope with a fixed frame rate.

    ``values`` holds per-frame amplitudes in ``[0, 1]``; ``fps`` is the frame
    rate those samples were computed at (usually the analyzer's ``fps``).
    :meth:`value_at` accepts an arbitrary absolute time in seconds and returns
    the nearest stored sample, clamped to zero outside the track.

    Using a pre-sampled track lets the bass envelope do its smoothing /
    normalisation once at render start instead of per frame in the hot loop.
    """

    values: np.ndarray
    fps: float

    def value_at(self, t: float) -> float:
        if self.values.size == 0:
            return 0.0
        fps = float(self.fps)
        if fps <= 0.0:
            return 0.0
        idx = int(math.floor(float(t) * fps))
        if idx < 0:
            return 0.0
        last = int(self.values.shape[0]) - 1
        if idx > last:
            idx = last
        return float(self.values[idx])


def _extract_spectrum_frames(
    analysis: Mapping[str, Any],
) -> tuple[np.ndarray, float] | None:
    """Return ``(frames x bands, fps)`` from ``analysis.json`` or ``None``.

    Accepts any value that can be coerced to a 2-D ``float32`` array with a
    positive number of frames and bands, and a strictly positive ``fps``
    sampling rate. Missing / malformed entries disable the bass envelope
    (caller falls back gracefully), they never raise: an analyzer that
    failed to emit a spectrum shouldn't take down the render.
    """
    spec = analysis.get("spectrum") if isinstance(analysis, Mapping) else None
    if not isinstance(spec, Mapping):
        return None
    values = spec.get("values")
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    fps_raw = spec.get("fps")
    if fps_raw is None:
        fps_raw = analysis.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fps) or fps <= 0.0:
        return None
    return arr, fps


def build_bass_pulse_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    num_bass_bands: int = DEFAULT_BASS_BANDS,
    decay_sec: float = DEFAULT_BASS_DECAY_SEC,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Build a :class:`PulseTrack` keyed off low-frequency spectrum energy.

    The envelope is shaped in four small steps so kicks / sub drops read
    clearly but sustained bass doesn't leave the logo permanently inflated:

    1. Average the first ``num_bass_bands`` log-mel bands per frame — the
       analyzer's mel grid puts these roughly in the sub-bass / bass range.
    2. Take the *positive* first difference so only rising bass (attacks)
       contributes; flat or decaying bass collapses to 0.
    3. Drive a one-pole exponential decay filter (fast attack, ``decay_sec``
       tail) so every attack produces a clean, short pulse that doesn't
       retrigger on every subdivision.
    4. Normalise by the ``norm_percentile``-th value so energetic tracks
       peak near 1.0 without letting a single outlier crush every other
       kick. ``sensitivity`` then scales the final envelope (``> 1`` makes
       weaker kicks read stronger, ``< 1`` tames a bass-heavy mix).

    Returns ``None`` when ``analysis`` lacks a usable spectrum; callers
    should fall back to the beat-grid envelope (or disable the effect) in
    that case.
    """
    if sensitivity <= 0.0 or not math.isfinite(float(sensitivity)):
        return None
    extracted = _extract_spectrum_frames(analysis)
    if extracted is None:
        return None
    spectrum, fps = extracted
    total_bands = int(spectrum.shape[1])
    bands = max(1, min(int(num_bass_bands), total_bands))
    bass = spectrum[:, :bands].mean(axis=1).astype(np.float32, copy=False)

    diff = np.diff(bass, prepend=float(bass[0])).astype(np.float32, copy=False)
    onsets = np.clip(diff, 0.0, None)

    tau = max(1e-3, float(decay_sec))
    decay = math.exp(-1.0 / (tau * fps))
    env = np.empty_like(onsets)
    acc = 0.0
    for i in range(onsets.shape[0]):
        acc = max(acc * decay, float(onsets[i]))
        env[i] = acc

    p = float(np.clip(float(norm_percentile), 1.0, 99.9))
    peak = float(np.percentile(env, p)) if env.size else 0.0
    # Guard against near-silent tracks where the percentile collapses to
    # zero (would divide into NaN) and against a single dominant transient
    # hiding the rest of the track (use half the max as a sanity floor).
    peak = max(peak, float(env.max()) * 0.5, 1e-3)
    scaled = np.clip(env * (float(sensitivity) / peak), 0.0, 1.0)
    return PulseTrack(values=scaled.astype(np.float32, copy=False), fps=fps)


DEFAULT_LOGO_SUSTAIN_ATTACK_SEC = 0.022
"""Fast upward tracking of sustained sub / 808 level (seconds)."""

DEFAULT_LOGO_SUSTAIN_RELEASE_SEC = 0.68
"""Slow ease-out after the sub tail drops — keeps the logo “open” on long kicks."""

DEFAULT_SNARE_BAND_LO = 3
DEFAULT_SNARE_BAND_HI = 6
"""Inclusive-exclusive mel slice for snare / mid-percussion (8-band 40 Hz–16 kHz grid)."""

DEFAULT_SNARE_DECAY_SEC = 0.095
"""Short decay so the neon edge flashes on snare transients, not sustained mids."""


def build_logo_bass_pulse_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    num_bass_bands: int = DEFAULT_BASS_BANDS,
    attack_decay_sec: float = DEFAULT_BASS_DECAY_SEC,
    sustain_attack_sec: float = DEFAULT_LOGO_SUSTAIN_ATTACK_SEC,
    sustain_release_sec: float = DEFAULT_LOGO_SUSTAIN_RELEASE_SEC,
    sustain_weight: float = 0.78,
    attack_weight: float = 1.0,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Logo-oriented bass curve: sharp attacks plus a slow **sustain** follower.

    Long 808s stay elevated while low-end energy holds, then release gently
    when the sub falls — unlike :func:`build_bass_pulse_track`, which only
    rewards positive *changes* in bass.
    """
    if sensitivity <= 0.0 or not math.isfinite(float(sensitivity)):
        return None
    extracted = _extract_spectrum_frames(analysis)
    if extracted is None:
        return None
    spectrum, fps = extracted
    total_bands = int(spectrum.shape[1])
    bands = max(1, min(int(num_bass_bands), total_bands))
    bass = spectrum[:, :bands].mean(axis=1).astype(np.float32, copy=False)
    n = int(bass.shape[0])

    diff = np.diff(bass, prepend=float(bass[0])).astype(np.float32, copy=False)
    onsets = np.clip(diff, 0.0, None)
    tau_a = max(1e-3, float(attack_decay_sec))
    decay_a = math.exp(-1.0 / (tau_a * fps))
    attack = np.empty_like(onsets)
    acc = 0.0
    for i in range(onsets.shape[0]):
        acc = max(acc * decay_a, float(onsets[i]))
        attack[i] = acc

    ca = math.exp(-1.0 / (max(1e-4, float(sustain_attack_sec)) * fps))
    cr = math.exp(-1.0 / (max(1e-4, float(sustain_release_sec)) * fps))
    sustain = np.empty_like(bass, dtype=np.float32)
    s = 0.0
    for i in range(n):
        b = float(bass[i])
        c = ca if b > s else cr
        s = b + (s - b) * c
        sustain[i] = s

    p = float(np.clip(float(norm_percentile), 1.0, 99.9))

    def _norm(arr: np.ndarray) -> np.ndarray:
        pk = float(np.percentile(arr, p)) if arr.size else 0.0
        pk = max(pk, float(arr.max()) * 0.5, 1e-3)
        return np.clip(arr / pk, 0.0, 1.0)

    a_n = _norm(attack)
    s_n = _norm(sustain)
    raw = np.maximum(a_n * float(attack_weight), s_n * float(sustain_weight))
    peak = float(np.percentile(raw, p)) if raw.size else 0.0
    peak = max(peak, float(raw.max()) * 0.5, 1e-3)
    scaled = np.clip(raw * (float(sensitivity) / peak), 0.0, 1.0)
    return PulseTrack(values=scaled.astype(np.float32, copy=False), fps=fps)


def build_snare_glow_track(
    analysis: Mapping[str, Any],
    *,
    band_lo: int = DEFAULT_SNARE_BAND_LO,
    band_hi: int = DEFAULT_SNARE_BAND_HI,
    decay_sec: float = DEFAULT_SNARE_DECAY_SEC,
    sensitivity: float = 1.0,
    norm_percentile: float = 92.0,
) -> PulseTrack | None:
    """Percussive mid-band envelope for logo neon (snare / clap energy)."""
    if sensitivity <= 0.0 or not math.isfinite(float(sensitivity)):
        return None
    extracted = _extract_spectrum_frames(analysis)
    if extracted is None:
        return None
    spectrum, fps = extracted
    total = int(spectrum.shape[1])
    lo = max(0, min(int(band_lo), total - 1))
    hi = max(lo + 1, min(int(band_hi), total))
    mid = spectrum[:, lo:hi].mean(axis=1).astype(np.float32, copy=False)

    diff = np.diff(mid, prepend=float(mid[0])).astype(np.float32, copy=False)
    onsets = np.clip(diff, 0.0, None)
    tau = max(1e-3, float(decay_sec))
    decay = math.exp(-1.0 / (tau * fps))
    env = np.empty_like(onsets)
    acc = 0.0
    for i in range(onsets.shape[0]):
        acc = max(acc * decay, float(onsets[i]))
        env[i] = acc

    p = float(np.clip(float(norm_percentile), 1.0, 99.9))
    peak = float(np.percentile(env, p)) if env.size else 0.0
    peak = max(peak, float(env.max()) * 0.5, 1e-3)
    scaled = np.clip(env * (float(sensitivity) / peak), 0.0, 1.0)
    return PulseTrack(values=scaled.astype(np.float32, copy=False), fps=fps)


def scale_and_opacity_for_pulse(
    pulse: float,
    *,
    strength: float = 1.0,
    max_scale_delta: float = 0.085,
    max_opacity_boost: float = 0.18,
    max_scale_cap: float = 0.22,
    max_opacity_cap: float = 0.38,
) -> tuple[float, float]:
    """Turn a raw pulse value into ``(scale, opacity_multiplier)``.

    ``strength`` above ``1.0`` (UI slider max 2) increases ``p`` past the
    pulse peak so “full strength” reads as a **big** hit. Caps avoid extreme
    resize artefacts on sub-pixel logos.
    """
    p = max(0.0, min(1.0, float(pulse))) * max(0.0, float(strength))
    scale = 1.0 + min(max_scale_delta * p, max_scale_cap)
    opacity_mul = 1.0 + min(max_opacity_boost * p, max_opacity_cap)
    return scale, opacity_mul


__all__ = [
    "DEFAULT_BASS_BANDS",
    "DEFAULT_BASS_DECAY_SEC",
    "DEFAULT_BASS_NORM_PERCENTILE",
    "DEFAULT_FALLBACK_BPM",
    "DEFAULT_LOGO_SUSTAIN_ATTACK_SEC",
    "DEFAULT_LOGO_SUSTAIN_RELEASE_SEC",
    "DEFAULT_PULSE_TAU_FRACTION",
    "DEFAULT_SNARE_BAND_HI",
    "DEFAULT_SNARE_BAND_LO",
    "DEFAULT_SNARE_DECAY_SEC",
    "PulseTrack",
    "beat_pulse_envelope",
    "build_bass_pulse_track",
    "build_logo_bass_pulse_track",
    "build_snare_glow_track",
    "scale_and_opacity_for_pulse",
]
