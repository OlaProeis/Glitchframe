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

# ---------------------------------------------------------------------------
# Generalised band-transient defaults (used by the reactive shader layer).
#
# The bass defaults above already carve out [0, 2) with a 0.18 s decay; these
# pick sensible windows for the rest of the 8-band log-mel grid
# (40 Hz – 16 kHz). Decays get progressively shorter: low-end wants to "sit"
# for a bit, hats / cymbals want to snap.
# ---------------------------------------------------------------------------

DEFAULT_LO_BAND_LO = 0
DEFAULT_LO_BAND_HI = 2  # ~40 – 300 Hz (kick / sub)
DEFAULT_LO_DECAY_SEC = 0.34
"""Shader-side low-transient decay. Deliberately longer than the logo bass
pulse so backgrounds breathe on every kick instead of flickering."""

DEFAULT_MID_BAND_LO = 3
DEFAULT_MID_BAND_HI = 6  # ~800 Hz – 5 kHz (snare / mids)
DEFAULT_MID_DECAY_SEC = 0.12

DEFAULT_HI_BAND_LO = 6
DEFAULT_HI_BAND_HI = 8  # ~5 – 16 kHz (hats / cymbals / air)
DEFAULT_HI_DECAY_SEC = 0.06

DEFAULT_LOGO_PULSE_DYN_HI_PCT = 99.5
"""Upper percentile for the high end of the logo envelope stretch range.

``r_lo`` uses ``min(raw)`` so a single quiet intro (or digital silence) anchors
the floor. If ``r_lo`` were a low *percentile* instead, long sustained bass
sections dominate the histogram and the 5th–10th percentile lands **inside**
the 808 plateau — then the whole hold maps near 0 and only rare transients
hit 1 (inverted, tiny motion). The high end stays a robust percentile so one
hot sample does not swallow the range.
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


def _extract_rms_frames(
    analysis: Mapping[str, Any],
) -> tuple[np.ndarray, float] | None:
    """Return ``(1-D rms samples, fps)`` from ``analysis.json`` or ``None``."""
    rms = analysis.get("rms") if isinstance(analysis, Mapping) else None
    if not isinstance(rms, Mapping):
        return None
    values = rms.get("values")
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size < 4:
        return None
    fps_raw = rms.get("fps")
    if fps_raw is None:
        fps_raw = analysis.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fps) or fps <= 0.0:
        return None
    return arr.astype(np.float32, copy=False), fps


DEFAULT_IMPACT_SMOOTH_SEC = 0.11
"""Moving-average window on RMS before differencing (seconds)."""

DEFAULT_IMPACT_LAG_SEC = 0.19
"""Look-back horizon for positive RMS jumps — build-ups vs drops (seconds)."""

DEFAULT_IMPACT_DECAY_SEC = 0.15
"""Envelope decay after a detected impact (seconds)."""

DEFAULT_IMPACT_NORM_PCT = 87.0
"""Percentile normalisation on the impact envelope (``[0, 1]`` mapping)."""


def build_rms_impact_pulse_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    smooth_sec: float = DEFAULT_IMPACT_SMOOTH_SEC,
    lag_sec: float = DEFAULT_IMPACT_LAG_SEC,
    decay_sec: float = DEFAULT_IMPACT_DECAY_SEC,
    norm_percentile: float = DEFAULT_IMPACT_NORM_PCT,
) -> PulseTrack | None:
    """Per-frame ``[0, 1]`` envelope for loudness **jumps** (drops, impacts).

    Smooths RMS, takes positive differences versus a lagged copy (energy ramp),
    then peak-picks through a short decay — similar shaping to the snare glow
    track but keyed off overall level instead of mid mel bands.
    """
    if sensitivity <= 0.0 or not math.isfinite(float(sensitivity)):
        return None
    extracted = _extract_rms_frames(analysis)
    if extracted is None:
        return None
    rms, fps = extracted
    n = int(rms.shape[0])
    win = max(1, int(round(float(smooth_sec) * fps)))
    if win > 1:
        k = np.ones(win, dtype=np.float32) / float(win)
        smooth = np.convolve(rms, k, mode="same").astype(np.float32, copy=False)
    else:
        smooth = rms
    lag = max(1, int(round(float(lag_sec) * fps)))
    jump = np.zeros(n, dtype=np.float32)
    for i in range(lag, n):
        d = float(smooth[i]) - float(smooth[i - lag])
        if d > 0.0:
            jump[i] = d
    tau = max(1e-3, float(decay_sec))
    decay = math.exp(-1.0 / (tau * fps))
    env = np.empty_like(jump)
    acc = 0.0
    for i in range(n):
        acc = max(acc * decay, float(jump[i]))
        env[i] = acc
    p = float(np.clip(float(norm_percentile), 1.0, 99.9))
    peak = float(np.percentile(env, p)) if env.size else 0.0
    peak = max(peak, float(env.max()) * 0.5, 1e-3)
    scaled = np.clip(env * (float(sensitivity) / peak), 0.0, 1.0)
    return PulseTrack(values=scaled.astype(np.float32, copy=False), fps=fps)


def build_band_pulse_track(
    analysis: Mapping[str, Any],
    *,
    band_lo: int,
    band_hi: int,
    decay_sec: float,
    sensitivity: float = 1.0,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Generalised transient envelope for any contiguous slice of mel bands.

    The shape is identical to :func:`build_bass_pulse_track` but keyed off
    ``spectrum[:, band_lo:band_hi]`` instead of a fixed ``[0, num_bass_bands)``
    prefix. Used by the reactive shader layer to carve out low / mid / high
    transient envelopes the same way the logo already does for bass.

    Steps (same as the existing bass builder so behaviour stays familiar):

    1. Average the requested band slice per frame.
    2. Take the *positive* first difference so only rising energy (attacks)
       contributes; flat or decaying energy collapses to ``0``.
    3. One-pole exponential decay with time constant ``decay_sec`` — short
       values (~0.06 s) give hat-snappy envelopes, longer values (~0.34 s)
       give breathing low-end.
    4. Normalise by the ``norm_percentile``-th value and scale by
       ``sensitivity``; clip to ``[0, 1]``.

    Returns ``None`` when ``analysis`` lacks a usable spectrum or when the
    requested slice is empty.
    """
    if sensitivity <= 0.0 or not math.isfinite(float(sensitivity)):
        return None
    extracted = _extract_spectrum_frames(analysis)
    if extracted is None:
        return None
    spectrum, fps = extracted
    total_bands = int(spectrum.shape[1])
    lo = max(0, min(int(band_lo), total_bands - 1))
    hi = max(lo + 1, min(int(band_hi), total_bands))
    if hi <= lo:
        return None
    avg = spectrum[:, lo:hi].mean(axis=1).astype(np.float32, copy=False)

    diff = np.diff(avg, prepend=float(avg[0])).astype(np.float32, copy=False)
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
    # Guard against near-silent tracks (percentile collapses to 0) and against
    # a single dominant transient hiding the rest of the track (half-max floor).
    peak = max(peak, float(env.max()) * 0.5, 1e-3)
    scaled = np.clip(env * (float(sensitivity) / peak), 0.0, 1.0)
    return PulseTrack(values=scaled.astype(np.float32, copy=False), fps=fps)


def build_bass_pulse_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    num_bass_bands: int = DEFAULT_BASS_BANDS,
    decay_sec: float = DEFAULT_BASS_DECAY_SEC,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Build a :class:`PulseTrack` keyed off low-frequency spectrum energy.

    Thin wrapper over :func:`build_band_pulse_track` that preserves the
    historical ``num_bass_bands`` parameter name used by the logo pulse. New
    code (notably the reactive shader layer) should call
    :func:`build_band_pulse_track` directly.
    """
    return build_band_pulse_track(
        analysis,
        band_lo=0,
        band_hi=int(num_bass_bands),
        decay_sec=decay_sec,
        sensitivity=sensitivity,
        norm_percentile=norm_percentile,
    )


def build_lo_transient_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    decay_sec: float = DEFAULT_LO_DECAY_SEC,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Low-band (kick / sub) transient envelope for reactive shaders.

    Longer decay than the logo bass pulse (~0.34 s) so background motion
    breathes on every kick rather than flickering at spectrum frame rate.
    """
    return build_band_pulse_track(
        analysis,
        band_lo=DEFAULT_LO_BAND_LO,
        band_hi=DEFAULT_LO_BAND_HI,
        decay_sec=decay_sec,
        sensitivity=sensitivity,
        norm_percentile=norm_percentile,
    )


def build_mid_transient_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    decay_sec: float = DEFAULT_MID_DECAY_SEC,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Mid-band (snare / clap / body) transient envelope for shaders."""
    return build_band_pulse_track(
        analysis,
        band_lo=DEFAULT_MID_BAND_LO,
        band_hi=DEFAULT_MID_BAND_HI,
        decay_sec=decay_sec,
        sensitivity=sensitivity,
        norm_percentile=norm_percentile,
    )


def build_hi_transient_track(
    analysis: Mapping[str, Any],
    *,
    sensitivity: float = 1.0,
    decay_sec: float = DEFAULT_HI_DECAY_SEC,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """High-band (hats / cymbals / air) transient envelope for shaders."""
    return build_band_pulse_track(
        analysis,
        band_lo=DEFAULT_HI_BAND_LO,
        band_hi=DEFAULT_HI_BAND_HI,
        decay_sec=decay_sec,
        sensitivity=sensitivity,
        norm_percentile=norm_percentile,
    )


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
    sustain_weight: float = 0.30,
    attack_weight: float = 1.0,
    norm_percentile: float = DEFAULT_BASS_NORM_PERCENTILE,
) -> PulseTrack | None:
    """Logo-oriented bass curve: sharp attacks plus a slow **sustain** follower.

    Long 808s stay elevated while low-end energy holds, then release gently
    when the sub falls — unlike :func:`build_bass_pulse_track`, which only
    rewards positive *changes* in bass.

    ``sustain_weight`` defaults to ``0.30`` (attack-dominant): the sustain
    component keeps the logo from fully collapsing between kicks on long
    bass plateaus, but attacks drive the visible motion. Higher values push
    the envelope toward always-on (around ``0.78`` the mix saturates near
    ``1.0`` on bass-heavy tracks and the logo stops bouncing).
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

    hi_pct = float(np.clip(float(DEFAULT_LOGO_PULSE_DYN_HI_PCT), 1.0, 100.0))
    r_lo = float(np.min(raw)) if raw.size else 0.0
    r_hi = float(np.percentile(raw, hi_pct)) if raw.size else 1.0
    r_hi = max(r_hi, r_lo + 1e-5)
    span = r_hi - r_lo
    if span < 1e-5 or not math.isfinite(span):
        # Flat ``raw`` (common on long held 808s): percentile span vanishes. Do
        # not fall back to attack-only — that collapses the logo to silence while
        # the sub is still holding. Scale by peak so a sustained high ``raw``
        # reads as a steady elevated pulse instead.
        rh = float(np.max(raw)) if raw.size else 0.0
        if rh < 1e-6:
            stretched = np.zeros_like(raw, dtype=np.float32)
        else:
            stretched = np.clip(raw / rh, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        stretched = np.clip((raw - r_lo) / span, 0.0, 1.0).astype(
            np.float32, copy=False
        )

    peak = float(np.percentile(stretched, p)) if stretched.size else 0.0
    peak = max(peak, float(stretched.max()) * 0.5, 1e-3)
    scaled = np.clip(stretched * (float(sensitivity) / peak), 0.0, 1.0)
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


def apply_pulse_deadzone(
    pulse: float,
    *,
    deadzone: float = 0.12,
    soft_width: float = 0.08,
) -> float:
    """Map a pulse value through a soft deadzone → ``[0, 1]``.

    Very small pulses (``≤ deadzone``) collapse to ``0``; pulses in
    ``(deadzone, deadzone + soft_width)`` ramp up via smoothstep so the
    transition into visible motion stays gentle; values above the soft zone
    keep the *remaining* proportional energy (linear above the knee).

    This is the fix for **chill-part micro-shake**: the bass envelope idles in
    the 0.05--0.20 range during quiet sections, which the linear mapping in
    :func:`scale_and_opacity_for_pulse` turns into a visible constant wobble.
    Dropping the low tail to zero and smoothstepping the shoulder removes the
    jitter without flattening real kicks (peaks ~1.0).

    Passing ``deadzone <= 0`` returns the input clamped to ``[0, 1]`` so
    callers can restore the legacy behaviour without branching.
    """
    p = max(0.0, min(1.0, float(pulse)))
    dz = max(0.0, float(deadzone))
    if dz <= 0.0:
        return p
    lo = dz
    hi = dz + max(1e-6, float(soft_width))
    if p <= lo:
        return 0.0
    if p >= hi:
        # Linear above the knee, mapped so the peak (p=1) stays at 1.
        span = max(1e-6, 1.0 - lo)
        return float(min(1.0, (p - lo) / span))
    x = (p - lo) / (hi - lo)
    # Smoothstep shoulder: ``x*x*(3-2x)`` then scale back into the outer
    # linear ramp so there's no C0 discontinuity at ``p == hi``.
    eased = x * x * (3.0 - 2.0 * x)
    knee = (hi - lo) / max(1e-6, 1.0 - lo)
    return float(eased * knee)


def scale_and_opacity_for_pulse(
    pulse: float,
    *,
    strength: float = 1.0,
    deadzone: float = 0.12,
    soft_width: float = 0.08,
    max_scale_delta: float = 0.12,
    max_opacity_boost: float = 0.22,
    max_scale_cap: float = 0.45,
    max_opacity_cap: float = 0.60,
) -> tuple[float, float]:
    """Turn a raw pulse value into ``(scale, opacity_multiplier)``.

    ``strength`` scales the pulse (UI default 2, max 4) so higher settings read
    as a **bigger** hit. Caps avoid extreme resize artefacts on sub-pixel logos.

    ``deadzone`` + ``soft_width`` suppress chill-section micro-shake by pushing
    low-amplitude pulses (< ``deadzone``) to zero and smoothstepping the knee
    (``deadzone..deadzone+soft_width``). Pass ``deadzone=0.0`` for the pre-task
    linear mapping.
    """
    p = apply_pulse_deadzone(pulse, deadzone=deadzone, soft_width=soft_width)
    p *= max(0.0, float(strength))
    scale = 1.0 + min(max_scale_delta * p, max_scale_cap)
    opacity_mul = 1.0 + min(max_opacity_boost * p, max_opacity_cap)
    return scale, opacity_mul


__all__ = [
    "DEFAULT_BASS_BANDS",
    "DEFAULT_BASS_DECAY_SEC",
    "DEFAULT_BASS_NORM_PERCENTILE",
    "DEFAULT_FALLBACK_BPM",
    "DEFAULT_HI_BAND_HI",
    "DEFAULT_HI_BAND_LO",
    "DEFAULT_HI_DECAY_SEC",
    "DEFAULT_IMPACT_DECAY_SEC",
    "DEFAULT_IMPACT_LAG_SEC",
    "DEFAULT_IMPACT_NORM_PCT",
    "DEFAULT_IMPACT_SMOOTH_SEC",
    "DEFAULT_LO_BAND_HI",
    "DEFAULT_LO_BAND_LO",
    "DEFAULT_LO_DECAY_SEC",
    "DEFAULT_LOGO_SUSTAIN_ATTACK_SEC",
    "DEFAULT_LOGO_SUSTAIN_RELEASE_SEC",
    "DEFAULT_MID_BAND_HI",
    "DEFAULT_MID_BAND_LO",
    "DEFAULT_MID_DECAY_SEC",
    "DEFAULT_PULSE_TAU_FRACTION",
    "DEFAULT_SNARE_BAND_HI",
    "DEFAULT_SNARE_BAND_LO",
    "DEFAULT_SNARE_DECAY_SEC",
    "PulseTrack",
    "apply_pulse_deadzone",
    "beat_pulse_envelope",
    "build_band_pulse_track",
    "build_bass_pulse_track",
    "build_hi_transient_track",
    "build_lo_transient_track",
    "build_logo_bass_pulse_track",
    "build_mid_transient_track",
    "build_rms_impact_pulse_track",
    "build_snare_glow_track",
    "scale_and_opacity_for_pulse",
]
