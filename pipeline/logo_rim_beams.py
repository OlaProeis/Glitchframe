"""
Pre-choreographed rim-light beam emissions on high-impact moments.

The compositor already draws a travelling-wave rim field around the logo
(:mod:`pipeline.logo_rim_lights`); this module sits on top of that and fires
short **straight beams** outward from the logo centroid at musically earned
moments -- typically a classic **pre-drop snare roll + drop** pattern.

Design in one breath
--------------------

1. Scan ``analysis["events"]["drops"]`` once per render. For every drop ``T``
   peak-pick the snare envelope in ``[T - lead_in_window_sec, T)``; take the
   top ``lead_in_max_beams`` snares as "lead-in" beams, then add one thicker
   "drop" beam at ``T`` itself. A drop without a snare lead-in fires just the
   drop beam. Standalone big RMS impacts (drops never detected) get a single
   beam each.
2. Apply a global minimum-interval gate (default ``10 s``) between groups so
   the effect feels *earned*, not spammy.
3. At render time, :func:`compute_beam_patch` returns a padded premultiplied
   RGBA patch around the logo centroid for the currently active beams. The
   compositor blends it with :func:`pipeline.logo_composite._blend_premult_rgba_patch`
   the same way it blends the rim / neon / logo patches.

Colors
~~~~~~

Beam tints come from :func:`pipeline.logo_rim_lights._layer_srgb_tints` so
they automatically track the active rim layers and ``hue_drift_per_sec``.
This keeps beams visually wedded to the rim they seem to spring from.

Determinism
~~~~~~~~~~~

Angles, color-layer cycling, and angle jitter are seeded off a
``song_hash`` + event index, so the same render produces the same beam
choreography bit-for-bit.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from pipeline.beat_pulse import PulseTrack
from pipeline.logo_rim_lights import _layer_srgb_tints

# Internal supersample factor for beam rasterisation: draw + blur at this
# resolution, then box-average down to output pixels. Keeps length/thickness
# (in output px) identical while reducing stair-stepping on diagonals.
_BEAM_ANTIALIAS_SUPERSAMPLE: int = 2


def _downsample_accum_box(
    accum_srgb: np.ndarray,
    alpha_acc: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Box-average float RGB + alpha accumulators from ``factor``× to base size."""
    if factor <= 1:
        return accum_srgb, alpha_acc
    h, w = int(accum_srgb.shape[0]), int(accum_srgb.shape[1])
    fh = h // factor
    fw = w // factor
    if fh <= 0 or fw <= 0:
        return accum_srgb, alpha_acc
    acc_c = accum_srgb[: fh * factor, : fw * factor]
    a_c = alpha_acc[: fh * factor, : fw * factor]
    acc_out = acc_c.reshape(fh, factor, fw, factor, 3).mean(axis=(1, 3))
    a_out = a_c.reshape(fh, factor, fw, factor).mean(axis=(1, 3))
    return acc_out.astype(np.float32, copy=False), a_out.astype(
        np.float32, copy=False
    )


def _hsv_to_srgb_u8(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV in ``[0, 1]`` to an sRGB triple in ``0..255``."""
    r, g, b = colorsys.hsv_to_rgb(float(h) % 1.0, float(s), float(v))
    return (
        int(round(max(0.0, min(1.0, r)) * 255.0)),
        int(round(max(0.0, min(1.0, g)) * 255.0)),
        int(round(max(0.0, min(1.0, b)) * 255.0)),
    )


def _random_beam_tint(rng: np.random.Generator) -> tuple[int, int, int]:
    """Pick a bright, saturated beam tint from ``rng``.

    Full hue randomness, saturation pinned near 1.0 (keeps colours vivid even
    after ``core_white_boost`` mixes white into the axis), value pinned at 1.0
    so every beam reads at peak brightness. Call sites seed ``rng`` from
    ``song_hash`` + a per-beam index so renders stay bit-stable.
    """
    h = float(rng.uniform(0.0, 1.0))
    s = float(rng.uniform(0.82, 1.0))
    return _hsv_to_srgb_u8(h, s, 1.0)


@dataclass(frozen=True, slots=True)
class BeamConfig:
    """Tuning for beam scheduling + rendering.

    Distances are in **output-frame pixels** (same space as the composited
    video frame); times are in **seconds**. Defaults are tuned for a
    1080p render with a centered logo ~30 % of the shorter edge.
    """

    enabled: bool = True
    """Master switch. When ``False``, :func:`schedule_rim_beams` returns
    an empty list and the compositor skips the per-frame draw entirely."""

    min_group_interval_sec: float = 10.0
    """Minimum seconds between the **end of one group** and the **start of
    the next**. Enforced both within drops and between drop-and-impact groups."""

    lead_in_window_sec: float = 1.8
    """Look-back window before each drop for snare pre-hits."""

    lead_in_max_beams: int = 3
    """Maximum pre-drop snare beams per group (plus one drop beam)."""

    lead_in_snare_threshold: float = 0.55
    """Minimum peak strength on the snare pulse track (``[0, 1]``) to count
    as a lead-in beam."""

    lead_in_min_spacing_sec: float = 0.12
    """Minimum spacing between adjacent lead-in peaks (seconds). Prevents a
    single mid-band transient from picking up as 2--3 overlapping beams."""

    standalone_impact_threshold: float = 0.70
    """Minimum peak strength on the RMS impact track for a beam outside a
    drop window (a lone build-up or sudden level jump)."""

    standalone_impact_exclusion_sec: float = 1.0
    """Skip standalone impacts within this window of an already-scheduled drop
    (± this many seconds) -- the drop group already covers them."""

    snare_beam_length_px: float = 420.0
    """Floor on the rendered length of a lead-in snare beam. The actual
    length is ``max(snare_beam_length_px, snare_beam_length_frac_edge *
    distance_to_frame_edge_along_angle)`` so beams always reach the edge
    on reasonable frame sizes but never collapse to zero on tiny thumbs."""

    drop_beam_length_px: float = 680.0
    """Floor on the rendered length of the drop beam. See
    :attr:`drop_beam_length_frac_edge` for the edge-relative stretch."""

    snare_beam_thickness_px: float = 20.0
    """Gaussian sigma across the beam width (snare lead-ins)."""

    drop_beam_thickness_px: float = 32.0
    """Gaussian sigma across the beam width (drop)."""

    snare_beam_length_frac_edge: float = 1.35
    """Lead-in beam length as a fraction of the distance from the logo
    centroid to the frame edge along the beam angle. Values > 1.0 push the
    nominal beam tip past the frame edge so the along-axis falloff (which
    fades to zero at the nominal tip) still reads as a bright core when
    it crosses the actual frame edge."""

    drop_beam_length_frac_edge: float = 1.40
    """Drop beam length as a fraction of the distance to the frame edge.
    The drop is the "hero" ray and should read as a full-frame burst; we
    overshoot slightly so the visible bright core reaches the edge instead
    of fading out before it."""

    duration_sec: float = 0.75
    """Total visible life of a beam -- attack + decay combined. The
    exponential decay leaves roughly a ``0.5 s`` afterglow after the
    attack so the bloom hangs on screen after the hit instead of cutting."""

    attack_sec: float = 0.04
    """Envelope rise time (linear to 1.0)."""

    halo_width_mul: float = 2.75
    """Halo gaussian sigma as a multiple of the beam's core ``thickness_px``.
    The halo sits under the core at lower intensity so the beam feels
    bloomed / afterglowing instead of a hard stroke."""

    halo_intensity: float = 0.50
    """Peak intensity of the halo layer (``[0, 1]``) relative to the core.
    Lower values keep the core crisp; higher values push bloom."""

    core_white_boost: float = 0.55
    """Additional white energy pushed into the brightest core pixels
    (``0``: pure tinted beam; ``1``: core saturates fully to white).
    This makes beams read bright on **any** preset colour -- without it a
    dark shadow tint would render the beam barely visible against a black
    background even at full alpha. The boost decays with ``profile_core``
    so only the hot axis goes white while the halo keeps the tint."""

    edge_offset_frac: float = 0.70
    """Fraction of the supplied ``logo_radius_px`` used as the beam's
    visual starting offset from the centroid. Values < 1 keep the beam
    root just *inside* the logo edge so it clearly springs *from* the
    logo rather than from empty space next to it. Too high and the beam
    loses usable length before reaching the frame edge, so we default to
    a conservative 0.70 (just inside the rim)."""

    angle_jitter_rad: float = 0.45
    """± jitter applied to each beam's base angle (radians, ~26°)."""

    angle_group_spread_rad: float = 2.0 * math.pi / 3.0
    """Base spacing between consecutive beams in the same group (radians).
    Default ~120° so 3 beams spread evenly around the logo."""


@dataclass(frozen=True, slots=True)
class ScheduledBeam:
    """A single beam cue on the render timeline."""

    t_start: float
    """Absolute time in seconds when the beam begins fading in."""

    duration_s: float
    """Full visible duration (attack + decay)."""

    angle_rad: float
    """Outward direction from the centroid (0 = +x, pi/2 = +y = down)."""

    length_px: float
    """Beam length at peak envelope, in output-frame pixels."""

    thickness_px: float
    """Gaussian sigma across the beam width."""

    intensity: float
    """Envelope scale in ``[0, 1]``; 1.0 = full brightness at peak."""

    color_layer_idx: int
    """Index into the rim color tints (wraps with the active layer count)."""

    is_drop: bool
    """``True`` = drop (thicker / longer); ``False`` = snare lead-in."""

    tint_srgb_override: tuple[int, int, int] | None = None
    """Explicit sRGB tint (0--255) that bypasses the rim-layer palette.

    ``None`` (the default) keeps the analyser-scheduled behaviour: colour comes
    from :func:`pipeline.logo_rim_lights._layer_srgb_tints` indexed by
    :attr:`color_layer_idx`. User-placed ``BEAM`` clips whose settings carry a
    ``color_hex`` use this field so the picker in the effects editor actually
    drives the rendered colour instead of being silently discarded.
    """

    sustain_shaping: bool = False
    """When ``True`` (user timeline BEAM longer than :attr:`BeamConfig.duration_sec`),
    strength follows a ramp → plateau → hold, then **core and halo** both ramp
    up together (with a modest extra halo swell) through the rest of the clip,
    then a longer fade-out. Auto-scheduled beams and short floored user ticks
    use the default attack–decay envelope."""


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def _drops_sorted(analysis: Mapping[str, Any]) -> list[float]:
    """Return finite, sorted drop times (``analysis["events"]["drops"]``)."""
    events = analysis.get("events") if isinstance(analysis, Mapping) else None
    if not isinstance(events, Mapping):
        return []
    raw = events.get("drops") or []
    out: list[float] = []
    for item in raw:
        if isinstance(item, Mapping):
            t = item.get("t")
        else:
            t = item
        try:
            tv = float(t)
        except (TypeError, ValueError):
            continue
        if math.isfinite(tv) and tv >= 0.0:
            out.append(tv)
    out.sort()
    return out


def _peak_pick_track(
    track: PulseTrack,
    *,
    t_lo: float,
    t_hi: float,
    threshold: float,
    min_spacing_sec: float,
) -> list[tuple[float, float]]:
    """Local maxima in ``track.values`` inside ``[t_lo, t_hi)`` above threshold.

    Returns ``[(t, strength), ...]`` sorted descending by strength. Spacing is
    measured on the sample grid so the caller can trust the peaks are at least
    ``min_spacing_sec`` apart.
    """
    fps = float(track.fps)
    if fps <= 0.0 or track.values.size == 0:
        return []
    n = int(track.values.shape[0])
    i_lo = max(0, int(math.floor(t_lo * fps)))
    i_hi = min(n, int(math.ceil(t_hi * fps)))
    if i_hi - i_lo < 3:
        return []
    segment = np.asarray(track.values[i_lo:i_hi], dtype=np.float32)
    # Interior-only local maxima (strictly greater than neighbours) so a flat
    # plateau doesn't yield one "peak" per frame.
    interior = segment[1:-1]
    left = segment[:-2]
    right = segment[2:]
    is_peak = (interior > left) & (interior >= right) & (interior >= threshold)
    peak_offsets = np.nonzero(is_peak)[0]
    if peak_offsets.size == 0:
        return []
    min_gap = max(1, int(round(min_spacing_sec * fps)))
    # Greedy non-max suppression: sort by strength desc, accept peaks at least
    # min_gap apart from every already-accepted peak.
    order = np.argsort(-interior[peak_offsets])
    accepted: list[int] = []
    for idx in order:
        offset = int(peak_offsets[idx])
        if all(abs(offset - a) >= min_gap for a in accepted):
            accepted.append(offset)
    return [
        ((i_lo + 1 + off) / fps, float(interior[off]))
        for off in accepted
    ]


def _rng_seed(song_hash: str | None, salt: int) -> int:
    """Deterministic 32-bit seed from ``song_hash`` + an event index."""
    base = (song_hash or "").encode("utf-8", errors="replace")
    digest = hashlib.sha256(base + f"::beam::{int(salt)}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "little", signed=False) & 0x7FFFFFFF


def _color_rng(song_hash: str | None, salt: int) -> np.random.Generator:
    """Deterministic per-beam colour RNG, seeded independently of the angle
    RNG so adding / removing hue picks can't drift the angle sequence
    (bit-stability tests compare ``angle_rad`` for a fixed ``song_hash``)."""
    base = (song_hash or "").encode("utf-8", errors="replace")
    digest = hashlib.sha256(
        base + f"::beam::color::{int(salt)}".encode("ascii")
    ).digest()
    seed = int.from_bytes(digest[:4], "little", signed=False) & 0x7FFFFFFF
    return np.random.default_rng(seed)


def _beam_group_for_drop(
    drop_time: float,
    drop_index: int,
    *,
    snare_track: PulseTrack | None,
    cfg: BeamConfig,
    song_hash: str | None,
    n_color_layers: int,
) -> list[ScheduledBeam]:
    """Schedule 1--(max+1) beams for a single drop."""
    lead_ins: list[tuple[float, float]] = []
    if snare_track is not None and cfg.lead_in_max_beams > 0:
        t_lo = max(0.0, drop_time - float(cfg.lead_in_window_sec))
        peaks = _peak_pick_track(
            snare_track,
            t_lo=t_lo,
            t_hi=drop_time,
            threshold=float(cfg.lead_in_snare_threshold),
            min_spacing_sec=float(cfg.lead_in_min_spacing_sec),
        )
        peaks.sort(key=lambda p: -p[1])
        lead_ins = peaks[: int(cfg.lead_in_max_beams)]
        lead_ins.sort(key=lambda p: p[0])

    rng = np.random.default_rng(_rng_seed(song_hash, drop_index))
    base_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    spread = float(cfg.angle_group_spread_rad)
    jitter = float(cfg.angle_jitter_rad)
    n_layers = max(1, int(n_color_layers))
    # Separate colour RNG per group so each beam in the group gets a distinct
    # hue. Seeded from the same drop index the angle RNG uses, but on a
    # different domain (``::color::``) so it never drifts the angle sequence.
    crng = _color_rng(song_hash, drop_index)

    beams: list[ScheduledBeam] = []
    for i, (t_peak, strength) in enumerate(lead_ins):
        ang = base_angle + i * spread + float(rng.uniform(-jitter, jitter))
        beams.append(
            ScheduledBeam(
                t_start=float(t_peak),
                duration_s=float(cfg.duration_sec),
                angle_rad=float(ang % (2.0 * math.pi)),
                length_px=float(cfg.snare_beam_length_px),
                thickness_px=float(cfg.snare_beam_thickness_px),
                intensity=float(min(1.0, max(0.35, strength))),
                color_layer_idx=i % n_layers,
                is_drop=False,
                tint_srgb_override=_random_beam_tint(crng),
            )
        )
    # Drop beam: biased angle so the drop doesn't overlap the last lead-in.
    drop_ang = base_angle + len(lead_ins) * spread + float(rng.uniform(-jitter, jitter))
    beams.append(
        ScheduledBeam(
            t_start=float(drop_time),
            duration_s=float(cfg.duration_sec) * 1.25,
            angle_rad=float(drop_ang % (2.0 * math.pi)),
            length_px=float(cfg.drop_beam_length_px),
            thickness_px=float(cfg.drop_beam_thickness_px),
            intensity=1.0,
            color_layer_idx=len(lead_ins) % n_layers,
            is_drop=True,
            tint_srgb_override=_random_beam_tint(crng),
        )
    )
    return beams


def _standalone_impact_group(
    impact_time: float,
    impact_index: int,
    strength: float,
    *,
    cfg: BeamConfig,
    song_hash: str | None,
    n_color_layers: int,
) -> list[ScheduledBeam]:
    """Schedule one beam for a standalone RMS impact (no drop nearby)."""
    rng = np.random.default_rng(_rng_seed(song_hash, 100_000 + impact_index))
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    n_layers = max(1, int(n_color_layers))
    crng = _color_rng(song_hash, 100_000 + impact_index)
    return [
        ScheduledBeam(
            t_start=float(impact_time),
            duration_s=float(cfg.duration_sec) * 1.15,
            angle_rad=ang,
            length_px=float(cfg.drop_beam_length_px) * 0.9,
            thickness_px=float(cfg.drop_beam_thickness_px) * 0.9,
            intensity=float(min(1.0, max(0.6, strength))),
            color_layer_idx=impact_index % n_layers,
            is_drop=True,
            tint_srgb_override=_random_beam_tint(crng),
        )
    ]


def _group_bounds(group: Sequence[ScheduledBeam]) -> tuple[float, float]:
    """Return ``(t_first_start, t_last_end)`` for a group."""
    t_start = min(b.t_start for b in group)
    t_end = max(b.t_start + b.duration_s for b in group)
    return float(t_start), float(t_end)


def _apply_group_interval_gate(
    groups: list[list[ScheduledBeam]],
    *,
    min_interval_sec: float,
) -> list[list[ScheduledBeam]]:
    """Drop groups whose start is < ``min_interval_sec`` after previous group end."""
    if not groups:
        return []
    groups_sorted = sorted(groups, key=lambda g: _group_bounds(g)[0])
    accepted: list[list[ScheduledBeam]] = []
    last_end = -math.inf
    gate = float(min_interval_sec)
    for g in groups_sorted:
        start, end = _group_bounds(g)
        if start - last_end < gate:
            continue
        accepted.append(g)
        last_end = end
    return accepted


def schedule_rim_beams(
    analysis: Mapping[str, Any],
    *,
    snare_track: PulseTrack | None,
    impact_track: PulseTrack | None,
    cfg: BeamConfig = BeamConfig(),
    song_hash: str | None = None,
    n_color_layers: int = 2,
) -> list[ScheduledBeam]:
    """Return a flat, time-sorted beam schedule for one render.

    ``snare_track`` is used to find pre-drop lead-in peaks and can be ``None``
    (drops then fire a single beam each). ``impact_track`` triggers extra
    beams for large standalone RMS jumps outside any drop window.

    The schedule is deterministic given the same inputs and ``song_hash``.
    """
    if not cfg.enabled:
        return []

    drops = _drops_sorted(analysis)
    drop_groups: list[list[ScheduledBeam]] = []
    for i, t_drop in enumerate(drops):
        drop_groups.append(
            _beam_group_for_drop(
                t_drop,
                i,
                snare_track=snare_track,
                cfg=cfg,
                song_hash=song_hash,
                n_color_layers=n_color_layers,
            )
        )

    impact_groups: list[list[ScheduledBeam]] = []
    if impact_track is not None:
        # Peak-pick across the whole track, then discard any peak too close to
        # an existing drop (drop group already covers it).
        duration = float(impact_track.values.shape[0]) / float(impact_track.fps) \
            if impact_track.fps > 0.0 else 0.0
        if duration > 0.0:
            peaks = _peak_pick_track(
                impact_track,
                t_lo=0.0,
                t_hi=duration,
                threshold=float(cfg.standalone_impact_threshold),
                min_spacing_sec=max(0.25, float(cfg.min_group_interval_sec) * 0.5),
            )
            excl = float(cfg.standalone_impact_exclusion_sec)
            for j, (t_peak, strength) in enumerate(peaks):
                if any(abs(t_peak - d) <= excl for d in drops):
                    continue
                impact_groups.append(
                    _standalone_impact_group(
                        t_peak,
                        j,
                        strength,
                        cfg=cfg,
                        song_hash=song_hash,
                        n_color_layers=n_color_layers,
                    )
                )

    all_groups = drop_groups + impact_groups
    kept = _apply_group_interval_gate(
        all_groups, min_interval_sec=float(cfg.min_group_interval_sec)
    )
    flat: list[ScheduledBeam] = [b for g in kept for b in g]
    flat.sort(key=lambda b: b.t_start)
    return flat


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------


def _distance_to_frame_edge(
    cx: float, cy: float, angle_rad: float, frame_hw: tuple[int, int]
) -> float:
    """Distance from ``(cx, cy)`` to the frame edge along ``angle_rad``.

    Treats the frame as ``[0, w - 1] x [0, h - 1]``; returns ``0.0`` when
    the centroid is already outside the frame or the angle would never
    reach an edge (degenerate). Used to stretch beams so they read as
    full-frame rays instead of short nubs near the logo rim.
    """
    h, w = int(frame_hw[0]), int(frame_hw[1])
    if h <= 1 or w <= 1:
        return 0.0
    x_max = float(w - 1)
    y_max = float(h - 1)
    if cx < 0.0 or cy < 0.0 or cx > x_max or cy > y_max:
        return 0.0
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    ts: list[float] = []
    if cos_a > 1e-9:
        ts.append((x_max - cx) / cos_a)
    elif cos_a < -1e-9:
        ts.append((0.0 - cx) / cos_a)
    if sin_a > 1e-9:
        ts.append((y_max - cy) / sin_a)
    elif sin_a < -1e-9:
        ts.append((0.0 - cy) / sin_a)
    ts = [t for t in ts if t > 0.0]
    return float(min(ts)) if ts else 0.0


def _stretch_beam_to_edge(
    beam: ScheduledBeam,
    *,
    cx: float,
    cy: float,
    frame_hw: tuple[int, int],
    cfg: BeamConfig,
) -> ScheduledBeam:
    """Return ``beam`` with ``length_px`` stretched to the edge-frac target.

    Uses :attr:`BeamConfig.drop_beam_length_frac_edge` for drop beams and
    :attr:`BeamConfig.snare_beam_length_frac_edge` for lead-ins. The
    rendered length is ``max(length_px, frac * distance_to_edge)`` so
    short-frame renders still honour the px floor.
    """
    frac = float(
        cfg.drop_beam_length_frac_edge if beam.is_drop
        else cfg.snare_beam_length_frac_edge
    )
    if frac <= 0.0:
        return beam
    edge_d = _distance_to_frame_edge(cx, cy, beam.angle_rad, frame_hw)
    target = frac * edge_d
    if target <= float(beam.length_px):
        return beam
    return replace(beam, length_px=float(target))


def _beam_envelope(age: float, duration: float, attack: float) -> float:
    """Fast linear attack, exponential decay; returns ``[0, 1]``."""
    if age < 0.0 or age > duration:
        return 0.0
    a = max(1e-4, float(attack))
    if age < a:
        return float(age / a)
    # Decay so env(duration) ≈ 0.05 of peak.
    tau = max(1e-3, (duration - a) / math.log(20.0))
    return float(math.exp(-(age - a) / tau))


def _sustain_knots(D: float) -> tuple[float, float, float, float]:
    """Knots for a long user BEAM: ``(ramp, plateau, fade, t_plateau_end)``.

    After ``t_plateau_end = ramp + plateau``, the beam **core** and halo both
    ramp up (see :func:`_sustain_energy_mul` and :func:`_sustain_halo_scales`)
    until a slightly elongated end fade on long clips.
    """
    D = max(1e-4, float(D))
    # Slightly longer tail than before so a long cue "hangs" before release.
    fade = min(0.18, max(0.035, 0.065 * D))
    ramp = min(0.45, max(0.04, 0.12 * D))
    plateau = min(0.45, max(0.03, 0.10 * D))
    t2 = ramp + plateau
    if t2 > D * 0.55:
        scale = (D * 0.50) / t2
        ramp *= scale
        plateau *= scale
        t2 = ramp + plateau
    return ramp, plateau, fade, t2


def _sustain_strength(age: float, D: float) -> float:
    """Ramp + plateau (full) + optional end fade; values in ``[0, 1]``."""
    if age < 0.0 or age > D or D <= 0.0:
        return 0.0
    ramp, _plateau, fade, t2 = _sustain_knots(D)
    d_end = D - fade
    if age >= d_end:
        return max(0.0, 1.0 - (age - d_end) / max(fade, 1e-4))
    if age < ramp:
        return float(age / max(ramp, 1e-4))
    if age < t2:
        return 1.0
    return 1.0


def _sustain_glow_u(age: float, D: float) -> float:
    """0 during ramp+plateau, 1 by end of glow window (before strength fade)."""
    if age < 0.0 or age > D or D <= 0.0:
        return 0.0
    ramp, _plateau, fade, t2 = _sustain_knots(D)
    t_glow_end = D - fade
    if age < t2:
        return 0.0
    if age >= t_glow_end:
        return 1.0
    return (age - t2) / max(t_glow_end - t2, 1e-4)


def _sustain_energy_mul(age: float, D: float) -> float:
    """Extra scale on :func:`_sustain_strength` for long clips (``>= 1.0``).

    After the strength plateau, the **core and halo** both get louder together
    as ``g`` sweeps 0→1, so the tip at the screen edge and the hot axis
    brighten — not just the afterglo halo. Clamped to keep gains sane.
    """
    if age < 0.0 or age > D or D <= 0.0:
        return 1.0
    g = _sustain_glow_u(age, D)
    if g <= 0.0:
        return 1.0
    extra = max(0.0, float(D) - 0.75)
    # Bigger D → a bit more peak punch on very long hero beams.
    boost = 0.12 + 0.26 * min(1.0, extra / 3.25)
    return min(1.0 + g * float(boost), 1.40)


def _sustain_halo_scales(age: float, D: float) -> tuple[float, float, float]:
    """Extra halo width, halo intensity, and blur multipliers (base = 1.0)."""
    g = _sustain_glow_u(age, D)
    extra = max(0.0, float(D) - 0.75)
    # Modest halo swell on top of :func:`_sustain_energy_mul` (global brightening).
    w = 1.0 + g * (0.28 + 0.45 * min(1.0, extra / 4.0))
    hi = 1.0 + g * 0.38 * min(1.0, extra / 3.5)
    blur = 1.0 + 0.14 * min(2.0, extra) * (0.2 + 0.8 * g)
    # Cap very long BEAMs so a single cue does not smother the whole frame.
    w = min(w, 1.45)
    hi = min(hi, 1.25)
    blur = min(blur, 1.30)
    return w, hi, blur


def _active_beam_states(
    t: float, scheduled: Sequence[ScheduledBeam]
) -> list[
    tuple[ScheduledBeam, float, float, float, float]
]:
    """Return ``(beam, env, halo_w_mul, halo_i_mul, blur_mul)`` for active beams.

    ``halo_*`` and ``blur`` are multipliers on top of :class:`BeamConfig` defaults
    (``1.0`` = no change; used for long user-timeline sustain shaping).
    """
    out: list[tuple[ScheduledBeam, float, float, float, float]] = []
    tf = float(t)
    for b in scheduled:
        age = tf - float(b.t_start)
        D = float(b.duration_s)
        if age < 0.0 or age > D:
            continue
        if b.sustain_shaping:
            s = _sustain_strength(age, D)
            emul = _sustain_energy_mul(age, D)
            env = s * float(b.intensity) * emul
            if env <= 1e-4:
                continue
            hw, hi, br = _sustain_halo_scales(age, D)
            out.append((b, env, hw, hi, br))
        else:
            env = _beam_envelope(age, D, 0.04) * float(b.intensity)
            if env > 1e-4:
                out.append((b, env, 1.0, 1.0, 1.0))
    return out


def _gaussian_blur_footprint_pad_px(
    blur_sigma_base: float, *, max_br: float
) -> int:
    """Extra outer pad so per-beam ``ndimage.gaussian_filter`` does not clip.

    The beam is rasterised, then each layer is blurred with
    ``sigma = blur_sigma_base * br``. The 2D blur tail must decay to
    ~negligible *inside* the sub-patch before its bottom/right edges, or
    a straight cut (often a horizontal strip) appears in the mid-frame
    when the bloom is wide. 6.5--7σ+ margin is a practical minimum for
    high-``br`` / long-sustain looks.
    """
    sigma = max(0.0, float(blur_sigma_base)) * max(1.0, float(max_br))
    return max(1, int(math.ceil(6.6 * sigma + 1.0)))


def _patch_bounds(
    scheduled_active: Sequence[
        tuple[ScheduledBeam, float, float, float, float]
    ],
    *,
    centroid_xy: tuple[float, float],
    frame_hw: tuple[int, int],
    base_halo_width_mul: float,
    pad_px: int = 8,
    blur_pad_px: int = 0,
) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x0, y0, x1, y1)`` in frame coords covering all beams.

    The margin honours the widest halo sigma so the unblurred field does not
    clip, and ``blur_pad_px`` extends the box for the post-pass Gaussian
    (see :func:`_gaussian_blur_footprint_pad_px`). Returns ``None`` when no
    beam is active or the bbox falls outside the frame.
    """
    if not scheduled_active:
        return None
    h, w = int(frame_hw[0]), int(frame_hw[1])
    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    xs: list[float] = [cx]
    ys: list[float] = [cy]
    base_hm = max(1.0, float(base_halo_width_mul))
    for beam, _env, hw, _hi, _br in scheduled_active:
        ang = float(beam.angle_rad)
        L = float(beam.length_px)
        # Quad envelope plus a halo-sigma margin so bloom doesn't clip.
        eff_halo = base_hm * max(1.0, float(hw))
        margin = 3.5 * float(beam.thickness_px) * eff_halo
        tip_x = cx + math.cos(ang) * L
        tip_y = cy + math.sin(ang) * L
        xs.extend([tip_x - margin, tip_x + margin])
        ys.extend([tip_y - margin, tip_y + margin])
        xs.extend([cx - margin, cx + margin])
        ys.extend([cy - margin, cy + margin])
    bpad = int(max(0, int(blur_pad_px)))
    x0 = max(0, int(math.floor(min(xs))) - int(pad_px) - bpad)
    y0 = max(0, int(math.floor(min(ys))) - int(pad_px) - bpad)
    x1 = min(w, int(math.ceil(max(xs))) + int(pad_px) + bpad)
    y1 = min(h, int(math.ceil(max(ys))) + int(pad_px) + bpad)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _draw_beam_scalar_fields(
    h: int,
    w: int,
    *,
    beam: ScheduledBeam,
    env: float,
    cx_local: float,
    cy_local: float,
    edge_offset_px: float,
    halo_width_mul: float,
    halo_intensity: float,
    core_white_boost: float = 0.0,
    sustain_axial_white: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Compute the three **scalar** intensity fields one beam needs.

    Returns ``(c_core, c_halo, white)`` as ``(h, w)`` ``float32`` arrays
    (``white`` is ``None`` when the white-hot path is inactive). The caller
    blurs each scalar **once** and then expands to RGB via ``c * tint``,
    which saves five of the eight per-beam ``gaussian_filter`` calls the
    previous implementation paid (one blur per RGB channel × core/halo plus
    one alpha blur each). This optimisation is exact under linearity of
    convolution: ``gaussian_filter(scalar * tint) == tint *
    gaussian_filter(scalar)`` -- only floating-point rounding can introduce
    sub-LSB drift in the final ``uint8`` output. At 1080p × 2× supersample
    each saved blur is ~250 ms on CPU, so the cliff in
    :func:`compute_beam_patch` (was ~2.5 s/beam, now ~1 s/beam) is the
    dominant per-beam cost.

    Returns ``None`` when the beam contributes nothing at this frame (zero
    gain or all-zero profile) so the caller can skip every downstream stage
    for this beam, matching the previous in-place predecessor's early
    returns.

    The geometric reasoning -- core supergaussian for a *laser* core, soft
    halo with a slow axial rise so the bright spot peaks past the rim, the
    forward gate that hides the centroid attach (line 1.2 px under the
    mark), and the white-hot axial-along boost on sustain beams -- is
    unchanged from ``_draw_beam_into``.
    """
    if h == 0 or w == 0:
        return None
    ang = float(beam.angle_rad)
    L = max(1.0, float(beam.length_px))
    sigma = max(0.75, float(beam.thickness_px))
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)

    ys, xs = np.indices((h, w), dtype=np.float32)
    dx = xs - float(cx_local)
    dy = ys - float(cy_local)
    u_raw = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    offset = max(0.0, float(edge_offset_px))
    u = u_raw - offset
    L_eff = max(1.0, L - offset)

    # Along-axis: core and halo share the same length tail; core stays tight to u>=0.
    rise_core = np.clip(u / max(1.0, 0.08 * L_eff), 0.0, 1.0)
    tail = np.clip(1.0 - (u / L_eff), 0.0, 1.0)
    axial_core = rise_core * (tail ** 0.85)
    axial_core = np.where(u >= 0.0, axial_core, 0.0)

    # Flatter, straighter *laser* cross-section (p > 2 → narrower bright core).
    p_core = 3.0
    radial_core = np.exp(
        -0.5 * (np.abs(v / sigma) ** p_core)
    )

    # Halo: wider gaussian, **slow** rise along the ray so the rim is not the
    # brightest part of the bloom (peaks past ~0.1·L from the edge).
    sigma_halo = max(sigma + 1.0, sigma * float(halo_width_mul))
    radial_halo = np.exp(-0.5 * (v / sigma_halo) ** 2)
    halo_rise = 0.13 * L_eff
    rise_halo = (np.clip(u / max(1.0, halo_rise), 0.0, 1.0) ** 1.12)
    # Soft forward gate: removes the harshest half-plane for the diffuse term only
    # (1–1.2 px in ``u`` under the logo, occluded by the glyph).
    edge_w = 1.2
    halo_u_gate = (np.clip((u + 0.2 * edge_w) / edge_w, 0.0, 1.0) ** 1.05) * (u > -0.3)
    axial_halo = rise_halo * (tail ** 0.62) * halo_u_gate
    halo_gain = max(0.0, min(1.0, float(halo_intensity)))

    profile_core = (axial_core * radial_core).astype(np.float32, copy=False)
    profile_halo = (axial_halo * radial_halo * halo_gain).astype(np.float32, copy=False)
    if not np.any((profile_core + profile_halo) > 1e-5):
        return None

    gain = float(env)
    if gain <= 0.0:
        return None

    c_core = (profile_core * gain).astype(np.float32, copy=False)
    c_halo = (profile_halo * gain).astype(np.float32, copy=False)
    # White-hot core: default ``core_white_boost``; on long sustains, ``saw``
    # also lifts white **along the ray** (``u / L``), not at the attach point.
    wb = max(0.0, float(core_white_boost))
    saw = max(0.0, float(sustain_axial_white))
    base_wb = max(wb, saw * 0.48)
    white: np.ndarray | None = None
    if base_wb > 1e-5 or saw > 1e-4:
        u_norm = np.clip(u / L_eff, 0.0, 1.0)
        w_axial = base_wb * (1.0 + saw * (u_norm ** 1.35))
        white = (
            profile_core * gain * w_axial * np.where(u > 0.0, 1.0, 0.0)
        ).astype(np.float32, copy=False)
    return c_core, c_halo, white


def _finalize_patch(
    accum_srgb: np.ndarray, alpha_accum: np.ndarray
) -> np.ndarray:
    """Convert (straight-sRGB RGB + alpha) accumulators to premultiplied uint8."""
    a = np.clip(alpha_accum, 0.0, 1.0).astype(np.float32, copy=False)
    rgb = np.clip(accum_srgb, 0.0, 1.0)
    # Clamp each channel to ``alpha`` so the premultiplied invariant holds even
    # if the gaussian blur pushed colour slightly ahead of alpha in 1--2 pixels.
    rgb[..., 0] = np.minimum(rgb[..., 0], a)
    rgb[..., 1] = np.minimum(rgb[..., 1], a)
    rgb[..., 2] = np.minimum(rgb[..., 2], a)
    r_u8 = np.clip(rgb[..., 0] * 255.0, 0.0, 255.0).astype(np.uint8)
    g_u8 = np.clip(rgb[..., 1] * 255.0, 0.0, 255.0).astype(np.uint8)
    b_u8 = np.clip(rgb[..., 2] * 255.0, 0.0, 255.0).astype(np.uint8)
    a_u8 = np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([r_u8, g_u8, b_u8, a_u8], axis=-1)


@dataclass(frozen=True, slots=True)
class BeamPatchResult:
    """Padded premultiplied RGBA patch ready for ``_blend_premult_rgba_patch``."""

    patch: np.ndarray
    """``(h, w, 4)`` uint8 premultiplied RGBA; ``rgb <= alpha`` per-pixel."""

    x0: int
    """Top-left x in output-frame coordinates."""

    y0: int
    """Top-left y in output-frame coordinates."""


def compute_beam_patch(
    frame_hw: tuple[int, int],
    *,
    centroid_xy: tuple[float, float],
    t: float,
    scheduled: Sequence[ScheduledBeam],
    rim_rgb: tuple[int, int, int],
    cfg: BeamConfig = BeamConfig(),
    color_spread_rad: float = 2.0 * math.pi / 3.0,
    song_hash: Any = None,
    hue_drift_per_sec: float = 0.0,
    n_color_layers: int = 1,
    blur_sigma_px: float = 2.6,
    logo_radius_px: float = 0.0,
) -> BeamPatchResult | None:
    """Build a padded premultiplied-RGBA patch for currently active beams.

    Returns ``None`` when no beam is active at ``t`` -- the compositor can
    then skip all per-frame work. Blend the returned patch at
    ``(result.x0, result.y0)`` via
    :func:`pipeline.logo_composite._blend_premult_rgba_patch`.

    ``logo_radius_px`` is the approximate distance from the centroid to the
    logo edge (output-frame px). Beams start at ``logo_radius_px *
    cfg.edge_offset_frac`` outward along their angle, so they visually spring
    from the rim rather than piercing the logo's middle. Pass ``0`` (the
    default) to restore the centroid-rooted legacy behaviour.
    """
    active_raw = _active_beam_states(t, scheduled)
    if not active_raw:
        return None

    cx_frame = float(centroid_xy[0])
    cy_frame = float(centroid_xy[1])
    # Stretch each active beam to ``length_frac_edge * distance_to_edge`` so
    # beams read as proper full-frame rays instead of stubs on the rim. Done
    # per-frame because the centroid moves with the logo pulse.
    active = [
        (
            _stretch_beam_to_edge(
                b, cx=cx_frame, cy=cy_frame, frame_hw=frame_hw, cfg=cfg,
            ),
            env,
            hw,
            hi,
            br,
        )
        for b, env, hw, hi, br in active_raw
    ]

    H, W = int(frame_hw[0]), int(frame_hw[1])
    if H <= 0 or W <= 0:
        return None
    ss = max(1, int(_BEAM_ANTIALIAS_SUPERSAMPLE))
    # Always raster into a **full-size** off-screen buffer, then place at
    # ``(0, 0)``. Tight AABBs + ``gaussian_filter(..., cval=0)`` at any inner
    # patch edge were still producing a straight "shelf" for wide halos. Full
    # frame is O(frame) but is the only reliable fix for the inner clip; the
    # *video* border still tapers from the same constant edge mode, which
    # matches how other full-frame effects behave.
    x0, y0 = 0, 0
    ph, pw = H * ss, W * ss

    accum_srgb = np.zeros((ph, pw, 3), dtype=np.float32)
    alpha_acc = np.zeros((ph, pw), dtype=np.float32)

    # Sample active tints once per call (same time for all beams).
    n_layers = max(1, int(n_color_layers))
    tints = _layer_srgb_tints(
        n_layers,
        (int(rim_rgb[0]), int(rim_rgb[1]), int(rim_rgb[2])),
        color_spread_rad=float(color_spread_rad),
        t=float(t),
        song_hash=song_hash,
        hue_drift_per_sec=float(hue_drift_per_sec),
    )

    cx_local = (float(centroid_xy[0]) - float(x0)) * float(ss)
    cy_local = (float(centroid_xy[1]) - float(y0)) * float(ss)
    edge_offset = (
        max(0.0, float(logo_radius_px) * float(cfg.edge_offset_frac)) * float(ss)
    )

    for beam, env, hw, hi, br in active:
        # User BEAM clips in the effects editor may pin an explicit sRGB tint
        # via ``color_hex``; when set we bypass the layer palette so the
        # picked colour lands in the rendered beam. Otherwise fall back to the
        # preset-driven rim tints so analyser-scheduled beams keep their hue
        # drift.
        if beam.tint_srgb_override is not None:
            tint = (
                int(beam.tint_srgb_override[0]),
                int(beam.tint_srgb_override[1]),
                int(beam.tint_srgb_override[2]),
            )
        else:
            tint = tints[int(beam.color_layer_idx) % n_layers]
        sustain_ax = 0.0
        if bool(beam.sustain_shaping):
            age_b = t - float(beam.t_start)
            sustain_ax = 1.12 * _sustain_glow_u(age_b, float(beam.duration_s))
        beam_hi = replace(
            beam,
            length_px=float(beam.length_px) * float(ss),
            thickness_px=float(beam.thickness_px) * float(ss),
        )
        # Compute the three scalar intensity fields for this beam (no RGB
        # scratch, no separate alpha allocations). ``c_core`` and
        # ``c_halo`` *are* the alpha fields by construction (since the
        # tint is applied later as a multiply, and both alpha and RGB
        # come from the same scalar).
        fields = _draw_beam_scalar_fields(
            ph,
            pw,
            beam=beam_hi,
            env=env,
            cx_local=cx_local,
            cy_local=cy_local,
            edge_offset_px=edge_offset,
            halo_width_mul=float(cfg.halo_width_mul) * float(hw),
            halo_intensity=max(
                0.0,
                min(1.0, float(cfg.halo_intensity) * float(hi)),
            ),
            core_white_boost=float(cfg.core_white_boost),
            sustain_axial_white=sustain_ax,
        )
        if fields is None:
            continue
        c_core, c_halo, white = fields
        # Per-beam, asymmetric blur: tight core (laser), wider halo. Three
        # blurs total instead of eight (was: 3 RGB + 1 alpha per band × 2
        # bands). Output is identical up to one float32 ULP because of
        # convolution linearity -- see :func:`_draw_beam_scalar_fields`.
        # Sigmas are in **raster** pixels; scale with ``ss`` so blur
        # footprint matches the pre-downsample grid (output-sized
        # ``blur_sigma_px`` unchanged).
        eff_b = float(blur_sigma_px) * max(1.0, float(br)) * float(ss)
        sig_c = max(0.0, eff_b * 0.36)
        sig_h = max(0.0, eff_b * 1.0)
        if sig_c > 0.1:
            c_core = ndimage.gaussian_filter(
                c_core, sigma=sig_c, mode="constant", cval=0.0
            )
            if white is not None:
                white = ndimage.gaussian_filter(
                    white, sigma=sig_c, mode="constant", cval=0.0
                )
        if sig_h > 0.1:
            c_halo = ndimage.gaussian_filter(
                c_halo, sigma=sig_h, mode="constant", cval=0.0
            )
        # Alpha combine: c_core / c_halo *are* the per-band alpha by
        # construction (RGB == c * tint, so alpha == c).
        a_c = np.clip(c_core, 0.0, 1.0)
        a_h = np.clip(c_halo, 0.0, 1.0)
        a_comb = 1.0 - (1.0 - a_c) * (1.0 - a_h)
        # Tint expansion: blend the combined scalar into accum_srgb per
        # channel. ``c_combined = c_core + c_halo`` lets us compute the
        # tint multiply once instead of twice (RGB add was a separate
        # full-frame allocation in the legacy path).
        c_combined = c_core + c_halo
        r_t = float(tint[0]) / 255.0
        g_t = float(tint[1]) / 255.0
        b_t = float(tint[2]) / 255.0
        accum_srgb[..., 0] += c_combined * r_t
        accum_srgb[..., 1] += c_combined * g_t
        accum_srgb[..., 2] += c_combined * b_t
        if white is not None:
            accum_srgb[..., 0] += white
            accum_srgb[..., 1] += white
            accum_srgb[..., 2] += white
        np.maximum(alpha_acc, a_comb, out=alpha_acc)

    if not np.any(alpha_acc > 1e-4):
        return None

    if ss > 1:
        accum_srgb, alpha_acc = _downsample_accum_box(accum_srgb, alpha_acc, ss)
    patch = _finalize_patch(accum_srgb, alpha_acc)
    return BeamPatchResult(patch=patch, x0=int(x0), y0=int(y0))


__all__ = [
    "BeamConfig",
    "BeamPatchResult",
    "ScheduledBeam",
    "compute_beam_patch",
    "schedule_rim_beams",
]
