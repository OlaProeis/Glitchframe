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

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from pipeline.beat_pulse import PulseTrack
from pipeline.logo_rim_lights import _layer_srgb_tints


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

    snare_beam_length_px: float = 320.0
    """Floor on the rendered length of a lead-in snare beam. The actual
    length is ``max(snare_beam_length_px, snare_beam_length_frac_edge *
    distance_to_frame_edge_along_angle)`` so beams always reach the edge
    on reasonable frame sizes but never collapse to zero on tiny thumbs."""

    drop_beam_length_px: float = 520.0
    """Floor on the rendered length of the drop beam. See
    :attr:`drop_beam_length_frac_edge` for the edge-relative stretch."""

    snare_beam_thickness_px: float = 10.0
    """Gaussian sigma across the beam width (snare lead-ins)."""

    drop_beam_thickness_px: float = 16.0
    """Gaussian sigma across the beam width (drop)."""

    snare_beam_length_frac_edge: float = 0.85
    """Lead-in beam length as a fraction of the distance from the logo
    centroid to the frame edge along the beam angle. ``1.0`` reaches the
    edge exactly; default ``0.85`` keeps snare beams visibly shorter than
    the drop beam even when the logo sits near one edge of the frame."""

    drop_beam_length_frac_edge: float = 1.0
    """Drop beam length as a fraction of the distance to the frame edge.
    The drop is the "hero" ray and should read as a full-frame burst."""

    duration_sec: float = 0.32
    """Total visible life of a beam -- attack + decay combined."""

    attack_sec: float = 0.04
    """Envelope rise time (linear to 1.0)."""

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


def _active_beams(
    t: float, scheduled: Sequence[ScheduledBeam]
) -> list[tuple[ScheduledBeam, float]]:
    """Return ``(beam, envelope)`` pairs with ``envelope > 0`` at ``t``."""
    out: list[tuple[ScheduledBeam, float]] = []
    tf = float(t)
    for b in scheduled:
        age = tf - float(b.t_start)
        if age < 0.0 or age > float(b.duration_s):
            continue
        env = _beam_envelope(age, float(b.duration_s), 0.04) * float(b.intensity)
        if env > 1e-4:
            out.append((b, env))
    return out


def _patch_bounds(
    scheduled_active: Sequence[tuple[ScheduledBeam, float]],
    *,
    centroid_xy: tuple[float, float],
    frame_hw: tuple[int, int],
    pad_px: int = 8,
) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x0, y0, x1, y1)`` in frame coords covering all beams.

    Returns ``None`` when no beam is active or the bbox falls outside the frame.
    """
    if not scheduled_active:
        return None
    h, w = int(frame_hw[0]), int(frame_hw[1])
    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    xs: list[float] = [cx]
    ys: list[float] = [cy]
    for beam, _env in scheduled_active:
        ang = float(beam.angle_rad)
        L = float(beam.length_px)
        # Quad envelope plus a bit of sigma margin.
        margin = 3.0 * float(beam.thickness_px)
        tip_x = cx + math.cos(ang) * L
        tip_y = cy + math.sin(ang) * L
        # Bound four corners of the beam quad (tip ± sigma perpendicular, base
        # ± sigma perpendicular) -- simpler: expand by margin on all axes.
        xs.extend([tip_x - margin, tip_x + margin])
        ys.extend([tip_y - margin, tip_y + margin])
        xs.extend([cx - margin, cx + margin])
        ys.extend([cy - margin, cy + margin])
    x0 = max(0, int(math.floor(min(xs))) - int(pad_px))
    y0 = max(0, int(math.floor(min(ys))) - int(pad_px))
    x1 = min(w, int(math.ceil(max(xs))) + int(pad_px))
    y1 = min(h, int(math.ceil(max(ys))) + int(pad_px))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _draw_beam_into(
    accum_linear: np.ndarray,
    alpha_accum: np.ndarray,
    *,
    beam: ScheduledBeam,
    env: float,
    cx_local: float,
    cy_local: float,
    rim_rgb_srgb: tuple[int, int, int],
) -> None:
    """Add one beam's energy into linear-RGB + alpha accumulators (in place).

    The beam is drawn as a rectangular coordinate system: ``u`` along the beam
    axis (``0`` at the logo edge, ``1`` at the tip) and ``v`` across its width
    (``0`` on the axis, ``±`` outward). Along ``u`` we apply a fast rise from
    the centroid edge and a long tail toward the tip; across ``v`` a gaussian
    so the beam has a soft cylindrical look rather than a stamped rectangle.
    """
    h, w = accum_linear.shape[:2]
    if h == 0 or w == 0:
        return
    ang = float(beam.angle_rad)
    L = max(1.0, float(beam.length_px))
    sigma = max(0.75, float(beam.thickness_px))
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)

    ys, xs = np.indices((h, w), dtype=np.float32)
    dx = xs - float(cx_local)
    dy = ys - float(cy_local)
    u = dx * cos_a + dy * sin_a          # along beam axis, px from centroid
    v = -dx * sin_a + dy * cos_a         # perpendicular distance, px

    # Along-axis profile: rises sharply over ~10 % of L from the centroid,
    # plateaus briefly, then fades toward the tip. Negative ``u`` (behind the
    # centroid) is zero so the beam is one-sided.
    rise = np.clip(u / (0.10 * L), 0.0, 1.0)
    tail = np.clip(1.0 - (u / L), 0.0, 1.0)
    axial = rise * (tail ** 1.6)
    axial = np.where(u >= 0.0, axial, 0.0)

    # Across-axis profile: gaussian, cheap to evaluate.
    radial = np.exp(-0.5 * (v / sigma) ** 2)

    # Beam core is brighter and tighter than the soft halo; combine so the
    # edges bloom without the center washing out.
    profile = (axial * radial).astype(np.float32, copy=False)
    if not np.any(profile > 1e-4):
        return

    gain = float(env)
    if gain <= 0.0:
        return

    # Straight sRGB accumulation; the final patch stores ``rgb = alpha * tint``
    # so the premultiplied invariant (``rgb <= alpha``) holds by construction.
    # Working in sRGB keeps the math cheap and matches the downstream blend
    # helper (``_blend_premult_rgba_patch``) which does normalised float math
    # on uint8 bytes without an intermediate linearisation.
    r = float(rim_rgb_srgb[0]) / 255.0
    g = float(rim_rgb_srgb[1]) / 255.0
    b = float(rim_rgb_srgb[2]) / 255.0

    contrib = profile * gain
    accum_linear[..., 0] += contrib * r
    accum_linear[..., 1] += contrib * g
    accum_linear[..., 2] += contrib * b
    np.maximum(alpha_accum, contrib, out=alpha_accum)


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
    blur_sigma_px: float = 1.8,
) -> BeamPatchResult | None:
    """Build a padded premultiplied-RGBA patch for currently active beams.

    Returns ``None`` when no beam is active at ``t`` -- the compositor can
    then skip all per-frame work. Blend the returned patch at
    ``(result.x0, result.y0)`` via
    :func:`pipeline.logo_composite._blend_premult_rgba_patch`.
    """
    active_raw = _active_beams(t, scheduled)
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
        )
        for b, env in active_raw
    ]

    bounds = _patch_bounds(active, centroid_xy=centroid_xy, frame_hw=frame_hw)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    ph, pw = y1 - y0, x1 - x0
    if ph <= 0 or pw <= 0:
        return None

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

    cx_local = float(centroid_xy[0]) - float(x0)
    cy_local = float(centroid_xy[1]) - float(y0)

    for beam, env in active:
        tint = tints[int(beam.color_layer_idx) % n_layers]
        _draw_beam_into(
            accum_srgb,
            alpha_acc,
            beam=beam,
            env=env,
            cx_local=cx_local,
            cy_local=cy_local,
            rim_rgb_srgb=tint,
        )

    if not np.any(alpha_acc > 1e-4):
        return None

    # Light gaussian so the beam edges bloom rather than aliasing along the
    # pixel grid. ``scipy`` is already a project dep (used by the rim code).
    if blur_sigma_px > 0.25:
        for c in range(3):
            accum_srgb[..., c] = ndimage.gaussian_filter(
                accum_srgb[..., c], sigma=float(blur_sigma_px)
            )
        alpha_acc = ndimage.gaussian_filter(alpha_acc, sigma=float(blur_sigma_px))

    patch = _finalize_patch(accum_srgb, alpha_acc)
    return BeamPatchResult(patch=patch, x0=int(x0), y0=int(y0))


__all__ = [
    "BeamConfig",
    "BeamPatchResult",
    "ScheduledBeam",
    "compute_beam_patch",
    "schedule_rim_beams",
]
