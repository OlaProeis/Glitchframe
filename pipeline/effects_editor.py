"""
Backend helpers for a future **effects timeline** editor (Gradio + JS).

Mirrors the responsibilities of :mod:`pipeline.lyrics_editor` (state load, JSON
round-trip, cache-scoped identity) for :class:`EffectsTimeline` / clips under
``cache/<song_hash>/`` — no HTML / UI in this module.

Ghost markers follow the same analyser + scheduling inputs the compositor uses
for rim beams (see :func:`schedule_rim_beams`) plus RMS-impact peaks for
``LOGO_GLITCH`` hints, per-drop ``ZOOM_PUNCH`` suggestions from
``analysis["events"]["drops"]``, low-band transient peaks for ``SCREEN_SHAKE``,
and high-band transient peaks for ``CHROMATIC_ABERRATION`` — see the effects
timeline PRD.
"""

from __future__ import annotations

import html as html_lib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableSequence

from pipeline.audio_ingest import ANALYSIS_MONO_WAV_NAME, ORIGINAL_WAV_NAME
from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.beat_pulse import (
    build_hi_transient_track,
    build_lo_transient_track,
    build_rms_impact_pulse_track,
)
from pipeline.compositor import CompositorConfig, _snare_track_for_logo
from pipeline._waveform_peaks import DEFAULT_PEAK_WIDTH, compute_peaks
from pipeline.effects_timeline import (
    EFFECT_SETTINGS_KEYS,
    SCHEMA_VERSION,
    EffectClip,
    EffectKind,
    EffectsTimeline,
    _auto_enabled_to_json,
    _clip_to_json,
    _timeline_from_dict,
    load,
    save,
    validate_effects_timeline,
)
from pipeline.logo_rim_beams import (
    BeamConfig,
    ScheduledBeam,
    _drops_sorted,
    _peak_pick_track,
    schedule_rim_beams,
)

# Match Task Master: dedupe when an existing clip of the same kind starts within
# this many seconds of a candidate auto-sourced time.
DEDUPE_TOL_S = 0.02

BAKE_GLITCH_DURATION_S = 0.22
BAKE_ZOOM_PUNCH_ON_DROP_S = 0.38
# Low-band peaks (kick / sub) → short screen-shake clips. Duration is long
# enough to read as a camera jolt (~two frames at 30 fps) without bleeding
# into the next kick; peak threshold matches the impact-peak pipeline so only
# prominent low-end transients fire.
BAKE_SHAKE_DURATION_S = 0.18
# High-band peaks (hats / cymbals) → brief chromatic-aberration flashes.
# Shorter than the kick-driven shake because hats cluster much tighter; the
# minimum spacing below keeps the row readable instead of being a wall of
# overlapping clips.
BAKE_CHROMA_DURATION_S = 0.14

_GHOST_IMPACT_PICK_THRESHOLD = 0.42
_GHOST_IMPACT_MIN_SPACING_S = 0.14
_GHOST_KICK_PICK_THRESHOLD = 0.55
_GHOST_KICK_MIN_SPACING_S = 0.18
_GHOST_HAT_PICK_THRESHOLD = 0.62
_GHOST_HAT_MIN_SPACING_S = 0.22


def _resolve_wav_for_peaks(cache: Path) -> Path:
    mono = cache / ANALYSIS_MONO_WAV_NAME
    if mono.is_file():
        return mono
    orig = cache / ORIGINAL_WAV_NAME
    if orig.is_file():
        return orig
    raise FileNotFoundError(
        f"No {ANALYSIS_MONO_WAV_NAME} or {ORIGINAL_WAV_NAME} in {cache} — run ingest first."
    )


def _load_analysis_mapping(cache: Path) -> dict[str, Any]:
    path = cache / ANALYSIS_JSON_NAME
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ANALYSIS_JSON_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        return {}
    return data


def _n_color_layers_for_editor(cfg: CompositorConfig) -> int:
    rrc = cfg.logo_rim_light_config
    if rrc is not None:
        return max(1, int(rrc.rim_color_layers))
    return 2


def _schedule_beams_for_ghosts(
    analysis: Mapping[str, Any],
    song_hash: str,
    cfg: CompositorConfig,
) -> list[ScheduledBeam]:
    sn = _snare_track_for_logo(cfg, analysis)
    it = build_rms_impact_pulse_track(
        analysis, sensitivity=float(cfg.logo_impact_sensitivity)
    )
    bcfg: BeamConfig = cfg.rim_beams_config or BeamConfig()
    return schedule_rim_beams(
        analysis,
        snare_track=sn,
        impact_track=it,
        cfg=bcfg,
        song_hash=song_hash or None,
        n_color_layers=_n_color_layers_for_editor(cfg),
    )


def _peak_pick_whole_track(
    track, threshold: float, min_spacing_sec: float
) -> list[tuple[float, float]]:
    """Run :func:`_peak_pick_track` over the full range of ``track``.

    Shared helper used by the impact / kick / hat auto sources so they all
    apply identical non-max suppression semantics — only the thresholds and
    minimum spacings differ per band.
    """
    if track is None or track.fps <= 0.0 or track.values.size == 0:
        return []
    duration = float(track.values.shape[0]) / float(track.fps)
    return [
        (t, s)
        for t, s in _peak_pick_track(
            track,
            t_lo=0.0,
            t_hi=duration,
            threshold=threshold,
            min_spacing_sec=min_spacing_sec,
        )
    ]


def _impact_glitch_peaks(
    analysis: Mapping[str, Any], cfg: CompositorConfig
) -> list[tuple[float, float]]:
    tr = build_rms_impact_pulse_track(
        analysis, sensitivity=float(cfg.logo_impact_sensitivity)
    )
    return _peak_pick_whole_track(
        tr, _GHOST_IMPACT_PICK_THRESHOLD, _GHOST_IMPACT_MIN_SPACING_S
    )


def _kick_transient_peaks(
    analysis: Mapping[str, Any],
) -> list[tuple[float, float]]:
    """Low-band (kick / sub) transient peaks for SCREEN_SHAKE autopopulation."""
    tr = build_lo_transient_track(analysis)
    return _peak_pick_whole_track(
        tr, _GHOST_KICK_PICK_THRESHOLD, _GHOST_KICK_MIN_SPACING_S
    )


def _hat_transient_peaks(
    analysis: Mapping[str, Any],
) -> list[tuple[float, float]]:
    """High-band (hat / cymbal) transient peaks for CHROMATIC_ABERRATION."""
    tr = build_hi_transient_track(analysis)
    return _peak_pick_whole_track(
        tr, _GHOST_HAT_PICK_THRESHOLD, _GHOST_HAT_MIN_SPACING_S
    )


# --- drop + confidence: reuse _drops_sorted times; add confidence for UI -------
def _drop_rows_for_ghosts(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    ev = analysis.get("events")
    if not isinstance(ev, dict):
        return []
    raw = ev.get("drops")
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                t = float(item.get("t", 0.0))
            except (TypeError, ValueError):
                continue
            c_raw = item.get("confidence", 1.0)
            try:
                c = float(c_raw) if c_raw is not None else 1.0
            except (TypeError, ValueError):
                c = 1.0
        else:
            try:
                t = float(item)
            except (TypeError, ValueError):
                continue
            c = 1.0
        if not math.isfinite(t) or t < 0.0:
            continue
        rows.append(
            {
                "kind": "ZOOM_PUNCH",
                "t": t,
                "source": "drop",
                "confidence": c,
            }
        )
    rows.sort(key=lambda r: (float(r["t"]), r["kind"]))
    return rows


def build_ghost_events(
    analysis: Mapping[str, Any], *, song_hash: str, cfg: CompositorConfig | None = None
) -> list[dict[str, Any]]:
    """Derive auto / analyser hint rows for a future JS timeline (ghost markers)."""
    cfg = cfg or CompositorConfig()
    out: list[dict[str, Any]] = []
    h = str(song_hash) if song_hash else ""

    for b in _schedule_beams_for_ghosts(analysis, h, cfg):
        out.append(
            {
                "kind": "BEAM",
                "t": float(b.t_start),
                "source": "rim_beam",
                "is_drop": bool(b.is_drop),
                "duration_s": float(b.duration_s),
            }
        )
    for t, strength in _impact_glitch_peaks(analysis, cfg):
        out.append(
            {
                "kind": "LOGO_GLITCH",
                "t": float(t),
                "source": "impact_peak",
                "strength": float(strength),
            }
        )
    for t, strength in _kick_transient_peaks(analysis):
        out.append(
            {
                "kind": "SCREEN_SHAKE",
                "t": float(t),
                "source": "kick_peak",
                "strength": float(strength),
            }
        )
    for t, strength in _hat_transient_peaks(analysis):
        out.append(
            {
                "kind": "CHROMATIC_ABERRATION",
                "t": float(t),
                "source": "hat_peak",
                "strength": float(strength),
            }
        )
    out.extend(_drop_rows_for_ghosts(analysis))
    out.sort(
        key=lambda e: (float(e.get("t", 0.0)), str(e.get("kind", "")), e.get("source", ""))
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_editor_state(
    cache_dir: str | Path,
    *,
    target_peak_width: int = DEFAULT_PEAK_WIDTH,
    compositor_config: CompositorConfig | None = None,
) -> dict[str, Any]:
    """
    Assemble data for a future Gradio/JS effects timeline editor.

    Returns JSON-friendly dict keys: ``song_hash``, ``clips``, ``auto_enabled``,
    ``auto_reactivity_master``, ``peaks`` (``[min, max]`` column pairs in ``[-1, 1]``),
    ``duration`` (float seconds), ``sample_rate``, and ``ghost_events``.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")
    song_hash = cache.name

    timeline = load(cache)
    analysis = _load_analysis_mapping(cache)
    wav = _resolve_wav_for_peaks(cache)
    peaks, sample_rate, duration = compute_peaks(wav, target_peak_width)
    peaks_list = [[float(a), float(b)] for a, b in peaks]

    cfg = compositor_config or CompositorConfig()
    ghost_events = build_ghost_events(analysis, song_hash=song_hash, cfg=cfg)

    return {
        "song_hash": song_hash,
        "schema_version": SCHEMA_VERSION,
        "auto_reactivity_master": float(timeline.auto_reactivity_master),
        "auto_enabled": _auto_enabled_to_json(timeline.auto_enabled),
        "clips": [_clip_to_json(c) for c in timeline.clips],
        "peaks": peaks_list,
        "duration": float(duration),
        "sample_rate": int(sample_rate),
        "ghost_events": ghost_events,
    }


def save_edited_timeline(
    cache_dir: str | Path,
    json_payload: str | bytes | Mapping[str, Any],
    song_hash_from_dir: str | None = None,
) -> Path:
    """
    Validate a client ``effects_timeline``-shaped JSON payload and persist
    to ``cache_dir/effects_timeline.json`` (atomic write).

    Canonical song identity is always ``Path(cache_dir).name``. If the payload
    includes a ``song_hash`` field, it must match; optional ``song_hash_from_dir``
    (when passed) must also match, so the Gradio layer can assert wiring.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")
    canonical = cache.name
    if song_hash_from_dir is not None and song_hash_from_dir != canonical:
        raise ValueError(
            f"song_hash_from_dir {song_hash_from_dir!r} != cache directory {canonical!r}"
        )
    if isinstance(json_payload, Mapping):
        data: Any = json_payload
    else:
        raw = (
            json_payload.decode("utf-8")
            if isinstance(json_payload, (bytes, bytearray))
            else str(json_payload)
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Effects editor payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Effects editor payload must be a JSON object.")

    sh = data.get("song_hash")
    if sh is not None and str(sh) != canonical:
        raise ValueError(
            f"Payload song_hash {sh!r} does not match cache directory {canonical!r}."
        )

    # Keep only the on-disk fields (JS may post the full load_editor_state blob).
    file_keys = ("schema_version", "auto_reactivity_master", "auto_enabled", "clips")
    file_payload = {k: data[k] for k in file_keys if k in data}
    if "clips" not in file_payload:
        raise ValueError("Effects editor payload is missing a 'clips' array.")
    try:
        timeline = _timeline_from_dict(file_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid effects timeline payload: {exc}") from exc
    validate_effects_timeline(timeline)
    return save(cache, timeline)


def _dedupe_against(
    t_start: float, kind: EffectKind, clips: MutableSequence[EffectClip]
) -> bool:
    for c in clips:
        if c.kind is not kind:
            continue
        if abs(float(c.t_start) - t_start) <= DEDUPE_TOL_S:
            return True
    return False


def _bake_beam_clips(
    analysis: Mapping[str, Any], song_hash: str, cfg: CompositorConfig
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for b in _schedule_beams_for_ghosts(analysis, song_hash, cfg):
        eid = f"bake-beam-{uuid.uuid4().hex[:12]}"
        out.append(
            EffectClip(
                id=eid,
                kind=EffectKind.BEAM,
                t_start=float(b.t_start),
                duration_s=float(b.duration_s),
                settings={"strength": float(min(1.0, max(0.0, b.intensity)))},
                auto_source=True,
            )
        )
    return out


def _bake_glitch_clips(
    analysis: Mapping[str, Any], cfg: CompositorConfig
) -> list[EffectClip]:
    out: list[EffectClip] = []
    for t, _st in _impact_glitch_peaks(analysis, cfg):
        eid = f"bake-glitch-{uuid.uuid4().hex[:12]}"
        out.append(
            EffectClip(
                id=eid,
                kind=EffectKind.LOGO_GLITCH,
                t_start=float(t),
                duration_s=BAKE_GLITCH_DURATION_S,
                settings={"strength": 0.8},
                auto_source=True,
            )
        )
    return out


def _bake_shake_clips(analysis: Mapping[str, Any]) -> list[EffectClip]:
    """SCREEN_SHAKE clips anchored on low-band (kick) transient peaks.

    ``amplitude_px`` scales with the peak strength so louder kicks nudge the
    frame harder; ``frequency_hz`` is fixed at a kick-friendly 6 Hz so the
    burst reads as a single camera jolt instead of a sustained tremor.
    """
    out: list[EffectClip] = []
    for t, strength in _kick_transient_peaks(analysis):
        eid = f"bake-shake-{uuid.uuid4().hex[:12]}"
        amp = float(6.0 + 6.0 * min(1.0, max(0.0, strength)))
        out.append(
            EffectClip(
                id=eid,
                kind=EffectKind.SCREEN_SHAKE,
                t_start=float(t),
                duration_s=BAKE_SHAKE_DURATION_S,
                settings={"amplitude_px": amp, "frequency_hz": 6.0},
                auto_source=True,
            )
        )
    return out


def _bake_chroma_clips(analysis: Mapping[str, Any]) -> list[EffectClip]:
    """CHROMATIC_ABERRATION clips anchored on high-band (hat) transient peaks."""
    out: list[EffectClip] = []
    for t, strength in _hat_transient_peaks(analysis):
        eid = f"bake-chroma-{uuid.uuid4().hex[:12]}"
        shift = float(2.5 + 3.5 * min(1.0, max(0.0, strength)))
        out.append(
            EffectClip(
                id=eid,
                kind=EffectKind.CHROMATIC_ABERRATION,
                t_start=float(t),
                duration_s=BAKE_CHROMA_DURATION_S,
                settings={
                    "shift_px": shift,
                    "jitter": 0.35,
                    "direction_deg": 0.0,
                },
                auto_source=True,
            )
        )
    return out


def _bake_zoom_clips_from_drops(analysis: Mapping[str, Any]) -> list[EffectClip]:
    out: list[EffectClip] = []
    for t in _drops_sorted(analysis):
        eid = f"bake-zoom-{uuid.uuid4().hex[:12]}"
        out.append(
            EffectClip(
                id=eid,
                kind=EffectKind.ZOOM_PUNCH,
                t_start=float(t),
                duration_s=BAKE_ZOOM_PUNCH_ON_DROP_S,
                settings={},
                auto_source=True,
            )
        )
    return out


def bake_auto_schedule(
    cache_dir: str | Path,
    *,
    compositor_config: CompositorConfig | None = None,
) -> Path:
    """
    Turn the current **auto** hints (analyser + rim-beam schedule + impact
    peaks + drop times + kick / hat transient peaks) into real
    :class:`EffectClip` rows and append to the timeline. ``SCREEN_SHAKE`` clips
    are anchored on low-band peaks and ``CHROMATIC_ABERRATION`` clips on
    high-band peaks so the editor can scaffold a punchy starting pass per
    song without a user placing every clip by hand.

    For each :class:`EffectKind`, new clips are skipped when ``auto_enabled`` is
    ``False`` for that kind, or when any existing clip of that kind (manual or
    baked) has a start time within :data:`DEDUPE_TOL_S` of the candidate.

    Persists the merged timeline to ``effects_timeline.json`` and returns that path.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")
    song_hash = cache.name
    analysis = _load_analysis_mapping(cache)
    if not analysis:
        raise FileNotFoundError(
            f"{ANALYSIS_JSON_NAME} not found in {cache}; run audio analysis first."
        )

    timeline = load(cache)
    cfg = compositor_config or CompositorConfig()
    merged: list[EffectClip] = list(timeline.clips)

    candidates: list[EffectClip] = []
    if timeline.auto_enabled.get(EffectKind.BEAM, True):
        candidates.extend(_bake_beam_clips(analysis, song_hash, cfg))
    if timeline.auto_enabled.get(EffectKind.LOGO_GLITCH, True):
        candidates.extend(_bake_glitch_clips(analysis, cfg))
    if timeline.auto_enabled.get(EffectKind.ZOOM_PUNCH, True):
        candidates.extend(_bake_zoom_clips_from_drops(analysis))
    if timeline.auto_enabled.get(EffectKind.SCREEN_SHAKE, True):
        candidates.extend(_bake_shake_clips(analysis))
    if timeline.auto_enabled.get(EffectKind.CHROMATIC_ABERRATION, True):
        candidates.extend(_bake_chroma_clips(analysis))

    for c in candidates:
        if _dedupe_against(float(c.t_start), c.kind, merged):
            continue
        merged.append(c)

    new_t = EffectsTimeline(
        clips=merged,
        auto_enabled=timeline.auto_enabled,
        auto_reactivity_master=timeline.auto_reactivity_master,
    )
    return save(cache, new_t)


# ---------------------------------------------------------------------------
# HTML + JS editor (mirrors pipeline.lyrics_editor.build_editor_html)
# ---------------------------------------------------------------------------

# Row order in the editor. Matches the seven v1 EffectKind values; a PRD
# requirement that every kind (including SCANLINE_TEAR) keeps a row + toolbar
# button alongside the others so the UI shape doesn't churn.
_KIND_ORDER: tuple[str, ...] = (
    "BEAM",
    "LOGO_GLITCH",
    "SCREEN_SHAKE",
    "COLOR_INVERT",
    "CHROMATIC_ABERRATION",
    "SCANLINE_TEAR",
    "ZOOM_PUNCH",
)

# Per-kind palette. Colours are chosen for legibility on the dark editor
# background and to be easy to tell apart on the seven compact rows.
_KIND_COLORS: dict[str, str] = {
    "BEAM": "#f59e0b",
    "LOGO_GLITCH": "#a855f7",
    "SCREEN_SHAKE": "#ef4444",
    "COLOR_INVERT": "#22d3ee",
    "CHROMATIC_ABERRATION": "#14b8a6",
    "SCANLINE_TEAR": "#eab308",
    "ZOOM_PUNCH": "#22c55e",
}

# Short button label for each kind's toolbar "+ …" shortcut.
_KIND_LABELS: dict[str, str] = {
    "BEAM": "Beam",
    "LOGO_GLITCH": "Glitch",
    "SCREEN_SHAKE": "Shake",
    "COLOR_INVERT": "Invert",
    "CHROMATIC_ABERRATION": "Chromatic",
    "SCANLINE_TEAR": "Scanline",
    "ZOOM_PUNCH": "Zoom",
}

# Defaults used when the toolbar adds a new clip at the playhead. Kept in
# sync with the renderer _DEFAULT_* constants where one exists (see
# pipeline/zoom_punch.py, screen_shake.py, color_invert.py); the remaining
# values are just sensible starting points. Server-side
# :func:`validate_settings_for_kind` remains the authority, so out-of-range
# values get caught on save regardless of what the JS sends.
_KIND_DEFAULTS: dict[str, dict[str, Any]] = {
    "BEAM": {"duration_s": 0.5, "settings": {"strength": 0.8}},
    "LOGO_GLITCH": {
        "duration_s": BAKE_GLITCH_DURATION_S,
        "settings": {"strength": 0.8},
    },
    "SCREEN_SHAKE": {
        "duration_s": 0.3,
        "settings": {"amplitude_px": 6.0, "frequency_hz": 4.0},
    },
    "COLOR_INVERT": {
        "duration_s": 0.25,
        "settings": {"mix": 1.0, "intensity": 1.0},
    },
    "CHROMATIC_ABERRATION": {
        "duration_s": 0.4,
        "settings": {"shift_px": 4.0, "jitter": 0.5, "direction_deg": 0.0},
    },
    "SCANLINE_TEAR": {
        "duration_s": 0.4,
        "settings": {"intensity": 0.6, "band_count": 3, "band_height_px": 12},
    },
    "ZOOM_PUNCH": {
        "duration_s": BAKE_ZOOM_PUNCH_ON_DROP_S,
        "settings": {"peak_scale": 1.12, "ease_in_s": 0.08, "ease_out_s": 0.12},
    },
}


def _settings_keys_for_js() -> dict[str, list[str]]:
    """Expose :data:`EFFECT_SETTINGS_KEYS` as a JSON-friendly map for the JS.

    The JS gear panel uses this to build inputs dynamically, so adding a
    setting to a renderer in Python is enough to surface it in the editor.
    """
    return {
        k.name: sorted(EFFECT_SETTINGS_KEYS[k]) for k in EffectKind
    }


_EFFECTS_CSS = """
<style>
  .mv-fx { font-family: system-ui, sans-serif; color: #e5e7eb; }
  .mv-fx .mv-fx-toolbar { display: flex; gap: 6px; align-items: center;
    margin-bottom: 6px; font-size: 13px; flex-wrap: wrap; }
  .mv-fx .mv-fx-toolbar button { background: #1f2937; color: #f3f4f6;
    border: 1px solid #374151; border-radius: 4px; padding: 4px 8px;
    cursor: pointer; font-size: 12px; }
  .mv-fx .mv-fx-toolbar button:hover { background: #374151; }
  .mv-fx .mv-fx-toolbar .mv-fx-info { color: #9ca3af; margin-left: auto;
    font-size: 12px; }
  .mv-fx .mv-fx-master { display: flex; gap: 8px; align-items: center;
    margin: 4px 0 8px 0; font-size: 12px; color: #cbd5e1; }
  .mv-fx .mv-fx-master input[type=range] { flex: 1; max-width: 300px; }
  .mv-fx .mv-fx-master .mv-fx-master-value { min-width: 48px;
    text-align: right; color: #f3f4f6; font-variant-numeric: tabular-nums; }
  .mv-fx .mv-fx-body { display: flex; gap: 0; align-items: stretch;
    background: #0b1220; border: 1px solid #1f2937; border-radius: 6px;
    overflow: hidden; user-select: none; }
  .mv-fx .mv-fx-labels { flex: 0 0 156px; background: #0f172a;
    border-right: 1px solid #1f2937; display: flex; flex-direction: column; }
  .mv-fx .mv-fx-label-head { height: 120px;
    border-bottom: 1px solid #1f2937;
    display: flex; align-items: flex-end; padding: 4px 8px;
    color: #64748b; font-size: 11px; }
  .mv-fx .mv-fx-label { height: 36px; display: flex; align-items: center;
    gap: 6px; padding: 0 8px; border-bottom: 1px solid #111827;
    font-size: 11px; }
  .mv-fx .mv-fx-label .mv-fx-swatch { width: 10px; height: 10px;
    border-radius: 2px; flex: 0 0 10px; }
  .mv-fx .mv-fx-label .mv-fx-kind-name { flex: 1; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; color: #e5e7eb; }
  .mv-fx .mv-fx-label label { color: #94a3b8; display: flex; gap: 3px;
    align-items: center; cursor: pointer; }
  .mv-fx .mv-fx-scroller { flex: 1; overflow-x: auto; overflow-y: hidden;
    position: relative; }
  .mv-fx .mv-fx-stage { position: relative; }
  .mv-fx canvas.mv-fx-wave { display: block; width: 100%; height: 120px;
    background: #0b1220; border-bottom: 1px solid #1f2937; }
  .mv-fx .mv-fx-row { position: relative; height: 36px;
    border-bottom: 1px solid #111827; background: #0b1220; }
  .mv-fx .mv-fx-row:last-child { border-bottom: none; }
  .mv-fx .mv-fx-row.row-disabled { background:
    repeating-linear-gradient(135deg, #0b1220, #0b1220 6px,
    #111827 6px, #111827 10px); }
  .mv-fx .mv-fx-clip { position: absolute; top: 4px; height: 28px;
    border-radius: 3px; box-sizing: border-box;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.4) inset; cursor: grab;
    display: flex; align-items: center; overflow: hidden;
    -webkit-user-select: none; user-select: none;
    font-size: 10px; color: rgba(0, 0, 0, 0.85); padding: 0 4px; }
  .mv-fx .mv-fx-clip.dragging { cursor: grabbing; opacity: 0.85; }
  .mv-fx .mv-fx-clip.selected { outline: 2px solid #fde68a;
    outline-offset: -2px; }
  .mv-fx .mv-fx-clip.auto { background-image:
    repeating-linear-gradient(45deg, rgba(255,255,255,0.0),
    rgba(255,255,255,0.0) 4px, rgba(255,255,255,0.18) 4px,
    rgba(255,255,255,0.18) 8px); }
  .mv-fx .mv-fx-clip .mv-fx-handle { position: absolute; top: 0;
    width: 5px; height: 100%; cursor: ew-resize;
    background: rgba(0,0,0,0.30); }
  .mv-fx .mv-fx-clip .mv-fx-handle.left { left: 0; }
  .mv-fx .mv-fx-clip .mv-fx-handle.right { right: 0; }
  .mv-fx .mv-fx-clip .mv-fx-gear { position: absolute; top: 1px;
    right: 8px; width: 14px; height: 14px; border-radius: 2px;
    background: rgba(0,0,0,0.55); color: #f8fafc;
    font-size: 10px; line-height: 14px; text-align: center;
    cursor: pointer; user-select: none; }
  .mv-fx .mv-fx-clip .mv-fx-gear:hover { background: rgba(0,0,0,0.75); }
  .mv-fx .mv-fx-clip .mv-fx-clip-label { pointer-events: none;
    font-weight: 600; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; flex: 1; padding-right: 22px; }
  .mv-fx .mv-fx-ghost { position: absolute; top: 4px; bottom: 4px;
    width: 1px; background: currentColor; opacity: 0.28;
    pointer-events: none; }
  .mv-fx .mv-fx-playhead { position: absolute; top: 0; bottom: 0;
    width: 2px; background: #f43f5e; pointer-events: none; z-index: 5; }
  .mv-fx .mv-fx-drag-guide { position: absolute; top: 0; bottom: 0;
    width: 1px; background: #fde68a; box-shadow: 0 0 4px #fde68a;
    pointer-events: none; z-index: 6; display: none; }
  .mv-fx .mv-fx-band { position: absolute; top: 120px; bottom: 0;
    border: 1px dashed #f59e0b;
    background: rgba(245, 158, 11, 0.12);
    pointer-events: none; display: none; z-index: 4; }
  .mv-fx .mv-fx-audio { display: block; width: 100%; margin-top: 6px; }
  .mv-fx .mv-fx-help { color: #9ca3af; font-size: 11px; margin-top: 6px;
    line-height: 1.4; }
  .mv-fx .mv-fx-help kbd { background: #1f2937; color: #f3f4f6; border: 1px solid #4b5563;
    border-radius: 3px; padding: 1px 5px; font-size: 11px; font-weight: 500; }
  .mv-fx .mv-fx-settings { position: absolute; min-width: 240px;
    background: #111827; border: 1px solid #374151; border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); padding: 10px;
    z-index: 50; color: #e5e7eb; font-size: 12px; }
  .mv-fx .mv-fx-settings h4 { margin: 0 0 8px 0; font-size: 12px;
    color: #f3f4f6; }
  .mv-fx .mv-fx-settings .mv-fx-row-field { display: flex;
    align-items: center; gap: 8px; margin-bottom: 4px; }
  .mv-fx .mv-fx-settings label { flex: 0 0 96px; color: #cbd5e1;
    font-size: 11px; }
  .mv-fx .mv-fx-settings input { flex: 1; background: #0b1220;
    border: 1px solid #374151; border-radius: 3px; color: #e5e7eb;
    padding: 3px 5px; font-size: 11px; font-family: inherit; }
  .mv-fx .mv-fx-settings input[type=color] { padding: 0; height: 24px; }
  .mv-fx .mv-fx-settings .mv-fx-settings-close { position: absolute;
    top: 4px; right: 6px; background: none; border: none;
    color: #9ca3af; cursor: pointer; font-size: 14px; }
</style>
""".strip()


def build_editor_html(
    state: Mapping[str, Any],
    *,
    audio_url: str,
    container_id: str,
    state_js_var: str = "_glitchframe_effects_state",
    audio_element_id: str = "mv_fx_audio",
    pixels_per_second: float = 40.0,
) -> str:
    """Return CSS + markup + inline vanilla-JS for the effects-timeline editor.

    Mirrors :func:`pipeline.lyrics_editor.build_editor_html`: the editor is a
    single self-contained block injected into a ``gr.HTML`` component, and
    round-trips all state through ``window[state_js_var]`` so the Gradio Save
    button can post it verbatim to :func:`save_edited_timeline`.

    ``state`` is the dict returned by :func:`load_editor_state` (``song_hash``,
    ``clips``, ``auto_enabled``, ``auto_reactivity_master``, ``peaks``,
    ``duration``, ``sample_rate``, ``ghost_events``).
    """
    song_hash = str(state.get("song_hash", ""))
    duration = float(state.get("duration", 0.0))
    clips = list(state.get("clips") or [])
    ghost_events = list(state.get("ghost_events") or [])
    peaks_in = state.get("peaks") or []
    peaks = [[float(a), float(b)] for a, b in peaks_in]
    auto_enabled_in = state.get("auto_enabled") or {}
    auto_enabled = {k: bool(auto_enabled_in.get(k, True)) for k in _KIND_ORDER}
    master_raw = state.get("auto_reactivity_master", 1.0)
    try:
        master = float(master_raw)
    except (TypeError, ValueError):
        master = 1.0

    payload = {
        "schema_version": int(state.get("schema_version", SCHEMA_VERSION)),
        "song_hash": song_hash,
        "duration": duration,
        "sample_rate": int(state.get("sample_rate", 0) or 0),
        "peaks": peaks,
        "clips": clips,
        "auto_enabled": auto_enabled,
        "auto_reactivity_master": master,
        "ghost_events": ghost_events,
        "kind_order": list(_KIND_ORDER),
        "kind_colors": _KIND_COLORS,
        "kind_labels": _KIND_LABELS,
        "kind_defaults": _KIND_DEFAULTS,
        "settings_keys": _settings_keys_for_js(),
    }
    payload_json = json.dumps(payload)

    info = (
        f"{song_hash[:8] or '<no-hash>'} · {duration:.1f}s · "
        f"{len(clips)} clip(s) · {len(ghost_events)} ghost(s)"
    )
    info_html = html_lib.escape(info)

    toolbar_buttons = "".join(
        (
            f"<button type=\"button\" data-mv-fx-add=\"{k}\" "
            f"style=\"border-left: 3px solid {_KIND_COLORS[k]};\">"
            f"+ {html_lib.escape(_KIND_LABELS[k])}</button>"
        )
        for k in _KIND_ORDER
    )

    # Left-column label rows. Per-kind disable-auto checkbox lives here so
    # the user can silence an analyser path without deleting anything.
    label_rows = "".join(
        (
            f"<div class=\"mv-fx-label\" data-mv-fx-kind-label=\"{k}\">"
            f"  <span class=\"mv-fx-swatch\" style=\"background:{_KIND_COLORS[k]}\"></span>"
            f"  <span class=\"mv-fx-kind-name\">{html_lib.escape(_KIND_LABELS[k])}</span>"
            f"  <label title=\"Disable auto for this kind\">"
            f"    <input type=\"checkbox\" data-mv-fx-auto=\"{k}\" "
            f"    {'checked' if auto_enabled[k] else ''}> auto"
            f"  </label>"
            f"</div>"
        )
        for k in _KIND_ORDER
    )

    # Seven stage rows (one per EffectKind). The JS populates clips / ghost
    # ticks into these rows; they have no children at page-load time.
    stage_rows = "".join(
        f"<div class=\"mv-fx-row\" data-mv-fx-row=\"{k}\" "
        f"style=\"color:{_KIND_COLORS[k]}\"></div>"
        for k in _KIND_ORDER
    )

    script = _EFFECTS_JS
    for key, val in {
        "__MV_CONTAINER_ID__": container_id,
        "__MV_STATE_JS_VAR__": state_js_var,
        "__MV_AUDIO_ELEMENT_ID__": audio_element_id,
        "__MV_PAYLOAD_JSON__": payload_json,
        "__MV_PIXELS_PER_SECOND__": str(pixels_per_second),
    }.items():
        script = script.replace(key, val)
    # Defuse stray </script> sequences (defensive; JSON uses \/ via replace).
    code_blob = script.replace("</", "<\\/")
    code_tag_id = f"mv_fx_code_{container_id}"

    audio_src = html_lib.escape(audio_url, quote=True)
    master_pct = int(round(max(0.0, min(2.0, master)) * 100.0))

    return (
        f"{_EFFECTS_CSS}"
        f"<div class=\"mv-fx\" id=\"{container_id}\">"
        f"  <div class=\"mv-fx-toolbar\">"
        f"    <button type=\"button\" data-mv-fx-action=\"play\">▶ Play / Pause</button>"
        f"    <button type=\"button\" data-mv-fx-action=\"zoom-in\">+</button>"
        f"    <button type=\"button\" data-mv-fx-action=\"zoom-out\">−</button>"
        f"    <button type=\"button\" data-mv-fx-action=\"zoom-fit\">Fit</button>"
        f"    <span style=\"width: 8px;\"></span>"
        f"    {toolbar_buttons}"
        f"    <span class=\"mv-fx-info\">{info_html}</span>"
        f"  </div>"
        f"  <div class=\"mv-fx-master\">"
        f"    <span>Master reactivity</span>"
        f"    <input type=\"range\" min=\"0\" max=\"200\" step=\"1\" "
        f"value=\"{master_pct}\" data-mv-fx-master>"
        f"    <span class=\"mv-fx-master-value\" data-mv-fx-master-value>"
        f"{master_pct}%</span>"
        f"  </div>"
        f"  <div class=\"mv-fx-body\">"
        f"    <div class=\"mv-fx-labels\">"
        f"      <div class=\"mv-fx-label-head\">waveform</div>"
        f"      {label_rows}"
        f"    </div>"
        f"    <div class=\"mv-fx-scroller\" data-mv-fx-scroller>"
        f"      <div class=\"mv-fx-stage\" data-mv-fx-stage>"
        f"        <canvas class=\"mv-fx-wave\" data-mv-fx-wave></canvas>"
        f"        <div data-mv-fx-rows>{stage_rows}</div>"
        f"        <div class=\"mv-fx-playhead\" data-mv-fx-playhead></div>"
        f"        <div class=\"mv-fx-drag-guide\" data-mv-fx-drag-guide></div>"
        f"        <div class=\"mv-fx-band\" data-mv-fx-band></div>"
        f"      </div>"
        f"    </div>"
        f"  </div>"
        f"  <audio class=\"mv-fx-audio\" id=\"{audio_element_id}\" "
        f"src=\"{audio_src}\" controls preload=\"auto\"></audio>"
        f"  <div class=\"mv-fx-help\">"
        f"    Click a toolbar <b>+</b> button to add a clip of that kind at the "
        f"playhead, or while playing use <kbd>1</kbd>–<kbd>7</kbd> (rows top to "
        f"bottom: Beam … Zoom). <strong>Double-click</strong> an empty spot in a "
        f"row to add that effect at that time. Drag a clip to move; drag its edges "
        f"to resize; "
        f"click its <b>⚙</b> to edit settings. Click an empty row to seek. "
        f"<kbd>Shift</kbd>/<kbd>Ctrl</kbd>-click to multi-select; click-drag "
        f"empty timeline to rubber-band; <kbd>Del</kbd> deletes. "
        f"<kbd>Space</kbd> play/pause, <kbd>+</kbd>/<kbd>−</kbd> zoom, "
        f"<kbd>Esc</kbd> clear selection, <kbd>Ctrl</kbd>+<kbd>A</kbd> select all. "
        f"Faint ticks mark automatic analyser events for reference."
        f"  </div>"
        f"</div>"
        f"<script type=\"text/plain\" id=\"{code_tag_id}\">{code_blob}</script>"
        f"<img src=\"x\" alt=\"\" style=\"display:none\" "
        f"onerror=\"this.remove();"
        f"var _c=document.getElementById('{code_tag_id}');"
        f"if(_c){{try{{(new Function(_c.textContent))();}}"
        f"catch(_e){{console.error('mv-fx-editor init failed',_e);}}}}\">"
    )


# NOTE: placeholders are rewritten by :func:`build_editor_html` via
# ``str.replace``; this keeps literal ``%`` operators in the JS (e.g. row
# layout maths) working without Python printf escaping.
_EFFECTS_JS = r"""
(function () {
  const container = document.getElementById("__MV_CONTAINER_ID__");
  if (!container) return;
  const state = __MV_PAYLOAD_JSON__;
  window.__MV_STATE_JS_VAR__ = state;

  // ── Layout constants ──────────────────────────────────────────────────
  const ROW_H = 36;
  const WAVE_H = 120;
  const KINDS = state.kind_order;
  const COLORS = state.kind_colors;
  const LABELS = state.kind_labels;
  const SETTINGS_KEYS = state.settings_keys || {};
  const DEFAULTS = state.kind_defaults || {};
  let pxPerSec = __MV_PIXELS_PER_SECOND__;

  // ── DOM handles ───────────────────────────────────────────────────────
  const toolbar = container.querySelector(".mv-fx-toolbar");
  const scroller = container.querySelector("[data-mv-fx-scroller]");
  const stage = container.querySelector("[data-mv-fx-stage]");
  const wave = container.querySelector("[data-mv-fx-wave]");
  const playhead = container.querySelector("[data-mv-fx-playhead]");
  const guide = container.querySelector("[data-mv-fx-drag-guide]");
  const bandEl = container.querySelector("[data-mv-fx-band]");
  const masterSlider = container.querySelector("[data-mv-fx-master]");
  const masterValue = container.querySelector("[data-mv-fx-master-value]");
  const rowEls = {};
  KINDS.forEach((k) => {
    rowEls[k] = container.querySelector(`[data-mv-fx-row="${k}"]`);
  });
  function audio() {
    return container.querySelector("#__MV_AUDIO_ELEMENT_ID__")
        || document.getElementById("__MV_AUDIO_ELEMENT_ID__");
  }

  // ── Core helpers ──────────────────────────────────────────────────────
  function secondsToPx(t) { return t * pxPerSec; }
  function pxToSeconds(x) { return x / pxPerSec; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function newId() {
    // Match the server's "bake-*" style for readability; JS-side ids use a
    // "ui-" prefix so the server can tell at a glance that a clip came from
    // the browser (no special behaviour — purely for debugging save logs).
    return "ui-" + Math.random().toString(36).slice(2, 14);
  }

  function applyRowDisabled() {
    KINDS.forEach((k) => {
      const disabled = state.auto_enabled && state.auto_enabled[k] === false;
      rowEls[k].classList.toggle("row-disabled", !!disabled);
    });
  }

  function setStageWidth() {
    const w = Math.max(600, Math.round(state.duration * pxPerSec));
    stage.style.width = w + "px";
    wave.width = w;
    wave.height = WAVE_H;
    KINDS.forEach((k) => { rowEls[k].style.width = w + "px"; });
    drawWaveform();
  }

  function drawWaveform() {
    const ctx = wave.getContext("2d");
    const W = wave.width, H = wave.height;
    const mid = H / 2;
    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
    const peaks = state.peaks || [];
    if (peaks.length === 0) return;
    // Resolve each destination pixel against *all* buckets that fall inside
    // it (when peaks-per-pixel > 1 we'd otherwise alias loudly). When zoomed
    // in below one peak per pixel we just sample the nearest bucket; the
    // filled band then reads as a continuous waveform instead of a 1-px line.
    const top = new Float32Array(W);
    const bot = new Float32Array(W);
    const scale = peaks.length / W;
    for (let x = 0; x < W; x++) {
      const lo = Math.floor(x * scale);
      const hi = Math.max(lo + 1, Math.floor((x + 1) * scale));
      let mn = peaks[lo][0], mx = peaks[lo][1];
      for (let i = lo + 1; i < hi && i < peaks.length; i++) {
        const p = peaks[i];
        if (p[0] < mn) mn = p[0];
        if (p[1] > mx) mx = p[1];
      }
      top[x] = mid - mx * mid;
      bot[x] = mid - mn * mid;
    }
    // Filled min--max band (alpha ~0.9) for body, then solid outline on top.
    ctx.fillStyle = "#38bdf8";
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(0, top[0]);
    for (let x = 1; x < W; x++) ctx.lineTo(x, top[x]);
    for (let x = W - 1; x >= 0; x--) ctx.lineTo(x, bot[x]);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = "#7dd3fc";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < W; x++) {
      if (x === 0) ctx.moveTo(x + 0.5, top[x]);
      else ctx.lineTo(x + 0.5, top[x]);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let x = 0; x < W; x++) {
      if (x === 0) ctx.moveTo(x + 0.5, bot[x]);
      else ctx.lineTo(x + 0.5, bot[x]);
    }
    ctx.stroke();
  }

  // ── Ghost markers (non-interactive) ───────────────────────────────────
  function renderGhosts() {
    KINDS.forEach((k) => {
      const row = rowEls[k];
      row.querySelectorAll(".mv-fx-ghost").forEach((n) => n.remove());
    });
    const ev = Array.isArray(state.ghost_events) ? state.ghost_events : [];
    ev.forEach((g) => {
      const k = g && g.kind;
      if (!KINDS.includes(k)) return;
      const row = rowEls[k];
      const tick = document.createElement("div");
      tick.className = "mv-fx-ghost";
      tick.style.left = secondsToPx(Number(g.t) || 0) + "px";
      row.appendChild(tick);
    });
  }

  // ── Clips ─────────────────────────────────────────────────────────────
  // ``selected`` stores clip ids (string) because indexes shift when we add
  // or remove clips — ids are stable across mutations.
  const selected = new Set();

  function findClip(id) {
    return (state.clips || []).find((c) => c.id === id) || null;
  }

  function renderClips() {
    KINDS.forEach((k) => {
      const row = rowEls[k];
      row.querySelectorAll(".mv-fx-clip").forEach((n) => n.remove());
    });
    (state.clips || []).forEach((c) => createClipDom(c));
    applySelectionStyling();
  }

  function createClipDom(clip) {
    const row = rowEls[clip.kind];
    if (!row) return;
    const el = document.createElement("div");
    el.className = "mv-fx-clip" + (clip.auto_source ? " auto" : "");
    el.dataset.clipId = clip.id;
    el.style.background = COLORS[clip.kind] || "#64748b";
    positionClipDom(el, clip);

    const label = document.createElement("span");
    label.className = "mv-fx-clip-label";
    label.textContent = LABELS[clip.kind] || clip.kind;
    el.appendChild(label);

    const gear = document.createElement("span");
    gear.className = "mv-fx-gear";
    gear.textContent = "⚙";
    gear.title = "Edit settings";
    el.appendChild(gear);

    const lh = document.createElement("div");
    lh.className = "mv-fx-handle left";
    const rh = document.createElement("div");
    rh.className = "mv-fx-handle right";
    el.appendChild(lh); el.appendChild(rh);

    el.addEventListener("pointerdown", onClipPointerDown);
    gear.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    gear.addEventListener("click", (ev) => {
      ev.stopPropagation();
      openSettingsPanel(clip.id, el);
    });

    row.appendChild(el);
    return el;
  }

  function positionClipDom(el, clip) {
    const left = secondsToPx(Math.max(0, clip.t_start));
    const width = Math.max(6, secondsToPx(Math.max(0, clip.duration_s)));
    el.style.left = left + "px";
    el.style.width = width + "px";
  }

  function updateClipDom(id) {
    const el = container.querySelector(
      `.mv-fx-clip[data-clip-id="${CSS.escape(id)}"]`);
    const c = findClip(id);
    if (el && c) positionClipDom(el, c);
  }

  function applySelectionStyling() {
    container.querySelectorAll(".mv-fx-clip").forEach((el) => {
      el.classList.toggle("selected", selected.has(el.dataset.clipId));
    });
  }

  function clearSelection() {
    if (selected.size === 0) return;
    selected.clear();
    applySelectionStyling();
  }

  function showGuide(t) {
    guide.style.left = secondsToPx(Math.max(0, t)) + "px";
    guide.style.display = "block";
  }
  function hideGuide() { guide.style.display = "none"; }

  // ── Drag / resize ─────────────────────────────────────────────────────
  let drag = null;
  let suppressNextStageClick = false;

  function onClipPointerDown(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    const node = ev.currentTarget;
    const id = node.dataset.clipId;
    const clip = findClip(id);
    if (!clip) return;
    const additive = ev.shiftKey || ev.ctrlKey || ev.metaKey;
    const mode = ev.target.classList.contains("mv-fx-handle")
      ? (ev.target.classList.contains("left") ? "left" : "right")
      : "move";

    if (additive) {
      if (selected.has(id)) { selected.delete(id); applySelectionStyling(); return; }
      selected.add(id);
    } else if (!selected.has(id)) {
      selected.clear(); selected.add(id);
    }
    applySelectionStyling();

    const ids = (mode === "move") ? Array.from(selected) : [id];
    const origs = new Map();
    ids.forEach((i) => {
      const c = findClip(i);
      if (c) origs.set(i, { s: c.t_start, e: c.t_start + c.duration_s });
    });
    drag = {
      primary: id, mode, node,
      startX: ev.clientX, ids, origs,
    };
    try { node.setPointerCapture(ev.pointerId); } catch (_e) {}
    node.classList.add("dragging");
    showGuide(mode === "right"
      ? clip.t_start + clip.duration_s
      : clip.t_start);
    document.addEventListener("pointermove", onClipPointerMove);
    document.addEventListener("pointerup", onClipPointerUp, { once: true });
  }

  function onClipPointerMove(ev) {
    if (!drag) return;
    const dt = pxToSeconds(ev.clientX - drag.startX);
    const dur = Math.max(0, state.duration);
    const primary = findClip(drag.primary);
    if (drag.mode === "move") {
      // Single shared delta preserves relative spacing across the group.
      let dtc = dt;
      drag.ids.forEach((i) => {
        const o = drag.origs.get(i); if (!o) return;
        if (o.s + dtc < 0) dtc = -o.s;
        if (o.e + dtc > dur) dtc = dur - o.e;
      });
      drag.ids.forEach((i) => {
        const c = findClip(i); const o = drag.origs.get(i);
        if (!c || !o) return;
        c.t_start = o.s + dtc;
        updateClipDom(i);
      });
      if (primary) showGuide(primary.t_start);
    } else {
      const o = drag.origs.get(drag.primary);
      if (!primary || !o) return;
      if (drag.mode === "left") {
        const newStart = clamp(o.s + dt, 0, o.e - 0.02);
        primary.t_start = newStart;
        primary.duration_s = o.e - newStart;
        showGuide(primary.t_start);
      } else {
        const newEnd = clamp(o.e + dt, o.s + 0.02, dur);
        primary.duration_s = newEnd - primary.t_start;
        showGuide(newEnd);
      }
      updateClipDom(drag.primary);
    }
  }

  function onClipPointerUp(ev) {
    if (!drag) return;
    hideGuide();
    drag.node.classList.remove("dragging");
    const moved = Math.abs(ev.clientX - drag.startX);
    if (moved < 3 && drag.mode === "move" && drag.ids.length === 1) {
      const c = findClip(drag.primary);
      const a = audio();
      if (c && a) {
        try { a.currentTime = Math.max(0, c.t_start); } catch (_e) {}
      }
    }
    drag = null;
    document.removeEventListener("pointermove", onClipPointerMove);
  }

  // ── Rubber-band on empty row / waveform area ──────────────────────────
  // When the drag starts inside a specific effect row, the band is scoped
  // to that row so you can marquee-select e.g. just LOGO_GLITCH clips
  // without pulling in overlapping BEAM / COLOR_INVERT clips at the same
  // x-range. Starting the drag on the waveform (above all rows) keeps the
  // legacy all-row behaviour.
  let band = null;
  stage.addEventListener("pointerdown", (ev) => {
    // Clips / handles / gears stop propagation, so reaching here means the
    // user pressed on empty timeline background.
    if (ev.target.classList && (
        ev.target.classList.contains("mv-fx-clip") ||
        ev.target.classList.contains("mv-fx-handle") ||
        ev.target.classList.contains("mv-fx-gear") ||
        ev.target.classList.contains("mv-fx-clip-label"))) {
      return;
    }
    const rect = stage.getBoundingClientRect();
    const additive = ev.shiftKey || ev.ctrlKey || ev.metaKey;
    if (!additive) clearSelection();
    // Identify the row the pointer went down on (``null`` = waveform).
    const rowEl = (ev.target && ev.target.closest)
      ? ev.target.closest("[data-mv-fx-row]") : null;
    const rowKind = rowEl ? rowEl.getAttribute("data-mv-fx-row") : null;
    band = {
      startX: ev.clientX - rect.left,
      startClientX: ev.clientX,
      origSelected: new Set(selected),
      moved: false,
      kind: rowKind,
    };
    bandEl.style.display = "block";
    bandEl.style.left = band.startX + "px";
    bandEl.style.width = "0px";
    // When scoped to one row, shrink the visual band to that row's vertical
    // slice so the user sees what's actually being selected.
    if (rowEl) {
      const rowRect = rowEl.getBoundingClientRect();
      const top = rowRect.top - rect.top;
      const h = rowRect.height;
      bandEl.style.top = top + "px";
      bandEl.style.bottom = "auto";
      bandEl.style.height = h + "px";
    } else {
      bandEl.style.top = "";
      bandEl.style.bottom = "";
      bandEl.style.height = "";
    }
    document.addEventListener("pointermove", onBandMove);
    document.addEventListener("pointerup", onBandUp, { once: true });
  });

  function onBandMove(ev) {
    if (!band) return;
    const rect = stage.getBoundingClientRect();
    const curX = ev.clientX - rect.left;
    const left = Math.min(band.startX, curX);
    const right = Math.max(band.startX, curX);
    bandEl.style.left = left + "px";
    bandEl.style.width = (right - left) + "px";
    if (Math.abs(ev.clientX - band.startClientX) > 2) band.moved = true;
    const newSel = new Set(band.origSelected);
    (state.clips || []).forEach((c) => {
      if (band.kind && c.kind !== band.kind) return;
      const cx = secondsToPx(c.t_start + c.duration_s / 2);
      if (cx >= left && cx <= right) newSel.add(c.id);
    });
    selected.clear();
    newSel.forEach((i) => selected.add(i));
    applySelectionStyling();
  }

  function onBandUp(_ev) {
    if (!band) return;
    bandEl.style.display = "none";
    // Reset inline row-scope styling so the next full-height drag renders
    // against the CSS defaults.
    bandEl.style.top = "";
    bandEl.style.bottom = "";
    bandEl.style.height = "";
    if (band.moved) suppressNextStageClick = true;
    band = null;
    document.removeEventListener("pointermove", onBandMove);
  }

  // Click empty row → seek audio.
  stage.addEventListener("click", (ev) => {
    if (suppressNextStageClick) { suppressNextStageClick = false; return; }
    if (ev.target.classList && (
        ev.target.classList.contains("mv-fx-clip") ||
        ev.target.classList.contains("mv-fx-handle") ||
        ev.target.classList.contains("mv-fx-gear") ||
        ev.target.classList.contains("mv-fx-clip-label"))) {
      return;
    }
    const rect = stage.getBoundingClientRect();
    const t = pxToSeconds(ev.clientX - rect.left);
    const a = audio();
    if (a) { try { a.currentTime = Math.max(0, t); } catch (_e) {} }
  });

  stage.addEventListener("dblclick", (ev) => {
    const rowEl = ev.target.closest("[data-mv-fx-row]");
    if (!rowEl) return;
    if (ev.target.closest(".mv-fx-clip")) return;
    ev.preventDefault();
    ev.stopPropagation();
    const kind = rowEl.getAttribute("data-mv-fx-row");
    if (!kind || !KINDS.includes(kind)) return;
    const rect = stage.getBoundingClientRect();
    const t = clamp(pxToSeconds(ev.clientX - rect.left), 0, state.duration);
    addClipAtPlayhead(kind, t);
  });

  // ── Toolbar: play / zoom / add clip ───────────────────────────────────
  toolbar.addEventListener("click", (ev) => {
    const target = ev.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.getAttribute("data-mv-fx-action");
    const addKind = target.getAttribute("data-mv-fx-add");
    if (action === "play") {
      const a = audio(); if (a) { a.paused ? a.play() : a.pause(); }
    } else if (action === "zoom-in") {
      setZoom(pxPerSec * 1.25);
    } else if (action === "zoom-out") {
      setZoom(pxPerSec / 1.25);
    } else if (action === "zoom-fit") {
      setZoom(Math.max(20,
        (scroller.clientWidth - 8) / Math.max(1, state.duration)));
    } else if (addKind) {
      addClipAtPlayhead(addKind, null);
    }
  });

  function setZoom(v) {
    pxPerSec = Math.max(4, Math.min(600, v));
    setStageWidth();
    renderGhosts();
    renderClips();
  }

  function addClipAtPlayhead(kind, atTime) {
    const a = audio();
    let t0;
    if (atTime != null && isFinite(atTime)) {
      t0 = atTime;
    } else {
      t0 = (a && isFinite(a.currentTime)) ? a.currentTime : 0;
    }
    const d = (DEFAULTS[kind] && DEFAULTS[kind].duration_s) || 0.3;
    const settings = Object.assign({}, (DEFAULTS[kind] || {}).settings || {});
    const dur = Math.max(0, state.duration);
    const tStart = Math.max(0, Math.min(t0, Math.max(0, dur - d)));
    const clip = {
      id: newId(),
      kind,
      t_start: tStart,
      duration_s: d,
      settings,
      auto_source: false,
    };
    state.clips = state.clips || [];
    state.clips.push(clip);
    createClipDom(clip);
    selected.clear();
    selected.add(clip.id);
    applySelectionStyling();
  }

  // ── Delete selected ───────────────────────────────────────────────────
  function deleteSelected() {
    if (selected.size === 0) return;
    state.clips = (state.clips || []).filter((c) => !selected.has(c.id));
    selected.clear();
    renderClips();
  }

  // ── Master reactivity + per-row auto checkboxes ───────────────────────
  if (masterSlider) {
    masterSlider.addEventListener("input", () => {
      const pct = parseInt(masterSlider.value, 10);
      state.auto_reactivity_master = clamp(pct / 100.0, 0, 2);
      if (masterValue) masterValue.textContent = pct + "%";
    });
  }
  container.querySelectorAll("[data-mv-fx-auto]").forEach((cb) => {
    cb.addEventListener("change", () => {
      const k = cb.getAttribute("data-mv-fx-auto");
      state.auto_enabled = state.auto_enabled || {};
      state.auto_enabled[k] = !!cb.checked;
      applyRowDisabled();
    });
  });

  // ── Settings panel (floating, one-at-a-time) ──────────────────────────
  let panelEl = null;
  function closePanel() {
    if (panelEl) { panelEl.remove(); panelEl = null; }
    document.removeEventListener("pointerdown", onDocPointerDownForPanel, true);
  }
  function onDocPointerDownForPanel(ev) {
    if (!panelEl) return;
    if (panelEl.contains(ev.target)) return;
    // Clicking the gear of the SAME panel is handled by its own listener
    // (which just reopens) — we close and let it reopen cleanly.
    closePanel();
  }

  function openSettingsPanel(clipId, clipEl) {
    closePanel();
    const clip = findClip(clipId);
    if (!clip) return;
    const keys = SETTINGS_KEYS[clip.kind] || [];
    panelEl = document.createElement("div");
    panelEl.className = "mv-fx-settings";
    // Position the panel near the clip, within the scroller, and leave the
    // stage's coordinate space alone so horizontal scrolling works naturally.
    const rect = clipEl.getBoundingClientRect();
    const parentRect = container.getBoundingClientRect();
    panelEl.style.left = (rect.left - parentRect.left) + "px";
    panelEl.style.top = (rect.bottom - parentRect.top + 6) + "px";

    const title = document.createElement("h4");
    title.textContent = (LABELS[clip.kind] || clip.kind)
      + "  ·  " + clip.id.slice(0, 8);
    panelEl.appendChild(title);

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "mv-fx-settings-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", closePanel);
    panelEl.appendChild(closeBtn);

    // t_start / duration always render first so the user can nudge numbers
    // when dragging is too coarse.
    ["t_start", "duration_s"].forEach((field) => {
      const row = document.createElement("div");
      row.className = "mv-fx-row-field";
      const lab = document.createElement("label");
      lab.textContent = field;
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "0.01";
      inp.value = String(clip[field]);
      inp.addEventListener("input", () => {
        const v = parseFloat(inp.value);
        if (!isFinite(v)) return;
        if (field === "t_start") clip.t_start = Math.max(0, v);
        else clip.duration_s = Math.max(0.02, v);
        updateClipDom(clip.id);
      });
      row.appendChild(lab); row.appendChild(inp);
      panelEl.appendChild(row);
    });

    keys.forEach((k) => {
      const row = document.createElement("div");
      row.className = "mv-fx-row-field";
      const lab = document.createElement("label");
      lab.textContent = k;
      const inp = document.createElement("input");
      clip.settings = clip.settings || {};
      const cur = clip.settings[k];
      if (k.endsWith("_hex")) {
        inp.type = "color";
        inp.value = (typeof cur === "string" && /^#/.test(cur))
          ? cur : "#ffffff";
      } else if (k.endsWith("_mode")) {
        inp.type = "text";
        inp.value = (cur == null) ? "" : String(cur);
      } else {
        inp.type = "number";
        inp.step = "0.01";
        inp.value = (typeof cur === "number") ? String(cur)
                    : (cur == null ? "" : String(cur));
      }
      inp.addEventListener("input", () => {
        clip.settings = clip.settings || {};
        if (inp.type === "number") {
          const v = parseFloat(inp.value);
          // Writing NaN would fail server-side validation; keep the previous
          // value intact until the user types a parseable number.
          if (isFinite(v)) clip.settings[k] = v;
        } else if (inp.type === "color") {
          clip.settings[k] = inp.value;
        } else {
          clip.settings[k] = inp.value;
        }
      });
      row.appendChild(lab); row.appendChild(inp);
      panelEl.appendChild(row);
    });

    container.appendChild(panelEl);
    // Defer listener attach one tick so the click that opened the panel
    // doesn't immediately close it via the outside-click guard.
    setTimeout(() => {
      document.addEventListener(
        "pointerdown", onDocPointerDownForPanel, true);
    }, 0);
  }

  // ── Playhead tick ─────────────────────────────────────────────────────
  function tickPlayhead() {
    const a = audio();
    if (a && !a.paused && !a.ended) {
      const x = secondsToPx(a.currentTime);
      playhead.style.left = x + "px";
      const sl = scroller.scrollLeft, sw = scroller.clientWidth;
      if (x < sl + 40) scroller.scrollLeft = Math.max(0, x - 40);
      else if (x > sl + sw - 40) scroller.scrollLeft = x - sw + 40;
    }
    requestAnimationFrame(tickPlayhead);
  }

  // ── Keyboard ──────────────────────────────────────────────────────────
  document.addEventListener("keydown", (ev) => {
    if (ev.target && /INPUT|TEXTAREA|SELECT/.test(ev.target.tagName)) return;
    if (!container.offsetParent) return;  // hidden tab
    if (ev.code === "Space") {
      ev.preventDefault();
      const a = audio(); if (a) { a.paused ? a.play() : a.pause(); }
    } else if (ev.key === "+" || ev.key === "=") {
      setZoom(pxPerSec * 1.25);
    } else if (ev.key === "-" || ev.key === "_") {
      setZoom(pxPerSec / 1.25);
    } else if (ev.key === "Escape") {
      clearSelection();
      closePanel();
    } else if (ev.key === "Delete" || ev.key === "Backspace") {
      if (selected.size > 0) {
        ev.preventDefault();
        deleteSelected();
      }
    } else if ((ev.ctrlKey || ev.metaKey)
               && (ev.key === "a" || ev.key === "A")) {
      ev.preventDefault();
      selected.clear();
      (state.clips || []).forEach((c) => selected.add(c.id));
      applySelectionStyling();
    } else if (!ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      const d = ev.key;
      if (d >= "1" && d <= "7") {
        const idx = parseInt(d, 10) - 1;
        if (KINDS[idx]) {
          ev.preventDefault();
          addClipAtPlayhead(KINDS[idx], null);
        }
      }
    }
  });

  // ── Boot ──────────────────────────────────────────────────────────────
  setStageWidth();
  applyRowDisabled();
  renderGhosts();
  renderClips();
  requestAnimationFrame(tickPlayhead);
})();
"""


__all__ = [
    "DEDUPE_TOL_S",
    "bake_auto_schedule",
    "build_editor_html",
    "build_ghost_events",
    "load_editor_state",
    "save_edited_timeline",
]
