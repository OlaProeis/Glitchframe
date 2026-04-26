"""Coordinate pipeline stages; own per-song cache lifecycle and progress mapping.

Song identity for ``cache/<song_hash>/`` is always the SHA-256 of raw uploaded
bytes from :func:`pipeline.audio_ingest.hash_audio_file` / :func:`ingest_audio_file`.
Metadata in :class:`OrchestratorInputs` is cosmetic only and must not affect the
cache key. Run-scoped paths (e.g. ``outputs/<run_id>/``) should use a separate
``run_id`` if metadata must be distinguished — never mix that into ``song_hash``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from config import song_cache_dir
from pipeline.audio_analyzer import AnalysisResult, analyze_song
from pipeline.audio_ingest import IngestResult, ingest_audio_file
from pipeline.lyrics_aligner import AlignmentResult, align_lyrics
from pipeline.metadata import write_metadata_txt
from pipeline.effects_timeline import EffectsTimeline, load as load_effects_timeline
from pipeline.preset_colors import resolve_text_colors
from pipeline.preview import DEFAULT_PREVIEW_WINDOW_SEC

if TYPE_CHECKING:  # pragma: no cover - imports used only for static typing
    from pipeline.av_sync import AvSyncReport
    from pipeline.compositor import CompositorResult

LOGGER = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class OrchestratorInputs:
    """Inputs for an orchestrated run (metadata/presets are non-cache-key)."""

    audio_path: str | Path | None = None
    song_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    lyrics_text: str | None = None
    presets: dict[str, Any] = field(default_factory=dict)
    output_settings: dict[str, Any] = field(default_factory=dict)
    # Background compositor source (see ``pipeline.background``); not part of song hash.
    background_mode: str = "animatediff"
    static_background_image: str | Path | None = None
    # Cosmetic render settings (never influence the song cache key).
    preset_id: str | None = None
    reactive_intensity_pct: float = 50.0
    logo_path: str | Path | None = None
    logo_position: str = "center"
    logo_opacity_pct: float = 85.0
    # Cap on the logo's longest edge as a percent of the shorter frame edge.
    # Keeps the slider resolution-independent (30 % reads the same on 1080p
    # and 4K); ``<= 0`` falls back to the legacy "fit inside frame" behaviour.
    logo_max_size_pct: float = 30.0
    logo_beat_pulse: bool = True
    # ``bass`` keys the pulse off low-frequency energy in ``analysis.json``
    # (kick / sub-bass hits). ``beats`` preserves the original grid-locked
    # behaviour for tracks / users who prefer an ``every beat`` kick.
    logo_pulse_mode: str = "bass"
    logo_pulse_strength: float = 2.0
    logo_pulse_sensitivity: float = 1.0
    # Mid-band (snare-ish) neon halo behind the logo; colour from preset shadow.
    logo_snare_glow: bool = True
    logo_glow_strength: float = 2.0
    logo_glow_sensitivity: float = 1.0
    logo_snare_squeeze_strength: float = 0.40
    logo_impact_glitch_strength: float = 1.0
    logo_impact_sensitivity: float = 1.0
    # Traveling-wave logo rim (``pipeline.logo_rim_lights``); cosmetic only.
    # ``logo_rim_mode``: ``off`` | ``classic`` | ``rim`` (Gradio maps labels).
    logo_rim_mode: str = "rim"
    logo_rim_travel_speed: float = 0.5
    logo_rim_color_spread_deg: float = 120.0
    logo_rim_inward_mix: float = 0.5
    logo_rim_direction: str = "cw"
    logo_rim_audio_reactive: bool = True
    logo_rim_sync_snare: bool = True
    logo_rim_sync_bass: bool = True
    logo_rim_mod_strength: float = 1.0
    # Pre-choreographed rim-light beams on drops + snare lead-ins. One
    # UI-surfaced switch (no fine-grained tuning); beat tracking / thresholds
    # live in :class:`pipeline.logo_rim_beams.BeamConfig` defaults.
    rim_beams_enabled: bool = True
    # Logo motion stability deadzone for the beat pulse + snare squeeze.
    # ``0`` = legacy linear (noisy during chill sections); ``1`` = default
    # soft deadzone; ``2`` = extra stable (snare squeeze barely moves).
    logo_motion_stability: float = 1.0
    # Rim emissive gain (1.0 = engine default; 3.0 = 300% product default).
    # Drives ``RimLightConfig.intensity`` and also lifts ``halo_boost`` when > 1
    # so the outer halo reads on screen.
    logo_rim_brightness: float = 3.0
    # Outer halo falloff radius in patch pixels; maps to ``halo_spread_px``.
    logo_rim_halo_spread_px: float = 22.0
    # Wave-shape preset keyword, resolved to ``(waves, wave_sharpness)``:
    # ``comet`` → (1, 4.0) — single distinctly travelling bright spot;
    # ``twin``  → (2, 4.0) — two tight comets;
    # ``lobes`` → (2, 2.0) — gentle twin lobes;
    # ``ring``  → (3, 1.5) — nearly uniform soft ring (previous engine default).
    logo_rim_wave_shape: str = "comet"
    # Burned-in title card. ``show_title`` is the master switch; when the
    # orchestrator can't derive a non-empty ``Artist - Title`` from metadata
    # the overlay is skipped automatically regardless of the switch.
    show_title: bool = True
    title_position: str = "bottom-left"
    title_size: str = "small"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    include_lyrics: bool = True
    preview_window_sec: float = DEFAULT_PREVIEW_WINDOW_SEC
    # Effects timeline: load ``cache/<hash>/effects_timeline.json`` into the
    # compositor when true; when false, behaviour matches pre–effects-timeline
    # renders (no per-song JSON / compositor defaults).
    effects_timeline_enabled: bool = True
    # Run-level trim (default 1.0) combined with
    # ``EffectsTimeline.auto_reactivity_master`` from the loaded file; see
    # :func:`_effects_compositor_config`.
    auto_reactivity_master: float = 1.0


@dataclass(frozen=True)
class OrchestratorState:
    song_hash: str
    cache_dir: Path
    ingest_result: IngestResult | None
    analysis: AnalysisResult
    alignment: AlignmentResult | None


@dataclass(frozen=True)
class RenderResult:
    """Return payload for :func:`orchestrate_preview_10s` / :func:`orchestrate_full_render`."""

    state: OrchestratorState
    compositor: CompositorResult
    preset_id: str | None
    preset: Mapping[str, Any] | None
    metadata_path: Path | None
    av_sync: AvSyncReport | None
    start_sec: float
    duration_sec: float | None
    is_preview: bool


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _wrap_progress(progress: ProgressFn | None, lo: float, hi: float) -> ProgressFn:
    """Map sub-stage progress ``p in [0, 1]`` to ``[lo, hi]`` on the outer bar."""

    def _inner(p: float, msg: str) -> None:
        if progress is not None:
            t = lo + (hi - lo) * _clamp01(p)
            progress(t, msg)

    return _inner


def _has_lyrics(text: str | None) -> bool:
    return bool(text and text.strip())


def _ingest_with_optional_hash_check(
    path: str | Path,
    expected_hash: str | None,
) -> IngestResult:
    result = ingest_audio_file(path)
    if expected_hash is not None and expected_hash != result.song_hash:
        raise ValueError(
            f"song_hash {expected_hash!r} does not match ingest hash {result.song_hash!r}"
        )
    return result


def orchestrate_analysis(
    inputs: OrchestratorInputs,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
    include_lyrics: bool = False,
) -> OrchestratorState:
    """
    Run ingest (when ``audio_path`` is set), then analysis, then optional lyrics
    alignment when ``include_lyrics`` is true and ``lyrics_text`` is non-empty.

    Stages use existing cache-skip logic; ``force`` is passed through to
    :func:`analyze_song` and :func:`align_lyrics` only (ingest stays hash-idempotent).
    """
    do_lyrics = include_lyrics and _has_lyrics(inputs.lyrics_text)
    need_ingest = inputs.audio_path is not None and str(inputs.audio_path).strip() != ""

    ingest_result: IngestResult | None = None

    if need_ingest:
        in0, in1 = 0.0, 0.1
        an0, an1 = (0.1, 0.7) if do_lyrics else (0.1, 1.0)
        ly0, ly1 = 0.7, 1.0
        if progress is not None:
            progress(in0, "Ingesting audio…")
        ingest_result = _ingest_with_optional_hash_check(
            inputs.audio_path,  # type: ignore[arg-type]
            inputs.song_hash,
        )
        song_hash = ingest_result.song_hash
        cache_dir = ingest_result.cache_dir
        if progress is not None:
            progress(in1, "Ingest complete")
    else:
        an0, an1 = (0.0, 0.7) if do_lyrics else (0.0, 1.0)
        ly0, ly1 = 0.7, 1.0
        h = inputs.song_hash
        if h is None or not str(h).strip():
            raise ValueError("OrchestratorInputs requires audio_path or song_hash")
        song_hash = h
        cache_dir = song_cache_dir(song_hash)
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Cache dir does not exist: {cache_dir}")

    def _outer(p: float, msg: str) -> None:
        if progress is not None:
            progress(_clamp01(p), msg)

    analysis_cb = _wrap_progress(_outer, an0, an1)
    analysis = analyze_song(cache_dir, force=force, progress=analysis_cb)

    alignment: AlignmentResult | None = None
    if do_lyrics:
        lyrics_cb = _wrap_progress(_outer, ly0, ly1)
        alignment = align_lyrics(
            cache_dir,
            inputs.lyrics_text or "",
            force=force,
            progress=lyrics_cb,
        )

    return OrchestratorState(
        song_hash=song_hash,
        cache_dir=cache_dir,
        ingest_result=ingest_result,
        analysis=analysis,
        alignment=alignment,
    )


def write_run_metadata(
    run_output_dir: str | Path,
    *,
    inputs: OrchestratorInputs,
    analysis_doc: Mapping[str, Any],
    alignment: AlignmentResult | None = None,
    preset_id: str | None = None,
    preset: Mapping[str, Any] | None = None,
) -> Path:
    """
    Write ``outputs/<run_id>/metadata.txt`` for upload / copy-paste.

    Call after a full render once ``run_output_dir`` exists. Resolves preset
    tags from ``preset_id`` / ``preset`` or from ``inputs.presets``.
    """
    meta = dict(inputs.metadata or {})
    presets = inputs.presets or {}
    pid = preset_id or presets.get("preset_id") or presets.get("id")
    pdict: Mapping[str, Any] | None = preset
    if pdict is None:
        raw_preset = presets.get("preset")
        if isinstance(raw_preset, dict):
            pdict = raw_preset
    if pdict is None and pid:
        try:
            from config import get_preset

            pdict = get_preset(str(pid))
        except KeyError:
            pdict = None

    lyrics_lines: list[str] | None = None
    if alignment is not None:
        lyrics_lines = list(alignment.lines)
    elif inputs.lyrics_text and str(inputs.lyrics_text).strip():
        lyrics_lines = [ln for ln in str(inputs.lyrics_text).splitlines()]

    return write_metadata_txt(
        run_output_dir,
        song_metadata=meta,
        analysis=dict(analysis_doc),
        preset_id=str(pid) if pid else None,
        preset=dict(pdict) if pdict is not None else None,
        lyrics_lines=lyrics_lines,
    )


def _resolve_preset(
    inputs: OrchestratorInputs,
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Pull ``(preset_id, preset_dict)`` from ``inputs`` (typed field or legacy dict)."""
    from config import get_preset

    pid = inputs.preset_id or inputs.presets.get("preset_id") or inputs.presets.get("id")
    preset: Mapping[str, Any] | None = None
    raw = inputs.presets.get("preset") if isinstance(inputs.presets, dict) else None
    if isinstance(raw, dict):
        preset = raw
    if preset is None and pid:
        try:
            preset = get_preset(str(pid))
        except KeyError:
            preset = None
    return (str(pid) if pid else None), preset


def _thumbnail_line_from_metadata(meta: Mapping[str, Any] | None) -> str | None:
    if not meta:
        return None
    artist = str(meta.get("artist") or "").strip()
    title = str(meta.get("title") or "").strip()
    if artist and title:
        return f"{artist} - {title}"
    return title or artist or None


# Rim wave-shape presets → ``(waves, wave_sharpness)``. Kept small & explicit
# so the UI dropdown stays authoritative; unknown labels fall back to ``comet``
# because a single distinct travelling lobe is the most visible motion cue.
_RIM_WAVE_SHAPES: dict[str, tuple[int, float]] = {
    "comet": (1, 4.0),
    "twin": (2, 4.0),
    "lobes": (2, 2.0),
    "ring": (3, 1.5),
}


def _resolve_rim_wave_shape(raw: str | None) -> tuple[int, float]:
    """Return ``(waves, wave_sharpness)`` for a UI wave-shape keyword."""
    key = (raw or "").strip().lower()
    return _RIM_WAVE_SHAPES.get(key, _RIM_WAVE_SHAPES["comet"])


def resolve_logo_rim_compositor_fields(inputs: OrchestratorInputs) -> dict[str, Any]:
    """Map :class:`OrchestratorInputs` rim branding fields to :class:`CompositorConfig` kwargs.

    Used by :func:`_render_pipeline` and unit tests. Does not depend on analysis
    or preset colours (tint still comes from the compositor's shadow/base hex).
    """
    from pipeline.logo_composite import LogoGlowMode
    from pipeline.logo_rim_lights import RimLightConfig

    raw = (inputs.logo_rim_mode or "off").strip().lower()
    if raw in ("rim", "new", "traveling", "travel", "on", "yes"):
        mode = "rim"
    elif raw in ("classic", "classic_neon", "neon_only"):
        mode = "classic"
    else:
        mode = "off"

    if mode == "off":
        rim_enabled = False
        glow_mode = LogoGlowMode.AUTO
    elif mode == "classic":
        rim_enabled = False
        glow_mode = LogoGlowMode.CLASSIC
    else:
        rim_enabled = True
        glow_mode = LogoGlowMode.AUTO

    rim_cfg: RimLightConfig | None = None
    if rim_enabled:
        dirc = (inputs.logo_rim_direction or "cw").strip().lower()
        ccw = dirc in (
            "ccw",
            "counterclockwise",
            "counter-clockwise",
            "anticlockwise",
        )
        sign = -1.0 if ccw else 1.0
        speed = max(0.0, min(2.0, float(inputs.logo_rim_travel_speed)))
        phase_hz = sign * speed
        spread_deg = max(0.0, min(180.0, float(inputs.logo_rim_color_spread_deg)))
        spread_rad = spread_deg * (pi / 180.0)
        layers = 1 if spread_deg < 0.5 else 2
        inward = max(0.0, min(1.0, float(inputs.logo_rim_inward_mix)))
        # Brightness ≥ 0 scales emissive gain directly; the outer halo weight
        # only ever lifts above the engine default (never dimmed below 1.0) so
        # the ring stays anchored to the alpha edge even at low brightness.
        brightness = max(0.0, float(inputs.logo_rim_brightness))
        halo_boost = max(1.0, brightness)
        halo_spread_px = max(4.0, min(64.0, float(inputs.logo_rim_halo_spread_px)))
        waves, wave_sharpness = _resolve_rim_wave_shape(inputs.logo_rim_wave_shape)
        rim_cfg = RimLightConfig(
            phase_hz=phase_hz,
            color_spread_rad=spread_rad,
            rim_color_layers=layers,
            inward_mix=inward,
            intensity=brightness,
            halo_boost=halo_boost,
            halo_spread_px=halo_spread_px,
            waves=waves,
            wave_sharpness=wave_sharpness,
        )

    mod_strength = max(0.0, min(2.0, float(inputs.logo_rim_mod_strength)))

    return {
        "logo_rim_enabled": rim_enabled,
        "logo_glow_mode": glow_mode,
        "logo_rim_light_config": rim_cfg,
        "logo_rim_audio_reactive": bool(inputs.logo_rim_audio_reactive),
        "logo_rim_sync_snare": bool(inputs.logo_rim_sync_snare),
        "logo_rim_sync_bass": bool(inputs.logo_rim_sync_bass),
        "logo_rim_mod_strength": mod_strength,
    }


def _effects_compositor_config(
    inputs: OrchestratorInputs, cache_dir: Path
) -> tuple[EffectsTimeline | None, float]:
    """
    When ``inputs.effects_timeline_enabled``, load ``effects_timeline.json``
    and return ``(EffectsTimeline, effective_master)`` for
    :class:`pipeline.compositor.CompositorConfig`. When disabled, return
    ``(None, 1.0)`` (legacy render path, no JSON).

    **Master merge:** the compositor reads a single
    :attr:`CompositorConfig.auto_reactivity_master` scalar. The on-disk
    :attr:`EffectsTimeline.auto_reactivity_master` (Gradio / editor save) is
    multiplied by :attr:`OrchestratorInputs.auto_reactivity_master` (run
    default ``1.0``). The loaded timeline object still stores the file value
    for round-trip; the product is what scales auto reactivity in the engine.
    """
    if not bool(inputs.effects_timeline_enabled):
        return (None, 1.0)
    tl = load_effects_timeline(cache_dir)
    in_m = max(0.0, float(inputs.auto_reactivity_master))
    eff = in_m * float(tl.auto_reactivity_master)
    return (tl, eff)


def _render_pipeline(
    inputs: OrchestratorInputs,
    *,
    is_preview: bool,
    force: bool,
    progress: ProgressFn | None,
) -> RenderResult:
    """Shared implementation behind preview and full render.

    Heavy dependencies (moderngl, skia, ffmpeg helpers) are imported lazily so
    :mod:`orchestrator` stays importable in lightweight environments (tests,
    metadata-only scripts) that do not have the full GPU / compositor stack.
    """
    from config import new_run_id
    from pipeline.av_sync import DEFAULT_AV_SYNC_TOLERANCE_MS, ffprobe_av_sync
    from pipeline.background import (
        MODE_STATIC_KENBURNS,
        BackgroundSource,
        create_background_source,
        normalize_background_mode,
    )
    from pipeline.compositor import CompositorConfig, render_full_video
    from pipeline.kinetic_typography import DEFAULT_MOTION
    from pipeline.preview import pick_loudest_window_start
    from pipeline.reactive_shader import DEFAULT_SHADER

    mode = normalize_background_mode(inputs.background_mode)
    if mode == MODE_STATIC_KENBURNS and not (
        inputs.static_background_image
        and str(inputs.static_background_image).strip()
    ):
        raise ValueError(
            "static-kenburns background mode requires an uploaded image"
        )

    analysis_hi = 0.25 if is_preview else 0.20
    bg_lo = analysis_hi
    bg_hi = 0.45 if is_preview else 0.40
    render_lo = bg_hi
    render_hi = 0.95

    analysis_cb = _wrap_progress(progress, 0.0, analysis_hi)
    state = orchestrate_analysis(
        inputs,
        force=force,
        progress=analysis_cb,
        include_lyrics=bool(inputs.include_lyrics and _has_lyrics(inputs.lyrics_text)),
    )

    preset_id, preset = _resolve_preset(inputs)
    preset_prompt = str(preset.get("prompt")) if preset and preset.get("prompt") else ""
    palette = list(preset.get("colors") or []) if preset else None

    shader_name = str(preset.get("shader")) if preset and preset.get("shader") else DEFAULT_SHADER
    typo_motion = (
        str(preset.get("typo_style"))
        if preset and preset.get("typo_style")
        else DEFAULT_MOTION
    )
    colors = list(preset.get("colors") or []) if preset else []
    # Preset palettes are ordered dark -> bright for the shader (u_palette[5]).
    # Text needs the opposite: the brightest entry for the fill and a
    # saturated mid-tone for the glow. ``resolve_text_colors`` centralises
    # that rule so title overlay, kinetic typography, and the thumbnail all
    # agree on a single readable pair.
    base_color, shadow_color = resolve_text_colors(colors)
    intensity = max(0.0, min(1.0, float(inputs.reactive_intensity_pct) / 100.0))

    logo_path_obj: Path | None = None
    if inputs.logo_path is not None and str(inputs.logo_path).strip():
        logo_path_obj = Path(str(inputs.logo_path))

    title_line = (
        _thumbnail_line_from_metadata(inputs.metadata) if inputs.show_title else None
    )

    eff_timeline, eff_auto_master = _effects_compositor_config(
        inputs, state.cache_dir
    )

    cfg = CompositorConfig(
        fps=int(inputs.fps),
        width=int(inputs.width),
        height=int(inputs.height),
        shader_name=shader_name,
        intensity=intensity,
        shader_palette=colors or None,
        typography_motion=typo_motion,
        base_color=base_color,
        shadow_color=shadow_color,
        logo_path=logo_path_obj,
        logo_position=inputs.logo_position,
        logo_opacity_pct=float(inputs.logo_opacity_pct),
        logo_max_size_pct=float(inputs.logo_max_size_pct),
        logo_beat_pulse=bool(inputs.logo_beat_pulse),
        logo_pulse_mode=str(inputs.logo_pulse_mode),
        logo_pulse_strength=float(inputs.logo_pulse_strength),
        logo_pulse_sensitivity=float(inputs.logo_pulse_sensitivity),
        logo_snare_glow=bool(inputs.logo_snare_glow),
        logo_glow_strength=float(inputs.logo_glow_strength),
        logo_glow_sensitivity=float(inputs.logo_glow_sensitivity),
        logo_snare_squeeze_strength=float(inputs.logo_snare_squeeze_strength),
        logo_impact_glitch_strength=float(inputs.logo_impact_glitch_strength),
        logo_impact_sensitivity=float(inputs.logo_impact_sensitivity),
        **resolve_logo_rim_compositor_fields(inputs),
        rim_beams_enabled=bool(inputs.rim_beams_enabled),
        logo_motion_stability=max(0.0, float(inputs.logo_motion_stability)),
        title_text=title_line,
        title_position=str(inputs.title_position),
        title_size=str(inputs.title_size),
        effects_timeline=eff_timeline,
        auto_reactivity_master=float(eff_auto_master),
    )
    if preset_id == "voidcat-laser":
        from pipeline.voidcat_ascii import build_voidcat_ascii_context

        _ad = dict(state.analysis.analysis)
        _ad.setdefault("song_hash", state.song_hash)
        cfg.voidcat_ascii_ctx = build_voidcat_ascii_context(
            _ad, list(colors) if colors else None
        )
        cfg.voidcat_ascii_sharp_cat = True

    bg_cb = _wrap_progress(progress, bg_lo, bg_hi)
    if progress is not None:
        progress(bg_lo, "Preparing background…")
    background: BackgroundSource = create_background_source(
        mode,
        state.cache_dir,
        preset_id=preset_id or "default",
        preset_prompt=preset_prompt,
        static_image_path=inputs.static_background_image,
        width=cfg.width,
        height=cfg.height,
    )
    try:
        background.ensure(force=False, progress=bg_cb)

        window_sec: float | None
        t_start: float
        if is_preview:
            window_sec = float(inputs.preview_window_sec)
            t_start = pick_loudest_window_start(
                state.analysis.analysis, window_sec=window_sec
            )
            if progress is not None:
                progress(
                    bg_hi,
                    f"Preview window: {window_sec:.1f}s @ t={t_start:.1f}s",
                )
        else:
            window_sec = None
            t_start = 0.0

        rid_base = new_run_id(song_hash=state.song_hash)
        rid = f"preview_{rid_base}" if is_preview else rid_base

        thumb_line = (
            None if is_preview else _thumbnail_line_from_metadata(inputs.metadata)
        )

        render_cb = _wrap_progress(progress, render_lo, render_hi)
        aligned_words = state.alignment.words if state.alignment is not None else None

        if progress is not None:
            progress(
                render_lo,
                "Compositing video (frames + encode) — GPU memory freed after AnimateDiff…",
            )

        compositor = render_full_video(
            state.cache_dir,
            background=background,
            analysis=state.analysis.analysis,
            aligned_words=aligned_words,
            run_id=rid,
            config=cfg,
            progress=render_cb,
            thumbnail_line=thumb_line,
            thumbnail_palette=palette if thumb_line else None,
            start_sec=t_start,
            duration_sec=window_sec,
        )
    finally:
        try:
            background.close()
        except Exception as exc:  # noqa: BLE001 - background close is best-effort
            LOGGER.debug("Background close raised: %s", exc)

    metadata_path: Path | None = None
    if not is_preview:
        if progress is not None:
            progress(0.96, "Writing metadata.txt…")
        try:
            metadata_path = write_run_metadata(
                compositor.output_dir,
                inputs=inputs,
                analysis_doc=state.analysis.analysis,
                alignment=state.alignment,
                preset_id=preset_id,
                preset=preset,
            )
        except Exception as exc:  # noqa: BLE001 - log but keep render successful
            LOGGER.warning("Failed to write metadata.txt: %s", exc)

    if progress is not None:
        progress(0.98, "Validating A/V sync…")
    av_sync: AvSyncReport | None = None
    try:
        av_sync = ffprobe_av_sync(
            compositor.output_mp4,
            tolerance_ms=DEFAULT_AV_SYNC_TOLERANCE_MS,
        )
    except Exception as exc:  # noqa: BLE001 - sync check is advisory
        LOGGER.warning("A/V sync check raised: %s", exc)

    if progress is not None:
        progress(1.0, "Done")

    return RenderResult(
        state=state,
        compositor=compositor,
        preset_id=preset_id,
        preset=preset,
        metadata_path=metadata_path,
        av_sync=av_sync,
        start_sec=float(t_start),
        duration_sec=float(window_sec) if window_sec is not None else None,
        is_preview=is_preview,
    )


def orchestrate_preview_10s(
    inputs: OrchestratorInputs,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> RenderResult:
    """
    Render a short preview (~``inputs.preview_window_sec`` seconds) starting
    at the loudest RMS window of the track. Reuses all cached artifacts
    (analysis, lyrics alignment, background) so re-runs stay fast.
    """
    return _render_pipeline(inputs, is_preview=True, force=force, progress=progress)


def orchestrate_full_render(
    inputs: OrchestratorInputs,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> RenderResult:
    """
    Render the full music video to ``outputs/<run_id>/output.mp4`` and write
    ``metadata.txt`` alongside. Runs a post-encode ffprobe A/V sync check and
    includes the report in :class:`RenderResult`.
    """
    return _render_pipeline(inputs, is_preview=False, force=force, progress=progress)
