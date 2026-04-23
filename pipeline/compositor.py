"""
Frame compositor pipeline.

Per-frame loop that pulls/interpolates the background, runs the reactive
shader pass on top of it, blends kinetic-typography and optional logo
layers, and streams raw ``bgr24`` bytes to an ``ffmpeg`` process (NVENC by
default). A bounded :class:`queue.Queue` between the producer thread
(rendering) and the consumer (ffmpeg ``stdin`` writer) overlaps encode with
render and provides natural backpressure.

The compositor is deliberately decoupled from the orchestrator: callers
pass in a ready :class:`~pipeline.background.BackgroundSource`, the
analysis mapping (typically loaded from ``analysis.json``), and optional
aligned lyrics / logo settings. All GPU/Skia resources are created inside
the producer thread so driver contexts do not leak across threads.
"""

from __future__ import annotations

import errno
import logging
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from config import (
    OUTPUTS_DIR,
    default_title_font_path,
    default_ui_font_path,
    new_run_id,
)
from pipeline.audio_ingest import ORIGINAL_WAV_NAME
from pipeline.background import BackgroundSource
from pipeline.ffmpeg_tools import require_ffmpeg, select_video_codec
from pipeline.kinetic_typography import (
    DEFAULT_FONT_SIZE,
    DEFAULT_MOTION,
    AlignedWord,
    KineticTypographyLayer,
)
from pipeline.beat_pulse import (
    DEFAULT_HI_DECAY_SEC,
    DEFAULT_LO_DECAY_SEC,
    DEFAULT_MID_DECAY_SEC,
    PulseTrack,
    apply_pulse_deadzone,
    beat_pulse_envelope,
    build_bass_pulse_track,
    build_hi_transient_track,
    build_lo_transient_track,
    build_logo_bass_pulse_track,
    build_mid_transient_track,
    build_rms_impact_pulse_track,
    build_snare_glow_track,
    scale_and_opacity_for_pulse,
)
from pipeline.musical_events import (
    DEFAULT_DROP_HOLD_DECAY_SEC,
    sample_drop_hold,
)
from pipeline.logo_rim_lights import (
    RimAudioModulation,
    RimAudioTuning,
    RimLightConfig,
    RimModulationState,
    advance_rim_audio_modulation,
    compute_logo_rim_prep,
    rim_base_rgb_from_preset,
)
from pipeline.logo_rim_beams import (
    BeamConfig,
    ScheduledBeam,
    compute_beam_patch,
    schedule_rim_beams,
)
from pipeline.logo_composite import (
    LogoGlowMode,
    _blend_premult_rgba_patch,
    _origin_for_position,
    composite_logo_onto_frame,
    glitch_seed_for_time,
    load_logo_rgba,
    normalize_logo_position,
    prepare_logo_rgba,
    resolve_logo_glow_rgb,
)
from pipeline.reactive_shader import (
    DEFAULT_NUM_BANDS,
    DEFAULT_SHADER,
    ReactiveShader,
    composite_premultiplied_rgba_over_rgb,
    resolve_builtin_shader_stem,
    uniforms_at_time,
)
from pipeline.title_overlay import (
    normalize_title_position,
    normalize_title_size,
    render_title_rgba,
)
from pipeline.renderer import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    _build_ffmpeg_cmd,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 4

# Smoother kick envelope for reactive shaders (longer decay than logo pulse).
DEFAULT_SHADER_BASS_SENSITIVITY = 0.72
DEFAULT_SHADER_BASS_DECAY_SEC = 0.34

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class CompositorResult:
    """Return value of :func:`render_full_video`."""

    run_id: str
    output_dir: Path
    output_mp4: Path
    frame_count: int
    audio_path: Path
    thumbnail_png: Path | None = None


@dataclass
class CompositorConfig:
    """
    All tunables for :func:`render_full_video`.

    The defaults mirror the M1 spectrum renderer (1920×1080 @ 30 fps, NVENC)
    so switching from the spectrum path to the full compositor does not
    require re-tuning the encoder.

    Reactive shader uniforms injected by this config (see
    :func:`_render_compositor_frame`):

    * ``bass_hit`` — ``shader_bass_*`` knobs → :func:`build_bass_pulse_track`.
    * ``transient_lo`` / ``transient_mid`` / ``transient_hi`` —
      ``shader_transient_*`` knobs → ``build_lo/mid/hi_transient_track`` in
      :mod:`pipeline.beat_pulse`.
    * ``drop_hold`` — ``shader_drop_hold_decay_sec`` →
      :func:`pipeline.musical_events.sample_drop_hold` over
      ``analysis["events"]["drops"]``.
    """

    fps: int = DEFAULT_FPS
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    shader_name: str = DEFAULT_SHADER
    num_bands: int = DEFAULT_NUM_BANDS
    intensity: float = 1.0
    # Preset hex palette fed to ``ReactiveShader`` as the ``u_palette`` uniform.
    # ``None`` uses :data:`pipeline.reactive_shader.DEFAULT_PALETTE`.
    shader_palette: Sequence[str] | None = None
    font_path: Path | None = field(default_factory=default_ui_font_path)
    # Heavier face for burned-in title + thumbnail line (lyrics use ``font_path``).
    title_font_path: Path | None = field(default_factory=default_title_font_path)
    font_size: float = DEFAULT_FONT_SIZE
    typography_motion: str = DEFAULT_MOTION
    base_color: str = "#FFFFFF"
    shadow_color: str | None = None
    logo_path: Path | None = None
    logo_position: str = "center"
    logo_opacity_pct: float = 100.0
    # Audio-reactive logo pulse: size + brightness kick on low-frequency hits
    # (``bass``, the default) or on every analyzer beat (``beats``). See
    # ``pipeline.beat_pulse`` for the envelope math.
    logo_beat_pulse: bool = False
    logo_pulse_mode: str = "bass"
    logo_pulse_strength: float = 2.0
    logo_pulse_sensitivity: float = 1.0
    # Snare / mid-perc reactive neon halo behind the logo (see
    # :func:`pipeline.beat_pulse.build_snare_glow_track`).
    logo_snare_glow: bool = True
    logo_glow_strength: float = 2.0
    logo_glow_sensitivity: float = 1.0
    # Mid-band envelope also drives a brief **scale contract** on snare hits
    # (independent of the neon glow toggle).
    logo_snare_squeeze_strength: float = 0.40
    # RMS jump envelope (drops / impacts) → RGB-split glitch on the logo.
    logo_impact_glitch_strength: float = 1.0
    logo_impact_sensitivity: float = 1.0
    # Pre-shaped bass/kick curve for fragment shaders (``bass_hit`` uniform).
    # Longer decay than the logo defaults so backgrounds breathe instead of
    # jittering on every spectral frame.
    shader_bass_sensitivity: float = DEFAULT_SHADER_BASS_SENSITIVITY
    shader_bass_decay_sec: float = DEFAULT_SHADER_BASS_DECAY_SEC
    # Low / mid / high band transient envelopes for fragment shaders
    # (``transient_lo``, ``transient_mid``, ``transient_hi`` uniforms). Built
    # once per render from :mod:`pipeline.beat_pulse` band-pulse helpers; the
    # decay constants default to the shader-oriented presets documented on
    # ``build_lo/mid/hi_transient_track`` (0.34 / 0.12 / 0.06 s).
    # ``shader_transient_sensitivity`` scales all three bands together, like
    # the existing ``shader_bass_sensitivity`` but applied post-rectification.
    shader_transient_sensitivity: float = 1.0
    shader_transient_lo_decay_sec: float = DEFAULT_LO_DECAY_SEC
    shader_transient_mid_decay_sec: float = DEFAULT_MID_DECAY_SEC
    shader_transient_hi_decay_sec: float = DEFAULT_HI_DECAY_SEC
    # Post-drop afterglow time constant for the ``drop_hold`` shader uniform.
    # Mirrors :data:`pipeline.musical_events.DEFAULT_DROP_HOLD_DECAY_SEC`
    # (~8 bars @ 120 bpm); override for genre-specific feel.
    shader_drop_hold_decay_sec: float = DEFAULT_DROP_HOLD_DECAY_SEC
    # Task 27--28: optional traveling-wave rim behind the logo; audio reactive
    # modulation when ``logo_rim_audio_reactive`` (same snare/bass tracks as neon).
    logo_rim_enabled: bool = True
    logo_rim_light_config: RimLightConfig | None = None
    logo_glow_mode: LogoGlowMode = LogoGlowMode.AUTO
    logo_rim_audio_reactive: bool = True
    logo_rim_sync_snare: bool = True
    logo_rim_sync_bass: bool = True
    logo_rim_mod_strength: float = 1.0
    # Pre-choreographed rim-light beams on drops + snare lead-ins; see
    # :mod:`pipeline.logo_rim_beams`. When ``rim_beams_enabled`` is true the
    # compositor pre-bakes a schedule once per render (drops + nearby snares,
    # 10 s gated) and blends a padded premultiplied RGBA patch around the logo
    # centroid on active beams.
    rim_beams_enabled: bool = True
    rim_beams_config: BeamConfig = field(default_factory=BeamConfig)
    # Logo motion stability deadzone. ``0`` = legacy (every micro-pulse moves
    # the logo); ``1`` = default soft deadzone (chill-section noise collapses
    # to zero); ``2`` = extra stable. Scales the deadzone passed to
    # :func:`scale_and_opacity_for_pulse` and the snare-squeeze gate.
    logo_motion_stability: float = 1.0
    # Persistent title/artist overlay burned into every frame. When
    # ``title_text`` is empty / None the overlay pass is skipped entirely.
    title_text: str | None = None
    title_position: str = "bottom-left"
    title_size: str = "small"
    title_opacity: float = 0.90
    video_codec: str | None = None
    queue_size: int = DEFAULT_QUEUE_SIZE


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


# 1 MiB keeps every os.write() well below any Win32 anonymous-pipe limit
# (Windows caps a single synchronous WriteFile on a pipe at ~32 MiB, and we've
# observed OSError(EINVAL) from BufferedWriter on Python 3.13 when passing a
# full 4K BGR24 frame in one call). 1 MiB is also large enough that syscall
# overhead is negligible versus the actual encode cost.
_PIPE_CHUNK_BYTES = 1 << 20


def _pipe_write_all(fd: int, data: bytes) -> None:
    """Write ``data`` to ``fd`` in small chunks, handling short writes.

    ``os.write`` on Windows pipes can return fewer bytes than requested when
    ffmpeg's read side is slower than our producer — loop until the whole
    buffer is drained. Raises :class:`BrokenPipeError` or :class:`OSError`
    (EPIPE/EINVAL) if the child process has closed its stdin.
    """
    view = memoryview(data)
    total = len(view)
    offset = 0
    while offset < total:
        end = min(offset + _PIPE_CHUNK_BYTES, total)
        written = os.write(fd, view[offset:end])
        if written <= 0:
            raise BrokenPipeError(
                errno.EPIPE, "ffmpeg stdin closed mid-write"
            )
        offset += written


def _validate_background_frame(
    arr: np.ndarray, width: int, height: int, t: float
) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"background.background_frame({t:.3f}) returned shape={arr.shape} "
            f"dtype={arr.dtype}; expected (H, W, 3) uint8"
        )
    if arr.shape[0] != height or arr.shape[1] != width:
        raise ValueError(
            f"background.background_frame({t:.3f}) returned {arr.shape[1]}×{arr.shape[0]} "
            f"but compositor is {width}×{height}"
        )
    return arr


PulseFn = Callable[[float], float]


def _shader_bass_track_for_analysis(
    analysis: Mapping[str, Any], cfg: CompositorConfig
) -> PulseTrack | None:
    return build_bass_pulse_track(
        analysis,
        sensitivity=float(cfg.shader_bass_sensitivity),
        decay_sec=float(cfg.shader_bass_decay_sec),
    )


def _shader_transient_tracks_for_analysis(
    analysis: Mapping[str, Any], cfg: CompositorConfig
) -> tuple[PulseTrack | None, PulseTrack | None, PulseTrack | None]:
    """Build the lo / mid / hi band transient envelopes for shader uniforms.

    Counterpart of :func:`_shader_bass_track_for_analysis` — one call per
    render, zero per-frame work. Each band shares ``shader_transient_sensitivity``
    but gets its own decay constant so kicks linger while hats snap off.
    Returns ``(None, None, None)`` when the analysis lacks the ``spectrum``
    block the builders need; individual entries may still be ``None`` if a
    band slice happens to be empty.
    """
    sensitivity = float(cfg.shader_transient_sensitivity)
    lo = build_lo_transient_track(
        analysis,
        sensitivity=sensitivity,
        decay_sec=float(cfg.shader_transient_lo_decay_sec),
    )
    mid = build_mid_transient_track(
        analysis,
        sensitivity=sensitivity,
        decay_sec=float(cfg.shader_transient_mid_decay_sec),
    )
    hi = build_hi_transient_track(
        analysis,
        sensitivity=sensitivity,
        decay_sec=float(cfg.shader_transient_hi_decay_sec),
    )
    return lo, mid, hi


def _drop_hold_fn_for_analysis(
    analysis: Mapping[str, Any], decay_sec: float
) -> Callable[[float], float] | None:
    """Return a scalar ``t -> drop_hold`` closure, or ``None`` when unused.

    Closes over the drops list once so the hot path is a single
    :func:`pipeline.musical_events.sample_drop_hold` call per frame with no
    per-frame dict walks. Returns ``None`` when the analysis has no drops so
    the compositor can skip the call entirely and fall back to ``0.0``.
    """
    events = analysis.get("events") or {}
    raw = events.get("drops") or []
    if not raw:
        return None
    drops: tuple[Mapping[str, Any], ...] = tuple(
        d for d in raw if isinstance(d, Mapping)
    )
    if not drops:
        return None
    tau = float(decay_sec)
    return lambda t, _d=drops, _tau=tau: sample_drop_hold(
        float(t), _d, decay_sec=_tau
    )


def _active_layers_label(
    *,
    has_typography: bool,
    has_title: bool,
    has_logo: bool,
    has_pulse: bool,
) -> str:
    """Return e.g. ``bg+shader+typo+title+logo+pulse`` for progress messages.

    The compositor always renders the background and reactive shader; the rest
    are optional. A compact label lets the UI show which stages are active so
    users can tell whether a slow run is from (say) kinetic typography or a
    heavy shader, rather than guessing.
    """
    parts = ["bg", "shader"]
    if has_typography:
        parts.append("typo")
    if has_title:
        parts.append("title")
    if has_logo:
        parts.append("logo")
    if has_pulse:
        parts.append("pulse")
    return "+".join(parts)


def _format_render_fps(fps: float) -> str:
    """Pick a compact format that still distinguishes slow renders."""
    if fps <= 0.0:
        return "0 fps"
    if fps < 1.0:
        return f"{fps:.2f} fps"
    if fps < 10.0:
        return f"{fps:.1f} fps"
    return f"{fps:.0f} fps"


def _effective_rim_light_config(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> RimLightConfig | None:
    """Single per-render rim config (``song_hash`` from analysis when omitted)."""
    if not bool(cfg.logo_rim_enabled):
        return None
    song_hash_raw = analysis.get("song_hash")
    song_hash = str(song_hash_raw) if song_hash_raw is not None else ""
    song_hash_opt: str | None = song_hash if song_hash else None
    tint = rim_base_rgb_from_preset(cfg.shadow_color, cfg.base_color)
    base = cfg.logo_rim_light_config
    if base is None:
        return RimLightConfig(rim_rgb=tint, song_hash=song_hash_opt)
    if base.song_hash is not None:
        return base
    return replace(base, song_hash=song_hash_opt)


@dataclass(frozen=True, slots=True)
class _BeamRenderContext:
    """Per-render state needed to draw the rim-beam patch each frame.

    Built once in :func:`_build_beam_render_context` (or ``None`` when beams
    are disabled / unavailable) and consumed inside the frame loop.
    """

    schedule: tuple[ScheduledBeam, ...]
    cfg: BeamConfig
    rim_rgb: tuple[int, int, int]
    color_spread_rad: float
    n_color_layers: int
    hue_drift_per_sec: float
    song_hash: str | None
    centroid_xy_base: tuple[float, float]
    logo_base_hw: tuple[int, int]


def _build_beam_render_context(
    cfg: CompositorConfig,
    analysis: Mapping[str, Any],
    *,
    logo_rgba_prepared: np.ndarray | None,
    resolved_rim_config: RimLightConfig | None,
) -> _BeamRenderContext | None:
    """Schedule beams + capture color params once per render; ``None`` to skip.

    Skips when ``rim_beams_enabled`` is false, the logo isn't available, or
    the schedule is empty (no qualifying drops / impacts in the track).
    """
    if not bool(cfg.rim_beams_enabled) or logo_rgba_prepared is None:
        return None
    beam_cfg = cfg.rim_beams_config or BeamConfig()
    if not bool(beam_cfg.enabled):
        return None

    # Rim color context: prefer the resolved rim config (so beams track the
    # same hue drift / layer count as the travelling rim) and fall back to
    # preset defaults when the rim is disabled but beams are still on.
    rim_rgb = rim_base_rgb_from_preset(cfg.shadow_color, cfg.base_color)
    if resolved_rim_config is not None:
        color_spread = float(resolved_rim_config.color_spread_rad)
        n_layers = max(1, int(resolved_rim_config.rim_color_layers))
        hue_drift = float(resolved_rim_config.hue_drift_per_sec)
    else:
        color_spread = 2.0 * math.pi / 3.0
        n_layers = 2
        hue_drift = 0.0

    song_hash_raw = analysis.get("song_hash") if isinstance(analysis, Mapping) else None
    song_hash = str(song_hash_raw) if song_hash_raw else None

    snare_track = _snare_track_for_logo(cfg, analysis)
    impact_track: PulseTrack | None = None
    if float(cfg.logo_impact_glitch_strength) > 1e-6 or bool(cfg.rim_beams_enabled):
        impact_track = build_rms_impact_pulse_track(
            analysis, sensitivity=float(cfg.logo_impact_sensitivity)
        )
    schedule = schedule_rim_beams(
        analysis,
        snare_track=snare_track,
        impact_track=impact_track,
        cfg=beam_cfg,
        song_hash=song_hash,
        n_color_layers=n_layers,
    )
    if not schedule:
        return None

    prep = compute_logo_rim_prep(logo_rgba_prepared)
    lh, lw = int(logo_rgba_prepared.shape[0]), int(logo_rgba_prepared.shape[1])
    return _BeamRenderContext(
        schedule=tuple(schedule),
        cfg=beam_cfg,
        rim_rgb=rim_rgb,
        color_spread_rad=color_spread,
        n_color_layers=n_layers,
        hue_drift_per_sec=hue_drift,
        song_hash=song_hash,
        centroid_xy_base=(float(prep.centroid_xy[0]), float(prep.centroid_xy[1])),
        logo_base_hw=(lh, lw),
    )


def _draw_beam_patch_onto_frame(
    dst: np.ndarray,
    t: float,
    *,
    ctx: _BeamRenderContext,
    position: str,
    logo_scale: float,
) -> None:
    """Blend active beams onto ``dst`` (uint8 RGB) in place."""
    fh, fw = int(dst.shape[0]), int(dst.shape[1])
    base_h, base_w = ctx.logo_base_hw
    scale = max(1e-3, float(logo_scale))
    lh = max(1, int(round(base_h * scale)))
    lw = max(1, int(round(base_w * scale)))
    x0, y0 = _origin_for_position(position, fh, fw, lh, lw)
    cx = x0 + ctx.centroid_xy_base[0] * scale
    cy = y0 + ctx.centroid_xy_base[1] * scale

    result = compute_beam_patch(
        (fh, fw),
        centroid_xy=(cx, cy),
        t=float(t),
        scheduled=ctx.schedule,
        rim_rgb=ctx.rim_rgb,
        cfg=ctx.cfg,
        color_spread_rad=ctx.color_spread_rad,
        song_hash=ctx.song_hash,
        hue_drift_per_sec=ctx.hue_drift_per_sec,
        n_color_layers=ctx.n_color_layers,
    )
    if result is None:
        return
    _blend_premult_rgba_patch(dst, result.patch, int(result.x0), int(result.y0))


def _render_compositor_frame(
    t: float,
    *,
    background: BackgroundSource,
    analysis: Mapping[str, Any],
    cfg: CompositorConfig,
    reactive: ReactiveShader,
    typo_layer: KineticTypographyLayer | None,
    logo_rgba_prepared: np.ndarray | None,
    logo_position_norm: str | None,
    title_rgba: np.ndarray | None = None,
    pulse_fn: PulseFn | None = None,
    shader_bass_track: PulseTrack | None = None,
    shader_transient_lo_track: PulseTrack | None = None,
    shader_transient_mid_track: PulseTrack | None = None,
    shader_transient_hi_track: PulseTrack | None = None,
    drop_hold_fn: PulseFn | None = None,
    snare_fn: PulseFn | None = None,
    impact_fn: PulseFn | None = None,
    rim_mod_step: RimModStepFn | None = None,
    resolved_rim_config: RimLightConfig | None = None,
    beam_ctx: _BeamRenderContext | None = None,
) -> np.ndarray:
    """One RGB frame: background → reactive → typography → title → logo.

    Title and logo are drawn last so they always sit on top of the reactive
    shader and lyrics. Logo goes *above* the title so branding is never
    occluded by an artist/song label that shares its edge.
    """
    bg_rgb = _validate_background_frame(
        background.background_frame(t),
        cfg.width,
        cfg.height,
        t,
    )
    uniforms = uniforms_at_time(
        analysis,
        float(t),
        num_bands=cfg.num_bands,
        intensity=cfg.intensity,
    )
    if shader_bass_track is not None:
        uniforms["bass_hit"] = float(shader_bass_track.value_at(float(t)))
    else:
        uniforms["bass_hit"] = 0.0
    if shader_transient_lo_track is not None:
        uniforms["transient_lo"] = float(
            shader_transient_lo_track.value_at(float(t))
        )
    else:
        uniforms["transient_lo"] = 0.0
    if shader_transient_mid_track is not None:
        uniforms["transient_mid"] = float(
            shader_transient_mid_track.value_at(float(t))
        )
    else:
        uniforms["transient_mid"] = 0.0
    if shader_transient_hi_track is not None:
        uniforms["transient_hi"] = float(
            shader_transient_hi_track.value_at(float(t))
        )
    else:
        uniforms["transient_hi"] = 0.0
    if drop_hold_fn is not None:
        uniforms["drop_hold"] = float(drop_hold_fn(float(t)))
    else:
        uniforms["drop_hold"] = 0.0
    composited = reactive.render_frame_composited_rgb(uniforms, bg_rgb)
    if typo_layer is not None:
        typo_rgba = typo_layer.render_frame(float(t), uniforms)
        composited = composite_premultiplied_rgba_over_rgb(typo_rgba, composited)
    if title_rgba is not None:
        composited = composite_premultiplied_rgba_over_rgb(title_rgba, composited)
    if logo_rgba_prepared is not None and logo_position_norm is not None:
        logo_scale = 1.0
        logo_opacity_pct = float(cfg.logo_opacity_pct)
        stability = max(0.0, float(cfg.logo_motion_stability))
        pulse_deadzone = 0.12 * stability
        if pulse_fn is not None:
            # The pulse function is pre-built once per render (``beats`` grid
            # or ``bass`` envelope) and already encodes the mode + analysis
            # shape, so the hot path stays a single scalar lookup per frame.
            pulse = pulse_fn(float(t))
            logo_scale, opacity_mul = scale_and_opacity_for_pulse(
                pulse,
                strength=float(cfg.logo_pulse_strength),
                deadzone=pulse_deadzone,
            )
            logo_opacity_pct = max(
                0.0, min(100.0, float(cfg.logo_opacity_pct) * opacity_mul)
            )
        snare_val = 0.0
        if snare_fn is not None:
            snare_val = float(snare_fn(float(t)))
        if snare_fn is not None and float(cfg.logo_snare_squeeze_strength) > 1e-6:
            sq = float(cfg.logo_snare_squeeze_strength)
            # Apply the same deadzone to the snare-squeeze so mid-band noise
            # during chill sections doesn't keep nudging the logo smaller.
            sv = apply_pulse_deadzone(snare_val, deadzone=pulse_deadzone)
            logo_scale *= max(0.68, 1.0 - sq * sv * 0.42)
        glow_amt = 0.0
        if (
            cfg.logo_snare_glow
            and snare_fn is not None
            and float(cfg.logo_glow_strength) > 1e-6
        ):
            glow_amt = snare_val * float(cfg.logo_glow_strength)
        glow_rgb = resolve_logo_glow_rgb(cfg.shadow_color, cfg.base_color)
        glitch_amt = 0.0
        glitch_seed = 0
        if impact_fn is not None and float(cfg.logo_impact_glitch_strength) > 1e-6:
            imp = float(impact_fn(float(t)))
            g = max(0.0, min(1.0, imp)) * float(cfg.logo_impact_glitch_strength)
            if g > 1e-4:
                glitch_amt = g
                glitch_seed = glitch_seed_for_time(
                    str(analysis.get("song_hash") or ""), float(t)
                )
        rim_audio_mod: RimAudioModulation | None = None
        if rim_mod_step is not None:
            rim_audio_mod = rim_mod_step(float(t))
            LOGGER.debug(
                "Rim audio modulation t=%.4fs glow=%.3f phase=%.3frad inward=%.3f",
                float(t),
                rim_audio_mod.glow_strength_mul,
                rim_audio_mod.phase_offset_rad,
                rim_audio_mod.inward_strength_mul,
            )
        composited = composite_logo_onto_frame(
            composited,
            logo_rgba_prepared,
            logo_position_norm,
            logo_opacity_pct,
            inplace=True,
            scale=logo_scale,
            glow_amount=glow_amt,
            glow_rgb=glow_rgb if glow_amt > 1e-5 else None,
            glitch_amount=glitch_amt,
            glitch_seed=glitch_seed,
            t_sec=float(t),
            song_hash=str(analysis.get("song_hash") or "") or None,
            rim_light_config=resolved_rim_config,
            rim_audio_mod=rim_audio_mod,
            logo_glow_mode=cfg.logo_glow_mode,
        )
        if beam_ctx is not None:
            _draw_beam_patch_onto_frame(
                composited,
                float(t),
                ctx=beam_ctx,
                position=logo_position_norm,
                logo_scale=float(logo_scale),
            )
    return composited


def _prerender_title_layer(cfg: CompositorConfig) -> np.ndarray | None:
    """Rasterise the configured title line into a cached premultiplied RGBA.

    Returns ``None`` when the title is blank so the per-frame render path
    can short-circuit without checking for an empty array.
    """
    text = (cfg.title_text or "").strip()
    if not text:
        return None
    title_face = cfg.title_font_path if cfg.title_font_path is not None else cfg.font_path
    return render_title_rgba(
        text,
        width=cfg.width,
        height=cfg.height,
        font_path=title_face,
        font_size=cfg.font_size,
        size=normalize_title_size(cfg.title_size),
        position=normalize_title_position(cfg.title_position),
        fill_hex=cfg.base_color,
        shadow_hex=cfg.shadow_color,
        alpha=float(cfg.title_opacity),
    )


PULSE_MODE_BASS = "bass"
PULSE_MODE_BEATS = "beats"
_PULSE_MODES = (PULSE_MODE_BASS, PULSE_MODE_BEATS)


def _normalize_pulse_mode(raw: str | None) -> str:
    mode = (raw or PULSE_MODE_BASS).strip().lower()
    if mode not in _PULSE_MODES:
        raise ValueError(
            f"Unknown logo_pulse_mode {raw!r}; expected one of {_PULSE_MODES}"
        )
    return mode


def _beats_from_analysis(analysis: Mapping[str, Any]) -> tuple[list[float], float | None]:
    raw_beats = analysis.get("beats")
    beats: list[float] = []
    if isinstance(raw_beats, list):
        for b in raw_beats:
            try:
                beats.append(float(b))
            except (TypeError, ValueError):
                continue
    tempo = analysis.get("tempo")
    bpm_raw: Any = None
    if isinstance(tempo, Mapping):
        bpm_raw = tempo.get("bpm")
    if bpm_raw is None:
        bpm_raw = analysis.get("bpm")
    try:
        bpm = float(bpm_raw) if bpm_raw is not None else None
    except (TypeError, ValueError):
        bpm = None
    return beats, bpm


def _build_pulse_fn(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> PulseFn | None:
    """Return a ``t -> [0, 1]`` callable for the active pulse mode.

    Returns ``None`` (so the frame loop can skip the envelope call entirely)
    when the feature is disabled, when the pulse mode doesn't have usable
    analysis features (e.g. ``bass`` mode on an analyzer output that lacks a
    spectrum), or when the user picked ``beats`` mode but the tracker
    produced no beats.
    """
    if not cfg.logo_beat_pulse:
        return None
    mode = _normalize_pulse_mode(cfg.logo_pulse_mode)
    if mode == PULSE_MODE_BASS:
        track = build_logo_bass_pulse_track(
            analysis, sensitivity=float(cfg.logo_pulse_sensitivity)
        )
        if track is None:
            return None
        return track.value_at
    # ``beats`` mode: preserve the original grid-locked behaviour.
    beats, bpm = _beats_from_analysis(analysis)
    if not beats:
        return None
    beats_tuple: tuple[float, ...] = tuple(beats)
    return lambda t, _b=beats_tuple, _bpm=bpm: beat_pulse_envelope(
        float(t), _b, bpm=_bpm
    )


def _snare_track_for_logo(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> PulseTrack | None:
    need_glow = bool(cfg.logo_snare_glow) and float(cfg.logo_glow_strength) > 1e-6
    need_squeeze = float(cfg.logo_snare_squeeze_strength) > 1e-6
    if not need_glow and not need_squeeze:
        return None
    return build_snare_glow_track(
        analysis, sensitivity=float(cfg.logo_glow_sensitivity)
    )


def _snare_envelope_fn(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> PulseFn | None:
    track = _snare_track_for_logo(cfg, analysis)
    if track is None:
        return None
    return track.value_at


def _impact_envelope_fn(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> PulseFn | None:
    if float(cfg.logo_impact_glitch_strength) <= 1e-6:
        return None
    track = build_rms_impact_pulse_track(
        analysis, sensitivity=float(cfg.logo_impact_sensitivity)
    )
    if track is None:
        return None
    return track.value_at


RimModStepFn = Callable[[float], RimAudioModulation]


def _create_rim_modulation_stepper(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> RimModStepFn | None:
    """Per-render closure: absolute ``t`` → rim scalers; mutates a single state.

    When ``logo_rim_audio_reactive`` is false, returns ``None`` (compositor
    skips; behaviour unchanged from before task 27).
    """
    if not bool(cfg.logo_rim_audio_reactive):
        return None
    state = RimModulationState()
    snare_t: PulseTrack | None = None
    if bool(cfg.logo_rim_sync_snare):
        snare_t = _snare_track_for_logo(cfg, analysis)
    bass_t: PulseTrack | None = None
    if bool(cfg.logo_rim_sync_bass):
        bt = build_logo_bass_pulse_track(
            analysis, sensitivity=float(cfg.logo_pulse_sensitivity)
        )
        bass_t = bt
    dt = 1.0 / max(1, float(cfg.fps))
    strength = max(0.0, float(cfg.logo_rim_mod_strength))
    tuning = RimAudioTuning(global_strength=strength)

    def step(t: float) -> RimAudioModulation:
        s = float(snare_t.value_at(t)) if snare_t is not None else 0.0
        b = float(bass_t.value_at(t)) if bass_t is not None else 0.0
        return advance_rim_audio_modulation(
            state,
            snare_env=s,
            bass_env=b,
            dt_sec=dt,
            tuning=tuning,
        )

    return step


def render_single_frame(
    t: float,
    *,
    background: BackgroundSource,
    analysis: Mapping[str, Any],
    aligned_words: Sequence[AlignedWord] | None = None,
    config: CompositorConfig | None = None,
) -> np.ndarray:
    """
    Render a single fully composited RGB frame at time ``t`` (seconds).

    Uses the same stacking order as :func:`render_full_video`: background,
    reactive shader, optional kinetic typography, optional logo. The returned
    ``(H, W, 3) uint8`` array is a copy owned by the caller.
    """
    cfg = config or CompositorConfig()
    if cfg.width <= 0 or cfg.height <= 0:
        raise ValueError(f"Invalid resolution: {cfg.width}x{cfg.height}")
    if cfg.num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {cfg.num_bands}")

    bg_w, bg_h = background.size
    if bg_w != cfg.width or bg_h != cfg.height:
        raise ValueError(
            f"Background size {bg_w}×{bg_h} does not match compositor "
            f"{cfg.width}×{cfg.height}"
        )

    resolve_builtin_shader_stem(cfg.shader_name)

    logo_position_norm: str | None = None
    logo_rgba_prepared: np.ndarray | None = None
    if cfg.logo_path is not None and str(cfg.logo_path).strip():
        logo_position_norm = normalize_logo_position(cfg.logo_position)
        raw_logo = load_logo_rgba(cfg.logo_path)
        logo_rgba_prepared = prepare_logo_rgba(raw_logo, cfg.height, cfg.width)

    title_rgba = _prerender_title_layer(cfg)
    pulse_fn = _build_pulse_fn(cfg, analysis)
    shader_bass_track = _shader_bass_track_for_analysis(analysis, cfg)
    (
        shader_transient_lo_track,
        shader_transient_mid_track,
        shader_transient_hi_track,
    ) = _shader_transient_tracks_for_analysis(analysis, cfg)
    drop_hold_fn = _drop_hold_fn_for_analysis(
        analysis, float(cfg.shader_drop_hold_decay_sec)
    )
    snare_fn = _snare_envelope_fn(cfg, analysis)
    impact_fn = _impact_envelope_fn(cfg, analysis)
    rim_mod_step = _create_rim_modulation_stepper(cfg, analysis)
    resolved_rim_config = _effective_rim_light_config(cfg, analysis)
    beam_ctx = _build_beam_render_context(
        cfg,
        analysis,
        logo_rgba_prepared=logo_rgba_prepared,
        resolved_rim_config=resolved_rim_config,
    )

    has_typography = bool(aligned_words)

    with ReactiveShader(
        cfg.shader_name,
        width=cfg.width,
        height=cfg.height,
        num_bands=cfg.num_bands,
        palette=cfg.shader_palette,
    ) as reactive:
        typo_layer: KineticTypographyLayer | None = None
        if has_typography:
            typo_layer = KineticTypographyLayer(
                list(aligned_words or ()),
                motion=cfg.typography_motion,
                font_path=cfg.font_path,
                width=cfg.width,
                height=cfg.height,
                font_size=cfg.font_size,
                base_color=cfg.base_color,
                shadow_color=cfg.shadow_color,
            )
        try:
            out = _render_compositor_frame(
                t,
                background=background,
                analysis=analysis,
                cfg=cfg,
                reactive=reactive,
                typo_layer=typo_layer,
                logo_rgba_prepared=logo_rgba_prepared,
                logo_position_norm=logo_position_norm,
                title_rgba=title_rgba,
                pulse_fn=pulse_fn,
                shader_bass_track=shader_bass_track,
                shader_transient_lo_track=shader_transient_lo_track,
                shader_transient_mid_track=shader_transient_mid_track,
                shader_transient_hi_track=shader_transient_hi_track,
                drop_hold_fn=drop_hold_fn,
                snare_fn=snare_fn,
                impact_fn=impact_fn,
                rim_mod_step=rim_mod_step,
                resolved_rim_config=resolved_rim_config,
                beam_ctx=beam_ctx,
            )
        finally:
            if typo_layer is not None:
                typo_layer.close()

    return np.ascontiguousarray(out)


def render_full_video(
    cache_dir: Path | str,
    *,
    background: BackgroundSource,
    analysis: Mapping[str, Any],
    aligned_words: Sequence[AlignedWord] | None = None,
    audio_path: Path | None = None,
    run_id: str | None = None,
    outputs_dir: Path | None = None,
    config: CompositorConfig | None = None,
    progress: ProgressFn | None = None,
    thumbnail_line: str | None = None,
    thumbnail_palette: Sequence[str] | None = None,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> CompositorResult:
    """
    Render a full-length video: background + reactive shader + kinetic
    typography + optional logo → ``outputs/<run_id>/output.mp4``.

    Parameters
    ----------
    cache_dir:
        Per-song cache directory (``cache/<song_hash>/``). Used to locate
        ``original.wav`` by default and to seed the run id.
    background:
        A prepared :class:`BackgroundSource` whose :meth:`ensure` has already
        been called. The compositor does **not** close it.
    analysis:
        ``analysis.json``-shaped mapping driving reactive uniforms.
    aligned_words:
        Word list from ``lyrics.aligned.json``; pass ``None``/empty to skip
        the typography layer entirely.
    audio_path:
        Override for the muxed audio. Defaults to ``<cache_dir>/original.wav``.
    run_id:
        Output folder name under ``outputs/``. Auto-generated when omitted.
    outputs_dir:
        Override for :data:`config.OUTPUTS_DIR` (useful for tests).
    config:
        :class:`CompositorConfig` with resolution / fps / codec / layer
        settings. Defaults apply when omitted.
    progress:
        Optional ``progress(fraction, message)`` callback. ``fraction`` is in
        ``[0, 1]`` and reports per-frame render progress.
    thumbnail_line:
        When non-empty after stripping, writes ``thumbnail.png`` (1920×1080)
        under ``outputs/<run_id>/`` after a successful encode. The image uses
        a representative frame plus this text (e.g. ``Artist — Title``).
    thumbnail_palette:
        Optional list of ``#RRGGBB`` colors (preset ``colors``); the first
        entry is the main title fill and the second (when present) is the
        shadow. When omitted, :class:`CompositorConfig` ``base_color`` /
        ``shadow_color`` are used.
    start_sec:
        Optional time offset into the audio (seconds). Frame times passed to
        the background, reactive shader, and typography layers are shifted by
        this amount, and the muxed audio is seeked to ``start_sec`` so the
        video slice starts exactly on the same sample. Defaults to ``0.0``.
    duration_sec:
        When set, render only this many seconds beginning at ``start_sec``
        (instead of the full audio). Frame count follows
        ``floor(duration_sec * fps)`` and the muxed audio is trimmed to
        match. Must be positive; must fit inside the audio's remaining
        duration after ``start_sec``.

    Raises
    ------
    FileNotFoundError
        Missing audio or cache directory.
    ValueError
        Invalid config, size mismatch between background and compositor,
        unknown shader stem, or malformed per-frame arrays.
    RuntimeError
        ``ffmpeg`` missing, encoder failure, or an exception raised in the
        render producer thread.
    """
    cfg = config or CompositorConfig()
    if cfg.fps <= 0:
        raise ValueError(f"fps must be positive, got {cfg.fps}")
    if cfg.width <= 0 or cfg.height <= 0:
        raise ValueError(f"Invalid resolution: {cfg.width}x{cfg.height}")
    if cfg.queue_size <= 0:
        raise ValueError(f"queue_size must be positive, got {cfg.queue_size}")
    if cfg.num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {cfg.num_bands}")

    ffmpeg_bin = require_ffmpeg()

    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir does not exist: {cache}")

    audio_p = Path(audio_path) if audio_path is not None else cache / ORIGINAL_WAV_NAME
    if not audio_p.is_file():
        raise FileNotFoundError(
            f"Missing {ORIGINAL_WAV_NAME} at {audio_p}; run audio ingest first"
        )

    bg_w, bg_h = background.size
    if bg_w != cfg.width or bg_h != cfg.height:
        raise ValueError(
            f"Background size {bg_w}×{bg_h} does not match compositor "
            f"{cfg.width}×{cfg.height}"
        )

    # Raises on unknown/missing shader stem before we spawn any thread.
    resolve_builtin_shader_stem(cfg.shader_name)

    logo_position_norm: str | None = None
    logo_rgba_prepared: np.ndarray | None = None
    if cfg.logo_path is not None and str(cfg.logo_path).strip():
        # Validate the logo up-front so the producer thread can't silently
        # skip it on disk errors.
        logo_position_norm = normalize_logo_position(cfg.logo_position)
        raw_logo = load_logo_rgba(cfg.logo_path)
        logo_rgba_prepared = prepare_logo_rgba(raw_logo, cfg.height, cfg.width)

    title_rgba = _prerender_title_layer(cfg)
    pulse_fn = _build_pulse_fn(cfg, analysis)
    shader_bass_track = _shader_bass_track_for_analysis(analysis, cfg)
    (
        shader_transient_lo_track,
        shader_transient_mid_track,
        shader_transient_hi_track,
    ) = _shader_transient_tracks_for_analysis(analysis, cfg)
    drop_hold_fn = _drop_hold_fn_for_analysis(
        analysis, float(cfg.shader_drop_hold_decay_sec)
    )
    snare_fn = _snare_envelope_fn(cfg, analysis)
    impact_fn = _impact_envelope_fn(cfg, analysis)
    rim_mod_step = _create_rim_modulation_stepper(cfg, analysis)
    resolved_rim_config = _effective_rim_light_config(cfg, analysis)
    beam_ctx = _build_beam_render_context(
        cfg,
        analysis,
        logo_rgba_prepared=logo_rgba_prepared,
        resolved_rim_config=resolved_rim_config,
    )

    if start_sec < 0:
        raise ValueError(f"start_sec must be non-negative, got {start_sec!r}")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError(
            f"duration_sec must be positive when set, got {duration_sec!r}"
        )

    audio_total = _audio_duration(audio_p)
    if start_sec >= audio_total:
        raise ValueError(
            f"start_sec {start_sec!r} is beyond audio duration {audio_total:.3f}s"
        )
    remaining = audio_total - float(start_sec)
    if duration_sec is None:
        window_sec = remaining
    else:
        if duration_sec > remaining + 1e-6:
            raise ValueError(
                f"duration_sec {duration_sec!r} exceeds remaining audio "
                f"{remaining:.3f}s after start_sec {start_sec!r}"
            )
        window_sec = float(duration_sec)

    frame_count = max(1, int(np.floor(window_sec * float(cfg.fps) + 1e-9)))
    audio_start_arg: float | None = float(start_sec) if start_sec > 0 else None
    audio_duration_arg: float | None = (
        float(duration_sec) if duration_sec is not None else None
    )

    codec = select_video_codec(cfg.video_codec)

    rid = run_id or new_run_id(song_hash=cache.name)
    out_root = Path(outputs_dir) if outputs_dir is not None else OUTPUTS_DIR
    out_dir = out_root / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "output.mp4"

    cmd = _build_ffmpeg_cmd(
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        audio_path=audio_p,
        output_mp4=out_mp4,
        video_codec=codec,
        audio_start_sec=audio_start_arg,
        audio_duration_sec=audio_duration_arg,
        ffmpeg_bin=ffmpeg_bin,
    )

    frame_q: queue.Queue[bytes | None] = queue.Queue(maxsize=cfg.queue_size)
    producer_error: list[BaseException] = []
    stop_event = threading.Event()

    has_typography = bool(aligned_words)
    has_title = title_rgba is not None
    has_logo = logo_rgba_prepared is not None and logo_position_norm is not None
    has_pulse = pulse_fn is not None

    layer_label = _active_layers_label(
        has_typography=has_typography,
        has_title=has_title,
        has_logo=has_logo,
        has_pulse=has_pulse,
    )
    # Kinetic-typography renders roughly 1 Skia draw per visible word per
    # frame, so it dominates the per-frame cost by a lot on lyrics-heavy
    # tracks; surface that in the log so a slow run is self-explanatory.
    LOGGER.info(
        "Compositor starting: %d frames · %dx%d @ %d fps · layers=%s",
        frame_count,
        cfg.width,
        cfg.height,
        cfg.fps,
        layer_label,
    )
    if has_typography and aligned_words:
        # One-line fingerprint of the first few timings the compositor is
        # about to bake into frames. This is the final checkpoint before
        # rendering — if these don't match what Save fingerprinted earlier,
        # the edits got lost somewhere in the align_lyrics cache path.
        _probe = ", ".join(
            f"[{i}]{w.word!r}@{w.t_start:.3f}s"
            for i, w in enumerate(list(aligned_words)[:3])
        )
        LOGGER.info(
            "Compositor typography source: %d words; first: %s",
            len(aligned_words),
            _probe,
        )

    def _produce() -> None:
        try:
            if progress is not None:
                progress(0.0, "Initializing GPU shader context…")
            with ReactiveShader(
                cfg.shader_name,
                width=cfg.width,
                height=cfg.height,
                num_bands=cfg.num_bands,
                palette=cfg.shader_palette,
            ) as reactive:
                typo_layer: KineticTypographyLayer | None = None
                if has_typography:
                    if progress is not None:
                        progress(
                            0.0,
                            f"Preparing kinetic typography ({len(aligned_words or [])} words)…",
                        )
                    typo_layer = KineticTypographyLayer(
                        list(aligned_words or ()),
                        motion=cfg.typography_motion,
                        font_path=cfg.font_path,
                        width=cfg.width,
                        height=cfg.height,
                        font_size=cfg.font_size,
                        base_color=cfg.base_color,
                        shadow_color=cfg.shadow_color,
                    )
                try:
                    render_started = time.monotonic()
                    last_log_t = render_started
                    if progress is not None:
                        progress(
                            0.0,
                            f"Rendering frame 0/{frame_count} · warming up · {layer_label}",
                        )
                    for i in range(frame_count):
                        if stop_event.is_set():
                            return
                        # Frame-centered time matches `pipeline.renderer`; it
                        # makes interpolation symmetric and avoids double-
                        # counting either endpoint. ``start_sec`` shifts the
                        # window for preview renders so the analysis / shader
                        # / typography all sample the same absolute timeline
                        # as the trimmed audio.
                        t = float(start_sec) + (i + 0.5) / float(cfg.fps)

                        composited = _render_compositor_frame(
                            t,
                            background=background,
                            analysis=analysis,
                            cfg=cfg,
                            reactive=reactive,
                            typo_layer=typo_layer,
                            logo_rgba_prepared=logo_rgba_prepared,
                            logo_position_norm=logo_position_norm,
                            title_rgba=title_rgba,
                            pulse_fn=pulse_fn,
                            shader_bass_track=shader_bass_track,
                            shader_transient_lo_track=shader_transient_lo_track,
                            shader_transient_mid_track=shader_transient_mid_track,
                            shader_transient_hi_track=shader_transient_hi_track,
                            drop_hold_fn=drop_hold_fn,
                            snare_fn=snare_fn,
                            impact_fn=impact_fn,
                            rim_mod_step=rim_mod_step,
                            resolved_rim_config=resolved_rim_config,
                            beam_ctx=beam_ctx,
                        )

                        bgr = np.ascontiguousarray(composited[:, :, ::-1])
                        frame_q.put(bgr.tobytes())

                        done = i + 1
                        now = time.monotonic()
                        elapsed = max(1e-6, now - render_started)
                        avg_fps = done / elapsed
                        if progress is not None:
                            remaining = max(0, frame_count - done)
                            eta_sec = remaining / avg_fps if avg_fps > 0 else 0.0
                            progress(
                                done / float(frame_count),
                                (
                                    f"Rendering frame {done}/{frame_count} "
                                    f"· {_format_render_fps(avg_fps)} · "
                                    f"{layer_label} · frame ETA {int(eta_sec)}s"
                                ),
                            )
                        # Throttle INFO logs to roughly once every 5 seconds
                        # so terminal output stays readable on long renders
                        # without hiding progress on minute-long stalls.
                        if now - last_log_t >= 5.0:
                            LOGGER.info(
                                "Compositor: %d/%d frames (%.1f%%) · %s",
                                done,
                                frame_count,
                                100.0 * done / float(frame_count),
                                _format_render_fps(avg_fps),
                            )
                            last_log_t = now
                finally:
                    if typo_layer is not None:
                        typo_layer.close()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            producer_error.append(exc)
        finally:
            frame_q.put(None)

    producer = threading.Thread(
        target=_produce, name="compositor-producer", daemon=True
    )

    with tempfile.TemporaryFile() as stderr_file:
        # bufsize=0 gives us an unbuffered RawIOBase on proc.stdin so we can
        # feed the pipe via os.write directly. Python 3.13 on Windows has been
        # observed to raise OSError(EINVAL) from BufferedWriter.write() when a
        # single frame buffer (a few MB for 1080p, ~24 MB for 4K) is passed in
        # one shot; chunked os.write + explicit short-write handling is the
        # reliable pattern used by most ffmpeg-pipe integrations.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=stderr_file,
            bufsize=0,
        )
        assert proc.stdin is not None
        stdin_fd = proc.stdin.fileno()
        producer.start()
        try:
            while True:
                chunk = frame_q.get()
                if chunk is None:
                    break
                try:
                    _pipe_write_all(stdin_fd, chunk)
                except (BrokenPipeError, OSError) as exc:
                    # ffmpeg died early (BrokenPipe / EINVAL after pipe close
                    # on Windows); stop the producer and surface the encoder
                    # error below via the non-zero exit code.
                    if isinstance(exc, OSError) and exc.errno not in (
                        errno.EPIPE,
                        errno.EINVAL,
                    ):
                        raise
                    stop_event.set()
                    break
        except BaseException:
            stop_event.set()
            raise
        finally:
            try:
                proc.stdin.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Ignoring stdin close error: %s", exc)
            producer.join()

        code = proc.wait()
        stderr_file.seek(0)
        err_tail = stderr_file.read().decode("utf-8", errors="replace").strip()

    if producer_error:
        raise RuntimeError("Compositor producer thread failed") from producer_error[0]

    if code != 0:
        msg = f"ffmpeg exited with code {code}"
        if err_tail:
            msg = f"{msg}\n--- ffmpeg stderr ---\n{err_tail}"
        raise RuntimeError(msg)

    thumb_path: Path | None = None
    line = thumbnail_line.strip() if thumbnail_line else ""
    if line:
        from pipeline.thumbnail import save_thumbnail_png

        thumb_path = out_dir / "thumbnail.png"
        save_thumbnail_png(
            thumb_path,
            line=line,
            analysis=analysis,
            background=background,
            config=cfg,
            palette=list(thumbnail_palette) if thumbnail_palette is not None else None,
        )

    return CompositorResult(
        run_id=rid,
        output_dir=out_dir,
        output_mp4=out_mp4,
        frame_count=frame_count,
        audio_path=audio_p,
        thumbnail_png=thumb_path,
    )


__all__: Sequence[str] = [
    "DEFAULT_QUEUE_SIZE",
    "PULSE_MODE_BASS",
    "PULSE_MODE_BEATS",
    "CompositorConfig",
    "CompositorResult",
    "ProgressFn",
    "PulseFn",
    "render_full_video",
    "render_single_frame",
]
