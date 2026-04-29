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
import hashlib
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
from pipeline.chromatic_aberration import apply_chromatic_aberration
from pipeline.color_invert import apply_invert_mix, invert_mix
from pipeline.scanline_tear import apply_scanline_tear
from pipeline.effects_timeline import EffectClip, EffectKind, EffectsTimeline
from pipeline.ffmpeg_tools import require_ffmpeg, select_video_codec
from pipeline.screen_shake import apply_shake_offset, shake_offset
from pipeline.zoom_punch import apply_zoom_scale, zoom_scale
from pipeline.kinetic_typography import (
    DEFAULT_FONT_SIZE,
    DEFAULT_KINETIC_BASELINE_RATIO,
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
    kick_punch_scale_and_opacity,
    scale_and_opacity_for_pulse,
    stable_pulse_value,
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
    _random_beam_tint,
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
from pipeline.voidcat_ascii import (
    VoidcatAsciiContext,
    render_voidcat_ascii_rgba,
    render_voidcat_cat_overlay_rgba,
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

# How often the encoder-feed (request thread) loop emits a progress(...) call
# to the UI. 250 ms keeps the bar lively without spamming Gradio's internal
# update queue. Throttling is critical: at 1080p with NVENC we hit ~30 fps,
# 30 progress calls/s is enough to make Gradio's WebSocket choke noticeably.
_PROGRESS_TICK_SEC = 0.25


def _format_eta_compositor(seconds: float) -> str:
    """Human-readable ETA used inside the compositor progress message.

    Duplicated from ``app._format_eta`` because the compositor is imported
    from contexts (CLI, tests) that don't pull in Gradio. Kept identical in
    output so the progress-bar text matches whatever the UI shows already.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return "<1s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@dataclass
class _CompositorStats:
    """Cross-thread state shared between the render producer and the
    encoder-feed consumer.

    Why this exists: the producer renders frames on a daemon thread, but the
    encoder-feed loop runs on the request thread (``render_full_video``'s
    caller). Gradio's :class:`gr.Progress` object only reliably propagates
    updates when called from the request thread — calls made from a daemon
    background thread are silently dropped by Gradio's internal asyncio
    queue. Symptom: UI stuck on the orchestrator's outer "Compositing
    video..." message at 40 % for the entire render even though the
    producer's per-frame ``progress(...)`` calls are firing.

    Solution: producer writes its phase string + a frame counter into this
    object (no progress callback), consumer reads the object every
    ``_PROGRESS_TICK_SEC`` seconds and forwards to ``progress``. All fields
    are scalars whose individual reads/writes are atomic under the GIL, so
    no lock is needed for these particular access patterns. The string is
    treated as opaque (Python ``str`` rebinding is atomic — no torn read
    is possible).

    ``frames_encoded`` is the canonical "done" counter for the UI because
    it's measured at the encoder pipe — that's where the actual bottleneck
    sits when NVENC is unavailable and we fall back to libx264.
    """

    total_frames: int
    started_at: float
    frames_produced: int = 0  # how many frames the renderer has finished
    frames_encoded: int = 0  # how many frames have been pushed to ffmpeg stdin
    layer_label: str = ""
    phase: str = "starting"  # human-readable phase ("warming up", "encoding")

    def progress_pair(self) -> tuple[float, str]:
        """Compute ``(p, msg)`` for the UI from the current stats snapshot.

        Reads several scalar fields without a lock; under the GIL each read
        is atomic but the *combination* could in principle be torn (e.g.
        frames_encoded read just after producer bumped frames_produced).
        That's fine — the UI value is approximate and self-corrects on the
        next tick.
        """
        total = max(1, self.total_frames)
        encoded = self.frames_encoded
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        p = encoded / total
        if encoded > 0:
            fps = encoded / elapsed
            remaining = max(0, self.total_frames - encoded)
            eta = remaining / fps if fps > 0 else 0.0
            msg = (
                f"Compositing {encoded}/{self.total_frames} "
                f"({100.0 * p:.1f}%) - {fps:.2f} fps - "
                f"ETA {_format_eta_compositor(eta)}"
            )
            if self.layer_label:
                msg = f"{msg} - {self.layer_label}"
        else:
            # No frames encoded yet -- show producer phase (e.g. "warming
            # up", "preparing typography") so the user knows something is
            # happening during the warmup that can take 10-30s on big runs.
            msg = (
                f"Compositing 0/{self.total_frames} - {self.phase}"
                if self.phase
                else f"Compositing 0/{self.total_frames}"
            )
            if self.layer_label:
                msg = f"{msg} - {self.layer_label}"
        return p, msg


@dataclass(frozen=True)
class CompositorRenderStats:
    """Headline stats from one ``render_full_video`` invocation.

    Attached to :class:`CompositorResult` so :mod:`app` can append a
    one-liner like
    ``Compositor: 4923 frames in 41m23s - avg 1.98 fps - encoder=h264_nvenc``
    to the in-app run log when the render finishes.
    Pinokio terminals are read-only on Windows so this is the **only** way
    fork users can see whether NVENC actually engaged.
    """

    frame_count: int
    elapsed_sec: float
    avg_fps: float
    video_codec: str
    ffmpeg_path: str | None


@dataclass(frozen=True)
class CompositorResult:
    """Return value of :func:`render_full_video`."""

    run_id: str
    output_dir: Path
    output_mp4: Path
    frame_count: int
    audio_path: Path
    thumbnail_png: Path | None = None
    render_stats: CompositorRenderStats | None = None


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
    # Lyrics baseline Y as fraction of frame height (larger = lower on screen).
    typography_baseline_y_ratio: float = DEFAULT_KINETIC_BASELINE_RATIO
    base_color: str = "#FFFFFF"
    shadow_color: str | None = None
    logo_path: Path | None = None
    logo_position: str = "center"
    logo_opacity_pct: float = 100.0
    # Cap the logo's longest edge at this percent of the **shorter** frame edge
    # (so the slider behaves the same on 720p / 1080p / 4K). ``<= 0`` disables
    # the cap and falls back to the legacy "fit inside the frame" behaviour.
    # Default 30 % keeps a center-aligned logo from covering the kinetic-type
    # band while leaving room for rim beams to shoot past the rim.
    logo_max_size_pct: float = 30.0
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
    # Kick (low-band) contribution to the same neon halo. Uses the attack-only
    # bass envelope (:func:`pipeline.beat_pulse.build_bass_pulse_track`) so the
    # glow pulses on every kick transient in addition to snare hits; toggle off
    # to restore the snare-only behaviour.
    logo_kick_glow: bool = True
    logo_kick_glow_strength: float = 1.6
    # Dedicated kick punch on the **logo scale** driven by the separated
    # low-band transient (``build_lo_transient_track`` → ``transient_lo``).
    # Complements the sustain-aware :func:`build_logo_bass_pulse_track` curve
    # by giving cleanly-separated kicks a bigger visual bump than the blended
    # pulse can produce on its own. Combined with the existing pulse via
    # ``max`` of deltas so one signal never cancels the other — whichever is
    # larger on the current frame wins the bounce. Set to ``0`` to disable.
    logo_kick_punch_strength: float = 1.0
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
    #
    # Lowered from ``1.0`` to ``0.75`` (2026-04) alongside the build-time
    # :func:`pipeline.beat_pulse.shape_reactive_envelope` gate to calm the
    # "wild across every preset" feeling: three full-amplitude envelopes
    # slamming into every shader's motion terms at once produced constant
    # low-amplitude wobble during non-hit sections. Shape gate kills the
    # wobble; the sensitivity trim brings the stack closer to the
    # ``shader_bass_sensitivity=0.72`` budget so peaks and shader mixes
    # stay in the same ballpark.
    shader_transient_sensitivity: float = 0.75
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
    # Task 49: optional per-song effects timeline (see
    # :mod:`pipeline.effects_timeline`). When ``None`` the compositor renders
    # exactly as before (regression guard); when set, user ``EffectClip`` rows
    # drive the post-stack frame effects (zoom punch, screen shake, colour
    # invert, future chromatic / scanline) and merge additively into the beam
    # + logo-glitch auto paths.
    effects_timeline: EffectsTimeline | None = None
    # Master multiplier applied to the **auto** reactivity envelopes (bass
    # pulse, snare glow, impact glitch, rim audio modulation). User clip
    # contributions are not scaled. ``1.0`` preserves today's behaviour; the
    # Gradio editor writes ``EffectsTimeline.auto_reactivity_master`` and the
    # orchestrator threads it here.
    auto_reactivity_master: float = 1.0
    # voidcat-ASCII: CPU grid layer (sides + hero-drop cat) over the reactive
    # pass, under lyrics and logo. Set by the orchestrator for preset
    # ``voidcat-laser``; ``None`` skips the pass (default).
    voidcat_ascii_ctx: VoidcatAsciiContext | None = None
    # When True with a voidcat context, the side cat (and * trace) is
    # composited again **after** the frame-effects stack (chroma, scanline,
    # etc.) so it stays legible. The ASCII grid is still only drawn in the
    # first pass, preserving the "glitchy field" look.
    voidcat_ascii_sharp_cat: bool = False


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
    has_voidcat_ascii: bool = False,
) -> str:
    """Return e.g. ``bg+shader+ascii+typo+title+logo+pulse`` for progress messages.

    The compositor always renders the background and reactive shader; the rest
    are optional. A compact label lets the UI show which stages are active so
    users can tell whether a slow run is from (say) kinetic typography or a
    heavy shader, rather than guessing.
    """
    parts = ["bg", "shader"]
    if has_voidcat_ascii:
        parts.append("ascii")
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


def _auto_reactivity_master(cfg: CompositorConfig) -> float:
    """Non-negative finite scalar; out-of-range/NaN collapses to ``1.0``."""
    m = float(cfg.auto_reactivity_master)
    if not math.isfinite(m) or m < 0.0:
        return 1.0
    return m


def _scaled_pulse_fn(fn: PulseFn | None, k: float) -> PulseFn | None:
    """Return ``t -> k * fn(t)`` or pass-through when ``k == 1`` / ``fn`` empty."""
    if fn is None:
        return None
    if abs(k - 1.0) < 1e-9:
        return fn
    return lambda t, _fn=fn, _k=float(k): float(_k) * float(_fn(float(t)))


def _timeline_clips(cfg: CompositorConfig) -> tuple[EffectClip, ...]:
    tl = cfg.effects_timeline
    if tl is None:
        return ()
    return tuple(tl.clips)


def _clips_of_kind(
    clips: Sequence[EffectClip], kind: EffectKind
) -> tuple[EffectClip, ...]:
    return tuple(c for c in clips if c.kind is kind)


def _auto_enabled_for(cfg: CompositorConfig, kind: EffectKind) -> bool:
    """``True`` when the effects-timeline hasn't switched off the auto path
    for ``kind`` (default when no timeline is set)."""
    tl = cfg.effects_timeline
    if tl is None:
        return True
    return bool(tl.auto_enabled.get(kind, True))


def _clip_id_u32(clip_id: str) -> int:
    """Stable 32-bit seed from a clip's id (uses :mod:`zlib` like shake)."""
    import zlib as _zlib  # local import keeps the main module import fast

    return _zlib.adler32(clip_id.encode("utf-8", errors="ignore")) & 0xFFFFFFFF


def _parse_srgb_hex(value: Any) -> tuple[int, int, int] | None:
    """Parse ``"#RRGGBB"`` (with or without ``#``) into an sRGB triple.

    Returns ``None`` when ``value`` is not a recognisable 6-hex string so the
    caller can fall back to the preset palette. Case-insensitive; whitespace
    trimmed.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return None
    return (r, g, b)


def _user_beam_schedule(
    clips: Sequence[EffectClip],
    *,
    beam_cfg: BeamConfig,
    song_hash: str | None,
    n_color_layers: int,
) -> list[ScheduledBeam]:
    """Convert user ``BEAM`` :class:`EffectClip` rows into :class:`ScheduledBeam`.

    User beams are treated as hero drops (full edge-stretch, drop thickness)
    with a deterministic per-clip angle and colour layer so re-renders are
    bit-stable. The 10 s group gate in :func:`schedule_rim_beams` is **not**
    applied — user cues are authoritative and must fire where placed.
    """
    beams: list[ScheduledBeam] = []
    layers = max(1, int(n_color_layers))
    for c in clips:
        if c.kind is not EffectKind.BEAM:
            continue
        s = c.settings
        raw_strength = s.get("strength", 1.0)
        try:
            strength = float(raw_strength) if raw_strength is not None else 1.0
        except (TypeError, ValueError):
            strength = 1.0
        strength = max(0.0, min(1.0, strength))
        if strength <= 1e-6:
            continue
        raw_thick = s.get("thickness_px", beam_cfg.drop_beam_thickness_px)
        try:
            thickness = (
                float(raw_thick)
                if raw_thick is not None
                else float(beam_cfg.drop_beam_thickness_px)
            )
        except (TypeError, ValueError):
            thickness = float(beam_cfg.drop_beam_thickness_px)
        if not math.isfinite(thickness) or thickness <= 0.0:
            thickness = float(beam_cfg.drop_beam_thickness_px)
        # Deterministic angle + colour layer from the clip id, salted with song.
        cid = _clip_id_u32(c.id)
        seed_src = (
            f"{song_hash or ''}::user_beam::{cid}".encode("utf-8", errors="ignore")
        )
        rng_seed = int.from_bytes(
            hashlib.sha256(seed_src).digest()[:4], "little", signed=False
        ) & 0x7FFFFFFF
        rng = np.random.default_rng(rng_seed)
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        layer_idx = cid % layers
        # Colour pick: an explicit ``color_hex`` in settings always wins
        # (user's picker); otherwise draw a deterministic random hue so
        # user-added beams don't all collapse onto the 2-layer preset
        # palette. Seeded from the same ``rng`` so the same clip id on the
        # same song renders to the same colour.
        tint_override = _parse_srgb_hex(s.get("color_hex"))
        if tint_override is None:
            tint_override = _random_beam_tint(rng)
        # Floor the rendered beam lifetime at ``BeamConfig.duration_sec``. Users
        # routinely place BEAM clips as short ticks on the effects timeline; a
        # clip with ``duration_s`` shorter than the envelope's attack (40 ms)
        # would collapse to a single bright pixel and read as a "point" rather
        # than a ray. Flooring preserves the hand-placed ``t_start`` cue while
        # still letting the full attack + decay + afterglow play out.
        raw_duration = float(c.duration_s)
        floor_duration = float(beam_cfg.duration_sec)
        render_duration = raw_duration if raw_duration >= floor_duration else floor_duration
        # Longer-than-default clips use ramp + plateau + growing halo/afterglow
        # (see :attr:`ScheduledBeam.sustain_shaping`); floored short ticks keep
        # the original hit-style envelope.
        sustain_shaping = raw_duration > floor_duration + 1e-9
        beams.append(
            ScheduledBeam(
                t_start=float(c.t_start),
                duration_s=render_duration,
                angle_rad=angle,
                length_px=float(beam_cfg.drop_beam_length_px),
                thickness_px=thickness,
                intensity=strength,
                color_layer_idx=layer_idx,
                is_drop=True,
                tint_srgb_override=tint_override,
                sustain_shaping=sustain_shaping,
            )
        )
    return beams


def _user_glitch_envelope_fn(
    clips: Sequence[EffectClip],
) -> PulseFn | None:
    """Return a ``t -> [0, 1]`` callable summing user ``LOGO_GLITCH`` clip
    strengths active at ``t`` (clamped). ``None`` when no clips exist so the
    frame loop can skip the combine step."""
    items: list[tuple[float, float, float]] = []
    for c in clips:
        if c.kind is not EffectKind.LOGO_GLITCH:
            continue
        raw = c.settings.get("strength", 1.0)
        try:
            strength = float(raw) if raw is not None else 1.0
        except (TypeError, ValueError):
            strength = 1.0
        strength = max(0.0, min(1.0, strength))
        if strength <= 1e-6:
            continue
        t0 = float(c.t_start)
        items.append((t0, t0 + float(c.duration_s), strength))
    if not items:
        return None
    snap = tuple(items)

    def fn(t: float, _snap=snap) -> float:
        tf = float(t)
        if not math.isfinite(tf):
            return 0.0
        total = 0.0
        for t0, t1, s in _snap:
            if t0 <= tf < t1:
                total += s
        if total <= 0.0:
            return 0.0
        if total >= 1.0:
            return 1.0
        return total

    return fn


def _combined_glitch_fn(
    auto_fn: PulseFn | None, user_fn: PulseFn | None
) -> PulseFn | None:
    """Sum of ``auto_fn`` and ``user_fn`` clamped to ``[0, 1]``; ``None`` when
    both are ``None`` so the frame loop short-circuits."""
    if auto_fn is None and user_fn is None:
        return None
    if user_fn is None:
        return auto_fn
    if auto_fn is None:
        return user_fn
    return lambda t, _a=auto_fn, _u=user_fn: max(
        0.0, min(1.0, float(_a(float(t))) + float(_u(float(t))))
    )


@dataclass(frozen=True, slots=True)
class _FrameEffectsContext:
    """Per-render cache of non-logo ``EffectClip`` rows + frame-pass toggles.

    Built once in :func:`_build_frame_effects_context` from
    :attr:`CompositorConfig.effects_timeline`. The per-frame pass reads these
    tuples in fixed order: ``zoom_punch`` → ``screen_shake`` →
    ``chromatic_aberration`` → ``scanline_tear`` → ``color_invert`` (see
    :func:`_apply_frame_effects`).
    """

    zoom_clips: tuple[EffectClip, ...]
    shake_clips: tuple[EffectClip, ...]
    color_invert_clips: tuple[EffectClip, ...]
    chromatic_clips: tuple[EffectClip, ...]
    scanline_clips: tuple[EffectClip, ...]
    song_hash: str


def _build_frame_effects_context(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> _FrameEffectsContext | None:
    """Return a frame-effects ctx or ``None`` when nothing post-logo fires.

    ``None`` is the fast path: the frame loop then skips the whole pass and
    the output is byte-identical to the pre-effects compositor (regression
    guard required by the Task 49 acceptance criteria).
    """
    clips = _timeline_clips(cfg)
    if not clips:
        return None
    zoom = _clips_of_kind(clips, EffectKind.ZOOM_PUNCH)
    shake = _clips_of_kind(clips, EffectKind.SCREEN_SHAKE)
    inv = _clips_of_kind(clips, EffectKind.COLOR_INVERT)
    chrom = _clips_of_kind(clips, EffectKind.CHROMATIC_ABERRATION)
    scan = _clips_of_kind(clips, EffectKind.SCANLINE_TEAR)
    if not (zoom or shake or inv or chrom or scan):
        return None
    song_hash = str(analysis.get("song_hash") or "") if isinstance(analysis, Mapping) else ""
    return _FrameEffectsContext(
        zoom_clips=zoom,
        shake_clips=shake,
        color_invert_clips=inv,
        chromatic_clips=chrom,
        scanline_clips=scan,
        song_hash=song_hash,
    )


def _apply_frame_effects(
    frame: np.ndarray, t: float, fx: _FrameEffectsContext | None
) -> np.ndarray:
    """Fixed-order post-stack effects pass (task 49).

    Order: zoom_punch → screen_shake → chromatic_aberration → scanline_tear
    → color_invert. Each stage short-circuits to the input array
    when its clip set is inactive at ``t``, so an empty timeline produces a
    byte-identical frame.
    """
    if fx is None:
        return frame
    out = frame
    if fx.zoom_clips:
        scale = zoom_scale(t, fx.zoom_clips)
        if scale > 1.0 + 1e-9:
            out = apply_zoom_scale(out, scale)
    if fx.shake_clips:
        dx, dy = shake_offset(t, fx.shake_clips, fx.song_hash)
        if abs(dx) >= 0.5 or abs(dy) >= 0.5:
            out = apply_shake_offset(out, dx, dy)
    if fx.chromatic_clips:
        out = apply_chromatic_aberration(out, t, fx.chromatic_clips, fx.song_hash)
    if fx.scanline_clips:
        out = apply_scanline_tear(out, t, fx.scanline_clips, fx.song_hash)
    if fx.color_invert_clips:
        mix = invert_mix(t, fx.color_invert_clips)
        if mix > 1e-4:
            out = apply_invert_mix(out, mix)
    return out


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
    the combined auto + user schedule is empty. Task 49: user ``BEAM`` clips
    on :attr:`CompositorConfig.effects_timeline` are merged into the schedule
    **without** the 10 s group gate (``schedule_rim_beams`` only gates the
    analyser-driven path).
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

    auto_beams: list[ScheduledBeam] = []
    if _auto_enabled_for(cfg, EffectKind.BEAM):
        snare_track = _snare_track_for_logo(cfg, analysis)
        impact_track: PulseTrack | None = None
        if float(cfg.logo_impact_glitch_strength) > 1e-6 or bool(cfg.rim_beams_enabled):
            impact_track = build_rms_impact_pulse_track(
                analysis, sensitivity=float(cfg.logo_impact_sensitivity)
            )
        auto_beams = list(
            schedule_rim_beams(
                analysis,
                snare_track=snare_track,
                impact_track=impact_track,
                cfg=beam_cfg,
                song_hash=song_hash,
                n_color_layers=n_layers,
            )
        )

    user_beams = _user_beam_schedule(
        _timeline_clips(cfg),
        beam_cfg=beam_cfg,
        song_hash=song_hash,
        n_color_layers=n_layers,
    )
    schedule = auto_beams + user_beams
    if not schedule:
        return None
    schedule.sort(key=lambda b: b.t_start)

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
    # Half of the shorter logo side is a cheap, position-independent estimate
    # of the centroid-to-edge distance. Beams use this to start outside the
    # logo rim instead of piercing its middle.
    logo_radius_px = 0.5 * float(min(lh, lw))

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
        logo_radius_px=logo_radius_px,
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
    kick_glow_fn: PulseFn | None = None,
    kick_punch_fn: PulseFn | None = None,
    impact_fn: PulseFn | None = None,
    rim_mod_step: RimModStepFn | None = None,
    resolved_rim_config: RimLightConfig | None = None,
    beam_ctx: _BeamRenderContext | None = None,
    frame_effects: _FrameEffectsContext | None = None,
    stage_timings: dict[str, float] | None = None,
) -> np.ndarray:
    """One RGB frame: background → reactive → [ascii] → typography → title →
    [rim beams] → logo → FX.

    Optional voidcat ASCII (``cfg.voidcat_ascii_ctx``) runs after the reactive
    pass so glyphs sit on the background wash but under lyrics and logo, leaving
    the centre column transparent for a centre logo.

    Rim beams (when active) are blended before the logo so they read as light
    from behind the mark; the logo compositing occludes the beam in the
    glyph area. Title and logo are drawn so branding stays on top of the
    reactive shader and lyrics; the logo is drawn *above* the title so the
    mark is not occluded by an artist/song label on a shared edge. Task 49 adds a
    final fixed-order frame-effects pass (``zoom_punch → screen_shake →
    chromatic_aberration → scanline_tear → color_invert``) applied after all
    layers when an ``EffectsTimeline`` is set on the config; the pass is
    skipped entirely when no clip of any frame-effect kind is active at
    ``t``.
    """
    # Per-stage timing capture: when ``stage_timings`` is provided we record
    # ``time.perf_counter()`` deltas around each major stage. The producer
    # passes a dict only for one warmup frame so the regular render path
    # pays zero cost (no perf_counter() calls, no dict ops). Diagnoses
    # which stage owns the per-frame budget on slow Pinokio renders. See
    # the call site in :func:`render_full_video`.
    if stage_timings is not None:
        _t0 = time.perf_counter()
    bg_rgb = _validate_background_frame(
        background.background_frame(t),
        cfg.width,
        cfg.height,
        t,
    )
    if stage_timings is not None:
        _t1 = time.perf_counter()
        stage_timings["bg"] = _t1 - _t0
        _t0 = _t1
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
    if stage_timings is not None:
        _t1 = time.perf_counter()
        stage_timings["uniforms"] = _t1 - _t0
        _t0 = _t1
    composited = reactive.render_frame_composited_rgb(uniforms, bg_rgb)
    if stage_timings is not None:
        _t1 = time.perf_counter()
        stage_timings["shader"] = _t1 - _t0
        _t0 = _t1
    if cfg.voidcat_ascii_ctx is not None:
        vox = render_voidcat_ascii_rgba(
            cfg.width,
            cfg.height,
            float(t),
            uniforms=uniforms,
            ctx=cfg.voidcat_ascii_ctx,
            omit_cat=bool(cfg.voidcat_ascii_sharp_cat),
        )
        composited = composite_premultiplied_rgba_over_rgb(vox, composited)
        if stage_timings is not None:
            _t1 = time.perf_counter()
            stage_timings["voidcat"] = _t1 - _t0
            _t0 = _t1
    if typo_layer is not None:
        typo_rgba = typo_layer.render_frame(float(t), uniforms)
        if stage_timings is not None:
            _t1 = time.perf_counter()
            stage_timings["typo_render"] = _t1 - _t0
            _t0 = _t1
        composited = composite_premultiplied_rgba_over_rgb(typo_rgba, composited)
        if stage_timings is not None:
            _t1 = time.perf_counter()
            stage_timings["typo_composite"] = _t1 - _t0
            _t0 = _t1
    if title_rgba is not None:
        composited = composite_premultiplied_rgba_over_rgb(title_rgba, composited)
        if stage_timings is not None:
            _t1 = time.perf_counter()
            stage_timings["title_composite"] = _t1 - _t0
            _t0 = _t1
    if logo_rgba_prepared is not None and logo_position_norm is not None:
        logo_scale = 1.0
        logo_opacity_pct = float(cfg.logo_opacity_pct)
        stability = max(0.0, float(cfg.logo_motion_stability))
        pulse_deadzone = 0.22 * stability
        # Smart controls: stability knob scales a short asymmetric smoother
        # on the pulse. At stability=1.0 we sample a 60 ms look-back window
        # and smooth the release; at stability=0.0 we pass the raw pulse
        # through (legacy behaviour). Stability > 1 lets a user push the
        # window out for ultra-calm logos during mellow sections.
        pulse_smooth_sec = 0.06 * stability
        if pulse_fn is not None:
            # ``stable_pulse_value`` preserves attack (kick still hits in one
            # frame) but averages the release to kill sub-kick jitter that
            # otherwise shows up as 1-pixel logo shake after integer origin
            # rounding.
            pulse = stable_pulse_value(pulse_fn, float(t), smooth_sec=pulse_smooth_sec)
            logo_scale, opacity_mul = scale_and_opacity_for_pulse(
                pulse,
                strength=float(cfg.logo_pulse_strength),
                deadzone=pulse_deadzone,
            )
            logo_opacity_pct = max(
                0.0, min(100.0, float(cfg.logo_opacity_pct) * opacity_mul)
            )
        # Kick punch: clean low-band transient on top of the sustain-aware
        # pulse, combined via ``max`` of deltas so the punch channel only
        # wins when it's genuinely bigger (kick attack) and the existing
        # pulse owns the sustained bounce. Independent of ``pulse_fn`` so
        # a user who disables the mode-based pulse still gets kick reactivity.
        if (
            kick_punch_fn is not None
            and float(cfg.logo_kick_punch_strength) > 1e-6
        ):
            k_val = stable_pulse_value(
                kick_punch_fn, float(t), smooth_sec=pulse_smooth_sec
            )
            # ``kick_punch_scale_and_opacity`` applies its own deadzone (0.12)
            # which is lower than the logo pulse deadzone on purpose — the
            # envelope already has the build-time shape gate, so extra margin
            # from ``pulse_deadzone`` would double-gate real kicks.
            k_scale, k_opacity_mul = kick_punch_scale_and_opacity(
                k_val,
                strength=float(cfg.logo_kick_punch_strength)
                * float(cfg.logo_pulse_strength),
            )
            if k_scale > logo_scale:
                logo_scale = k_scale
            k_opacity_pct = max(
                0.0, min(100.0, float(cfg.logo_opacity_pct) * k_opacity_mul)
            )
            if k_opacity_pct > logo_opacity_pct:
                logo_opacity_pct = k_opacity_pct
        snare_val = 0.0
        if snare_fn is not None:
            snare_val = stable_pulse_value(
                snare_fn, float(t), smooth_sec=pulse_smooth_sec
            )
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
        if (
            cfg.logo_kick_glow
            and kick_glow_fn is not None
            and float(cfg.logo_kick_glow_strength) > 1e-6
        ):
            # Kick envelope is already master-scaled (_scaled_pulse_fn). Adds on
            # top of the snare glow so the halo pumps on every low-end transient
            # even on tracks with sparse / absent snare energy.
            glow_amt += float(kick_glow_fn(float(t))) * float(
                cfg.logo_kick_glow_strength
            )
        glow_rgb = resolve_logo_glow_rgb(cfg.shadow_color, cfg.base_color)
        glitch_amt = 0.0
        glitch_seed = 0
        glitch_tilt_seed = 0
        if impact_fn is not None and float(cfg.logo_impact_glitch_strength) > 1e-6:
            imp = float(impact_fn(float(t)))
            g = max(0.0, min(1.0, imp)) * float(cfg.logo_impact_glitch_strength)
            if g > 1e-4:
                glitch_amt = g
                song_hash_str = str(analysis.get("song_hash") or "")
                glitch_seed = glitch_seed_for_time(song_hash_str, float(t))
                # Quantise ``t`` into ~0.4 s buckets so every frame of a
                # single impact event shares the same tilt direction (a
                # typical glitch lasts <0.3 s, so the whole envelope lands in
                # one bucket). Per-frame randomness only drives the RGB /
                # tear noise -- the tilt itself now holds one sign for the
                # full attack-and-release, reading as a crisp camera nudge
                # instead of left/right shake.
                _BUCKET_S = 0.4
                bucket_t = math.floor(float(t) / _BUCKET_S) * _BUCKET_S
                glitch_tilt_seed = glitch_seed_for_time(song_hash_str, bucket_t)
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
        # Rim beams read as light from *behind* the mark: draw before the logo
        # composite so the RGBA logo occludes the beam instead of washing out.
        if beam_ctx is not None:
            _draw_beam_patch_onto_frame(
                composited,
                float(t),
                ctx=beam_ctx,
                position=logo_position_norm,
                logo_scale=float(logo_scale),
            )
            if stage_timings is not None:
                _t1 = time.perf_counter()
                stage_timings["beams"] = _t1 - _t0
                _t0 = _t1
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
            glitch_tilt_seed=glitch_tilt_seed,
            t_sec=float(t),
            song_hash=str(analysis.get("song_hash") or "") or None,
            rim_light_config=resolved_rim_config,
            rim_audio_mod=rim_audio_mod,
            logo_glow_mode=cfg.logo_glow_mode,
        )
        if stage_timings is not None:
            _t1 = time.perf_counter()
            stage_timings["logo"] = _t1 - _t0
            _t0 = _t1
    if frame_effects is not None:
        composited = _apply_frame_effects(composited, float(t), frame_effects)
        if stage_timings is not None:
            _t1 = time.perf_counter()
            stage_timings["effects"] = _t1 - _t0
            _t0 = _t1
    if (
        cfg.voidcat_ascii_sharp_cat
        and cfg.voidcat_ascii_ctx is not None
    ):
        cat_only = render_voidcat_cat_overlay_rgba(
            cfg.width,
            cfg.height,
            float(t),
            uniforms=uniforms,
            ctx=cfg.voidcat_ascii_ctx,
        )
        composited = composite_premultiplied_rgba_over_rgb(
            cat_only, composited
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


def _kick_glow_envelope_fn(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> PulseFn | None:
    """Attack-only bass envelope used to pulse the neon halo on kicks.

    Keyed off :func:`pipeline.beat_pulse.build_bass_pulse_track` so it spikes
    on each low-band onset and decays back to zero between kicks — distinct
    from :func:`build_logo_bass_pulse_track` (which adds a sustain follower
    for the logo *scale* pulse and would keep the halo lit on long 808s).
    Returns ``None`` when the feature is disabled or the analyser lacks a
    usable spectrum so the frame loop can skip the lookup entirely.
    """
    if not bool(cfg.logo_kick_glow):
        return None
    if float(cfg.logo_kick_glow_strength) <= 1e-6:
        return None
    track = build_bass_pulse_track(
        analysis, sensitivity=float(cfg.logo_pulse_sensitivity)
    )
    if track is None:
        return None
    return track.value_at


def _kick_punch_envelope_fn(
    cfg: CompositorConfig, analysis: Mapping[str, Any]
) -> PulseFn | None:
    """Low-band transient envelope used to punch the logo *scale* on kicks.

    Distinct from the sustain-aware :func:`build_logo_bass_pulse_track`:
    this channel keys off :func:`build_lo_transient_track` so it only
    fires on actual kick attacks (no sustain follower), with the build-
    time shape gate applied so between-kick wobble reads as zero. The
    compositor combines the kick punch with the existing pulse via
    ``max`` of scale / opacity deltas so one never cancels the other.

    Returns ``None`` when the feature is disabled or the analyser lacks a
    usable spectrum so the per-frame path can skip the lookup entirely.
    """
    if float(cfg.logo_kick_punch_strength) <= 1e-6:
        return None
    track = build_lo_transient_track(
        analysis, sensitivity=float(cfg.logo_pulse_sensitivity)
    )
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
    # Task 49: ``auto_reactivity_master`` scales the rim-audio driver alongside
    # bass / snare / impact so one knob tames the full auto reactivity stack.
    master = _auto_reactivity_master(cfg)
    strength = max(0.0, float(cfg.logo_rim_mod_strength)) * master
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
        logo_rgba_prepared = prepare_logo_rgba(raw_logo, cfg.height, cfg.width, max_size_pct=cfg.logo_max_size_pct)

    title_rgba = _prerender_title_layer(cfg)
    master = _auto_reactivity_master(cfg)
    # Auto reactivity envelopes: bass pulse (logo), snare, impact — each is a
    # scalar ``t -> [0, 1]`` closure multiplied by ``master`` so the effects
    # timeline slider damps the full auto stack without touching user clips.
    pulse_fn = _scaled_pulse_fn(_build_pulse_fn(cfg, analysis), master)
    shader_bass_track = _shader_bass_track_for_analysis(analysis, cfg)
    (
        shader_transient_lo_track,
        shader_transient_mid_track,
        shader_transient_hi_track,
    ) = _shader_transient_tracks_for_analysis(analysis, cfg)
    drop_hold_fn = _drop_hold_fn_for_analysis(
        analysis, float(cfg.shader_drop_hold_decay_sec)
    )
    snare_fn = _scaled_pulse_fn(_snare_envelope_fn(cfg, analysis), master)
    kick_glow_fn = _scaled_pulse_fn(_kick_glow_envelope_fn(cfg, analysis), master)
    kick_punch_fn = _scaled_pulse_fn(_kick_punch_envelope_fn(cfg, analysis), master)
    auto_impact_fn = _scaled_pulse_fn(_impact_envelope_fn(cfg, analysis), master)
    user_glitch_fn = _user_glitch_envelope_fn(_timeline_clips(cfg))
    impact_fn = _combined_glitch_fn(auto_impact_fn, user_glitch_fn)
    rim_mod_step = _create_rim_modulation_stepper(cfg, analysis)
    resolved_rim_config = _effective_rim_light_config(cfg, analysis)
    beam_ctx = _build_beam_render_context(
        cfg,
        analysis,
        logo_rgba_prepared=logo_rgba_prepared,
        resolved_rim_config=resolved_rim_config,
    )
    frame_effects = _build_frame_effects_context(cfg, analysis)

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
                baseline_y_ratio=float(cfg.typography_baseline_y_ratio),
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
                kick_glow_fn=kick_glow_fn,
                kick_punch_fn=kick_punch_fn,
                impact_fn=impact_fn,
                rim_mod_step=rim_mod_step,
                resolved_rim_config=resolved_rim_config,
                beam_ctx=beam_ctx,
                frame_effects=frame_effects,
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
        logo_rgba_prepared = prepare_logo_rgba(raw_logo, cfg.height, cfg.width, max_size_pct=cfg.logo_max_size_pct)

    title_rgba = _prerender_title_layer(cfg)
    master = _auto_reactivity_master(cfg)
    # Mirror of the same auto-envelope wiring in ``render_single_frame``; the
    # ``master`` multiplier is applied once here (one scalar lookup per frame
    # stays the hot path). User ``LOGO_GLITCH`` contributions are additive
    # after the scale so manual cues keep their intended strength.
    pulse_fn = _scaled_pulse_fn(_build_pulse_fn(cfg, analysis), master)
    shader_bass_track = _shader_bass_track_for_analysis(analysis, cfg)
    (
        shader_transient_lo_track,
        shader_transient_mid_track,
        shader_transient_hi_track,
    ) = _shader_transient_tracks_for_analysis(analysis, cfg)
    drop_hold_fn = _drop_hold_fn_for_analysis(
        analysis, float(cfg.shader_drop_hold_decay_sec)
    )
    snare_fn = _scaled_pulse_fn(_snare_envelope_fn(cfg, analysis), master)
    kick_glow_fn = _scaled_pulse_fn(_kick_glow_envelope_fn(cfg, analysis), master)
    kick_punch_fn = _scaled_pulse_fn(_kick_punch_envelope_fn(cfg, analysis), master)
    auto_impact_fn = _scaled_pulse_fn(_impact_envelope_fn(cfg, analysis), master)
    user_glitch_fn = _user_glitch_envelope_fn(_timeline_clips(cfg))
    impact_fn = _combined_glitch_fn(auto_impact_fn, user_glitch_fn)
    rim_mod_step = _create_rim_modulation_stepper(cfg, analysis)
    resolved_rim_config = _effective_rim_light_config(cfg, analysis)
    beam_ctx = _build_beam_render_context(
        cfg,
        analysis,
        logo_rgba_prepared=logo_rgba_prepared,
        resolved_rim_config=resolved_rim_config,
    )
    frame_effects = _build_frame_effects_context(cfg, analysis)

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
    # ``select_video_codec`` may have promoted a different ffmpeg into the
    # resolver cache when the highest-priority binary lacked NVENC but a
    # lower-priority candidate had it. Re-resolve so the encode command and
    # the NVENC-preset probe both see the same (working) binary; otherwise
    # ``_ffmpeg_video_args`` could probe ffmpeg X for ``-preset p5`` while
    # the encode pipes through ffmpeg Y, producing a parse error mid-render.
    ffmpeg_bin = require_ffmpeg()
    # Make the encoder identity visible in logs at INFO. Without this the
    # only signal is silence (probe ok -> no log) vs noise ("Promoting...",
    # "falling back to CPU encoder...") and the user can't tell from the
    # transcript whether their slow render is using NVENC or libx264.
    LOGGER.info(
        "Compositor encoder: codec=%s ffmpeg=%s",
        codec,
        ffmpeg_bin,
    )

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

    # Shared progress state. Initialised on the request thread *before* the
    # producer starts so the consumer's progress poll has valid totals from
    # the very first tick. ``started_at`` is reset in the producer once it
    # begins the actual frame loop (so warmup time isn't billed against the
    # rendering FPS shown to the user).
    stats = _CompositorStats(
        total_frames=frame_count,
        started_at=time.monotonic(),
        layer_label="",
        phase="initializing",
    )

    has_typography = bool(aligned_words)
    has_title = title_rgba is not None
    has_logo = logo_rgba_prepared is not None and logo_position_norm is not None
    has_pulse = pulse_fn is not None

    layer_label = _active_layers_label(
        has_typography=has_typography,
        has_title=has_title,
        has_logo=has_logo,
        has_pulse=has_pulse,
        has_voidcat_ascii=bool(cfg.voidcat_ascii_ctx is not None),
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

    # Update stats.layer_label up-front so the consumer's first progress
    # tick already shows the correct layer info.
    stats.layer_label = layer_label

    def _produce() -> None:
        try:
            # NB: NEVER call ``progress(...)`` from this function. Producer
            # runs on a daemon thread and Gradio drops cross-thread progress
            # calls. Update ``stats.phase`` instead -- the consumer (request
            # thread) polls stats and emits progress at ``_PROGRESS_TICK_SEC``.
            stats.phase = "initializing GPU shader context"
            with ReactiveShader(
                cfg.shader_name,
                width=cfg.width,
                height=cfg.height,
                num_bands=cfg.num_bands,
                palette=cfg.shader_palette,
            ) as reactive:
                typo_layer: KineticTypographyLayer | None = None
                if has_typography:
                    stats.phase = (
                        f"preparing kinetic typography "
                        f"({len(aligned_words or [])} words)"
                    )
                    typo_layer = KineticTypographyLayer(
                        list(aligned_words or ()),
                        motion=cfg.typography_motion,
                        font_path=cfg.font_path,
                        width=cfg.width,
                        height=cfg.height,
                        font_size=cfg.font_size,
                        baseline_y_ratio=float(cfg.typography_baseline_y_ratio),
                        base_color=cfg.base_color,
                        shadow_color=cfg.shadow_color,
                    )
                try:
                    render_started = time.monotonic()
                    last_log_t = render_started
                    # Reset the FPS-measurement clock now that warmup is
                    # done -- the displayed fps should reflect rendering
                    # rate, not (renderer + warmup) rate.
                    stats.started_at = render_started
                    stats.phase = "warming up"
                    # Per-stage breakdown samples. Goal: catch the
                    # Pinokio cliff where the producer goes from 4 fps
                    # (~225 ms/frame) to 0.2 fps (~5 s/frame) within a
                    # 2-frame window even though per-stage cost is
                    # constant in *all* observed samples. Profiling
                    # every Nth frame creates blind spots straddling
                    # the cliff, so after warmup we profile **every
                    # frame** -- the cost of the ~12 ``perf_counter()``
                    # calls is sub-microsecond per frame, far below
                    # the 200 ms baseline. Logging is throttled: we
                    # always emit on slow frames (so the cliff is
                    # immediately visible) and otherwise only every
                    # ``_LOG_EVERY`` frames so the terminal stays
                    # readable.
                    _PROFILE_OFFSET = 8  # warmup margin
                    _LOG_EVERY = 30
                    _LOG_SLOW_MS = 500.0
                    # Optional RSS readout via psutil. Used to spot the
                    # "memory grows then everything stalls" pattern --
                    # if process RSS climbs steadily into the multi-GB
                    # range as the producer slows, the OS is paging
                    # frame buffers to disk. Best-effort: skipped when
                    # psutil isn't installed or the host blocks the
                    # query.
                    try:
                        import psutil  # type: ignore
                        _proc = psutil.Process()
                    except Exception:  # noqa: BLE001 - diagnostics only
                        _proc = None
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

                        # Allocate a timings dict for every post-warmup
                        # frame. ``_render_compositor_frame`` populates
                        # it; ``perf_counter()`` is sub-microsecond so
                        # this costs ~10 us against the 200 ms+ frame
                        # baseline (i.e. <0.01% overhead).
                        stage_timings: dict[str, float] | None = (
                            {} if i >= _PROFILE_OFFSET else None
                        )
                        _frame_t0 = (
                            time.perf_counter()
                            if stage_timings is not None else 0.0
                        )

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
                            kick_glow_fn=kick_glow_fn,
                            kick_punch_fn=kick_punch_fn,
                            impact_fn=impact_fn,
                            rim_mod_step=rim_mod_step,
                            resolved_rim_config=resolved_rim_config,
                            beam_ctx=beam_ctx,
                            frame_effects=frame_effects,
                            stage_timings=stage_timings,
                        )

                        if stage_timings is not None:
                            _bgr_t0 = time.perf_counter()
                        bgr = np.ascontiguousarray(composited[:, :, ::-1])
                        if stage_timings is not None:
                            _bgr_t1 = time.perf_counter()
                            stage_timings["bgr_convert"] = _bgr_t1 - _bgr_t0
                            _tb_t0 = _bgr_t1
                        # Splitting tobytes from qput reveals encoder
                        # back-pressure: a slow consumer keeps the
                        # bounded queue full, so producer ``put`` blocks
                        # until the encoder drains a slot. When we see
                        # ``qput`` ballooning while every other stage
                        # stays flat, the bottleneck is downstream of
                        # the producer (ffmpeg / pipe / disk).
                        frame_bytes = bgr.tobytes()
                        if stage_timings is not None:
                            _tb_t1 = time.perf_counter()
                            stage_timings["tobytes"] = _tb_t1 - _tb_t0
                            _q_t0 = _tb_t1
                        frame_q.put(frame_bytes)
                        if stage_timings is not None:
                            _q_t1 = time.perf_counter()
                            stage_timings["qput"] = _q_t1 - _q_t0
                            total_ms = (_q_t1 - _frame_t0) * 1000.0
                            # Log on every slow frame so a sudden
                            # cliff (e.g. wall jumping from 225 ms to
                            # 5 s) is immediately attributable to a
                            # stage. Otherwise emit only every
                            # ``_LOG_EVERY`` frames -- enough to
                            # confirm steady state without spamming.
                            offset_i = i - _PROFILE_OFFSET
                            should_log = (
                                total_ms >= _LOG_SLOW_MS
                                or offset_i == 0
                                or (offset_i > 0
                                    and offset_i % _LOG_EVERY == 0)
                            )
                            if should_log:
                                parts = [
                                    f"{k}={v * 1000.0:.1f}ms"
                                    for k, v in stage_timings.items()
                                ]
                                rss_str = ""
                                if _proc is not None:
                                    try:
                                        rss_mb = (
                                            _proc.memory_info().rss
                                            / (1024.0 * 1024.0)
                                        )
                                        rss_str = f" rss={rss_mb:.0f}MB"
                                    except Exception:  # noqa: BLE001
                                        rss_str = ""
                                LOGGER.info(
                                    "Compositor stage breakdown "
                                    "(frame %d, wall=%.1fms, "
                                    "qsize=%d/%d%s): %s",
                                    i,
                                    total_ms,
                                    frame_q.qsize(),
                                    cfg.queue_size,
                                    rss_str,
                                    " ".join(parts),
                                )

                        done = i + 1
                        # Producer-side counter for the throttled INFO log
                        # below; the UI uses ``frames_encoded`` (consumer).
                        stats.frames_produced = done
                        # First produced frame -> exit the "warming up" phase.
                        if done == 1:
                            stats.phase = "encoding"

                        now = time.monotonic()
                        elapsed = max(1e-6, now - render_started)
                        avg_fps = done / elapsed
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

        # Consumer feed loop. Two responsibilities:
        # 1. Pull frames off the queue and pipe them to ffmpeg stdin.
        # 2. Emit progress(...) updates to the UI from THIS thread (request
        #    thread). Gradio's progress mechanism only reliably propagates
        #    cross-thread when called from the request thread, hence the
        #    move from producer-side progress() calls.
        #
        # We use ``frame_q.get(timeout=_PROGRESS_TICK_SEC)`` so the consumer
        # wakes up even when the queue is empty (during warmup before the
        # first frame is produced, or during a long ffmpeg stall) and can
        # still tick the progress bar -- otherwise the UI would freeze on
        # the last-shown message until the next frame arrives.
        last_progress_t = 0.0

        def _emit_progress(force: bool = False) -> None:
            nonlocal last_progress_t
            if progress is None:
                return
            now = time.monotonic()
            if not force and (now - last_progress_t) < _PROGRESS_TICK_SEC:
                return
            last_progress_t = now
            p, msg = stats.progress_pair()
            try:
                progress(p, msg)
            except Exception as exc:  # noqa: BLE001 - UI must never break encode
                LOGGER.debug("Compositor progress callback raised: %s", exc)

        # Push an initial tick so the bar shows the warmup phase string
        # immediately rather than the orchestrator's stale "Compositing
        # video..." label.
        _emit_progress(force=True)

        try:
            while True:
                try:
                    chunk = frame_q.get(timeout=_PROGRESS_TICK_SEC)
                except queue.Empty:
                    # No frame yet (warmup or encoder stall) -- still tick
                    # the UI so the user sees "warming up", "preparing
                    # typography", etc., as the producer transitions phases.
                    _emit_progress()
                    continue
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
                stats.frames_encoded += 1
                _emit_progress()
            # Flush a final 100%-ish tick so the bar's last visible state
            # matches the actual encoded count rather than whatever was
            # there ~250ms before the queue closed.
            _emit_progress(force=True)
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

    elapsed_sec = max(1e-6, time.monotonic() - stats.started_at)
    avg_fps = stats.frames_encoded / elapsed_sec if elapsed_sec > 0 else 0.0
    render_stats = CompositorRenderStats(
        frame_count=stats.frames_encoded,
        elapsed_sec=elapsed_sec,
        avg_fps=avg_fps,
        video_codec=codec,
        ffmpeg_path=ffmpeg_bin,
    )

    return CompositorResult(
        run_id=rid,
        output_dir=out_dir,
        output_mp4=out_mp4,
        frame_count=frame_count,
        audio_path=audio_p,
        thumbnail_png=thumb_path,
        render_stats=render_stats,
    )


__all__: Sequence[str] = [
    "DEFAULT_QUEUE_SIZE",
    "PULSE_MODE_BASS",
    "PULSE_MODE_BEATS",
    "CompositorConfig",
    "CompositorRenderStats",
    "CompositorResult",
    "ProgressFn",
    "PulseFn",
    "render_full_video",
    "render_single_frame",
]
