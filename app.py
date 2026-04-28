"""Gradio entrypoint: Glitchframe UI.

Provides the tabbed layout, upload / analyze / align / preview / render actions,
and progress callbacks. Long-running buttons (Analyze, Preview 10 s, Render
full video) dispatch to :mod:`orchestrator` and report an ETA derived from
elapsed time / progress ratio alongside each stage message.
"""

from __future__ import annotations

# --- Speechbrain LazyModule Windows compat (must precede any heavyweight import) ---
# whisperx → pyannote-audio → speechbrain>=1.0,<1.1 registers many ``LazyModule``
# objects (``speechbrain.integrations.k2_fsa``, ``...integrations.nlp``,
# multiple ``deprecated_redirect`` shims, ...). Their ``__getattr__`` force-imports
# the target on ANY attribute access — including ``hasattr(mod, '__file__')``.
# CPython's ``inspect.getmodule`` walks ``sys.modules`` and probes ``__file__`` on
# every entry; ``librosa`` (audio ingest) calls ``inspect.stack()`` which triggers
# that walk. Speechbrain has a guard intended to short-circuit ``inspect.py``
# probes, but the guard hard-codes ``"/inspect.py"`` (forward slash) which is a
# **no-op on Windows** (path is ``...\\Lib\\inspect.py``). Result: every audio
# upload force-imports every speechbrain integration, and whichever integration
# is missing its optional dep (k2 / flair / ...) crashes the ingest.
#
# ``pipeline._speechbrain_compat.patch_speechbrain_lazy_module`` replaces
# ``LazyModule.ensure_module`` with a separator-aware version (the one-character
# upstream fix for issue #2995). It runs after speechbrain is loaded — see the
# call in ``main()``. As a belt-and-braces fallback we also pre-stub ``k2`` in
# ``sys.modules`` so speechbrain's k2_fsa integration imports cleanly even if
# the patch ever fails to apply (e.g. a future speechbrain refactors the class).
import sys as _sys
import types as _types

_sys.modules.setdefault("k2", _types.ModuleType("k2"))
del _sys, _types
# --- end speechbrain compat ---

import json
import logging
import sys
import time
import traceback
import urllib.parse
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("glitchframe.app")


def _format_exception(label: str, exc: BaseException) -> str:
    """Return ``label: ExcType(msg)`` + last frame so the Gradio log has a hint."""
    tb = traceback.extract_tb(exc.__traceback__)
    origin = ""
    if tb:
        frame = tb[-1]
        fname = frame.filename.replace("\\", "/").rsplit("/", 2)[-1]
        origin = f" @ {fname}:{frame.lineno} in {frame.name}"
    return f"{label}: {type(exc).__name__}: {exc}{origin}"

from config import ensure_runtime_dirs, get_preset, load_preset_registry, song_cache_dir
from orchestrator import (
    OrchestratorInputs,
    RenderResult,
    orchestrate_analysis,
    orchestrate_full_render,
    orchestrate_preview_10s,
)
from pipeline.audio_ingest import (
    ANALYSIS_MONO_WAV_NAME,
    ORIGINAL_WAV_NAME,
    ingest_audio_file,
    preview_value_for_gradio,
)
from pipeline.effects_editor import (
    bake_auto_schedule,
    build_editor_html as build_effects_editor_html,
    load_editor_state as load_effects_editor_state,
    save_edited_timeline,
)
from pipeline.effects_timeline import (
    load as load_effects_timeline,
    save as save_effects_timeline,
)
from pipeline.background import (
    BACKGROUND_MODES,
    MODE_ANIMATEDIFF,
    MODE_SDXL_STILLS,
    MODE_STATIC_KENBURNS,
    normalize_background_mode,
)
from pipeline.builtin_shaders import BUILTIN_SHADERS
from pipeline.beat_pulse import (
    build_bass_pulse_track,
    build_hi_transient_track,
    build_lo_transient_track,
    build_mid_transient_track,
)
from pipeline.musical_events import sample_drop_hold
from pipeline.compositor import DEFAULT_SHADER_BASS_DECAY_SEC, DEFAULT_SHADER_BASS_SENSITIVITY
from pipeline.logo_composite import composite_logo_from_path
from pipeline.lyrics_editor import (
    build_editor_html,
    load_editor_state,
    revert_manual_edits,
    save_edited_alignment,
)
from pipeline.reactive_shader import (
    ReactiveShader,
    composite_premultiplied_rgba_over_rgb,
    resolve_builtin_shader_stem,
    uniforms_at_time,
)
from pipeline.voidcat_ascii import build_voidcat_ascii_context, render_voidcat_ascii_rgba

# PRD §3.1 preset ids when no YAML files are present yet
_DEFAULT_PRESET_CHOICES: list[str] = [
    "neon-synthwave",
    "minimal-mono",
    "organic-liquid",
    "glitch-vhs",
    "cosmic-flow",
    "lofi-warm",
]

_LOG_MAX_CHARS = 12_000

_RESOLUTION_CHOICES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("1080p (1920×1080)", (1920, 1080)),
    ("4K (3840×2160)", (3840, 2160)),
)
_RESOLUTION_MAP: dict[str, tuple[int, int]] = dict(_RESOLUTION_CHOICES)
_RESOLUTION_DEFAULT_LABEL = _RESOLUTION_CHOICES[0][0]


def _parse_resolution(label: str | None) -> tuple[int, int]:
    if label and label in _RESOLUTION_MAP:
        return _RESOLUTION_MAP[label]
    return _RESOLUTION_MAP[_RESOLUTION_DEFAULT_LABEL]


def _format_eta(seconds: float) -> str:
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


class _EtaProgress:
    """Wrap ``gr.Progress`` with an elapsed/ETA suffix in the stage message."""

    def __init__(self, progress: gr.Progress | None) -> None:
        self._progress = progress
        self._start: float | None = None

    def __call__(self, p: float, msg: str) -> None:
        now = time.monotonic()
        if self._start is None:
            self._start = now
        elapsed = now - self._start
        desc = msg
        if p > 0.02:
            eta = max(0.0, (elapsed / p) - elapsed)
            desc = f"{msg} · elapsed {_format_eta(elapsed)} · ETA {_format_eta(eta)}"
        if self._progress is not None:
            self._progress(max(0.0, min(1.0, float(p))), desc=desc)


def _preset_choices(reg: dict[str, dict] | None = None) -> list[str]:
    r = reg if reg is not None else load_preset_registry()
    return sorted(r.keys()) if r else list(_DEFAULT_PRESET_CHOICES)


def _apply_preset(preset_id: str | None) -> tuple[str, str, str, str]:
    """Fill prompt, shader, typography, and palette from the preset registry."""
    if not preset_id:
        return "", BUILTIN_SHADERS[0], "", ""
    try:
        p = get_preset(str(preset_id))
    except KeyError:
        return "", BUILTIN_SHADERS[0], "", ""
    return (
        str(p["prompt"]),
        str(p["shader"]),
        str(p["typo_style"]),
        ", ".join(str(c) for c in p["colors"]),
    )


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _append_log(existing: str, message: str) -> str:
    line = f"[{_ts()}] {message}\n"
    combined = (existing or "").rstrip() + "\n" + line if existing else line
    if len(combined) > _LOG_MAX_CHARS:
        combined = combined[-_LOG_MAX_CHARS:]
    return combined


def _analyze(
    song_hash: str | None,
    log: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[float | None, str]:
    """Run the audio analyzer against the current song_hash and refresh UI."""
    if not song_hash:
        progress(1.0, desc="Idle")
        return None, _append_log(log, "Analyze: no audio ingested yet.")

    progress(0.0, desc="Analyze")
    cb = _EtaProgress(progress)

    try:
        state = orchestrate_analysis(
            OrchestratorInputs(song_hash=song_hash),
            progress=cb,
            include_lyrics=False,
        )
        result = state.analysis
    except Exception as exc:
        progress(1.0, desc="Idle")
        return None, _append_log(log, f"Analyze failed: {exc}")

    tempo = result.analysis.get("tempo", {}) or {}
    bpm = float(tempo.get("bpm") or 0.0)
    source = tempo.get("source", "unknown")
    vocals_note = (
        f" | vocals={result.vocals_wav.name}"
        if result.vocals_wav is not None
        else (
            " | vocals=skipped (demucs unavailable — on Windows `py -m app` uses the "
            "global Python, not `.venv`. Run: `.venv\\Scripts\\python.exe -m app` or "
            "`python -m app` after `Activate.ps1`, then install: "
            "`python -m pip install demucs torchaudio`)"
        )
    )
    summary = (
        f"Analyze done — hash={result.song_hash} | bpm={bpm:.2f} (via {source}) | "
        f"segments={len(result.analysis.get('segments', []))}"
        f"{vocals_note} | json={result.analysis_json}"
    )
    progress(1.0, desc="Idle")
    return (bpm if bpm > 0 else None), _append_log(log, summary)


def _align_lyrics(
    song_hash: str | None,
    lyrics_text: str,
    log: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Align pasted lyrics to the current song's ``vocals.wav`` via WhisperX."""
    if not song_hash:
        progress(1.0, desc="Idle")
        return _append_log(log, "Align lyrics: no audio ingested yet.")
    if not lyrics_text or not lyrics_text.strip():
        progress(1.0, desc="Idle")
        return _append_log(log, "Align lyrics: lyrics textarea is empty.")

    progress(0.0, desc="Align lyrics")
    cb = _EtaProgress(progress)

    try:
        state = orchestrate_analysis(
            OrchestratorInputs(song_hash=song_hash, lyrics_text=lyrics_text),
            progress=cb,
            include_lyrics=True,
        )
        result = state.alignment
        if result is None:
            raise RuntimeError("Lyrics alignment produced no result")
    except Exception as exc:
        progress(1.0, desc="Idle")
        return _append_log(log, f"Align lyrics failed: {exc}")

    summary = (
        f"Align lyrics done — hash={result.song_hash} | "
        f"model={result.model} | lang={result.language} | "
        f"lines={len(result.lines)} | words={len(result.words)} | "
        f"json={result.aligned_json}"
    )
    progress(1.0, desc="Idle")
    return _append_log(log, summary)


def _coerce_path(value: object) -> str | None:
    path = getattr(value, "name", value) if value else None
    if not path or not str(path).strip():
        return None
    return str(path)


def _coerce_gradio_file_path(file: object) -> str | None:
    """Normalize ``gr.File`` / ``gr.FileData`` payloads to a path string."""
    if file is None:
        return None
    if isinstance(file, str):
        s = file.strip()
        return s or None
    if isinstance(file, (list, tuple)):
        if not file:
            return None
        return _coerce_gradio_file_path(file[0])
    if isinstance(file, dict):
        for key in ("path", "name"):
            v = file.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    path_attr = getattr(file, "path", None)
    if isinstance(path_attr, str) and path_attr.strip():
        return path_attr.strip()
    name_attr = getattr(file, "name", None)
    if isinstance(name_attr, str) and name_attr.strip():
        return name_attr.strip()
    return None


def _build_render_inputs(
    *,
    song_hash: str | None,
    bg_mode: str,
    static_bg: object,
    preset_id: str | None,
    reactive_intensity_pct: float,
    logo_file: object,
    logo_position: str,
    logo_opacity_pct: float,
    logo_max_size_pct: float,
    logo_beat_pulse: bool,
    logo_pulse_mode: str,
    logo_pulse_strength: float,
    logo_pulse_sensitivity: float,
    logo_snare_glow: bool,
    logo_glow_strength_pct: float,
    logo_snare_squeeze_pct: float,
    logo_impact_glitch_pct: float,
    logo_impact_sensitivity: float,
    logo_rim_mode: str,
    logo_rim_travel_speed: float,
    logo_rim_color_spread_deg: float,
    logo_rim_inward_mix_pct: float,
    logo_rim_direction: str,
    logo_rim_audio_reactive: bool,
    logo_rim_sync_snare: bool,
    logo_rim_sync_bass: bool,
    logo_rim_mod_strength: float,
    logo_rim_brightness_pct: float,
    logo_rim_halo_size_px: float,
    logo_rim_wave_shape: str,
    rim_beams_enabled: bool,
    logo_motion_stability_pct: float,
    show_title: bool,
    title_position: str,
    title_size: str,
    lyrics_text: str,
    enable_typography: bool,
    resolution_label: str,
    fps: int,
    artist: str,
    title: str,
    album: str,
    year: str,
    genre: str,
) -> OrchestratorInputs:
    """Assemble :class:`OrchestratorInputs` from raw Gradio values."""
    if not song_hash:
        raise ValueError("ingest audio first (no song_hash yet)")
    mode = normalize_background_mode(bg_mode)
    static_path = _coerce_path(static_bg)
    if mode == MODE_STATIC_KENBURNS and not static_path:
        raise ValueError("static-kenburns requires a background image upload")

    width, height = _parse_resolution(resolution_label)
    metadata = {
        "artist": (artist or "").strip(),
        "title": (title or "").strip(),
        "album": (album or "").strip(),
        "year": (year or "").strip(),
        "genre": (genre or "").strip(),
    }

    return OrchestratorInputs(
        song_hash=song_hash,
        background_mode=mode,
        static_background_image=static_path,
        preset_id=(preset_id or None),
        reactive_intensity_pct=float(reactive_intensity_pct),
        logo_path=_coerce_path(logo_file),
        logo_position=str(logo_position or "center"),
        logo_opacity_pct=float(logo_opacity_pct),
        logo_max_size_pct=float(np.clip(logo_max_size_pct, 0.0, 100.0)),
        logo_beat_pulse=bool(logo_beat_pulse),
        logo_pulse_mode=str(logo_pulse_mode or "bass").strip().lower(),
        logo_pulse_strength=float(logo_pulse_strength),
        logo_pulse_sensitivity=float(logo_pulse_sensitivity),
        logo_snare_glow=bool(logo_snare_glow),
        logo_glow_strength=float(np.clip(logo_glow_strength_pct, 0.0, 400.0)) / 100.0,
        logo_snare_squeeze_strength=float(np.clip(logo_snare_squeeze_pct, 0.0, 100.0))
        / 100.0,
        logo_impact_glitch_strength=float(np.clip(logo_impact_glitch_pct, 0.0, 100.0))
        / 100.0,
        logo_impact_sensitivity=float(logo_impact_sensitivity),
        logo_rim_mode=str(logo_rim_mode or "off").strip().lower(),
        logo_rim_travel_speed=float(
            np.clip(logo_rim_travel_speed, 0.0, 2.0)
        ),
        logo_rim_color_spread_deg=float(
            np.clip(logo_rim_color_spread_deg, 0.0, 180.0)
        ),
        logo_rim_inward_mix=float(np.clip(logo_rim_inward_mix_pct, 0.0, 100.0))
        / 100.0,
        logo_rim_direction=str(logo_rim_direction or "cw").strip().lower(),
        logo_rim_audio_reactive=bool(logo_rim_audio_reactive),
        logo_rim_sync_snare=bool(logo_rim_sync_snare),
        logo_rim_sync_bass=bool(logo_rim_sync_bass),
        logo_rim_mod_strength=float(np.clip(logo_rim_mod_strength, 0.0, 200.0))
        / 100.0,
        logo_rim_brightness=float(np.clip(logo_rim_brightness_pct, 0.0, 500.0))
        / 100.0,
        logo_rim_halo_spread_px=float(np.clip(logo_rim_halo_size_px, 4.0, 64.0)),
        logo_rim_wave_shape=str(logo_rim_wave_shape or "comet").strip().lower(),
        rim_beams_enabled=bool(rim_beams_enabled),
        # UI exposes 0--200 %; compositor expects a raw scale factor where
        # ``1.0`` is the default deadzone and ``2.0`` is extra stable.
        logo_motion_stability=float(np.clip(logo_motion_stability_pct, 0.0, 200.0))
        / 100.0,
        show_title=bool(show_title),
        title_position=str(title_position or "bottom-left"),
        title_size=str(title_size or "small"),
        width=int(width),
        height=int(height),
        fps=int(fps or 30),
        include_lyrics=bool(enable_typography),
        lyrics_text=lyrics_text or None,
        metadata=metadata,
    )


def _summarise_render(result: RenderResult) -> str:
    tag = "Preview" if result.is_preview else "Render"
    mp4 = result.compositor.output_mp4
    parts = [
        f"{tag} done — {mp4}",
        f"frames={result.compositor.frame_count}",
    ]
    if result.is_preview:
        parts.append(
            f"window={result.start_sec:.1f}s+{result.duration_sec or 0.0:.1f}s"
        )
    if result.metadata_path is not None:
        parts.append(f"metadata={result.metadata_path.name}")
    if result.compositor.thumbnail_png is not None:
        parts.append(f"thumbnail={result.compositor.thumbnail_png.name}")
    if result.av_sync is not None:
        parts.append(result.av_sync.message)
    return " | ".join(parts)


def _run_preview(
    song_hash: str | None,
    bg_mode: str,
    static_bg: object,
    preset_id: str | None,
    reactive_intensity_pct: float,
    logo_file: object,
    logo_position: str,
    logo_opacity_pct: float,
    logo_max_size_pct: float,
    logo_beat_pulse: bool,
    logo_pulse_mode: str,
    logo_pulse_strength: float,
    logo_pulse_sensitivity: float,
    logo_snare_glow: bool,
    logo_glow_strength_pct: float,
    logo_snare_squeeze_pct: float,
    logo_impact_glitch_pct: float,
    logo_impact_sensitivity: float,
    logo_rim_mode: str,
    logo_rim_travel_speed: float,
    logo_rim_color_spread_deg: float,
    logo_rim_inward_mix_pct: float,
    logo_rim_direction: str,
    logo_rim_audio_reactive: bool,
    logo_rim_sync_snare: bool,
    logo_rim_sync_bass: bool,
    logo_rim_mod_strength: float,
    logo_rim_brightness_pct: float,
    logo_rim_halo_size_px: float,
    logo_rim_wave_shape: str,
    rim_beams_enabled: bool,
    logo_motion_stability_pct: float,
    show_title: bool,
    title_position: str,
    title_size: str,
    lyrics_text: str,
    enable_typography: bool,
    resolution_label: str,
    fps: int,
    artist: str,
    title: str,
    album: str,
    year: str,
    genre: str,
    log: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    progress(0.0, desc="Preview 10 s")
    try:
        inputs = _build_render_inputs(
            song_hash=song_hash,
            bg_mode=bg_mode,
            static_bg=static_bg,
            preset_id=preset_id,
            reactive_intensity_pct=reactive_intensity_pct,
            logo_file=logo_file,
            logo_position=logo_position,
            logo_opacity_pct=logo_opacity_pct,
            logo_max_size_pct=logo_max_size_pct,
            logo_beat_pulse=logo_beat_pulse,
            logo_pulse_mode=logo_pulse_mode,
            logo_pulse_strength=logo_pulse_strength,
            logo_pulse_sensitivity=logo_pulse_sensitivity,
            logo_snare_glow=logo_snare_glow,
            logo_glow_strength_pct=logo_glow_strength_pct,
            logo_snare_squeeze_pct=logo_snare_squeeze_pct,
            logo_impact_glitch_pct=logo_impact_glitch_pct,
            logo_impact_sensitivity=logo_impact_sensitivity,
            logo_rim_mode=logo_rim_mode,
            logo_rim_travel_speed=logo_rim_travel_speed,
            logo_rim_color_spread_deg=logo_rim_color_spread_deg,
            logo_rim_inward_mix_pct=logo_rim_inward_mix_pct,
            logo_rim_direction=logo_rim_direction,
            logo_rim_audio_reactive=logo_rim_audio_reactive,
            logo_rim_sync_snare=logo_rim_sync_snare,
            logo_rim_sync_bass=logo_rim_sync_bass,
            logo_rim_mod_strength=logo_rim_mod_strength,
            logo_rim_brightness_pct=logo_rim_brightness_pct,
            logo_rim_halo_size_px=logo_rim_halo_size_px,
            logo_rim_wave_shape=logo_rim_wave_shape,
            rim_beams_enabled=rim_beams_enabled,
            logo_motion_stability_pct=logo_motion_stability_pct,
            show_title=show_title,
            title_position=title_position,
            title_size=title_size,
            lyrics_text=lyrics_text,
            enable_typography=enable_typography,
            resolution_label=resolution_label,
            fps=fps,
            artist=artist,
            title=title,
            album=album,
            year=year,
            genre=genre,
        )
    except ValueError as exc:
        progress(1.0, desc="Idle")
        return _append_log(log, f"Preview: {exc}")

    cb = _EtaProgress(progress)
    try:
        result = orchestrate_preview_10s(inputs, progress=cb)
    except Exception as exc:
        progress(1.0, desc="Idle")
        LOGGER.exception("Preview pipeline failed")
        return _append_log(log, _format_exception("Preview failed", exc))
    progress(1.0, desc="Idle")
    return _append_log(log, _summarise_render(result))


def _run_render(
    song_hash: str | None,
    bg_mode: str,
    static_bg: object,
    preset_id: str | None,
    reactive_intensity_pct: float,
    logo_file: object,
    logo_position: str,
    logo_opacity_pct: float,
    logo_max_size_pct: float,
    logo_beat_pulse: bool,
    logo_pulse_mode: str,
    logo_pulse_strength: float,
    logo_pulse_sensitivity: float,
    logo_snare_glow: bool,
    logo_glow_strength_pct: float,
    logo_snare_squeeze_pct: float,
    logo_impact_glitch_pct: float,
    logo_impact_sensitivity: float,
    logo_rim_mode: str,
    logo_rim_travel_speed: float,
    logo_rim_color_spread_deg: float,
    logo_rim_inward_mix_pct: float,
    logo_rim_direction: str,
    logo_rim_audio_reactive: bool,
    logo_rim_sync_snare: bool,
    logo_rim_sync_bass: bool,
    logo_rim_mod_strength: float,
    logo_rim_brightness_pct: float,
    logo_rim_halo_size_px: float,
    logo_rim_wave_shape: str,
    rim_beams_enabled: bool,
    logo_motion_stability_pct: float,
    show_title: bool,
    title_position: str,
    title_size: str,
    lyrics_text: str,
    enable_typography: bool,
    resolution_label: str,
    fps: int,
    artist: str,
    title: str,
    album: str,
    year: str,
    genre: str,
    log: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    progress(0.0, desc="Render full video")
    try:
        inputs = _build_render_inputs(
            song_hash=song_hash,
            bg_mode=bg_mode,
            static_bg=static_bg,
            preset_id=preset_id,
            reactive_intensity_pct=reactive_intensity_pct,
            logo_file=logo_file,
            logo_position=logo_position,
            logo_opacity_pct=logo_opacity_pct,
            logo_max_size_pct=logo_max_size_pct,
            logo_beat_pulse=logo_beat_pulse,
            logo_pulse_mode=logo_pulse_mode,
            logo_pulse_strength=logo_pulse_strength,
            logo_pulse_sensitivity=logo_pulse_sensitivity,
            logo_snare_glow=logo_snare_glow,
            logo_glow_strength_pct=logo_glow_strength_pct,
            logo_snare_squeeze_pct=logo_snare_squeeze_pct,
            logo_impact_glitch_pct=logo_impact_glitch_pct,
            logo_impact_sensitivity=logo_impact_sensitivity,
            logo_rim_mode=logo_rim_mode,
            logo_rim_travel_speed=logo_rim_travel_speed,
            logo_rim_color_spread_deg=logo_rim_color_spread_deg,
            logo_rim_inward_mix_pct=logo_rim_inward_mix_pct,
            logo_rim_direction=logo_rim_direction,
            logo_rim_audio_reactive=logo_rim_audio_reactive,
            logo_rim_sync_snare=logo_rim_sync_snare,
            logo_rim_sync_bass=logo_rim_sync_bass,
            logo_rim_mod_strength=logo_rim_mod_strength,
            logo_rim_brightness_pct=logo_rim_brightness_pct,
            logo_rim_halo_size_px=logo_rim_halo_size_px,
            logo_rim_wave_shape=logo_rim_wave_shape,
            rim_beams_enabled=rim_beams_enabled,
            logo_motion_stability_pct=logo_motion_stability_pct,
            show_title=show_title,
            title_position=title_position,
            title_size=title_size,
            lyrics_text=lyrics_text,
            enable_typography=enable_typography,
            resolution_label=resolution_label,
            fps=fps,
            artist=artist,
            title=title,
            album=album,
            year=year,
            genre=genre,
        )
    except ValueError as exc:
        progress(1.0, desc="Idle")
        return _append_log(log, f"Render: {exc}")

    cb = _EtaProgress(progress)
    try:
        result = orchestrate_full_render(inputs, progress=cb)
    except Exception as exc:
        progress(1.0, desc="Idle")
        LOGGER.exception("Render pipeline failed")
        return _append_log(log, _format_exception("Render failed", exc))
    progress(1.0, desc="Idle")
    return _append_log(log, _summarise_render(result))


def _toggle_static_bg_visibility(mode: str):
    try:
        m = normalize_background_mode(mode)
    except ValueError:
        m = MODE_SDXL_STILLS
    return gr.update(visible=(m == MODE_STATIC_KENBURNS))


def _clear_log() -> str:
    return ""


def _preview_background_rgb(height: int, width: int) -> np.ndarray:
    """Horizontal + vertical gradient for reactive compositing spot-checks.

    Each channel expression must produce a full ``(height, width)`` array before
    ``np.stack`` can combine them — an earlier version only used one axis per
    channel (e.g. ``g = 0.12 + 0.20 * xs`` stayed ``(1, w)``) which raised
    ``ValueError: all input arrays must have the same shape`` at stack time.
    Assigning into a pre-allocated ``(H, W, 3)`` array broadcasts implicitly.
    """
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    out = np.empty((height, width, 3), dtype=np.float32)
    # Keep the gradient dark so the reactive overlay reads the same way it
    # would over a real (usually nighttime / moody) SDXL background. A too-
    # bright test gradient caused presets like synth_grid to wash out.
    out[..., 0] = 0.04 + 0.18 * ys + 0.08 * xs
    out[..., 1] = 0.06 + 0.12 * xs
    out[..., 2] = 0.14 + 0.20 * (1.0 - ys)
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def _preview_logo_on_test_frame(
    logo_file: str | gr.utils.NamedString | None,
    position: str,
    opacity_pct: float,
    max_size_pct: float,
    log: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[np.ndarray | None, str]:
    """RGB gradient frame with optional logo overlay (no audio/GPU required)."""
    progress(0.0, desc="Logo preview")
    w, h = 960, 540
    base = _preview_background_rgb(h, w)
    path = getattr(logo_file, "name", logo_file) if logo_file else None
    if not path or not str(path).strip():
        progress(1.0, desc="Idle")
        return None, _append_log(log, "Logo preview: upload a PNG in Branding first.")
    try:
        out = composite_logo_from_path(
            base,
            path,
            position,
            opacity_pct,
            max_size_pct=float(max_size_pct),
        )
    except Exception as exc:
        progress(1.0, desc="Idle")
        return None, _append_log(log, f"Logo preview failed: {exc}")
    progress(1.0, desc="Idle")
    msg = (
        f"Logo preview OK — position={position!r} opacity={opacity_pct:.0f}% "
        f"size={max_size_pct:.0f}% of frame · canvas {w}×{h}"
    )
    return out, _append_log(log, msg)


def _parse_palette_hex(palette_hex: str | None) -> list[str]:
    """Split the preset's comma-separated palette textbox into a hex list.

    Whitespace and empty tokens are discarded. Further validation (exact
    ``#RRGGBB`` format) happens inside :class:`ReactiveShader` so bad input
    surfaces via the same error channel as any other uniform problem.
    """
    if not palette_hex:
        return []
    return [tok.strip() for tok in str(palette_hex).split(",") if tok.strip()]


def _preview_reactive_frame(
    song_hash: str | None,
    shader_stem: str,
    palette_hex: str,
    intensity_pct: float,
    logo_file: str | gr.utils.NamedString | None,
    logo_position: str,
    logo_opacity: float,
    logo_max_size_pct: float,
    log: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[np.ndarray | None, str]:
    """One GPU frame: preset shader + palette + analysis uniforms + intensity over a test background.

    The whole body is wrapped in one outer ``try/except`` so any unexpected
    exception (GL compile, uniform type mismatch, missing driver, logo decode
    error, bad palette hex, etc.) lands as a readable log entry instead of
    rendering Gradio's generic "Error" chip in both outputs — we always
    return a valid ``(None, log)`` tuple on failure. User-facing precondition
    guards (no song ingested, no shader picked, unknown stem, missing
    analysis) still produce targeted messages before we enter the wrapped
    block.
    """
    if not song_hash:
        progress(1.0, desc="Idle")
        return None, _append_log(log, "Reactive preview: ingest audio first.")

    stem_input = str(shader_stem or "").strip()
    if not stem_input:
        progress(1.0, desc="Idle")
        return None, _append_log(log, "Reactive preview: pick a Reactive shader first.")

    try:
        try:
            stem = resolve_builtin_shader_stem(stem_input)
        except (ValueError, FileNotFoundError) as exc:
            progress(1.0, desc="Idle")
            return None, _append_log(log, f"Reactive preview: {exc}")

        analysis_path = song_cache_dir(song_hash) / "analysis.json"
        if not analysis_path.is_file():
            progress(1.0, desc="Idle")
            return None, _append_log(
                log,
                f"Reactive preview: missing {analysis_path} — run Analyze first.",
            )

        progress(0.2, desc="Reactive preview")
        with analysis_path.open(encoding="utf-8") as f:
            analysis = json.load(f)

        # 960×540 keeps UI responsive; same row-major RGB as full HD pipeline.
        w, h = 960, 540
        bg = _preview_background_rgb(h, w)
        intensity = float(np.clip(intensity_pct, 0.0, 100.0)) / 100.0
        uniforms = uniforms_at_time(analysis, 0.0, intensity=intensity)
        _bt = build_bass_pulse_track(
            analysis,
            sensitivity=DEFAULT_SHADER_BASS_SENSITIVITY,
            decay_sec=DEFAULT_SHADER_BASS_DECAY_SEC,
        )
        uniforms["bass_hit"] = float(_bt.value_at(0.0)) if _bt is not None else 0.0

        # Match the compositor: inject band-transient + drop-hold signals at
        # t=0 so the preview doesn't read zero for every new shader uniform
        # that only exists on the per-frame injection path.
        _lo = build_lo_transient_track(analysis)
        _mid = build_mid_transient_track(analysis)
        _hi = build_hi_transient_track(analysis)
        uniforms["transient_lo"] = float(_lo.value_at(0.0)) if _lo is not None else 0.0
        uniforms["transient_mid"] = float(_mid.value_at(0.0)) if _mid is not None else 0.0
        uniforms["transient_hi"] = float(_hi.value_at(0.0)) if _hi is not None else 0.0
        uniforms["drop_hold"] = float(
            sample_drop_hold(0.0, (analysis.get("events") or {}).get("drops") or [])
        )

        palette = _parse_palette_hex(palette_hex)

        progress(0.5, desc="Reactive preview (GPU)")
        with ReactiveShader(stem, width=w, height=h, palette=palette or None) as reactive:
            rgb = reactive.render_frame_composited_rgb(uniforms, bg)
        if stem == "void_ascii_bg":
            ad = dict(analysis)
            if song_hash:
                ad.setdefault("song_hash", song_hash)
            vctx = build_voidcat_ascii_context(ad, palette or None)
            vox = render_voidcat_ascii_rgba(
                w, h, 0.0, uniforms=uniforms, ctx=vctx
            )
            rgb = composite_premultiplied_rgba_over_rgb(vox, rgb)

        logo_path = getattr(logo_file, "name", logo_file) if logo_file else None
        if logo_path and str(logo_path).strip():
            rgb = composite_logo_from_path(
                rgb,
                logo_path,
                logo_position,
                logo_opacity,
                max_size_pct=float(logo_max_size_pct),
            )

        progress(1.0, desc="Idle")
        palette_label = f"{len(palette)} colors" if palette else "default palette"
        msg = (
            f"Reactive preview OK — shader={stem!r} intensity={intensity_pct:.0f}% "
            f"palette={palette_label} t=0s resolution={w}×{h}"
            + (" + ascii" if stem == "void_ascii_bg" else "")
            + (
                f" | logo={logo_position!r} @ {logo_opacity:.0f}%"
                if logo_path and str(logo_path).strip()
                else ""
            )
        )
        return rgb, _append_log(log, msg)
    except Exception as exc:  # noqa: BLE001
        progress(1.0, desc="Idle")
        return None, _append_log(log, _format_exception("Reactive preview failed", exc))


def _on_audio_upload(
    file: str | gr.utils.NamedString | None,
    log: str,
) -> tuple[str | None, str | None, str]:
    """Load upload into cache, expose 44.1 kHz mono preview for ``gr.Audio``."""
    if file is None:
        return None, None, _append_log(log, "Audio cleared.")
    path = _coerce_gradio_file_path(file)
    if not path:
        return None, None, _append_log(log, "Audio cleared.")
    try:
        result = ingest_audio_file(path)
        preview = preview_value_for_gradio(result)
        msg = (
            f"Audio ingested — hash={result.song_hash} | "
            f"duration={result.duration_sec:.2f}s | cache={result.cache_dir} | "
            f"original.wav + analysis_mono.wav written"
        )
        return preview, result.song_hash, _append_log(log, msg)
    except Exception as exc:
        LOGGER.exception("Audio ingest failed (upload type=%s)", type(file).__name__)
        return None, None, _append_log(log, _format_exception("Audio ingest failed", exc))


# ---------------------------------------------------------------------------
# Lyrics-timeline editor handlers
# ---------------------------------------------------------------------------

_EDITOR_AUDIO_ELEM_ID = "mv_editor_audio"
_EDITOR_CONTAINER_ID = "mv_editor_root"
_EDITOR_STATE_JS_VAR = "_glitchframe_editor_state"

_EDITOR_EMPTY_HTML = (
    "<div style='color:#9ca3af;font-family:system-ui,sans-serif;padding:12px;'>"
    "Click <b>Load timeline</b> after ingesting audio and running "
    "<b>Align lyrics</b> to inspect and edit per-word timings.</div>"
)

_EFFECTS_EDITOR_CONTAINER_ID = "mv_fx_root"
_EFFECTS_EDITOR_AUDIO_ELEM_ID = "mv_fx_audio"
_EFFECTS_EDITOR_STATE_JS_VAR = "_glitchframe_effects_state"
_EFFECTS_EDITOR_EMPTY_HTML = (
    "<div style='color:#9ca3af;font-family:system-ui,sans-serif;padding:12px;'>"
    "Click <b>Load timeline</b> after ingesting audio and running <b>Analyze</b> "
    "to inspect and edit the effects timeline.</div>"
)


def _editor_audio_url(abs_path: Path) -> str:
    """Build a Gradio ``/file=...`` URL for a file inside ``allowed_paths``.

    Gradio 4.x serves any file under ``blocks.allowed_paths`` at
    ``/file={path:path}``. We normalise to forward slashes so Windows
    paths like ``G:\\DEV\\...\\vocals.wav`` don't bake backslashes into
    the URL, and percent-encode everything except ``/`` and ``:`` so the
    browser can still address drive letters verbatim.
    """
    as_posix = str(abs_path).replace("\\", "/")
    return "/file=" + urllib.parse.quote(as_posix, safe="/:")


def _load_editor(song_hash: str | None, log: str) -> tuple[str, str]:
    """Render the editor HTML (with its own ``<audio>``) into the tab."""
    if not song_hash:
        return (
            _EDITOR_EMPTY_HTML,
            _append_log(log, "Lyrics timeline: no audio ingested yet."),
        )
    try:
        cache_dir = song_cache_dir(song_hash)
        state = load_editor_state(cache_dir)
        audio_abs = (cache_dir / state.vocals_rel_path).resolve()
        html_blob = build_editor_html(
            state,
            audio_url=_editor_audio_url(audio_abs),
            audio_element_id=_EDITOR_AUDIO_ELEM_ID,
            container_id=_EDITOR_CONTAINER_ID,
            state_js_var=_EDITOR_STATE_JS_VAR,
        )
        msg = (
            f"Lyrics timeline ready — hash={state.song_hash} | "
            f"words={len(state.words)} | duration={state.duration_sec:.1f}s"
            + (" | manually-edited" if state.manually_edited else "")
        )
        return html_blob, _append_log(log, msg)
    except Exception as exc:  # noqa: BLE001
        return (
            _EDITOR_EMPTY_HTML,
            _append_log(log, _format_exception("Lyrics timeline load failed", exc)),
        )


def _save_editor(
    song_hash: str | None, edited_json: str, lyrics_text: str, log: str
) -> str:
    # Always leave a breadcrumb in the Python logger — the UI log string is
    # easy to miss (and worse, a silent "saved with no error" made us waste
    # 2 hours rendering with un-saved edits once already).
    LOGGER.info(
        "Save timeline clicked: song_hash=%s, payload_len=%d",
        (song_hash[:8] if song_hash else "<none>"),
        len(edited_json or ""),
    )
    if not song_hash:
        msg = "Save timeline: no song_hash (ingest audio first)."
        LOGGER.warning(msg)
        return _append_log(log, msg)
    if not edited_json or not edited_json.strip() or edited_json.strip() == "{}":
        msg = (
            "Save timeline: editor payload was empty — click 'Load timeline' "
            "first so the browser has a state object to read."
        )
        LOGGER.warning(msg)
        return _append_log(log, msg)

    payload_summary = _summarise_editor_payload(edited_json)

    try:
        cache_dir = song_cache_dir(song_hash)
        path = save_edited_alignment(
            cache_dir, edited_json, lyrics_text_snapshot=lyrics_text
        )
        LOGGER.info(
            "Save timeline succeeded: %s (%s)", path, payload_summary
        )
        return _append_log(
            log,
            f"Saved edited timings to {path} (manually_edited=true) — "
            f"{payload_summary}",
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Save timeline failed")
        return _append_log(log, _format_exception("Save timeline failed", exc))


def _summarise_editor_payload(edited_json: str) -> str:
    """Return ``"N words; [0]=<word>@<t>, [1]=..., [2]=..."`` for save logs.

    Invalid JSON / unexpected shape yields a short descriptive string rather
    than raising, because this is strictly diagnostic noise — the real save
    path does full validation separately.
    """
    try:
        data = json.loads(edited_json)
    except Exception:  # noqa: BLE001
        return f"payload unparseable (len={len(edited_json)})"
    if not isinstance(data, dict):
        return "payload not a JSON object"
    words = data.get("words")
    if not isinstance(words, list):
        return "payload has no 'words' list"
    probe: list[str] = []
    for idx, w in enumerate(words[:3]):
        if not isinstance(w, dict):
            continue
        try:
            token = str(w.get("word", "?"))
            ts = float(w.get("t_start", 0.0))
        except (TypeError, ValueError):
            continue
        probe.append(f"[{idx}]{token!r}@{ts:.3f}s")
    manual = bool(data.get("manually_edited"))
    return (
        f"{len(words)} words, manual={manual}, first: "
        + (", ".join(probe) if probe else "(none)")
    )


def _revert_editor(song_hash: str | None, log: str) -> str:
    if not song_hash:
        return _append_log(log, "Revert timeline: no song_hash.")
    try:
        cache_dir = song_cache_dir(song_hash)
        path = revert_manual_edits(cache_dir)
        if path is None:
            return _append_log(
                log, "Revert timeline: no lyrics.aligned.json to revert."
            )
        return _append_log(
            log,
            f"Cleared manually_edited on {path}. The next Align click will "
            "regenerate timings from WhisperX.",
        )
    except Exception as exc:  # noqa: BLE001
        return _append_log(log, _format_exception("Revert timeline failed", exc))


# ---------------------------------------------------------------------------
# Effects-timeline editor handlers
# ---------------------------------------------------------------------------


def _resolve_wav_path_for_effects_editor(cache_dir: Path) -> Path:
    """Prefer ``analysis_mono.wav``, then ``original.wav`` (same as ``_resolve_wav_for_peaks``)."""
    mono = cache_dir / ANALYSIS_MONO_WAV_NAME
    if mono.is_file():
        return mono
    orig = cache_dir / ORIGINAL_WAV_NAME
    if orig.is_file():
        return orig
    raise FileNotFoundError(
        f"No {ANALYSIS_MONO_WAV_NAME} or {ORIGINAL_WAV_NAME} in {cache_dir} — run ingest first."
    )


def _load_effects_editor(song_hash: str | None, log: str) -> tuple[str, str]:
    """Render the effects editor HTML (with its own ``<audio>``) into the tab."""
    if not song_hash:
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, "Effects timeline: no audio ingested yet."),
        )
    try:
        cache_dir = song_cache_dir(song_hash)
        state = load_effects_editor_state(cache_dir)
        audio_abs = _resolve_wav_path_for_effects_editor(cache_dir).resolve()
        html_blob = build_effects_editor_html(
            state,
            audio_url=_editor_audio_url(audio_abs),
            container_id=_EFFECTS_EDITOR_CONTAINER_ID,
            state_js_var=_EFFECTS_EDITOR_STATE_JS_VAR,
            audio_element_id=_EFFECTS_EDITOR_AUDIO_ELEM_ID,
        )
        n_clips = len(state.get("clips") or [])
        n_ghost = len(state.get("ghost_events") or [])
        msg = (
            f"Effects timeline ready — hash={state.get('song_hash', song_hash)} | "
            f"clips={n_clips} | ghost markers={n_ghost}"
        )
        return html_blob, _append_log(log, msg)
    except Exception as exc:  # noqa: BLE001
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, _format_exception("Effects timeline load failed", exc)),
        )


def _save_effects_editor(
    song_hash: str | None, edited_json: str, log: str
) -> tuple[str, str]:
    LOGGER.info(
        "Save effects timeline: song_hash=%s, payload_len=%d",
        (song_hash[:8] if song_hash else "<none>"),
        len(edited_json or ""),
    )
    if not song_hash:
        msg = "Save effects: no song_hash (ingest audio first)."
        LOGGER.warning(msg)
        return _EFFECTS_EDITOR_EMPTY_HTML, _append_log(log, msg)
    if not edited_json or not edited_json.strip() or edited_json.strip() == "{}":
        msg = (
            "Save effects: editor payload was empty — click 'Load timeline' first "
            "so the browser has a state object to read."
        )
        LOGGER.warning(msg)
        return _load_effects_editor(song_hash, _append_log(log, msg))
    try:
        cache_dir = song_cache_dir(song_hash)
        path = save_edited_timeline(
            cache_dir, edited_json, song_hash_from_dir=song_hash
        )
        LOGGER.info("Save effects timeline succeeded: %s", path)
        return _load_effects_editor(
            song_hash,
            _append_log(
                log, f"Saved effects timeline to {path} (manually edited timings)."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Save effects timeline failed")
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, _format_exception("Save effects timeline failed", exc)),
        )


def _bake_effects_editor(song_hash: str | None, log: str) -> tuple[str, str]:
    if not song_hash:
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, "Bake auto events: no song_hash (ingest audio first)."),
        )
    try:
        cache_dir = song_cache_dir(song_hash)
        path = bake_auto_schedule(cache_dir)
        return _load_effects_editor(
            song_hash,
            _append_log(
                log,
                f"Baked auto events into {path} (new clips merged; duplicates within "
                f"20 ms skipped).",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, _format_exception("Bake auto events failed", exc)),
        )


def _clear_effects_editor(song_hash: str | None, log: str) -> tuple[str, str]:
    if not song_hash:
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, "Clear all: no song_hash (ingest audio first)."),
        )
    try:
        cache_dir = song_cache_dir(song_hash)
        timeline = load_effects_timeline(cache_dir)
        timeline.clips = []
        path = save_effects_timeline(cache_dir, timeline)
        return _load_effects_editor(
            song_hash,
            _append_log(
                log,
                f"Cleared all clips on {path} (auto toggles and master reactivity "
                "unchanged).",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _EFFECTS_EDITOR_EMPTY_HTML,
            _append_log(log, _format_exception("Clear all failed", exc)),
        )


def build_ui() -> gr.Blocks:
    preset_registry = load_preset_registry()
    preset_choices = _preset_choices(preset_registry)
    first_id = preset_choices[0] if preset_choices else None
    first = preset_registry[first_id] if first_id and first_id in preset_registry else {}
    shader0 = first.get("shader", BUILTIN_SHADERS[0])
    if shader0 not in BUILTIN_SHADERS:
        shader0 = BUILTIN_SHADERS[0]

    with gr.Blocks(title="Glitchframe") as demo:
        gr.Markdown(
            "# Glitchframe\n"
            "Local music video generator — upload, analyze, style, and render."
        )

        with gr.Tabs():
            with gr.Tab("Audio"):
                audio_in = gr.File(
                    label="Audio file",
                    file_types=[".mp3", ".wav", ".flac", ".ogg", ".m4a"],
                )
                song_hash_state = gr.State(value=None)
                gr.Markdown(
                    "Upload writes **`cache/<hash>/original.wav`** (native rate, stereo preserved) "
                    "and **`analysis_mono.wav`** (44.1 kHz mono). Preview uses the mono file."
                )
                audio_preview = gr.Audio(
                    label="Waveform preview (44.1 kHz mono)",
                    type="filepath",
                    interactive=True,
                )

            with gr.Tab("Metadata"):
                meta_artist = gr.Textbox(label="Artist")
                meta_title = gr.Textbox(label="Title")
                meta_album = gr.Textbox(label="Album")
                meta_year = gr.Textbox(label="Year", placeholder="e.g. 2024")
                meta_genre = gr.Textbox(label="Genre")
                bpm_number = gr.Number(
                    label="BPM",
                    precision=2,
                    info="Auto-filled after Analyze.",
                )

            with gr.Tab("Branding"):
                logo_file = gr.File(
                    label="Logo (PNG, transparent recommended)",
                    file_types=[".png"],
                )
                logo_position = gr.Dropdown(
                    label="Logo position",
                    choices=[
                        "Top-left",
                        "Top-right",
                        "Bottom-left",
                        "Bottom-right",
                        "Center",
                    ],
                    value="Center",
                )
                logo_opacity = gr.Slider(
                    label="Logo opacity",
                    minimum=0,
                    maximum=100,
                    value=85,
                    step=1,
                    info="Percent (0–100; multiplied with PNG alpha when blending)",
                )
                logo_max_size_pct = gr.Slider(
                    label="Logo size",
                    minimum=5,
                    maximum=60,
                    value=30,
                    step=1,
                    info=(
                        "Percent of the shorter frame edge the logo's longest side "
                        "is capped to. Keeps the logo the same visual size across "
                        "720p / 1080p / 4K and prevents it from covering the "
                        "kinetic-type band. 30% ≈ ⅓ of the screen."
                    ),
                )
                logo_beat_pulse = gr.Checkbox(
                    label="Pulse logo on audio",
                    value=True,
                    info="Master switch for the size + brightness kick on the logo overlay.",
                )
                logo_pulse_mode = gr.Radio(
                    label="Pulse signal",
                    choices=[
                        ("Bass / kick energy (recommended)", "bass"),
                        ("Every beat (analyzer grid)", "beats"),
                    ],
                    value="bass",
                    info=(
                        "Bass keys off low-frequency hits in analysis.json so the logo"
                        " only kicks on actual kicks / sub drops. Beats mirrors every"
                        " tracked subdivision (the legacy behaviour)."
                    ),
                )
                logo_pulse_sensitivity = gr.Slider(
                    label="Bass sensitivity",
                    minimum=0.25,
                    maximum=3.0,
                    value=1.0,
                    step=0.05,
                    info=(
                        "Scales the bass envelope before clipping. < 1 tames a"
                        " bass-heavy mix; > 1 lets weaker kicks read as full pulses."
                        " Ignored when ‘Every beat’ is selected."
                    ),
                )
                logo_pulse_strength = gr.Slider(
                    label="Pulse strength",
                    minimum=0.0,
                    maximum=4.0,
                    value=2.0,
                    step=0.05,
                    info=(
                        "How big each bass hit reads (scale + brightness). "
                        "2.0 is the default; 4.0 is maximum punch for heavy drops."
                    ),
                )
                logo_snare_glow = gr.Checkbox(
                    label="Snare-reactive neon glow (behind logo)",
                    value=True,
                    info=(
                        "Blurred halo keyed off mid-band spectral hits (~snare/clap range "
                        "in the 8-band analyser). Colour follows the preset’s second "
                        "palette swatch (title shadow colour)."
                    ),
                )
                logo_glow_strength_pct = gr.Slider(
                    label="Neon glow strength",
                    minimum=0,
                    maximum=400,
                    value=200,
                    step=5,
                    info="Percent · multiplied with the snare envelope (100% = base engine level; 200% = default).",
                )
                logo_snare_squeeze_pct = gr.Slider(
                    label="Snare squeeze (logo scale)",
                    minimum=0,
                    maximum=100,
                    value=40,
                    step=5,
                    info=(
                        "Briefly shrinks the logo on mid-band (snare/clap) hits. "
                        "Uses the same detector as neon glow; 0% = off."
                    ),
                )
                logo_impact_glitch_pct = gr.Slider(
                    label="Drop / impact glitch",
                    minimum=0,
                    maximum=100,
                    value=100,
                    step=5,
                    info=(
                        "RGB-split / tear on loudness jumps (build-up → drop). "
                        "From RMS in analysis.json; 0% = off."
                    ),
                )
                logo_impact_sensitivity = gr.Slider(
                    label="Impact sensitivity",
                    minimum=0.25,
                    maximum=3.0,
                    value=1.0,
                    step=0.05,
                    info="Boosts weaker drops so they still trigger glitch (when glitch > 0%).",
                )
                with gr.Accordion(
                    "Traveling rim light (optional)",
                    open=True,
                ):
                    gr.Markdown(
                        "Adds a moving multi-colour rim behind the logo (see "
                        "`docs/technical/logo-rim-compositing.md`). **Traveling rim + "
                        "neon** is the default. **Classic** keeps only the blurred snare "
                        "neon. Rim settings do not change the song cache key."
                    )
                    logo_rim_mode = gr.Dropdown(
                        label="Rim mode",
                        choices=[
                            ("Off (no traveling rim)", "off"),
                            ("Classic neon only", "classic"),
                            ("Traveling rim + neon", "rim"),
                        ],
                        value="rim",
                        info=(
                            "Classic maps to `LogoGlowMode.CLASSIC` (no traveling wave). "
                            "Traveling rim uses `AUTO` stacking with snare neon when both apply."
                        ),
                    )
                    logo_rim_travel_speed = gr.Slider(
                        label="Rim travel speed",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.05,
                        info=(
                            "Wave phase speed in Hz (revolutions per second); "
                            "0 freezes the pattern. With the Traveling-comet "
                            "shape, 0.25-0.5 Hz reads as a clearly moving "
                            "bright spot orbiting the logo."
                        ),
                    )
                    logo_rim_color_spread_deg = gr.Slider(
                        label="Rim colour spread",
                        minimum=0.0,
                        maximum=180.0,
                        value=120.0,
                        step=1.0,
                        info=(
                            "Hue separation between rim layers (degrees). "
                            "0° uses a single tint; ≥~1° enables two layers."
                        ),
                    )
                    logo_rim_inward_mix_pct = gr.Slider(
                        label="Rim inward bleed",
                        minimum=0.0,
                        maximum=100.0,
                        value=50.0,
                        step=1.0,
                        info="How much light bleeds inward from the edge (0 = halo only, 100 = full inward mix).",
                    )
                    logo_rim_direction = gr.Radio(
                        label="Rim travel direction",
                        choices=[
                            ("Clockwise", "cw"),
                            ("Counter-clockwise", "ccw"),
                        ],
                        value="cw",
                        info="Flips the sign of travel speed (wave rotation around the logo centroid).",
                    )
                    logo_rim_audio_reactive = gr.Checkbox(
                        label="Audio-reactive rim",
                        value=True,
                        info=(
                            "When on, snare/bass envelopes from analysis.json modulate rim "
                            "intensity, phase, and inward spread (see logo-rim-audio-modulation.md)."
                        ),
                    )
                    with gr.Row():
                        logo_rim_sync_snare = gr.Checkbox(
                            label="Link rim to snare",
                            value=True,
                            info="Snare track drives rim glow multiplier and brief phase nudge.",
                        )
                        logo_rim_sync_bass = gr.Checkbox(
                            label="Link rim to bass",
                            value=True,
                            info="Bass envelope modulates inward bleed strength.",
                        )
                    logo_rim_mod_strength = gr.Slider(
                        label="Rim audio modulation strength",
                        minimum=0.0,
                        maximum=200.0,
                        value=100.0,
                        step=5.0,
                        info="Scales audio-driven rim modulation when “Audio-reactive rim” is on (100% = default).",
                    )
                    logo_rim_brightness_pct = gr.Slider(
                        label="Rim brightness",
                        minimum=0.0,
                        maximum=500.0,
                        value=300.0,
                        step=5.0,
                        info=(
                            "Scales the rim's emissive gain (`intensity`) and, "
                            "above 100%, the outer halo weight. 100% = engine "
                            "default; 300% is the product default; use 400-500% on "
                            "very bright or busy backgrounds where the rim needs to cut through."
                        ),
                    )
                    logo_rim_halo_size_px = gr.Slider(
                        label="Rim halo size",
                        minimum=4.0,
                        maximum=64.0,
                        value=22.0,
                        step=1.0,
                        info=(
                            "Outward halo falloff distance (`halo_spread_px`, "
                            "in patch pixels). Larger = softer, wider glow "
                            "around the logo silhouette."
                        ),
                    )
                    logo_rim_wave_shape = gr.Dropdown(
                        label="Rim wave shape",
                        choices=[
                            ("Traveling comet (1 bright spot)", "comet"),
                            ("Twin comets (2 bright spots)", "twin"),
                            ("Gentle lobes (2 soft)", "lobes"),
                            ("Smooth ring (3 soft, legacy)", "ring"),
                        ],
                        value="comet",
                        info=(
                            "Picks `(waves, wave_sharpness)`. Comet = single "
                            "distinctly travelling lobe (best when you want "
                            "to *see* the light move). Ring = near-uniform "
                            "glow, little perceptible motion."
                        ),
                    )
                    rim_beams_enabled = gr.Checkbox(
                        label="Rim beams on drops & snare rolls",
                        value=True,
                        info=(
                            "Emits short straight beams outward from the rim "
                            "on detected drops (and up to 3 pre-drop snare "
                            "lead-ins). Globally rate-limited to ~1 burst per "
                            "10 s so it stays earned. See "
                            "`docs/technical/logo-rim-beams.md`."
                        ),
                    )
                logo_motion_stability_pct = gr.Slider(
                    label="Logo stability (ignore micro-shake)",
                    minimum=0,
                    maximum=200,
                    value=100,
                    step=5,
                    info=(
                        "Soft deadzone for the beat-pulse + snare squeeze. "
                        "0% = legacy (every tiny pulse moves the logo); "
                        "100% = default (chill-section noise collapses to zero); "
                        "200% = extra stable. Only real kicks / snare hits "
                        "survive at higher settings."
                    ),
                )
                btn_logo_preview = gr.Button("Preview logo on test frame")
                logo_preview_image = gr.Image(
                    label="Logo overlay (test gradient)",
                    type="numpy",
                )

                gr.Markdown("### Title card (Artist - Title)")
                show_title = gr.Checkbox(
                    label="Burn Artist - Title onto every frame",
                    value=True,
                    info="Uses the Metadata tab's Artist + Title. Skipped automatically when both are blank.",
                )
                title_position = gr.Dropdown(
                    label="Title position",
                    choices=[
                        "Top-left",
                        "Top-center",
                        "Top-right",
                        "Middle-left",
                        "Center",
                        "Middle-right",
                        "Bottom-left",
                        "Bottom-center",
                        "Bottom-right",
                    ],
                    value="Bottom-left",
                )
                title_size = gr.Dropdown(
                    label="Title size",
                    choices=["Small", "Medium", "Large"],
                    value="Small",
                )

            with gr.Tab("Lyrics"):
                lyrics_text = gr.Textbox(
                    label="Lyrics (one line per line)",
                    lines=12,
                    placeholder=(
                        "Paste lyrics here…\n\n"
                        "Tip: insert a line containing just --- between "
                        "verse / pre-chorus / chorus / etc. blocks to help "
                        "the aligner separate repeated sections."
                    ),
                )
                enable_typo = gr.Checkbox(
                    label="Enable kinetic typography",
                    value=True,
                    info="Aligns pasted lyrics on demand; also controls whether render includes the typography layer.",
                )
                gr.Markdown(
                    "Align runs **WhisperX large-v3** on `cache/<hash>/vocals.wav` "
                    "and writes per-word timings to `cache/<hash>/lyrics.aligned.json`. "
                    "Analyze with demucs available must run first.\n\n"
                    "**Section markers:** put a line with just `---` between "
                    "musical sections (e.g. verse → pre-chorus → chorus). The "
                    "marker never appears on-screen — it tells the aligner "
                    "where each section starts so a repeated chorus can't bleed "
                    "into the pre-chorus buildup before it. Markers are "
                    "optional; lyrics without them still align the same as "
                    "before."
                )
                btn_align_lyrics = gr.Button("Align lyrics", variant="primary")

            with gr.Tab("Lyrics timeline"):
                gr.Markdown(
                    "Visual editor for `lyrics.aligned.json`. Drag word bars to "
                    "fix mis-aligned words before rendering; colours encode "
                    "wav2vec2 confidence (green = strong, yellow = weak, "
                    "red = very weak, grey = no score). Saved edits set "
                    "`manually_edited: true` so the next **Align lyrics** click "
                    "won't overwrite them — use **Re-align from scratch** to "
                    "discard edits and rerun WhisperX.\n\n"
                    "**Inline anchors (in the Lyrics tab):** start a line with "
                    "`[m:ss]` or `[m:ss.mmm]` to hard-pin its first word, e.g. "
                    "`[1:23] Chorus line`. The aligner uses the anchor as a hard "
                    "forced-alignment boundary so sections never drift past it."
                )
                btn_load_editor = gr.Button("Load timeline", variant="primary")
                # The editor ships its own ``<audio controls>`` tag, served
                # by Gradio's ``/file=`` proxy. No separate ``gr.Audio``
                # component here on purpose — keeping the audio inside the
                # editor puts the scrubber next to the waveform where
                # editing happens, and avoids puppeting WaveSurfer's DOM.
                editor_html = gr.HTML(value=_EDITOR_EMPTY_HTML)
                editor_state_buffer = gr.Textbox(
                    visible=False, value="", interactive=True
                )
                with gr.Row():
                    btn_save_editor = gr.Button("Save edited timings")
                    btn_revert_editor = gr.Button("Re-align from scratch")

            with gr.Tab("Effects timeline"):
                gr.Markdown(
                    "Visual editor for `effects_timeline.json`: seven colour-coded effect "
                    "rows over a full-song waveform, ghost markers from the analyser, and "
                    "a master reactivity slider. **Load timeline** needs ingest plus "
                    "**Analyze** so `analysis.json` and a cached WAV exist. **Save edits** "
                    "writes clips and auto settings to the cache. **Bake auto events** "
                    "turns analyser hints (beams, impact glitches, drop zooms) into clips "
                    "where auto is enabled, skipping times that already have a clip. "
                    "**Clear all** removes every clip but keeps your per-row auto "
                    "checkboxes and the master slider."
                )
                btn_load_fx = gr.Button("Load timeline", variant="primary")
                effects_editor_html = gr.HTML(value=_EFFECTS_EDITOR_EMPTY_HTML)
                effects_state_buffer = gr.Textbox(visible=False, value="", interactive=True)
                with gr.Row():
                    btn_save_fx = gr.Button("Save edits")
                    btn_bake_fx = gr.Button("Bake auto events")
                    btn_clear_fx = gr.Button("Clear all")

            with gr.Tab("Visual style"):
                preset_dd = gr.Dropdown(
                    label="Preset",
                    choices=preset_choices,
                    value=first_id,
                )
                with gr.Accordion("What is a preset? (read me)", open=False):
                    gr.Markdown(
                        """
A **preset** is a coherent look made of four pieces that you can still tweak
independently:

| Field | Drives | Used by |
|---|---|---|
| **Prompt** | Scene description | SDXL stills **and** AnimateDiff (prepended to per-loop motion language). Ignored by *Static image + Ken Burns*. |
| **Reactive shader** | GPU overlay on top of the background | Every background mode. Reads the **palette** below for its colors. |
| **Typography style** | Title/artist animation style | Every background mode. |
| **Color palette** | Up to 5 `#RRGGBB` colors | Fed to the shader as `u_palette[5]`. Earlier slots = base tones, later slots = accents / beat flashes. |

**How the backgrounds differ**

- *SDXL stills*: the prompt plus a structural hint (`scene N of M, t=X.Xs`) produces one keyframe per 8 s, cross-faded over the song.
- *Static image + Ken Burns*: the prompt is ignored — upload your own image and RMS drives zoom/pan/tilt.
- *AnimateDiff loops*: the prompt is combined with a preset-specific **motion flavor** (e.g. `slow cosmic drift, subtle parallax…`) plus a pacing cue (`establishing shot` → `steady motion` → `slower fade-out motion`). Each song segment renders one short loop, cross-faded at boundaries.

**The six built-in presets**

- **cosmic-flow** — deep-space nebula + `nebula_flow` shader (bar-synced drift, pre-drop build and drop bloom, Milkdrop-style dynamics).
- **glitch-vhs** — 90s tube TV + `vhs_tracking` shader (scanlines, RGB split, onset-triggered tracking bursts).
- **lofi-warm** — cozy study at golden hour + `paper_grain` shader (soft bokeh, film grain, warm vignette).
- **minimal-mono** — Swiss brutalist + `geometry_pulse` shader (concentric rings, clean monochrome).
- **neon-synthwave** — 80s retrowave skyline + `synth_grid` shader (perspective grid, sliced sun, horizon glow).
- **organic-liquid** — iridescent ink macro + `liquid_chrome` shader (domain-warped fluid, onset ripples).

See `docs/technical/visual-style-presets.md` for the full schema and
`docs/technical/background-modes.md` for the AnimateDiff prompt details.
"""
                    )
                custom_prompt = gr.Textbox(
                    label="Custom prompt (overrides preset SD prompt)",
                    lines=3,
                    value=first.get("prompt", ""),
                )
                shader_dd = gr.Dropdown(
                    label="Reactive shader",
                    choices=list(BUILTIN_SHADERS),
                    value=shader0,
                )
                typo_style = gr.Textbox(
                    label="Typography style",
                    value=first.get("typo_style", ""),
                )
                palette_hex = gr.Textbox(
                    label="Color palette (#RRGGBB, comma-separated)",
                    value=", ".join(str(c) for c in first.get("colors", [])),
                )
                bg_mode = gr.Radio(
                    label="Background mode",
                    choices=[
                        ("SDXL AI stills", MODE_SDXL_STILLS),
                        ("Static image + Ken Burns (RMS)", MODE_STATIC_KENBURNS),
                        ("AnimateDiff loops (SDXL, GPU)", MODE_ANIMATEDIFF),
                    ],
                    value=MODE_ANIMATEDIFF,
                    info=f"Canonical values: {', '.join(BACKGROUND_MODES)}",
                )
                static_bg_file = gr.File(
                    label="Static background image (Ken Burns)",
                    file_types=[".png", ".jpg", ".jpeg", ".webp"],
                    visible=False,
                )
                reactive_intensity = gr.Slider(
                    label="Reactive intensity",
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    info="Percent (uniform intensity = value ÷ 100)",
                )
                btn_reactive_preview = gr.Button("Preview reactive frame (t = 0 s)")
                reactive_preview_image = gr.Image(
                    label="Reactive + test background (GPU)",
                    type="numpy",
                )

            with gr.Tab("Output"):
                out_resolution = gr.Dropdown(
                    label="Resolution",
                    choices=[label for label, _ in _RESOLUTION_CHOICES],
                    value=_RESOLUTION_DEFAULT_LABEL,
                )
                out_fps = gr.Radio(
                    label="FPS",
                    choices=[30, 60],
                    value=30,
                )
                gr.Textbox(
                    label="Filename prefix",
                    value="musicvid",
                    info="Informational only; output path is outputs/<run_id>/output.mp4.",
                )

            with gr.Tab("Actions"):
                gr.Markdown(
                    "Long runs use **Queue**. Each action shows progress, elapsed time, and an ETA while running. "
                    "Preview renders ~10 s starting at the loudest RMS window; full render writes "
                    "`outputs/<run_id>/output.mp4`, `thumbnail.png`, and `metadata.txt`, then runs an ffprobe A/V sync check."
                )
                with gr.Row():
                    btn_analyze = gr.Button("Analyze", variant="primary")
                    btn_preview = gr.Button("Preview 10 s")
                    btn_render = gr.Button("Render full video", variant="stop")

        gr.Markdown("### Progress & log")
        run_log = gr.Textbox(
            label="Run log",
            lines=14,
            max_lines=24,
            placeholder="Action output and pipeline messages will appear here.",
        )
        with gr.Row():
            btn_clear_log = gr.Button("Clear log")

        btn_analyze.click(
            fn=_analyze,
            inputs=[song_hash_state, run_log],
            outputs=[bpm_number, run_log],
            show_progress="full",
        )
        render_inputs = [
            song_hash_state,
            bg_mode,
            static_bg_file,
            preset_dd,
            reactive_intensity,
            logo_file,
            logo_position,
            logo_opacity,
            logo_max_size_pct,
            logo_beat_pulse,
            logo_pulse_mode,
            logo_pulse_strength,
            logo_pulse_sensitivity,
            logo_snare_glow,
            logo_glow_strength_pct,
            logo_snare_squeeze_pct,
            logo_impact_glitch_pct,
            logo_impact_sensitivity,
            logo_rim_mode,
            logo_rim_travel_speed,
            logo_rim_color_spread_deg,
            logo_rim_inward_mix_pct,
            logo_rim_direction,
            logo_rim_audio_reactive,
            logo_rim_sync_snare,
            logo_rim_sync_bass,
            logo_rim_mod_strength,
            logo_rim_brightness_pct,
            logo_rim_halo_size_px,
            logo_rim_wave_shape,
            rim_beams_enabled,
            logo_motion_stability_pct,
            show_title,
            title_position,
            title_size,
            lyrics_text,
            enable_typo,
            out_resolution,
            out_fps,
            meta_artist,
            meta_title,
            meta_album,
            meta_year,
            meta_genre,
            run_log,
        ]
        btn_preview.click(
            fn=_run_preview,
            inputs=render_inputs,
            outputs=[run_log],
            show_progress="full",
        )
        btn_render.click(
            fn=_run_render,
            inputs=render_inputs,
            outputs=[run_log],
            show_progress="full",
        )
        bg_mode.change(
            fn=_toggle_static_bg_visibility,
            inputs=[bg_mode],
            outputs=[static_bg_file],
        )
        btn_align_lyrics.click(
            fn=_align_lyrics,
            inputs=[song_hash_state, lyrics_text, run_log],
            outputs=[run_log],
            show_progress="full",
        )

        btn_load_editor.click(
            fn=_load_editor,
            inputs=[song_hash_state, run_log],
            outputs=[editor_html, run_log],
            show_progress="full",
        )
        # Save: use Gradio's canonical single-step ``js=`` pattern where the
        # JS callback returns an *array* that becomes the Python fn's inputs.
        # A previous two-step chain (``.click(js=...).then(fn=..., inputs=[
        # editor_state_buffer, ...])``) silently dropped the browser-side
        # payload under certain Gradio 4.x conditions — the buffer textbox
        # never got updated before ``.then()`` read it, so Python received an
        # empty string and bailed out via the "editor payload was empty" log
        # line. That made 2 hours of careful timeline edits look like they
        # "saved without error" while actually saving nothing. The single-
        # step variant below has no such race: the returned array IS the
        # input tuple to ``_save_editor``, so there's no intermediate state.
        btn_save_editor.click(
            fn=_save_editor,
            inputs=[song_hash_state, editor_state_buffer, lyrics_text, run_log],
            outputs=[run_log],
            show_progress="full",
            js=(
                "(song_hash, _buf, lyrics_text, log) => ["
                "song_hash, "
                "JSON.stringify(window." + _EDITOR_STATE_JS_VAR + " || {}), "
                "lyrics_text, "
                "log"
                "]"
            ),
        ).then(
            fn=_load_editor,
            inputs=[song_hash_state, run_log],
            outputs=[editor_html, run_log],
        )
        btn_revert_editor.click(
            fn=_revert_editor,
            inputs=[song_hash_state, run_log],
            outputs=[run_log],
        )

        btn_load_fx.click(
            fn=_load_effects_editor,
            inputs=[song_hash_state, run_log],
            outputs=[effects_editor_html, run_log],
            show_progress="full",
        )
        btn_save_fx.click(
            fn=_save_effects_editor,
            inputs=[song_hash_state, effects_state_buffer, run_log],
            outputs=[effects_editor_html, run_log],
            show_progress="full",
            js=(
                "(song_hash, _buf, log) => ["
                "song_hash, "
                "JSON.stringify(window." + _EFFECTS_EDITOR_STATE_JS_VAR + " || {}), "
                "log"
                "]"
            ),
        )
        btn_bake_fx.click(
            fn=_bake_effects_editor,
            inputs=[song_hash_state, run_log],
            outputs=[effects_editor_html, run_log],
            show_progress="full",
        )
        btn_clear_fx.click(
            fn=_clear_effects_editor,
            inputs=[song_hash_state, run_log],
            outputs=[effects_editor_html, run_log],
            show_progress="full",
        )

        btn_clear_log.click(fn=_clear_log, inputs=None, outputs=[run_log])

        audio_in.change(
            fn=_on_audio_upload,
            inputs=[audio_in, run_log],
            outputs=[audio_preview, song_hash_state, run_log],
        )

        preset_dd.change(
            fn=_apply_preset,
            inputs=[preset_dd],
            outputs=[custom_prompt, shader_dd, typo_style, palette_hex],
        )
        btn_logo_preview.click(
            fn=_preview_logo_on_test_frame,
            inputs=[
                logo_file,
                logo_position,
                logo_opacity,
                logo_max_size_pct,
                run_log,
            ],
            outputs=[logo_preview_image, run_log],
            show_progress="full",
        )
        btn_reactive_preview.click(
            fn=_preview_reactive_frame,
            inputs=[
                song_hash_state,
                shader_dd,
                palette_hex,
                reactive_intensity,
                logo_file,
                logo_position,
                logo_opacity,
                logo_max_size_pct,
                run_log,
            ],
            outputs=[reactive_preview_image, run_log],
            show_progress="full",
        )

    return demo


def _log_runtime_python_and_optional_deps() -> None:
    """Log which interpreter runs the UI (Windows ``py -m app`` often ignores ``.venv``)."""
    LOGGER.info("Python executable (use this for pip install): %s", sys.executable)
    # So Pinokio users see versions in the app console without a shell (and so
    # ctranslate2 can resolve torch\\lib before whisperx/ctranslate2 import).
    try:
        from pipeline.win_cuda_path import ensure_windows_cuda_dll_paths

        ensure_windows_cuda_dll_paths()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        LOGGER.info("torch %s (CUDA: %s)", torch.__version__, torch.cuda.is_available())
        # Track A pins torch 2.2.2 which predates torch.xpu (added in 2.3).
        # Newer diffusers references torch.xpu at import time without a
        # hasattr() guard — installing a stub here keeps AnimateDiff SDXL
        # imports working on Track A while leaving real torch.xpu untouched
        # on Track B (torch >= 2.4). See pipeline/_torch_xpu_compat.py.
        try:
            from pipeline._torch_xpu_compat import patch_torch_xpu

            patch_torch_xpu()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Could not apply torch.xpu compat stub: %s — newer "
                "diffusers (AnimateDiff SDXL) may fail to import on "
                "PyTorch < 2.3 (Track A).",
                exc,
            )
    except Exception:  # noqa: BLE001
        pass
    try:
        import demucs  # noqa: F401
        import torchaudio  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "demucs/torchaudio not importable: %s — install into this env: "
            "%s -m pip install demucs torchaudio",
            exc,
            sys.executable,
        )
    else:
        LOGGER.info("demucs + torchaudio import OK (vocal stem can run)")
    try:
        import whisperx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "whisperx not importable: %s — install into this env: "
            "%s -m pip install -e .[lyrics]",
            exc,
            sys.executable,
        )
    else:
        LOGGER.info("whisperx import OK (lyrics alignment can run)")
        # whisperx pulls speechbrain transitively. Patch its LazyModule guard
        # NOW, before any user-triggered code calls inspect.stack() / librosa.
        # See pipeline/_speechbrain_compat.py for the upstream-bug rationale.
        try:
            from pipeline._speechbrain_compat import (
                patch_speechbrain_lazy_module,
            )

            patch_speechbrain_lazy_module()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Could not apply speechbrain LazyModule compat patch: %s — "
                "audio ingest may crash on Windows when speechbrain "
                "integrations have missing optional deps (k2 / flair).",
                exc,
            )
        try:
            import ctranslate2

            cv = str(getattr(ctranslate2, "__version__", "")).strip() or "unknown"
            LOGGER.info("ctranslate2 %s", cv)
            if sys.platform == "win32" and cv != "unknown":
                try:
                    parts = cv.split(".")
                    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
                except (ValueError, IndexError):
                    major, minor = 0, 0
                py_lt_313 = sys.version_info < (3, 13)
                # Track A (README / pyproject): Win + py3.11–3.12 pins ctranslate2 4.4.x + torch cu121.
                # Track B: Win + py3.13 uses ctranslate2>=4.5 + cu124 — do not tell Track A users to "upgrade".
                if py_lt_313 and major == 4 and minor == 4:
                    LOGGER.info(
                        "ctranslate2 %s matches the Windows pinned lyrics stack (WhisperX 3.3.0 / cu121). "
                        "If Align lyrics still fails with cudnn DLL errors, your Pinokio env differs from "
                        "a working local venv — run `pip freeze` in both and compare; see README troubleshooting.",
                        cv,
                    )
                elif not py_lt_313 and (major < 4 or (major == 4 and minor < 5)):
                    LOGGER.warning(
                        "ctranslate2 %s — for Python 3.13+ on Windows use >=4.5 with PyTorch cu124: "
                        '%s -m pip install -U "ctranslate2>=4.5.0,<5"',
                        cv,
                        sys.executable,
                    )
                elif py_lt_313 and (major < 4 or (major == 4 and minor < 4)):
                    LOGGER.warning(
                        "ctranslate2 %s is unusually old — reinstall extras: %s -m pip install -e \".[all]\"",
                        cv,
                        sys.executable,
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read ctranslate2 version: %s", exc)


def main() -> None:
    ensure_runtime_dirs()
    _log_runtime_python_and_optional_deps()
    app = build_ui()
    app.queue()
    # The lyrics-timeline editor plays ``cache/<hash>/vocals.wav`` via
    # ``gr.Audio(value=<path>)``. Gradio only serves files from paths in
    # its allowed list, so we explicitly whitelist the cache directory.
    from config import CACHE_DIR

    app.launch(allowed_paths=[str(CACHE_DIR)])


if __name__ == "__main__":
    main()
