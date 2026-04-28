"""
M1 spectrum visualizer: solid background, 8-band bars from ``analysis.json``,
raw ``bgr24`` frames into ffmpeg (NVENC by default), mux ``original.wav``.

CI / hosts without NVENC: set ``GLITCHFRAME_FFMPEG_VIDEO_CODEC=libx264`` (or legacy
``MUSICVIDS_FFMPEG_VIDEO_CODEC``) and optional ``GLITCHFRAME_FFMPEG_VIDEO_ARGS`` / legacy
``MUSICVIDS_FFMPEG_VIDEO_ARGS`` (space-separated extra args after ``-c:v``), e.g.
``-preset medium -crf 20``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from config import OUTPUTS_DIR, new_run_id
from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.audio_ingest import ORIGINAL_WAV_NAME
from pipeline.ffmpeg_tools import (
    require_ffmpeg,
    select_nvenc_preset,
    select_video_codec,
)

DEFAULT_FPS = 30
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


@dataclass(frozen=True)
class SpectrumRenderResult:
    run_id: str
    output_dir: Path
    output_mp4: Path
    frame_count: int
    audio_path: Path
    analysis_path: Path


def _load_analysis(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid analysis.json (expected object): {path}")
    return data


def _spectrum_array(analysis: dict[str, Any]) -> tuple[np.ndarray, float]:
    spec = analysis.get("spectrum")
    if not isinstance(spec, dict):
        raise KeyError("analysis.json missing 'spectrum' object")
    values = spec.get("values")
    if not isinstance(values, list) or not values:
        raise KeyError("spectrum.values missing or empty")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"spectrum.values must be 2D (frames, bands), got {arr.shape}")
    fps = float(spec.get("fps", analysis.get("fps", DEFAULT_FPS)))
    if fps <= 0:
        raise ValueError(f"Invalid spectrum fps: {fps}")
    return arr, fps


def _bands_at_time(values: np.ndarray, t: float, spec_fps: float) -> np.ndarray:
    """Linear interpolation along the spectrum time axis (seconds)."""
    n_frames = int(values.shape[0])
    if n_frames == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    idx_f = float(t) * spec_fps
    if idx_f <= 0.0:
        row = values[0]
    elif idx_f >= n_frames - 1:
        row = values[-1]
    else:
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, n_frames - 1)
        frac = idx_f - i0
        row = (1.0 - frac) * values[i0] + frac * values[i1]
    return np.clip(row, 0.0, 1.0)


def _draw_frame_rgb(
    width: int,
    height: int,
    bands: np.ndarray,
    *,
    background_rgb: tuple[int, int, int],
    bar_rgb: tuple[int, int, int],
    margin_x: int = 120,
    margin_bottom: int = 140,
    max_bar_frac: float = 0.55,
    gap_px: int = 12,
) -> np.ndarray:
    """Return ``uint8`` RGB array ``(H, W, 3)`` via Pillow (solid BG + bars)."""
    img = Image.new("RGB", (width, height), background_rgb)
    draw = ImageDraw.Draw(img)
    n = int(bands.shape[0])
    usable_w = width - 2 * margin_x
    total_gap = gap_px * max(0, n - 1)
    bar_w = max(1, (usable_w - total_gap) // max(1, n))
    base_y = height - margin_bottom
    max_h = int(float(height) * max_bar_frac)

    for i in range(n):
        bh = int(float(bands[i]) * float(max_h))
        x0 = margin_x + i * (bar_w + gap_px)
        x1 = x0 + bar_w
        y0 = base_y - bh
        draw.rectangle([x0, y0, x1, base_y], fill=bar_rgb)

    return np.asarray(img, dtype=np.uint8)


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def _ffmpeg_video_args(codec: str, *, ffmpeg_bin: str | None = None) -> list[str]:
    """Encoder-specific args for *codec*, adapting to the active ffmpeg build.

    NVENC's preset name family changed in FFmpeg 4.4 / NVENC SDK 11
    (``slow``/``medium``/``hp``/``hq`` → ``p1``..``p7``). When the user
    has a pre-4.4 ffmpeg as their resolved binary (common with Pinokio's
    conda-env-bundled ffmpeg), the modern ``p5`` preset is rejected at
    encode time with ``Undefined constant or missing '(' in 'p5'`` and
    the whole render aborts. :func:`select_nvenc_preset` probes the
    binary once and returns ``"slow"`` instead so the encode still
    succeeds at near-equivalent quality. *ffmpeg_bin* lets callers
    inject the resolved path; ``None`` falls back to whatever
    :func:`resolve_ffmpeg` returns.
    """
    if codec == "h264_nvenc":
        preset = select_nvenc_preset(ffmpeg_bin)
        return ["-preset", preset, "-rc", "vbr", "-cq", "19", "-b:v", "12M"]
    extra = (
        os.environ.get("GLITCHFRAME_FFMPEG_VIDEO_ARGS", "").strip()
        or os.environ.get("MUSICVIDS_FFMPEG_VIDEO_ARGS", "").strip()
    )
    if extra:
        return extra.split()
    if codec == "libx264":
        return ["-preset", "medium", "-crf", "23"]
    return ["-b:v", "12M"]


def _build_ffmpeg_cmd(
    *,
    width: int,
    height: int,
    fps: int,
    audio_path: Path,
    output_mp4: Path,
    video_codec: str,
    audio_start_sec: float | None = None,
    audio_duration_sec: float | None = None,
    ffmpeg_bin: str | None = None,
) -> list[str]:
    """
    Build the ffmpeg command for the raw ``bgr24`` pipe + muxed audio encode.

    ``audio_start_sec`` / ``audio_duration_sec`` let callers trim the audio
    input (e.g. for a preview slice); both are applied as input options to
    the audio stream only so the raw video pipe stays frame-aligned.
    ``ffmpeg_bin`` lets callers inject the resolved absolute path; when ``None``
    the legacy string ``"ffmpeg"`` is kept so unit tests stay hermetic.
    """
    audio_in: list[str] = []
    if audio_start_sec is not None:
        if audio_start_sec < 0:
            raise ValueError(
                f"audio_start_sec must be non-negative, got {audio_start_sec!r}"
            )
        audio_in += ["-ss", f"{float(audio_start_sec):.6f}"]
    if audio_duration_sec is not None:
        if audio_duration_sec <= 0:
            raise ValueError(
                f"audio_duration_sec must be positive, got {audio_duration_sec!r}"
            )
        audio_in += ["-t", f"{float(audio_duration_sec):.6f}"]

    return [
        ffmpeg_bin or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        *audio_in,
        "-i",
        str(audio_path),
        "-c:v",
        video_codec,
        *_ffmpeg_video_args(video_codec, ffmpeg_bin=ffmpeg_bin),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_mp4),
    ]


def render_spectrum_m1(
    cache_dir: Path | str,
    *,
    analysis_path: Path | None = None,
    audio_path: Path | None = None,
    run_id: str | None = None,
    outputs_dir: Path | None = None,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    background_rgb: tuple[int, int, int] = (18, 18, 28),
    bar_rgb: tuple[int, int, int] = (0, 210, 255),
    video_codec: str | None = None,
) -> SpectrumRenderResult:
    """
    Encode ``outputs/<run_id>/output.mp4``: 1080p spectrum bars + muxed audio.

    Expects ``cache/<song_hash>/analysis.json`` and ``original.wav`` unless paths
    are overridden. Frame count follows audio duration at ``fps`` (floor) so
    video length does not exceed audio; ``-shortest`` matches muxed output.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir does not exist: {cache}")

    analysis_p = Path(analysis_path) if analysis_path is not None else cache / ANALYSIS_JSON_NAME
    audio_p = Path(audio_path) if audio_path is not None else cache / ORIGINAL_WAV_NAME

    if not analysis_p.is_file():
        raise FileNotFoundError(f"Missing analysis: {analysis_p}")
    if not audio_p.is_file():
        raise FileNotFoundError(
            f"Missing {ORIGINAL_WAV_NAME} at {audio_p}; run audio ingest first"
        )

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    codec = select_video_codec(video_codec)
    # Resolve ffmpeg **after** the codec selection so we see any binary
    # promoted by the multi-candidate sweep (the highest-priority ffmpeg
    # might lack NVENC while a later candidate has it). Same reason as in
    # ``pipeline.compositor.render_full_video``: the NVENC-preset probe
    # must run against the binary we'll actually pipe frames into.
    ffmpeg_bin = require_ffmpeg()

    analysis = _load_analysis(analysis_p)
    spec_values, spec_fps = _spectrum_array(analysis)

    duration = _audio_duration(audio_p)
    frame_count = max(1, int(np.floor(duration * float(fps) + 1e-9)))

    rid = run_id or new_run_id(song_hash=cache.name)
    out_root = Path(outputs_dir) if outputs_dir is not None else OUTPUTS_DIR
    out_dir = out_root / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "output.mp4"

    cmd = _build_ffmpeg_cmd(
        width=width,
        height=height,
        fps=fps,
        audio_path=audio_p,
        output_mp4=out_mp4,
        video_codec=codec,
        ffmpeg_bin=ffmpeg_bin,
    )

    with tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=stderr_file,
        )
        assert proc.stdin is not None
        try:
            for i in range(frame_count):
                t = (i + 0.5) / float(fps)
                bands = _bands_at_time(spec_values, t, spec_fps)
                rgb = _draw_frame_rgb(
                    width,
                    height,
                    bands,
                    background_rgb=background_rgb,
                    bar_rgb=bar_rgb,
                )
                bgr = _rgb_to_bgr(rgb)
                proc.stdin.write(bgr.tobytes())
        finally:
            proc.stdin.close()

        code = proc.wait()
        stderr_file.seek(0)
        err_tail = stderr_file.read().decode("utf-8", errors="replace").strip()

    if code != 0:
        msg = f"ffmpeg exited with code {code}"
        if err_tail:
            msg = f"{msg}\n--- ffmpeg stderr ---\n{err_tail}"
        raise RuntimeError(msg)

    return SpectrumRenderResult(
        run_id=rid,
        output_dir=out_dir,
        output_mp4=out_mp4,
        frame_count=frame_count,
        audio_path=audio_p,
        analysis_path=analysis_p,
    )


__all__: Sequence[str] = [
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "SpectrumRenderResult",
    "render_spectrum_m1",
]
