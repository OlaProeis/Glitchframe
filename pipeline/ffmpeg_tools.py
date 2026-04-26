"""
Locate ``ffmpeg`` / ``ffprobe`` binaries even when they're not on ``PATH``.

On Windows, many installers (winget, Scoop, Chocolatey) put the binary on the
*user-level* ``PATH``, which freshly-opened shells inherit but already-running
processes (e.g. a Gradio app launched earlier) do not. That leads to
``shutil.which("ffmpeg")`` returning ``None`` even though the binary exists.

This module resolves in this order:

1. Environment override: ``GLITCHFRAME_FFMPEG`` / ``GLITCHFRAME_FFPROBE`` (or
   legacy ``MUSICVIDS_FFMPEG`` / ``MUSICVIDS_FFPROBE``) if set
   and pointing at an executable file).
2. ``shutil.which(name)`` — the ordinary PATH lookup.
3. A small set of well-known Windows install locations (winget, Scoop,
   Chocolatey, Program Files). Globs are resolved at call time so later
   installs are picked up without restarting the process.

The first match wins and is cached for the lifetime of the process. Callers
that need to defeat the cache (tests, re-discovery after an install) can call
:func:`clear_cache`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform.startswith("win")
_EXE_SUFFIX = ".exe" if _IS_WINDOWS else ""

_ENV_VAR_BY_NAME: dict[str, list[str]] = {
    "ffmpeg": ["GLITCHFRAME_FFMPEG", "MUSICVIDS_FFMPEG"],
    "ffprobe": ["GLITCHFRAME_FFPROBE", "MUSICVIDS_FFPROBE"],
}

_cache: dict[str, str | None] = {}


def _windows_candidate_dirs() -> list[Path]:
    """Directories where common Windows installers drop ``ffmpeg.exe``."""
    local_app = os.environ.get("LOCALAPPDATA") or ""
    program_files = os.environ.get("ProgramFiles") or r"C:\Program Files"
    program_data = os.environ.get("ProgramData") or r"C:\ProgramData"

    dirs: list[Path] = []
    if local_app:
        # winget packages keep the full build under a versioned subfolder; glob
        # matches any installed version of gyan.dev's build.
        winget_root = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.is_dir():
            for pkg_dir in winget_root.glob("Gyan.FFmpeg_*"):
                for build_dir in pkg_dir.glob("ffmpeg-*-full_build"):
                    dirs.append(build_dir / "bin")
                # Some builds drop bin/ directly under the package root.
                dirs.append(pkg_dir / "bin")
        # Scoop shims.
        dirs.append(Path(local_app) / "Scoop" / "shims")

    dirs.extend(
        [
            Path(program_files) / "ffmpeg" / "bin",
            Path(program_files) / "FFmpeg" / "bin",
            Path(program_data) / "chocolatey" / "bin",
        ]
    )
    return dirs


def _posix_candidate_dirs() -> list[Path]:
    """Common POSIX paths for completeness; most users have ffmpeg on PATH."""
    return [Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin")]


def _candidate_dirs() -> Iterable[Path]:
    return _windows_candidate_dirs() if _IS_WINDOWS else _posix_candidate_dirs()


def _env_override(name: str) -> str | None:
    for env_var in _ENV_VAR_BY_NAME.get(name, ()):
        raw = os.environ.get(env_var)
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.is_file():
            return str(p)
        LOGGER.warning(
            "%s=%r does not point at a file; falling back to PATH lookup",
            env_var, raw,
        )
    return None


def _resolve(name: str) -> str | None:
    if name in _cache:
        return _cache[name]

    override = _env_override(name)
    if override is not None:
        _cache[name] = override
        return override

    on_path = shutil.which(name)
    if on_path:
        _cache[name] = on_path
        return on_path

    exe_name = f"{name}{_EXE_SUFFIX}"
    for d in _candidate_dirs():
        candidate = d / exe_name
        if candidate.is_file():
            resolved = str(candidate)
            LOGGER.info(
                "Resolved %s via well-known location (%s); not on PATH", name, resolved
            )
            _cache[name] = resolved
            return resolved

    _cache[name] = None
    return None


def resolve_ffmpeg() -> str | None:
    """Absolute path to ``ffmpeg`` (or ``None`` if it cannot be located)."""
    return _resolve("ffmpeg")


def resolve_ffprobe() -> str | None:
    """Absolute path to ``ffprobe`` (or ``None`` if it cannot be located)."""
    return _resolve("ffprobe")


def require_ffmpeg() -> str:
    """Return the ffmpeg path or raise ``RuntimeError`` with install guidance."""
    path = resolve_ffmpeg()
    if path is None:
        raise RuntimeError(_not_found_message("ffmpeg"))
    return path


def require_ffprobe() -> str:
    """Return the ffprobe path or raise ``RuntimeError`` with install guidance."""
    path = resolve_ffprobe()
    if path is None:
        raise RuntimeError(_not_found_message("ffprobe"))
    return path


def _not_found_message(name: str) -> str:
    keys = _ENV_VAR_BY_NAME.get(name, ())
    hint = ""
    if keys:
        primary = keys[0]
        hint = (
            f" Set {primary}=<full path to {name}{_EXE_SUFFIX}>"
            " or add the install dir to PATH and restart the app."
        )
    return (
        f"{name} not found on PATH or in common install locations."
        + hint
    )


def clear_cache() -> None:
    """Reset the resolver cache (mainly for tests or after installing ffmpeg)."""
    _cache.clear()
    _codec_cache.clear()


# Video-encoder capability probe ------------------------------------------------

DEFAULT_VIDEO_CODEC = "h264_nvenc"
FALLBACK_VIDEO_CODEC = "libx264"

_codec_cache: dict[str, bool] = {}


def _probe_encoder(codec: str) -> bool:
    """Return ``True`` when the locally installed ffmpeg can actually encode
    with *codec*.

    Some codecs (notably ``h264_nvenc``) are advertised by ``ffmpeg -encoders``
    but still fail at runtime with ``Driver does not support the required
    nvenc API version`` if the NVIDIA driver is too old for the ffmpeg build.
    NVENC also enforces a minimum frame size (often above 64×64); a tiny
    ``lavfi`` encode must use dimensions that satisfy that minimum.
    The cheapest reliable check is to ask ffmpeg to encode 1 frame of a null
    source and observe the exit code.
    """
    if codec in _codec_cache:
        return _codec_cache[codec]
    binary = resolve_ffmpeg()
    if binary is None:
        _codec_cache[codec] = False
        return False
    cmd = [
        binary,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=c=black:s=256x256:d=0.04",
        "-frames:v", "1",
        "-c:v", codec,
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        LOGGER.info("Encoder probe for %s failed to run: %s", codec, exc)
        _codec_cache[codec] = False
        return False
    ok = result.returncode == 0
    if not ok and result.stderr:
        # Truncate to keep the log readable; full stderr is rarely useful.
        tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
        LOGGER.info("Encoder %s unavailable: %s", codec, " | ".join(tail))
    _codec_cache[codec] = ok
    return ok


def select_video_codec(preferred: str | None = None) -> str:
    """Pick a working H.264 encoder, preferring ``preferred`` / NVENC / x264.

    Resolution order:

    1. An explicit *preferred* codec (from config or ``GLITCHFRAME_FFMPEG_VIDEO_CODEC`` /
       legacy ``MUSICVIDS_FFMPEG_VIDEO_CODEC``) — trusted as-is if the user opted in.
    2. ``h264_nvenc`` if the local ffmpeg can actually open the encoder.
    3. ``libx264`` as the portable CPU fallback.
    """
    explicit = preferred or os.environ.get("GLITCHFRAME_FFMPEG_VIDEO_CODEC")
    if not explicit:
        explicit = os.environ.get("MUSICVIDS_FFMPEG_VIDEO_CODEC")
    if explicit:
        return explicit
    if _probe_encoder(DEFAULT_VIDEO_CODEC):
        return DEFAULT_VIDEO_CODEC
    LOGGER.warning(
        "h264_nvenc unavailable (NVIDIA driver too old for this ffmpeg, or "
        "no NVIDIA GPU); falling back to CPU encoder %s. Update the NVIDIA "
        "driver to >=570 to re-enable NVENC.",
        FALLBACK_VIDEO_CODEC,
    )
    return FALLBACK_VIDEO_CODEC


__all__ = [
    "DEFAULT_VIDEO_CODEC",
    "FALLBACK_VIDEO_CODEC",
    "clear_cache",
    "require_ffmpeg",
    "require_ffprobe",
    "resolve_ffmpeg",
    "resolve_ffprobe",
    "select_video_codec",
]
