"""
Locate ``ffmpeg`` / ``ffprobe`` binaries even when they're not on ``PATH``.

On Windows, many installers (winget, Scoop, Chocolatey) put the binary on the
*user-level* ``PATH``, which freshly-opened shells inherit but already-running
processes (e.g. a Gradio app launched earlier) do not. That leads to
``shutil.which("ffmpeg")`` returning ``None`` even though the binary exists.
On Pinokio specifically, conda-env activation prepends ``env\\Library\\bin``
to ``PATH`` — but a transitive package's bundled ffmpeg can shadow whatever
the user installed system-wide, and that bundle is sometimes built without
NVENC support. So we enumerate **all** candidates in priority order and let
:func:`select_video_codec` pick a codec-capable one.

Discovery priority (highest first):

1. **Environment override**: ``GLITCHFRAME_FFMPEG`` / ``GLITCHFRAME_FFPROBE``
   (or legacy ``MUSICVIDS_FFMPEG`` / ``MUSICVIDS_FFPROBE``) if set and
   pointing at an executable file. User opt-in always wins.
2. **Active Python env's bin**: ``sys.prefix\\Library\\bin\\ffmpeg.exe``
   on Windows, ``sys.prefix/bin/ffmpeg`` on POSIX. This is what Pinokio's
   ``conda install -c conda-forge ffmpeg`` produces; checking it explicitly
   is more deterministic than relying on ``PATH`` order.
3. **PATH lookup** via ``shutil.which``.
4. **Well-known install locations** (winget, Scoop, Chocolatey, Program
   Files). Globs are resolved at call time so later installs are picked up
   without restarting the process.

:func:`resolve_ffmpeg` returns the highest-priority candidate. The resolution
is cached for the lifetime of the process; callers that need to defeat the
cache (tests, re-discovery after an install) can call :func:`clear_cache`.

For codec-aware selection see :func:`select_video_codec`, which can probe
**all** candidates and prefer one that supports a given codec
(e.g. ``h264_nvenc``) — that's the right choice when the highest-priority
binary lacks NVENC but a lower-priority one has it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

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


def _active_env_bindir() -> Path | None:
    """Return the active Python env's binary dir, if it has one we trust.

    On a conda env (Windows), executables live under ``sys.prefix\\Library\\bin``
    (and conda also drops them into ``sys.prefix\\Scripts`` for some packages,
    but Library\\bin is where ``conda install ffmpeg`` ends up). On POSIX
    venvs/conda, ``sys.prefix/bin``.

    We return ``None`` when ``sys.prefix == sys.base_prefix`` (i.e. we're not
    in a venv/conda env at all) so we don't accidentally treat the system
    Python's ``bin`` as a special-priority location.
    """
    try:
        prefix = Path(sys.prefix).resolve()
        base = Path(sys.base_prefix).resolve()
    except (OSError, RuntimeError):
        return None
    if prefix == base:
        return None
    if _IS_WINDOWS:
        # Library\bin is where conda-forge drops ffmpeg.exe; Scripts is the
        # other common Windows location. Prefer Library\bin first.
        for sub in ("Library/bin", "Scripts"):
            d = prefix / sub
            if d.is_dir():
                return d
        return None
    bin_dir = prefix / "bin"
    return bin_dir if bin_dir.is_dir() else None


class FfmpegCandidate(NamedTuple):
    """One discovered ffmpeg binary, with provenance for diagnostic logs."""

    path: str
    source: str  # e.g. "env override", "active env Library\\bin", "PATH", "well-known"


def _iter_candidates(name: str) -> Iterable[FfmpegCandidate]:
    """Yield ``FfmpegCandidate`` tuples in priority order, deduped by path.

    Used by :func:`_resolve` (takes the first) and by
    :func:`select_video_codec` when probing for a codec-capable binary.
    """
    seen: set[str] = set()

    def _emit(path: str | None, source: str) -> Iterable[FfmpegCandidate]:
        if not path:
            return
        try:
            normalised = str(Path(path).resolve())
        except (OSError, RuntimeError):
            normalised = path
        key = normalised.lower() if _IS_WINDOWS else normalised
        if key in seen:
            return
        seen.add(key)
        yield FfmpegCandidate(path=path, source=source)

    override = _env_override(name)
    if override is not None:
        yield from _emit(override, "env override")

    env_bin = _active_env_bindir()
    if env_bin is not None:
        candidate = env_bin / f"{name}{_EXE_SUFFIX}"
        if candidate.is_file():
            yield from _emit(str(candidate), f"active env ({env_bin.name})")

    on_path = shutil.which(name)
    if on_path:
        yield from _emit(on_path, "PATH")

    exe_name = f"{name}{_EXE_SUFFIX}"
    for d in _candidate_dirs():
        candidate = d / exe_name
        if candidate.is_file():
            yield from _emit(str(candidate), "well-known location")


def _resolve(name: str) -> str | None:
    """Return the highest-priority ``ffmpeg``/``ffprobe`` candidate, or ``None``.

    Priority follows :func:`_iter_candidates`: env override → active env's
    bin dir → PATH → well-known. Logs the resolution source so a Pinokio /
    fork-user log shows exactly which binary the process picked. Cached.
    """
    if name in _cache:
        return _cache[name]

    for cand in _iter_candidates(name):
        _cache[name] = cand.path
        LOGGER.info("Resolved %s -> %s (via %s)", name, cand.path, cand.source)
        return cand.path

    _cache[name] = None
    LOGGER.warning(
        "%s not found via env override, active env, PATH, or well-known "
        "install locations",
        name,
    )
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


_PROBE_TAIL_LINES = 14


def _probe_encoder_with_binary(binary: str, codec: str) -> tuple[bool, str]:
    """Run a 1-frame lavfi encode through *binary* and report success + stderr tail.

    Returns ``(ok, stderr_tail)`` where ``stderr_tail`` is the last
    ``_PROBE_TAIL_LINES`` lines of ffmpeg's stderr (joined with newlines).
    On success the tail is the empty string. Used both by the cached
    single-binary :func:`_probe_encoder` and by
    :func:`_pick_codec_capable_ffmpeg` which iterates over candidates.

    Some codecs (notably ``h264_nvenc``) are advertised by ``ffmpeg -encoders``
    but still fail at runtime with ``Driver does not support the required
    nvenc API version`` (driver too old for this ffmpeg build) or with
    ``Cannot load nvEncodeAPI64.dll`` (NVIDIA encode runtime not visible to
    the process — common on Pinokio's conda env which strips DLL search
    paths). NVENC also enforces a minimum frame size (often above 64×64);
    a 256×256 lavfi source comfortably clears that. ``-loglevel info`` is
    required: at ``error`` level ffmpeg suppresses the genuine NVENC
    diagnostic and only prints its useless wrapper message.
    """
    cmd = [
        binary,
        "-hide_banner",
        "-loglevel", "info",
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
        return False, f"<probe failed to run: {exc}>"
    if result.returncode == 0:
        return True, ""
    stderr_text = (
        result.stderr.decode("utf-8", errors="replace").strip()
        if result.stderr
        else "<no stderr>"
    )
    tail_lines = stderr_text.splitlines()[-_PROBE_TAIL_LINES:] or ["<no stderr>"]
    return False, "\n".join(tail_lines)


def _probe_encoder(codec: str) -> bool:
    """Backwards-compatible single-binary probe (uses :func:`resolve_ffmpeg`).

    Caches the answer per codec for the lifetime of the process. Logs the
    full stderr tail on failure so the real NVENC error is visible. For
    multi-candidate selection use :func:`_pick_codec_capable_ffmpeg`.
    """
    if codec in _codec_cache:
        return _codec_cache[codec]
    binary = resolve_ffmpeg()
    if binary is None:
        _codec_cache[codec] = False
        return False
    ok, tail = _probe_encoder_with_binary(binary, codec)
    if not ok:
        LOGGER.info(
            "Encoder %s probe failed (binary=%s); last stderr line(s):\n  %s",
            codec,
            binary,
            (tail or "<no stderr>").replace("\n", "\n  "),
        )
    _codec_cache[codec] = ok
    return ok


def _pick_codec_capable_ffmpeg(codec: str) -> str | None:
    """Probe every discovered ffmpeg candidate and return the first one that
    can actually encode with *codec*, or ``None`` if none can.

    This is the answer to the Pinokio "NVENC unavailable" failure mode where
    a low-quality bundled ffmpeg shadows a working one. Without this we'd
    silently fall back to ``libx264`` (CPU, ~5–10× slower at 1080p) even
    when a perfectly good NVENC-capable ffmpeg exists elsewhere on the
    system. Each failed candidate's stderr tail is logged at info level so
    the user can see *why* it was rejected.

    Side-effect: as a bonus, if the picked binary is **not** the one
    :func:`resolve_ffmpeg` would normally return, we update ``_cache`` so
    subsequent code (encoder commands, ffprobe via the same binary) uses
    the working one. This avoids the awkward situation where we correctly
    pick NVENC ffmpeg here but then run encode via a different ffmpeg.
    """
    candidates = list(_iter_candidates("ffmpeg"))
    if not candidates:
        return None

    for cand in candidates:
        ok, tail = _probe_encoder_with_binary(cand.path, codec)
        if ok:
            LOGGER.info(
                "Codec %s available via %s (source: %s)",
                codec,
                cand.path,
                cand.source,
            )
            # Promote this binary to be THE ffmpeg for the run. Without this
            # the encode command would use the priority-0 candidate that we
            # just proved cannot do NVENC.
            if _cache.get("ffmpeg") != cand.path:
                LOGGER.info(
                    "Promoting %s to active ffmpeg (was %s) because it "
                    "supports %s and the previous one did not.",
                    cand.path,
                    _cache.get("ffmpeg"),
                    codec,
                )
                _cache["ffmpeg"] = cand.path
            return cand.path
        LOGGER.info(
            "Codec %s rejected via %s (source: %s); last stderr line(s):\n  %s",
            codec,
            cand.path,
            cand.source,
            (tail or "<no stderr>").replace("\n", "\n  "),
        )
    return None


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
    # Try the highest-priority binary first (cheap when it works).
    if _probe_encoder(DEFAULT_VIDEO_CODEC):
        return DEFAULT_VIDEO_CODEC
    # The default ffmpeg can't do NVENC. Before giving up to libx264, sweep
    # every discovered candidate — Pinokio commonly has multiple ffmpegs
    # (conda env's bundle, the user's PATH ffmpeg, a winget install) and
    # only some of them support NVENC. Picking the working one here saves
    # the user a 5–10× slowdown at no UX cost.
    fallback_binary = _pick_codec_capable_ffmpeg(DEFAULT_VIDEO_CODEC)
    if fallback_binary is not None:
        # ``_pick_codec_capable_ffmpeg`` already promoted the working binary
        # into the cache, so the encode command will use it.
        _codec_cache[DEFAULT_VIDEO_CODEC] = True
        return DEFAULT_VIDEO_CODEC
    LOGGER.warning(
        "%s unavailable on every discovered ffmpeg candidate; falling back "
        "to CPU encoder %s. Encoder probe logs above contain ffmpeg's actual "
        "stderr for each candidate -- check them for the real cause (driver "
        "/ NVENC SDK mismatch, missing nvEncodeAPI64.dll, session limit, "
        "conda ffmpeg built without --enable-nvenc). CPU encode is ~5-10x "
        "slower than NVENC at 1080p.",
        DEFAULT_VIDEO_CODEC,
        FALLBACK_VIDEO_CODEC,
    )
    return FALLBACK_VIDEO_CODEC


def log_ffmpeg_diagnostics() -> None:
    """One-shot startup diagnostic: log resolved ffmpeg path + version banner.

    Called from ``app.py`` startup so every Pinokio / fork-user log shows
    *which* ffmpeg binary the process actually picked, **and** what other
    candidates we found (so the codec-capable picker's later choice has
    visible context). Without this, the "NVENC unavailable" path is
    undebuggable from a log alone — conda env activation can shadow the
    user's PATH ffmpeg silently.

    Never raises (best-effort logging only).
    """
    # Enumerate everything we could find, so the user sees the full picture.
    all_candidates = list(_iter_candidates("ffmpeg"))
    if len(all_candidates) > 1:
        LOGGER.info(
            "Discovered %d ffmpeg candidates (priority order):", len(all_candidates)
        )
        for i, cand in enumerate(all_candidates, 1):
            LOGGER.info("  %d. %s  [%s]", i, cand.path, cand.source)

    ffmpeg_path = resolve_ffmpeg()
    if ffmpeg_path is None:
        LOGGER.warning(
            "ffmpeg not found on PATH or in well-known install locations; "
            "video encode and the NVENC probe will fail. Install ffmpeg or "
            "set GLITCHFRAME_FFMPEG=<full path to ffmpeg.exe>."
        )
        return
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        LOGGER.warning("ffmpeg -version probe failed (%s): %s", ffmpeg_path, exc)
        return
    out = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    first_line = out.splitlines()[0] if out else "<empty>"
    LOGGER.info("ffmpeg banner: %s -- %s", ffmpeg_path, first_line)
    # Also log the configure line if present — it tells us at a glance whether
    # this build even has --enable-nvenc / --enable-cuda-llvm linked in.
    config_line = next(
        (
            ln.strip()
            for ln in out.splitlines()
            if ln.lstrip().startswith("configuration:")
        ),
        None,
    )
    if config_line:
        # Truncate to keep one log line readable; the nvenc/cuda/cuvid flags
        # are the diagnostic value here.
        flags_of_interest = [
            tok
            for tok in config_line.split()
            if any(
                marker in tok.lower()
                for marker in ("nvenc", "cuda", "cuvid", "nvdec", "nvidia")
            )
        ]
        if flags_of_interest:
            LOGGER.info(
                "ffmpeg NVIDIA-related configure flags: %s",
                " ".join(flags_of_interest),
            )
        else:
            LOGGER.warning(
                "ffmpeg %s appears to lack NVIDIA support in its configure "
                "line (no nvenc/cuda/nvdec flags). NVENC will fail; consider "
                "installing a build with --enable-nvenc.",
                ffmpeg_path,
            )


__all__ = [
    "DEFAULT_VIDEO_CODEC",
    "FALLBACK_VIDEO_CODEC",
    "clear_cache",
    "log_ffmpeg_diagnostics",
    "require_ffmpeg",
    "require_ffprobe",
    "resolve_ffmpeg",
    "resolve_ffprobe",
    "select_video_codec",
]
