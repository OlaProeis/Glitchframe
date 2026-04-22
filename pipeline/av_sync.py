"""
Post-encode A/V sync validation via ``ffprobe``.

Returns per-stream durations and the absolute video/audio drift in milliseconds
so callers (e.g. :func:`orchestrator.orchestrate_full_render`) can surface a
pass/fail line in the run log. The default tolerance is ~50 ms, which matches
what ``ffmpeg``'s ``-shortest`` typically leaves after a frame-aligned pipe
encode.

When ``ffprobe`` is not on PATH the report flags ``ffprobe_available=False`` and
``ok=False`` so the caller can downgrade the result to an informational message
rather than a hard failure.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.ffmpeg_tools import resolve_ffprobe

LOGGER = logging.getLogger(__name__)

DEFAULT_AV_SYNC_TOLERANCE_MS = 50.0


@dataclass(frozen=True)
class AvSyncReport:
    """Result of :func:`ffprobe_av_sync`."""

    video_duration_sec: float | None
    audio_duration_sec: float | None
    drift_ms: float | None
    tolerance_ms: float
    ffprobe_available: bool
    ok: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_duration_sec": self.video_duration_sec,
            "audio_duration_sec": self.audio_duration_sec,
            "drift_ms": self.drift_ms,
            "tolerance_ms": self.tolerance_ms,
            "ffprobe_available": self.ffprobe_available,
            "ok": self.ok,
            "message": self.message,
        }


def _stream_duration(stream: dict[str, Any]) -> float | None:
    dur = stream.get("duration")
    try:
        return float(dur) if dur is not None else None
    except (TypeError, ValueError):
        return None


def ffprobe_av_sync(
    video_path: Path | str,
    *,
    tolerance_ms: float = DEFAULT_AV_SYNC_TOLERANCE_MS,
) -> AvSyncReport:
    """
    Probe ``video_path`` with ``ffprobe`` and compare video vs audio duration.

    Returns an :class:`AvSyncReport`; never raises on non-zero ``ffprobe`` exit
    codes so a rendering pipeline can always log a human-readable message.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    ValueError
        If ``tolerance_ms`` is negative.
    """
    if tolerance_ms < 0:
        raise ValueError(f"tolerance_ms must be non-negative, got {tolerance_ms!r}")

    p = Path(video_path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing video for ffprobe: {p}")

    ffprobe_bin = resolve_ffprobe()
    if ffprobe_bin is None:
        return AvSyncReport(
            video_duration_sec=None,
            audio_duration_sec=None,
            drift_ms=None,
            tolerance_ms=tolerance_ms,
            ffprobe_available=False,
            ok=False,
            message=(
                "ffprobe not on PATH or in common install locations; "
                "A/V sync check skipped"
            ),
        )

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_entries",
        "stream=codec_type,duration:format=duration",
        str(p),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        LOGGER.warning("ffprobe invocation failed: %s", exc)
        return AvSyncReport(
            video_duration_sec=None,
            audio_duration_sec=None,
            drift_ms=None,
            tolerance_ms=tolerance_ms,
            ffprobe_available=True,
            ok=False,
            message=f"ffprobe invocation failed: {exc}",
        )

    if proc.returncode != 0:
        return AvSyncReport(
            video_duration_sec=None,
            audio_duration_sec=None,
            drift_ms=None,
            tolerance_ms=tolerance_ms,
            ffprobe_available=True,
            ok=False,
            message=f"ffprobe exited with code {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}",
        )

    try:
        doc = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return AvSyncReport(
            video_duration_sec=None,
            audio_duration_sec=None,
            drift_ms=None,
            tolerance_ms=tolerance_ms,
            ffprobe_available=True,
            ok=False,
            message=f"ffprobe output was not JSON: {exc}",
        )

    streams = doc.get("streams") or []
    v_dur: float | None = None
    a_dur: float | None = None
    for s in streams:
        kind = s.get("codec_type")
        d = _stream_duration(s)
        if kind == "video" and v_dur is None:
            v_dur = d
        elif kind == "audio" and a_dur is None:
            a_dur = d

    if v_dur is None or a_dur is None:
        missing = ", ".join(
            k for k, v in (("video", v_dur), ("audio", a_dur)) if v is None
        )
        return AvSyncReport(
            video_duration_sec=v_dur,
            audio_duration_sec=a_dur,
            drift_ms=None,
            tolerance_ms=tolerance_ms,
            ffprobe_available=True,
            ok=False,
            message=f"ffprobe did not report duration for: {missing}",
        )

    drift_ms = abs(float(v_dur) - float(a_dur)) * 1000.0
    ok = drift_ms <= tolerance_ms
    verdict = "PASS" if ok else "FAIL"
    return AvSyncReport(
        video_duration_sec=float(v_dur),
        audio_duration_sec=float(a_dur),
        drift_ms=drift_ms,
        tolerance_ms=tolerance_ms,
        ffprobe_available=True,
        ok=ok,
        message=(
            f"A/V sync {verdict}: video={v_dur:.3f}s audio={a_dur:.3f}s "
            f"drift={drift_ms:.1f}ms (tol {tolerance_ms:.0f}ms)"
        ),
    )


__all__ = [
    "DEFAULT_AV_SYNC_TOLERANCE_MS",
    "AvSyncReport",
    "ffprobe_av_sync",
]
