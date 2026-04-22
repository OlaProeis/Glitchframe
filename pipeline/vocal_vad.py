"""
Silero-VAD wrapper for vocal-stem voice activity detection.

Used by :mod:`pipeline.lyrics_aligner` to tighten whisper's VAD bounds
before forced alignment. Whisper's VAD is trained on general speech and
routinely emits single segments that span real vocals **plus** several
seconds of instrumental; Silero-VAD is a small ONNX model trained
specifically for voice detection and produces much cleaner windows on
singing / ad-lib / backing-vocal content.

The aligner calls :func:`detect_vocal_speech_spans`, which:

* Imports :mod:`silero_vad` lazily so the app still runs when it isn't
  installed. The caller treats an ``ImportError`` as "skip the tighten
  pass".
* Accepts either the ``torch.hub``-vendored helper or the newer
  ``silero-vad`` PyPI wheel (both expose the same ``get_speech_timestamps``
  API, modulo kwarg names).
* Returns a sorted list of ``(start_sec, end_sec)`` tuples, merged across
  tiny gaps so a single word isn't split in two.

Results are cached beside the vocal stem as ``vocal_vad.json`` keyed by
the stem's modification time so repeated aligner runs don't rerun Silero.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)

VOCAL_VAD_JSON_NAME = "vocal_vad.json"

# Silero VAD expects 16 kHz mono.
_SILERO_SAMPLE_RATE = 16_000

# Conservative defaults: require at least 120 ms of continuous speech, and
# bridge gaps below 250 ms so a single sung word isn't split across frames
# where the consonant decays into the vowel.
_MIN_SPEECH_MS = 120
_MIN_SILENCE_MS = 250
_SPEECH_PAD_MS = 30
# Silero's default threshold on its own speech-prob head; 0.5 is the
# documented sweet spot for pre-mixed vocals. Lower = more permissive.
_SPEECH_THRESHOLD = 0.5


def _cache_path(vocals_wav: Path) -> Path:
    return vocals_wav.parent / VOCAL_VAD_JSON_NAME


def _try_load_cached(vocals_wav: Path) -> list[tuple[float, float]] | None:
    cache = _cache_path(vocals_wav)
    if not cache.is_file():
        return None
    try:
        with cache.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read cached vocal VAD %s: %s", cache, exc)
        return None
    if not isinstance(data, dict):
        return None
    mtime_stored = data.get("vocals_mtime")
    try:
        mtime_current = vocals_wav.stat().st_mtime
    except OSError:
        return None
    if (
        not isinstance(mtime_stored, (int, float))
        or abs(float(mtime_stored) - float(mtime_current)) > 1e-6
    ):
        return None
    spans = data.get("spans")
    if not isinstance(spans, list):
        return None
    out: list[tuple[float, float]] = []
    for s in spans:
        if not isinstance(s, (list, tuple)) or len(s) != 2:
            return None
        out.append((float(s[0]), float(s[1])))
    return out


def _write_cache(
    vocals_wav: Path, spans: list[tuple[float, float]]
) -> None:
    cache = _cache_path(vocals_wav)
    try:
        mtime = vocals_wav.stat().st_mtime
    except OSError:
        return
    payload = {
        "vocals_mtime": float(mtime),
        "sample_rate": _SILERO_SAMPLE_RATE,
        "spans": [[float(a), float(b)] for a, b in spans],
    }
    tmp = cache.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        tmp.replace(cache)
    except OSError as exc:
        LOGGER.warning("Failed to write vocal-VAD cache %s: %s", cache, exc)


def _load_silero() -> tuple[object, object]:
    """Return ``(model, get_speech_timestamps)`` from whichever Silero flavour is installed.

    Supports three bindings:

    * ``silero_vad`` PyPI wheel — exposes :func:`load_silero_vad` + a
      top-level :func:`get_speech_timestamps`.
    * ``torch.hub.load("snakers4/silero-vad", ...)`` — the original
      distribution, still in active use.
    * ``silero-vad`` helper package exposing ``VADIterator`` plus the
      model — we only use ``get_speech_timestamps`` here.

    Raises ``ImportError`` with install guidance if none are available.
    """
    # 1) Modern PyPI package.
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore

        model = load_silero_vad()
        return model, get_speech_timestamps
    except Exception:  # noqa: BLE001
        pass

    # 2) torch.hub fallback — works offline once the hub repo has been
    # cached. We don't rely on a network round-trip here; the user must
    # have run `torch.hub.load("snakers4/silero-vad", ...)` at least once
    # (pip install silero-vad pulls the hub cache as a side effect).
    try:
        import torch  # type: ignore

        model, utils = torch.hub.load(  # type: ignore[attr-defined]
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]
        return model, get_speech_timestamps
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "silero-vad is not available. Install the wheel into the same "
            "Python that runs the app: "
            "`python -m pip install silero-vad` (small ONNX model, no GPU "
            "required). Falls back to whisper's VAD when absent."
        ) from exc


def detect_vocal_speech_spans(
    vocals_wav: Path | str,
) -> list[tuple[float, float]]:
    """Run Silero VAD on a vocal stem and return ``[(start_sec, end_sec), ...]``.

    The returned spans are sorted, non-overlapping, and padded by
    :data:`_SPEECH_PAD_MS` on each side so a word's consonant onset isn't
    clipped. Cached beside the stem keyed by its mtime.

    Raises :class:`ImportError` when Silero isn't installed — the lyrics
    aligner catches this and falls back to whisper's VAD so the pipeline
    still runs.
    """
    path = Path(vocals_wav)
    if not path.is_file():
        raise FileNotFoundError(f"Vocals stem missing: {path}")

    cached = _try_load_cached(path)
    if cached is not None:
        return cached

    model, get_speech_timestamps = _load_silero()

    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return []

    if sr != _SILERO_SAMPLE_RATE:
        # librosa's polyphase resampler is good enough for VAD (we don't
        # care about absolute fidelity here, just onset positions).
        import librosa  # local import keeps cold startup cheap

        y = librosa.resample(y, orig_sr=sr, target_sr=_SILERO_SAMPLE_RATE)
        sr = _SILERO_SAMPLE_RATE

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "silero-vad requires torch at runtime; install CUDA torch from "
            "the PyTorch index before installing silero-vad."
        ) from exc

    audio = torch.from_numpy(y).float()
    try:
        ts = get_speech_timestamps(
            audio,
            model,
            sampling_rate=_SILERO_SAMPLE_RATE,
            threshold=_SPEECH_THRESHOLD,
            min_speech_duration_ms=_MIN_SPEECH_MS,
            min_silence_duration_ms=_MIN_SILENCE_MS,
            speech_pad_ms=_SPEECH_PAD_MS,
            return_seconds=True,
        )
    except TypeError:
        # Older Silero signatures don't accept ``return_seconds``; fall
        # back to sample indices and convert manually.
        ts_samples = get_speech_timestamps(
            audio,
            model,
            sampling_rate=_SILERO_SAMPLE_RATE,
            threshold=_SPEECH_THRESHOLD,
            min_speech_duration_ms=_MIN_SPEECH_MS,
            min_silence_duration_ms=_MIN_SILENCE_MS,
            speech_pad_ms=_SPEECH_PAD_MS,
        )
        ts = [
            {
                "start": float(t["start"]) / _SILERO_SAMPLE_RATE,
                "end": float(t["end"]) / _SILERO_SAMPLE_RATE,
            }
            for t in ts_samples
        ]

    spans: list[tuple[float, float]] = []
    for t in ts:
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        if e > s:
            spans.append((s, e))
    spans.sort()

    _write_cache(path, spans)
    return spans


__all__ = ["detect_vocal_speech_spans", "VOCAL_VAD_JSON_NAME"]
