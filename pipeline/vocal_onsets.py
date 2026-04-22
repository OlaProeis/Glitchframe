"""
Vocal-specific onset detection for the lyrics aligner's snap pass.

The pipeline-wide ``pipeline.audio_analyzer`` runs onset detection on the
full mono mix, which is fine for shader / reactive layers but noisy for
word alignment: every snare hit / hi-hat tick reads as an "onset" and
can pull a word's ``t_start`` off the consonant by a few frames.

This module computes onsets on the **vocal stem** (:file:`vocals.wav`)
with a librosa configuration tuned for voice — higher hop resolution,
a low detection threshold so soft consonants aren't dropped, and a
short post-minimum-distance so rapid-fire syllables (``"la la la"``)
each get their own onset.

Results are cached beside the vocal stem as ``vocal_onsets.json`` keyed
by the stem's modification time so subsequent aligner runs reuse them
without redoing the librosa work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)

VOCAL_ONSETS_JSON_NAME = "vocal_onsets.json"

# librosa tuning for sung vocals. Smaller hop = ~11 ms resolution at 44.1k
# so consonant detection isn't quantised to 23 ms frames. pre_max / post_max
# govern the local-peak window; pre_avg / post_avg the local-mean window;
# delta is the threshold above the local mean. Wait = minimum gap between
# peaks, in frames, which on fast articulation we want under 60 ms.
_VOCAL_ONSET_HOP = 256
_VOCAL_ONSET_PRE_MAX = 3
_VOCAL_ONSET_POST_MAX = 3
_VOCAL_ONSET_PRE_AVG = 6
_VOCAL_ONSET_POST_AVG = 6
_VOCAL_ONSET_DELTA = 0.05
_VOCAL_ONSET_WAIT = 4


def _cache_path(vocals_wav: Path) -> Path:
    return vocals_wav.parent / VOCAL_ONSETS_JSON_NAME


def _try_load_cached(vocals_wav: Path) -> list[float] | None:
    cache = _cache_path(vocals_wav)
    if not cache.is_file():
        return None
    try:
        with cache.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read cached vocal onsets %s: %s", cache, exc)
        return None
    if not isinstance(data, dict):
        return None
    # Invalidate when the vocal stem has been rewritten (demucs re-run).
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
    times = data.get("times")
    if not isinstance(times, list):
        return None
    return [float(t) for t in times]


def _write_cache(vocals_wav: Path, times: list[float]) -> None:
    cache = _cache_path(vocals_wav)
    try:
        mtime = vocals_wav.stat().st_mtime
    except OSError:
        return
    payload = {
        "vocals_mtime": float(mtime),
        "hop_length": _VOCAL_ONSET_HOP,
        "times": [float(t) for t in times],
    }
    tmp = cache.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        tmp.replace(cache)
    except OSError as exc:
        LOGGER.warning("Failed to write vocal-onset cache %s: %s", cache, exc)


def compute_vocal_onsets(vocals_wav: Path | str) -> list[float]:
    """Return sorted onset times (seconds) detected on the vocal stem.

    Reuses :file:`vocal_onsets.json` beside ``vocals_wav`` when the stem
    hasn't been modified since the last run. Raises on missing file;
    returns an empty list on a silent / zero-length stem rather than
    raising, so downstream "no onsets found" is an OK no-op.
    """
    path = Path(vocals_wav)
    if not path.is_file():
        raise FileNotFoundError(f"Vocals stem missing: {path}")

    cached = _try_load_cached(path)
    if cached is not None:
        return cached

    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = librosa.to_mono(np.ascontiguousarray(y.T))
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return []

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_VOCAL_ONSET_HOP)
    if env.size == 0:
        return []

    peaks_frames = librosa.util.peak_pick(
        env,
        pre_max=_VOCAL_ONSET_PRE_MAX,
        post_max=_VOCAL_ONSET_POST_MAX,
        pre_avg=_VOCAL_ONSET_PRE_AVG,
        post_avg=_VOCAL_ONSET_POST_AVG,
        delta=_VOCAL_ONSET_DELTA,
        wait=_VOCAL_ONSET_WAIT,
    )
    times = librosa.frames_to_time(
        peaks_frames, sr=sr, hop_length=_VOCAL_ONSET_HOP
    ).tolist()
    times_f = sorted(float(t) for t in times)

    _write_cache(path, times_f)
    return times_f


__all__ = ["compute_vocal_onsets", "VOCAL_ONSETS_JSON_NAME"]
