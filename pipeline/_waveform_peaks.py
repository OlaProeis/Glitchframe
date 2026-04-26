"""Shared downsampling of WAV audio into min/max peak columns for canvas UIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# Width of the peaks array the editor renders. 6000 keeps each bucket around
# 30--40 ms for a typical 3--4 min song (0.03 s/bucket vs ~0.15 s at 1600),
# which is enough to read individual kicks / snares on the timeline without
# bloating the HTML payload (~140 KB of JSON numbers). The editor still scales
# the array across whatever stage width the user zooms to.
DEFAULT_PEAK_WIDTH = 6000
# Minimum number of samples per peak bucket. When the audio is short we
# cap the effective width so buckets don't collapse to <1 sample each.
_MIN_SAMPLES_PER_BUCKET = 4


def compute_peaks(
    wav_path: Path | str, target_width: int = DEFAULT_PEAK_WIDTH
) -> tuple[list[tuple[float, float]], int, float]:
    """Return ``(peaks, sample_rate, duration_sec)`` for a WAV file.

    ``peaks`` is a list of ``(min, max)`` pairs, each in ``[-1, 1]``, with
    exactly ``len(peaks)`` columns. The browser side draws them as
    vertical lines from ``min`` to ``max``. Mono-mixes stereo on load so
    the editor doesn't need to handle channel layout.
    """
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"WAV missing: {path}")
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    n = int(y.shape[0])
    if n == 0:
        return [], int(sr), 0.0

    # Pick effective width so each bucket has at least _MIN_SAMPLES_PER_BUCKET.
    max_width = max(1, n // _MIN_SAMPLES_PER_BUCKET)
    width = int(max(1, min(target_width, max_width)))
    # Normalise to the signal's own peak so quiet songs still read on screen.
    peak_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if peak_abs < 1e-6:
        peak_abs = 1.0

    bucket = np.linspace(0, n, num=width + 1, dtype=np.int64)
    peaks: list[tuple[float, float]] = []
    for i in range(width):
        lo, hi = int(bucket[i]), int(bucket[i + 1])
        if hi <= lo:
            peaks.append((0.0, 0.0))
            continue
        slice_ = y[lo:hi]
        mn = float(np.min(slice_)) / peak_abs
        mx = float(np.max(slice_)) / peak_abs
        peaks.append((mn, mx))
    duration = float(n) / float(sr)
    return peaks, int(sr), duration
