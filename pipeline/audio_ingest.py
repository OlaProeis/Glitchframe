"""Load uploaded audio, cache per-song artifacts, and produce analysis mono @ 44.1 kHz."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from config import song_cache_dir

ANALYSIS_SR = 44_100
ORIGINAL_WAV_NAME = "original.wav"
ANALYSIS_MONO_WAV_NAME = "analysis_mono.wav"


@dataclass(frozen=True)
class IngestResult:
    song_hash: str
    cache_dir: Path
    original_wav: Path
    analysis_mono_wav: Path
    analysis_sample_rate: int
    duration_sec: float


def hash_audio_file(path: Path) -> str:
    """SHA-256 of file bytes (hex digest) for cache folder naming."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_soundfile_layout(y: np.ndarray) -> np.ndarray:
    """Librosa (channels, samples) -> (samples, channels) float32."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    if y.ndim == 2:
        return np.ascontiguousarray(y.T)
    raise ValueError(f"Unexpected audio shape {y.shape}")


def _normalize_source_path(upload_path: str | Path) -> Path:
    """Absolute, resolved path for reads — avoids libsndfile *System error* on Windows."""
    raw = Path(upload_path)
    if raw.is_file():
        return raw.resolve()
    alt = raw.expanduser().resolve(strict=False)
    if alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(f"Not a file: {upload_path!r} (resolved attempt: {alt!r})")


def ingest_audio_file(
    upload_path: str | Path,
    *,
    cache_dir_root: Path | None = None,
) -> IngestResult:
    """
    Decode upload, write ``original.wav`` (native SR, channel layout preserved) and
    ``analysis_mono.wav`` (44.1 kHz mono). Idempotent when cache already populated
    for the same content hash.
    """
    path = _normalize_source_path(upload_path)

    song_hash = hash_audio_file(path)
    out_dir = (
        Path(cache_dir_root) / song_hash
        if cache_dir_root is not None
        else song_cache_dir(song_hash)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    original_wav = out_dir / ORIGINAL_WAV_NAME
    analysis_mono_wav = out_dir / ANALYSIS_MONO_WAV_NAME

    if original_wav.is_file() and analysis_mono_wav.is_file():
        info = sf.info(str(analysis_mono_wav.resolve()))
        return IngestResult(
            song_hash=song_hash,
            cache_dir=out_dir,
            original_wav=original_wav,
            analysis_mono_wav=analysis_mono_wav,
            analysis_sample_rate=int(info.samplerate),
            duration_sec=float(info.duration),
        )

    src_for_librosa = str(path)
    y, sr = librosa.load(src_for_librosa, sr=None, mono=False)
    y_sf = _to_soundfile_layout(y)
    orig_out = str(original_wav.resolve())
    try:
        sf.write(
            orig_out,
            y_sf,
            int(sr),
            subtype="FLOAT",
            format="WAV",
        )
    except Exception as exc:
        raise RuntimeError(
            f"soundfile could not write {orig_out!r} (FLOAT WAV). On Windows this "
            f"often follows a *System error* from libsndfile when the path is odd "
            f"or the folder is not writable — check cache dir {out_dir!r}. "
            f"Original: {type(exc).__name__}: {exc}"
        ) from exc

    y_mono = librosa.to_mono(y) if y.ndim > 1 else np.asarray(y, dtype=np.float32)
    y_mono_44 = librosa.resample(
        y_mono,
        orig_sr=int(sr),
        target_sr=ANALYSIS_SR,
        res_type="scipy",
    )
    y_mono_44 = np.clip(y_mono_44, -1.0, 1.0).astype(np.float32, copy=False)

    mono_out = str(analysis_mono_wav.resolve())
    try:
        sf.write(
            mono_out,
            y_mono_44.reshape(-1, 1),
            ANALYSIS_SR,
            subtype="FLOAT",
            format="WAV",
        )
    except Exception as exc:
        raise RuntimeError(
            f"soundfile could not write {mono_out!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    duration = float(y_mono_44.shape[0]) / ANALYSIS_SR
    return IngestResult(
        song_hash=song_hash,
        cache_dir=out_dir,
        original_wav=original_wav,
        analysis_mono_wav=analysis_mono_wav,
        analysis_sample_rate=ANALYSIS_SR,
        duration_sec=duration,
    )


def preview_value_for_gradio(result: IngestResult) -> Any:
    """Path to analysis mono WAV — ``gr.Audio`` can play/scrub and show waveform."""
    return str(result.analysis_mono_wav)
