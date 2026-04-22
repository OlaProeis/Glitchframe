"""
Audio analyzer: beats, onsets, 8-band log-mel, RMS, structural segments, and an
optional vocal stem. Persists the complete feature bundle as
``cache/<song_hash>/analysis.json`` alongside the ingest WAV artifacts.

Preferred beat detector is :mod:`BeatNet` (ML, downbeat-aware). :mod:`madmom`
is used as an intermediate fallback; ``librosa.beat.beat_track`` is always
available as a final fallback so the analyzer works in minimal environments.
Vocal separation uses :mod:`demucs` (``htdemucs_ft``) when installed and is
skipped gracefully otherwise.

The JSON schema is versioned so downstream stages (reactive layer, typography,
background planner) can guard against breaking changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf

from pipeline.audio_ingest import (
    ANALYSIS_MONO_WAV_NAME,
    ORIGINAL_WAV_NAME,
)

LOGGER = logging.getLogger(__name__)

ANALYSIS_JSON_NAME = "analysis.json"
VOCALS_WAV_NAME = "vocals.wav"

ANALYSIS_SCHEMA_VERSION = 1

DEFAULT_FPS = 30
DEFAULT_NUM_BANDS = 8
DEFAULT_ONSET_HOP = 512
DEFAULT_N_FFT = 2048
DEFAULT_SEGMENT_COUNT = 8
DEFAULT_MEL_FMIN = 40.0
DEFAULT_MEL_FMAX = 16_000.0
MEL_DB_FLOOR = 80.0  # dB below ref treated as silence (maps to 0.0 after norm)

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class AnalysisResult:
    """Return payload for :func:`analyze_song`."""

    song_hash: str
    cache_dir: Path
    analysis_json: Path
    analysis: dict[str, Any]
    vocals_wav: Path | None


# ---------------------------------------------------------------------------
# Beat / tempo estimators (BeatNet → madmom → librosa)
# ---------------------------------------------------------------------------


def _bpm_from_intervals(beat_times: np.ndarray) -> float:
    if beat_times.size < 2:
        return 0.0
    intervals = np.diff(beat_times)
    intervals = intervals[intervals > 1e-6]
    if intervals.size == 0:
        return 0.0
    return float(60.0 / np.median(intervals))


def _estimate_beats_beatnet(wav_path: Path) -> dict[str, Any] | None:
    try:
        from BeatNet.BeatNet import BeatNet  # type: ignore
    except Exception as exc:  # noqa: BLE001 - any import failure is a miss
        LOGGER.info("BeatNet not available, skipping: %s", exc)
        return None
    try:
        estimator = BeatNet(
            1, mode="offline", inference_model="DBN", plot=[], thread=False
        )
        raw = np.asarray(estimator.process(str(wav_path)))
        if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
            LOGGER.warning("BeatNet returned unexpected output shape %s", raw.shape)
            return None
        times = raw[:, 0].astype(float)
        downbeats = raw[raw[:, 1] == 1, 0].astype(float)
        return {
            "source": "beatnet",
            "bpm": _bpm_from_intervals(times),
            "beats": times.tolist(),
            "downbeats": downbeats.tolist(),
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("BeatNet inference failed, falling back: %s", exc)
        return None


def _estimate_beats_madmom(wav_path: Path) -> dict[str, Any] | None:
    try:
        from madmom.features.beats import (  # type: ignore
            DBNBeatTrackingProcessor,
            RNNBeatProcessor,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("madmom not available, skipping: %s", exc)
        return None
    try:
        activations = RNNBeatProcessor()(str(wav_path))
        tracker = DBNBeatTrackingProcessor(fps=100)
        beats = np.asarray(tracker(activations), dtype=float)
        return {
            "source": "madmom",
            "bpm": _bpm_from_intervals(beats),
            "beats": beats.tolist(),
            "downbeats": [],
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("madmom inference failed, falling back: %s", exc)
        return None


def _estimate_beats_librosa(y: np.ndarray, sr: int) -> dict[str, Any]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if tempo_val <= 0:
        tempo_val = _bpm_from_intervals(np.asarray(beat_times, dtype=float))
    return {
        "source": "librosa",
        "bpm": float(tempo_val),
        "beats": [float(t) for t in beat_times],
        "downbeats": [],
    }


# ---------------------------------------------------------------------------
# Spectral / RMS / onset / segmentation features
# ---------------------------------------------------------------------------


def _frame_hop_for_fps(sr: int, fps: int) -> int:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    hop = sr // fps
    if hop <= 0:
        raise ValueError(f"fps={fps} too high for sr={sr}")
    return int(hop)


def _spectral_bands(
    y: np.ndarray, sr: int, fps: int, n_bands: int
) -> np.ndarray:
    """Return ``(n_bands, frames)`` log-mel energies normalized to [0, 1]."""
    hop = _frame_hop_for_fps(sr, fps)
    fmax = min(float(sr) / 2.0, DEFAULT_MEL_FMAX)
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=DEFAULT_N_FFT,
        hop_length=hop,
        n_mels=n_bands,
        fmin=DEFAULT_MEL_FMIN,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=MEL_DB_FLOOR)
    norm = (mel_db + MEL_DB_FLOOR) / MEL_DB_FLOOR
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _rms_envelope(y: np.ndarray, sr: int, fps: int) -> np.ndarray:
    hop = _frame_hop_for_fps(sr, fps)
    rms = librosa.feature.rms(
        y=y, frame_length=DEFAULT_N_FFT, hop_length=hop, center=True
    )[0]
    return rms.astype(np.float32)


def _onset_features(y: np.ndarray, sr: int) -> dict[str, Any]:
    hop = DEFAULT_ONSET_HOP
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    peak_times = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="time"
    )
    return {
        "hop_length": hop,
        "frame_rate_hz": float(sr) / float(hop),
        "frames": int(env.shape[0]),
        "strength": env.astype(np.float32).tolist(),
        "peaks": [float(t) for t in peak_times],
    }


def _segments(
    y: np.ndarray, sr: int, max_segments: int = DEFAULT_SEGMENT_COUNT
) -> list[dict[str, Any]]:
    duration = float(y.shape[-1]) / float(sr)
    if duration < 4.0:
        return [{"t_start": 0.0, "t_end": duration, "label": 0}]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < 4:
        return [{"t_start": 0.0, "t_end": duration, "label": 0}]

    # Cap cluster count by feature length so very short clips don't blow up.
    k = int(min(max_segments, max(2, mfcc.shape[1] // 50)))
    boundary_frames = np.asarray(librosa.segment.agglomerative(mfcc, k=k))
    if boundary_frames.size == 0:
        return [{"t_start": 0.0, "t_end": duration, "label": 0}]

    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr).astype(float)
    boundary_times = np.concatenate([boundary_times, [duration]])
    # Deduplicate / sort defensively; agglomerative normally yields sorted output.
    boundary_times = np.unique(np.clip(boundary_times, 0.0, duration))

    segments: list[dict[str, Any]] = []
    for i in range(boundary_times.size - 1):
        t0 = float(boundary_times[i])
        t1 = float(boundary_times[i + 1])
        if t1 - t0 > 1e-3:
            segments.append({"t_start": t0, "t_end": t1, "label": i})
    if not segments:
        segments.append({"t_start": 0.0, "t_end": duration, "label": 0})
    return segments


# ---------------------------------------------------------------------------
# Vocal separation (demucs htdemucs_ft) — optional
# ---------------------------------------------------------------------------


def _separate_vocals_with_demucs(
    src_wav: Path, out_wav: Path, *, model_name: str = "htdemucs_ft"
) -> bool:
    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore
        from demucs.apply import apply_model  # type: ignore
        from demucs.pretrained import get_model  # type: ignore
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("demucs not available, skipping vocal stem: %s", exc)
        return False

    # htdemucs_ft on CUDA is ~1 GB of resident weights; we MUST release it
    # in all exit paths so the next stage (WhisperX, then SDXL) doesn't
    # start its allocation on top of our leftover model. Keep a local
    # reference we can explicitly drop in ``finally``.
    from pipeline.gpu_memory import move_to_cpu, release_cuda_memory

    model: Any = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(model_name)
        model.to(device)
        if hasattr(model, "eval"):
            model.eval()

        wav, sr = torchaudio.load(str(src_wav))
        target_sr = int(getattr(model, "samplerate", 44_100))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        channels_needed = int(getattr(model, "audio_channels", 2))
        if wav.shape[0] == 1 and channels_needed == 2:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > channels_needed:
            wav = wav[:channels_needed]

        with torch.no_grad():
            sources = apply_model(
                model, wav.unsqueeze(0).to(device), device=device, progress=False
            )
        vocals = sources[0, list(model.sources).index("vocals")].cpu()
        torchaudio.save(str(out_wav), vocals, target_sr)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("demucs vocal separation failed: %s", exc)
        return False
    finally:
        move_to_cpu(model)
        model = None
        release_cuda_memory("demucs htdemucs_ft")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_analysis_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = librosa.to_mono(np.ascontiguousarray(y.T))
    return np.asarray(y, dtype=np.float32), int(sr)


def analyze_song(
    cache_dir: Path | str,
    *,
    fps: int = DEFAULT_FPS,
    num_bands: int = DEFAULT_NUM_BANDS,
    force: bool = False,
    separate_vocals: bool = True,
    progress: ProgressFn | None = None,
) -> AnalysisResult:
    """
    Run the full analyzer for a song that has already been ingested.

    Expects ``cache_dir`` to contain ``analysis_mono.wav`` (44.1 kHz mono) and
    optionally ``original.wav`` (preferred input for demucs).
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir does not exist: {cache}")

    analysis_mono = cache / ANALYSIS_MONO_WAV_NAME
    original_wav = cache / ORIGINAL_WAV_NAME
    analysis_json_path = cache / ANALYSIS_JSON_NAME
    vocals_path = cache / VOCALS_WAV_NAME

    if not analysis_mono.is_file():
        raise FileNotFoundError(
            f"Missing {ANALYSIS_MONO_WAV_NAME} in {cache}; run audio ingest first"
        )

    song_hash = cache.name

    def _report(p: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, p)), msg)

    if analysis_json_path.is_file() and not force:
        with analysis_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        vocals_ref = data.get("vocals_wav")
        vocals_present = isinstance(vocals_ref, str) and vocals_path.is_file()
        # Earlier runs may have skipped demucs (not installed). Do not short-circuit
        # forever: if vocals are still missing, try separation once and merge into cache.
        if separate_vocals and not vocals_present:
            _report(0.5, "Cached analysis; separating vocals (demucs)…")
            src = original_wav if original_wav.is_file() else analysis_mono
            if _separate_vocals_with_demucs(src, vocals_path):
                data["vocals_wav"] = VOCALS_WAV_NAME
                tmp_path = analysis_json_path.with_suffix(".json.tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, separators=(",", ":"))
                tmp_path.replace(analysis_json_path)
                vocals_present = True
        _report(1.0, "Using cached analysis.json")
        return AnalysisResult(
            song_hash=song_hash,
            cache_dir=cache,
            analysis_json=analysis_json_path,
            analysis=data,
            vocals_wav=vocals_path if vocals_present else None,
        )

    _report(0.05, "Loading audio…")
    y, sr = _load_analysis_mono(analysis_mono)
    duration = float(y.shape[-1]) / float(sr)

    _report(0.15, "Detecting beats / tempo…")
    beats = (
        _estimate_beats_beatnet(analysis_mono)
        or _estimate_beats_madmom(analysis_mono)
        or _estimate_beats_librosa(y, sr)
    )

    _report(0.35, "Computing 8-band log-mel spectrum…")
    spectrum = _spectral_bands(y, sr, fps, num_bands)

    _report(0.5, "Computing RMS envelope…")
    rms = _rms_envelope(y, sr, fps)

    n_frames = int(min(spectrum.shape[1], rms.shape[0]))
    spectrum = spectrum[:, :n_frames]
    rms = rms[:n_frames]

    _report(0.65, "Detecting onsets…")
    onsets = _onset_features(y, sr)

    _report(0.8, "Segmenting structure…")
    segments = _segments(y, sr)

    vocals_wav_name: str | None = None
    if separate_vocals:
        _report(0.88, "Separating vocals (demucs)…")
        src = original_wav if original_wav.is_file() else analysis_mono
        if _separate_vocals_with_demucs(src, vocals_path):
            vocals_wav_name = VOCALS_WAV_NAME

    _report(0.97, "Writing analysis.json…")
    # Per-frame layout (frames, num_bands) is easier to consume in the shader loop.
    spectrum_per_frame = spectrum.T.astype(np.float32, copy=False)
    data: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "song_hash": song_hash,
        "sample_rate": sr,
        "duration_sec": duration,
        "fps": fps,
        "num_bands": num_bands,
        "tempo": {
            "bpm": float(beats.get("bpm", 0.0) or 0.0),
            "source": beats["source"],
        },
        "beats": [float(t) for t in beats["beats"]],
        "downbeats": [float(t) for t in beats.get("downbeats", [])],
        "onsets": onsets,
        "spectrum": {
            "num_bands": num_bands,
            "fps": fps,
            "frames": int(spectrum_per_frame.shape[0]),
            "fmin_hz": DEFAULT_MEL_FMIN,
            "fmax_hz": min(float(sr) / 2.0, DEFAULT_MEL_FMAX),
            "values": spectrum_per_frame.tolist(),
        },
        "rms": {
            "fps": fps,
            "frames": int(rms.shape[0]),
            "values": rms.tolist(),
        },
        "segments": segments,
        "vocals_wav": vocals_wav_name,
    }

    tmp_path = analysis_json_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp_path.replace(analysis_json_path)

    _report(1.0, "Analysis complete")
    return AnalysisResult(
        song_hash=song_hash,
        cache_dir=cache,
        analysis_json=analysis_json_path,
        analysis=data,
        vocals_wav=vocals_path if vocals_wav_name else None,
    )
