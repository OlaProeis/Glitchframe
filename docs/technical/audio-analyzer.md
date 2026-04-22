# Audio analyzer

Feature: extract tempo/beats, onsets, an 8-band log-mel spectrum, RMS envelope,
structural segments, and an optional vocal stem from an ingested song, then
persist everything as a single JSON cache the downstream stages can consume.

## Flow

1. Requires that audio ingest has already written `cache/<hash>/analysis_mono.wav`
   (44.1 kHz mono) and ideally `original.wav` (native SR, channels preserved).
2. Beat/tempo detector tries in order: **BeatNet** → **madmom** → **librosa**
   (`librosa.beat.beat_track`). The source is recorded under `tempo.source`.
3. Librosa computes:
   - `librosa.feature.melspectrogram` (8 bands, `power_to_db` normalized to [0, 1])
     at hop `sr // fps` — feeds the reactive shader's `band_energies[8]`.
   - `librosa.feature.rms` at the same hop for the global RMS envelope.
   - `librosa.onset.onset_strength` + `onset_detect` (hop 512) for onset peaks.
   - `librosa.segment.agglomerative` on MFCCs for up to 8 structural segments.
4. Vocal stem via **demucs** (`htdemucs_ft`, uses CUDA when available) prefers
   `original.wav`; writes `cache/<hash>/vocals.wav`. Skipped cleanly when
   demucs / torchaudio are not installed. The demucs model (≈1 GB on GPU)
   is moved to CPU and `pipeline.gpu_memory.release_cuda_memory()` is
   called in a `finally` block immediately after separation so the
   weights don't linger into the WhisperX / SDXL stages.
   If `analysis.json` already exists from an earlier run without vocals (e.g. before
   installing demucs), a normal Analyze still uses the cache but **attempts demucs
   once** to add `vocals.wav` and update `vocals_wav` without recomputing the full
   feature bundle. Use `force=True` only when you need to recompute beats/spectrum.
5. Output is streamed via a progress callback and saved atomically as
   `cache/<hash>/analysis.json` (a temporary `.json.tmp` is renamed over the final
   path so partial files can't surface).

## analysis.json schema (v1)

```json
{
  "schema_version": 1,
  "song_hash": "<sha256>",
  "sample_rate": 44100,
  "duration_sec": 212.43,
  "fps": 30,
  "num_bands": 8,
  "tempo": { "bpm": 128.02, "source": "beatnet" | "madmom" | "librosa" },
  "beats":     [t_sec, ...],
  "downbeats": [t_sec, ...],
  "onsets": {
    "hop_length": 512,
    "frame_rate_hz": 86.13,
    "frames": N,
    "strength": [float, ...],
    "peaks":    [t_sec, ...]
  },
  "spectrum": {
    "num_bands": 8, "fps": 30, "frames": F,
    "fmin_hz": 40.0, "fmax_hz": 16000.0,
    "values": [[b0, b1, ..., b7], ...]   // per-frame, [0, 1]
  },
  "rms":      { "fps": 30, "frames": F, "values": [float, ...] },
  "segments": [{ "t_start": 0.0, "t_end": 16.2, "label": 0 }, ...],
  "vocals_wav": "vocals.wav" | null
}
```

## Caching

- Keyed entirely by the per-song hash directory (`cache/<hash>/`). A second call
  reuses `analysis.json` unless `force=True` is passed.
- `vocals_wav` in the JSON is a relative filename and is only treated as present
  when `vocals.wav` still exists on disk.

## Code

| Piece | Location |
|-------|----------|
| `analyze_song`, beat estimators, feature helpers | `pipeline/audio_analyzer.py` |
| Analyze button handler, BPM wiring | `app.py` (`_analyze`, Metadata `bpm_number`) |
| Cache path helper | `config.song_cache_dir` |

## Dependencies

- Always used: `librosa`, `soundfile`, `numpy`, `scipy`.
- Preferred beat detector: `BeatNet>=1.1.0` (pulls `madmom`), optional extra `beats`.
- Fallback beat detector: `madmom>=0.16` alone (same extra).
- Vocal stem: `demucs>=4.0.0` + `torchaudio>=2.0.0` (optional extra `vocals`).

Listed in `pyproject.toml` under optional extras: `all` / `analysis` (vocals +
`whisperx` in one shot), `vocals` (stem only), `lyrics` (WhisperX only), `beats`
(BeatNet/madmom; madmom often needs a Cython toolchain or a wheel). On Windows
prefer `.venv\Scripts\python.exe -m pip install -e ".[all]"` so `pip` matches the
app interpreter; see `docs/technical/project-setup-and-config.md`.

## Fallback behavior (no silent failures)

- Missing BeatNet → logged at INFO, madmom attempted.
- Missing madmom → logged at INFO, librosa is always available.
- demucs import / inference failure → logged at WARNING, `vocals_wav` set to
  `null` and the UI reports `vocals=skipped (demucs unavailable)`.
- Any hard error (missing `analysis_mono.wav`, bad `fps`, write failure, etc.)
  raises; the Gradio handler converts it to an error line in the run log.

## Related

- Audio ingest and cache layout: `docs/technical/audio-ingest-and-cache.md`
- UI wiring and progress panel: `docs/technical/gradio-ui.md`
