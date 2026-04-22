# Audio ingest and per-song cache

Feature: decode uploaded tracks, normalize cache layout under a content hash, and drive the Gradio audio preview.

## What it does

- **Hash:** SHA-256 (hex) of the **raw uploaded file bytes** → folder name `cache/<hash>/`.
- **`original.wav`:** Full-quality decode at the file’s **native sample rate**; **channel layout preserved** (stereo/mono). Written as 32-bit float WAV via **soundfile**.
- **`analysis_mono.wav`:** **44.1 kHz mono** mixdown for downstream analysis and UI preview; resampling uses **librosa** with `res_type="scipy"` (requires **scipy**).
- **Idempotency:** If both WAVs already exist for that hash directory, ingest skips re-decoding and reuses them.

## Code

| Piece | Location |
|-------|----------|
| Ingest API, `IngestResult`, preview path helper | `pipeline/audio_ingest.py` |
| `song_cache_dir()` | `config.py` |
| Upload handler, `gr.Audio` preview, `gr.State` for hash | `app.py` (Audio tab) |

## UI

- **`gr.File`** change event calls ingest; **`gr.Audio`** is `type="filepath"`, **interactive**, and points at `analysis_mono.wav` so the browser can play, scrub, and show the waveform.
- Run log records full hash, duration, and cache path (or errors).

## Related

- UI overview: `docs/technical/gradio-ui.md`
- Global paths: `docs/technical/project-setup-and-config.md`
