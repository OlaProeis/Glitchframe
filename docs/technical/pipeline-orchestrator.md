# Pipeline orchestrator

Root module `orchestrator.py` coordinates ingest, audio analysis, and optional lyrics alignment. Later milestones will extend it with preview/full render entry points.

## Cache identity

- **`cache/<song_hash>/`** uses the SHA-256 hex digest of **raw uploaded file bytes** from `pipeline.audio_ingest` (`hash_audio_file` / `ingest_audio_file`).
- **`OrchestratorInputs.metadata`** (and similar cosmetic fields) must **not** change the song cache key. For export paths that need metadata, use a separate **`run_id`** under `outputs/<run_id>/` (see `config.OUTPUTS_DIR`).

## Public API

- **`OrchestratorInputs`**: `audio_path`, `song_hash`, `metadata`, `lyrics_text`, `presets`, `output_settings`. Provide **`audio_path`** to run ingest first, or **`song_hash`** alone when the song is already in cache (e.g. after Gradio upload).
- **`OrchestratorState`**: `song_hash`, `cache_dir`, optional `ingest_result`, **`analysis`** (`AnalysisResult`), optional **`alignment`** (`AlignmentResult`).
- **`orchestrate_analysis(..., force=False, progress=None, include_lyrics=False)`**: runs `analyze_song` on the cache dir; if `include_lyrics` and non-empty `lyrics_text`, runs `align_lyrics` after analysis. **`force`** is passed only to those two stages (ingest remains idempotent by hash).
- **`orchestrate_preview_10s(inputs, *, force=False, progress=None)`** / **`orchestrate_full_render(inputs, *, force=False, progress=None)`** return a `RenderResult` with the encoded MP4, optional thumbnail/metadata paths, and an `AvSyncReport`. See `docs/technical/preview-and-render.md` for the stage/progress map.

## Progress mapping

Sub-stages report `progress(float 0..1, msg)`. The orchestrator remaps them to a single Gradio bar:

- With **`audio_path`**: ingest **0–10%**, then analysis **10–70%** if lyrics run else **10–100%**, then lyrics **70–100%** when enabled.
- With **`song_hash` only**: analysis **0–70%** or **0–100%**, lyrics **70–100%** when enabled.

Stage messages are forwarded unchanged.

## UI wiring

`app.py` routes **Analyze** through `orchestrate_analysis` with `include_lyrics=False`. **Align lyrics** uses the same with `include_lyrics=True` so analysis runs first (cache-hit when already analyzed), then alignment. Exceptions are still caught and appended to the run log.

## Related docs

- `docs/technical/audio-ingest-and-cache.md` — WAV artifacts under `cache/<hash>/`.
- `docs/technical/audio-analyzer.md` — `analysis.json` and optional `vocals.wav`.
- `docs/technical/lyrics-aligner.md` — `lyrics.aligned.json` and `lyrics_sha256` cache key.
