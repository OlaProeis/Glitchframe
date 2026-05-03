# Preview 10 s, full render, progress, and A/V sync

Feature doc for the action buttons that build a music video end-to-end:
**Preview 10 s** renders a short slice starting at the loudest section, and
**Render full video** encodes the whole track with a post-render ffprobe A/V
sync check. Both share the same orchestrator pipeline; they only differ in
the time window and post-processing.

## Public API

- **`orchestrator.orchestrate_preview_10s(inputs, *, force=False, progress=None) -> RenderResult`**
- **`orchestrator.orchestrate_full_render(inputs, *, force=False, progress=None) -> RenderResult`**
- **`RenderResult`**: `state: OrchestratorState`, `compositor: CompositorResult`, `preset_id`, `preset`, `metadata_path`, `av_sync: AvSyncReport | None`, `start_sec`, `duration_sec`, `is_preview`.
- `OrchestratorInputs` carries render-time settings (`preset_id`, `reactive_intensity_pct`, `logo_path/position/opacity_pct`, `width`, `height`, `fps`, `include_lyrics`, `preview_window_sec`). None of these fields participate in the song cache key (see `docs/technical/pipeline-orchestrator.md`).

## Pipeline stages and progress mapping

Both entry points share `_render_pipeline`:

1. **Analysis (+ optional lyrics alignment)** via `orchestrate_analysis`. Cache-hits short-circuit this stage.
   - Preview: **0–25%**, Full: **0–20%**.
2. **Background preparation** via `create_background_source(...).ensure()`. SDXL stills (including optional **RIFE** morph cache under `background/rife_timeline/`), Ken Burns, and AnimateDiff caches are all reused when present.
   - Preview: **25–45%**, Full: **20–40%**.
   - AnimateDiff: after segments are generated or loaded from disk, the diffusion **pipeline is unloaded** so the compositor does not share VRAM with SDXL (first compositor frame used to look “stuck” at 40% while memory was contended).
3. **Compositor** via `render_full_video`.
   - Preview/Full: **45–95%**.
4. **Metadata** (`outputs/<run_id>/metadata.txt`) — full render only, **96%**.
5. **A/V sync check** (`ffprobe`) on the encoded MP4, **98%**.

All stage callbacks expose `(fraction in [0, 1], message)` which `app.py`
wraps with `_EtaProgress` to append elapsed time and ETA. Exceptions raised
by any stage propagate to the UI handler, which catches and appends a human
readable error to the run log.

## Preview 10 s: loudest window selection

- `pipeline/preview.py` → **`pick_loudest_window_start(analysis, window_sec=10.0)`** picks the start time that maximises a convolution of the smoothed RMS envelope with a box filter of length `round(window_sec * rms.fps)`.
- The compositor is then invoked with `start_sec=t_start` and `duration_sec=window_sec`; the time passed to the background, reactive uniforms, and typography layers is shifted so they sample the same absolute timeline as the trimmed audio (`-ss`/`-t` on the audio input).
- Falls back to `0.0` when the track is shorter than the window or RMS data is missing.

## A/V sync validation

- `pipeline/av_sync.py` → **`ffprobe_av_sync(mp4_path, tolerance_ms=50.0)`** runs `ffprobe -show_streams -show_entries stream=codec_type,duration` and compares the video vs audio durations.
- Returns an **`AvSyncReport`**: `{video_duration_sec, audio_duration_sec, drift_ms, tolerance_ms, ffprobe_available, ok, message}`.
- When `ffprobe` is not on PATH the report flags `ffprobe_available=False` and `ok=False`; callers (orchestrator + UI) surface the message to the run log rather than raising.

## Resumable / cached stages

- **Ingest** is idempotent by SHA-256 of raw upload bytes (`audio_ingest`).
- **Analysis** reuses `cache/<hash>/analysis.json` when present (see `audio-analyzer.md`).
- **Lyrics alignment** is keyed by `(song_hash, lyrics_sha256)`; editing the text invalidates it automatically.
- **Backgrounds** cache frames/keyframes under `cache/<hash>/background/` (see `background-modes.md`).
- **Output MP4** is re-encoded per run into `outputs/<run_id>/output.mp4`; `run_id` uses a timestamp prefix so repeated renders never collide, and failures do not clobber previously-successful artifacts.

## Related docs

- `docs/technical/pipeline-orchestrator.md` — stage order and cache-key rules.
- `docs/technical/frame-compositor.md` — per-frame render + NVENC pipe.
- `docs/technical/spectrum-renderer-ffmpeg.md` — ffmpeg args reused by the compositor, including the new optional `audio_start_sec` / `audio_duration_sec`.
- `docs/technical/thumbnail-generator.md` — `thumbnail.png` written beside `output.mp4` on full renders.
- `docs/technical/metadata-generator.md` — `metadata.txt` written beside `output.mp4`.
