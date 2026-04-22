# Frame compositor pipeline

End-to-end per-frame render for the full music video: pull the background RGB frame, run the reactive moderngl shader composited over it, alpha-blend kinetic typography and an optional logo, and stream raw `bgr24` bytes into an `ffmpeg` process.

## Entry point

- **`pipeline.compositor.render_full_video(cache_dir, *, background, analysis, aligned_words=None, audio_path=None, run_id=None, outputs_dir=None, config=None, progress=None)`**
- Returns **`CompositorResult`**: `run_id`, `output_dir`, `output_mp4`, `frame_count`, `audio_path`.
- Config via **`CompositorConfig`** (`fps`, `width`, `height`, shader/typography/logo settings, `font_path` / `title_font_path`, `video_codec`, `queue_size`).

## Per-frame flow

1. Frame-centered time `t = (i + 0.5) / fps` (consistent with `pipeline.renderer`).
2. `background.background_frame(t)` → `(H, W, 3) uint8` RGB (shape / dtype validated).
3. `uniforms_at_time(analysis, t, num_bands, intensity)` → reactive uniforms dict (`time`, `beat_phase`, `bar_phase`, `rms`, `onset_pulse`, `onset_env`, `build_tension`, `intensity`, `band_energies`).
4. Compositor-scope envelopes are merged into the same dict right after `uniforms_at_time`: `bass_hit` from `_shader_bass_track_for_analysis`, `transient_lo / transient_mid / transient_hi` from `_shader_transient_tracks_for_analysis`, and `drop_hold` from `_drop_hold_fn_for_analysis`. All four are built **once per render** outside the frame loop (`PulseTrack` / closure scalar lookups inside) to keep the hot path scalar; see `docs/technical/reactive-shader-layer.md` for how shaders should consume each signal and **[`musical-events.md`](musical-events.md)** for the underlying detector + decays.
5. `ReactiveShader.render_frame_composited_rgb(uniforms, bg_rgb)` blends the shader pass over the background on the GPU → `(H, W, 3) uint8` RGB.
6. When `aligned_words` are provided: `KineticTypographyLayer.render_frame(t, uniforms)` → `(H, W, 4) uint8` RGBA, premultiplied-alpha blended via `composite_premultiplied_rgba_over_rgb`.
7. When `config.title_text` is non-empty: a pre-rendered `(H, W, 4)` RGBA title card (rasterised once per render by `pipeline.title_overlay.render_title_rgba`) is alpha-blended on top. See **[`title-and-beat-pulse.md`](title-and-beat-pulse.md)** for position/size controls.
8. When `config.logo_path` is set: logo is loaded + resized once up front, then `composite_logo_onto_frame(..., inplace=True, scale=…)` blends it at the chosen corner/center. When `config.logo_beat_pulse` is enabled the per-frame `scale` and `opacity_pct` are modulated by a pulse function built once via `pipeline.compositor._build_pulse_fn(cfg, analysis)`: `bass` mode (default) keys off low-frequency energy via `pipeline.beat_pulse.build_bass_pulse_track`, `beats` mode keys off the analyzer's grid via `pipeline.beat_pulse.beat_pulse_envelope`. See **[`title-and-beat-pulse.md`](title-and-beat-pulse.md)**.
9. Final RGB frame is converted to a contiguous BGR buffer and pushed to the consumer.

## Threading / backpressure

- A single **producer thread** owns all GPU / Skia state: `ReactiveShader` and `KineticTypographyLayer` are constructed and closed inside `_produce()` so moderngl and Skia contexts never migrate across threads.
- The producer puts per-frame `bgr24` bytes onto a **bounded `queue.Queue`** (size `CompositorConfig.queue_size`, default **4**). The **main thread** is the consumer: it reads chunks and writes them to `ffmpeg`'s `stdin`.
- A sentinel `None` on the queue signals end-of-stream. A shared `stop_event` lets either side short-circuit the other on errors.
- `ffmpeg` is spawned with `bufsize=0`, and frame bytes are fed through `_pipe_write_all()` which loops `os.write(fd, view[:1 MiB])` until each frame is drained. This avoids `OSError(EINVAL)` observed on **Python 3.13 / Windows** when `BufferedWriter.write()` is handed a full 4K BGR24 frame (~24 MiB) in one call, *and* lets us surface short-writes on slow readers without losing bytes.
- If `ffmpeg` dies early the main thread catches `BrokenPipeError` / `OSError(EPIPE/EINVAL)`, sets `stop_event`, and surfaces the encoder's stderr tail through the final `RuntimeError`.
- Producer exceptions are captured and re-raised on the main thread after `proc.wait()` so the encoder is always cleaned up.

## Progress reporting

The `progress(fraction, message)` callback reports three distinct pre-loop stages and a live per-frame status so a slow render never looks "hung":

1. `Initializing GPU shader context…` — moderngl standalone context + shader compile (can be 1–5 s on first run of a session).
2. `Preparing kinetic typography (N words)…` — Skia font/typeface setup when `aligned_words` is non-empty. Skipped when lyrics are disabled.
3. `Rendering frame i/total · X fps · bg+shader[+typo][+title][+logo][+pulse] · frame ETA Ns` — per frame. `fps` is the cumulative render rate (`frames_done / elapsed`), the label lists which layers are active in this run so users can tell at a glance which stages are costing them time (lyric renders typically drop to ~1–2 fps), and the ETA is derived from that rate.

The compositor also emits an `INFO` log on start (`N frames · WxH @ fps · layers=…`) and a throttled ~5 s progress line so terminal output stays readable on hour-long renders without hiding stalls.

## ffmpeg

- Command built with the same helper used by the M1 spectrum renderer: `rawvideo` → H.264 encoder selected via `pipeline.ffmpeg_tools.select_video_codec()`. Default preference is `h264_nvenc` (NVENC args `-preset p5 -rc vbr -cq 19 -b:v 12M`); if the probe detects that the local ffmpeg cannot actually open NVENC (e.g. NVIDIA driver older than what the ffmpeg build requires), the compositor transparently falls back to `libx264` (`-preset medium -crf 23`). Users can still force a specific codec via `CompositorConfig.video_codec` or `MUSICVIDS_FFMPEG_VIDEO_CODEC` env var (override skips the probe). See **`docs/technical/spectrum-renderer-ffmpeg.md`** for CI overrides (`MUSICVIDS_FFMPEG_VIDEO_CODEC`, `MUSICVIDS_FFMPEG_VIDEO_ARGS`).
- Frame count is `floor(audio_duration × fps)` from `soundfile.info` on the muxed WAV; `-shortest` trims any residual drift in the muxed output.

## Errors

- Validates shader stem via `resolve_builtin_shader_stem` before spawning the producer (no fallback to another shader).
- Enforces background size matches the compositor resolution; rejects malformed per-frame arrays (dtype / shape).
- Requires `ffmpeg` on PATH; subprocess stderr is collected into a `TemporaryFile` to avoid pipe deadlocks, and the tail is included in the `RuntimeError` on non-zero exit.
- The caller owns the `BackgroundSource` — the compositor does not call `close()` on it.

## See also

- `docs/technical/background-modes.md` — `BackgroundSource` factory and caches.
- `docs/technical/reactive-composite-and-gradio-preview.md` — GPU compositing math and `u_comp_background`.
- `docs/technical/reactive-shader-layer.md` — per-uniform shader authoring guide for the signals injected in step 4.
- `docs/technical/musical-events.md` — drop / build-up / band-transient detectors feeding `drop_hold` + `transient_lo/mid/hi`.
- `docs/technical/kinetic-typography.md` — per-word RGBA layer and motion presets.
- `docs/technical/logo-composite.md` — logo placement / opacity / alpha blend.
- `docs/technical/spectrum-renderer-ffmpeg.md` — ffmpeg / NVENC pipe patterns reused here.
