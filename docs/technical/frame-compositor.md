# Frame compositor pipeline

End-to-end per-frame render for the full music video: pull the background RGB frame, run the reactive moderngl shader composited over it, alpha-blend kinetic typography and an optional logo, and stream raw `bgr24` bytes into an `ffmpeg` process.

## Entry point

- **`pipeline.compositor.render_full_video(cache_dir, *, background, analysis, aligned_words=None, audio_path=None, run_id=None, outputs_dir=None, config=None, progress=None)`**
- Returns **`CompositorResult`**: `run_id`, `output_dir`, `output_mp4`, `frame_count`, `audio_path`, `thumbnail_png`, **`render_stats`** (`CompositorRenderStats`).
- `render_stats` exposes the post-run headline numbers: `frame_count`, `elapsed_sec`, `avg_fps`, `video_codec` (e.g. `"h264_nvenc"` or `"libx264"`), `ffmpeg_path`. Used by `app._summarise_render` to append a one-line summary to the in-app run log so users on read-only Pinokio terminals can confirm whether NVENC actually engaged.
- Config via **`CompositorConfig`** (`fps`, `width`, `height`, shader/typography/logo settings, `font_path` / `title_font_path`, `video_codec`, `queue_size`).

## Per-frame flow

1. Frame-centered time `t = (i + 0.5) / fps` (consistent with `pipeline.renderer`).
2. `background.background_frame(t)` → `(H, W, 3) uint8` RGB (shape / dtype validated).
3. `uniforms_at_time(analysis, t, num_bands, intensity)` → reactive uniforms dict (`time`, `beat_phase`, `bar_phase`, `rms`, `onset_pulse`, `onset_env`, `build_tension`, `intensity`, `band_energies`).
4. Compositor-scope envelopes are merged into the same dict right after `uniforms_at_time`: `bass_hit` from `_shader_bass_track_for_analysis`, `transient_lo / transient_mid / transient_hi` from `_shader_transient_tracks_for_analysis`, and `drop_hold` from `_drop_hold_fn_for_analysis`. All four are built **once per render** outside the frame loop (`PulseTrack` / closure scalar lookups inside) to keep the hot path scalar; see `docs/technical/reactive-shader-layer.md` for how shaders should consume each signal and **[`musical-events.md`](musical-events.md)** for the underlying detector + decays.
5. `ReactiveShader.render_frame_composited_rgb(uniforms, bg_rgb)` blends the shader pass over the background on the GPU → `(H, W, 3) uint8` RGB.
6. When `config.voidcat_ascii_ctx` is set (**void ASCII** reactive shader via orchestrator): `render_voidcat_ascii_rgba` in `pipeline/voidcat_ascii` produces premultiplied RGBA and is blended over the composited frame **before** typography. The field **fills the frame** with per-cell coloured glyphs (time + beat/bar phase, onsets, transients, build tension, etc.); a **centre column band** uses a lower alpha than the flanks so a centre logo stays readable, unless `sides_only` is enabled (fully transparent middle). A small **cat** still appears in the side panes on the top-confidence `events.drops` hit only.
6a. When `config.audio_vignette_enabled` is true (default): a subtle radial dark-edge vignette pulses with `bass_hit` + `drop_hold` + `rms` over the picture (bg + shader + ascii) before typography runs, so lyrics / title / logo stay clean. See **[`audio-vignette.md`](audio-vignette.md)**.
7. When `aligned_words` are provided: `KineticTypographyLayer.render_frame(t, uniforms)` → `(H, W, 4) uint8` RGBA, premultiplied-alpha blended via `composite_premultiplied_rgba_over_rgb`.
8. When `config.title_text` is non-empty: a pre-rendered `(H, W, 4)` RGBA title card (rasterised once per render by `pipeline.title_overlay.render_title_rgba`) is alpha-blended on top. See **[`title-and-beat-pulse.md`](title-and-beat-pulse.md)** for position/size controls.
9. When `config.logo_path` is set: logo is loaded + resized once up front, then `composite_logo_onto_frame(..., inplace=True, scale=…)` blends it at the chosen corner/center. When `config.logo_beat_pulse` is enabled the per-frame `scale` and `opacity_pct` are modulated by a pulse function built once via `pipeline.compositor._build_pulse_fn(cfg, analysis)`: `bass` mode (default) keys off low-frequency energy via `pipeline.beat_pulse.build_bass_pulse_track`, `beats` mode keys off the analyzer's grid via `pipeline.beat_pulse.beat_pulse_envelope`. See **[`title-and-beat-pulse.md`](title-and-beat-pulse.md)**.
10. Final RGB frame is converted to a contiguous BGR buffer and pushed to the consumer.

## Threading / backpressure

- A single **producer thread** owns all GPU / Skia state: `ReactiveShader` and `KineticTypographyLayer` are constructed and closed inside `_produce()` so moderngl and Skia contexts never migrate across threads.
- The producer puts per-frame `bgr24` bytes onto a **bounded `queue.Queue`** (size `CompositorConfig.queue_size`, default **4**). The **main thread** is the consumer: it reads chunks and writes them to `ffmpeg`'s `stdin`.
- The producer also drives the rendering itself but **never calls `progress(...)` directly.** Gradio's `gr.Progress` silently drops cross-thread updates (the request thread it was created on is the only thread it forwards from reliably), so doing so would freeze the UI bar mid-render even though the producer was happily emitting per-frame progress. Instead the producer writes scalar counters (`frames_produced`, `frames_encoded`, `phase`) into a shared `_CompositorStats` dataclass and the consumer polls those scalars at a fixed cadence (see *Progress reporting* below). Lock-free: every field is a simple type whose individual reads / writes are atomic under the GIL.
- A sentinel `None` on the queue signals end-of-stream. A shared `stop_event` lets either side short-circuit the other on errors.
- `ffmpeg` is spawned with `bufsize=0`, and frame bytes are fed through `_pipe_write_all()` which loops `os.write(fd, view[:1 MiB])` until each frame is drained. This avoids `OSError(EINVAL)` observed on **Python 3.13 / Windows** when `BufferedWriter.write()` is handed a full 4K BGR24 frame (~24 MiB) in one call, *and* lets us surface short-writes on slow readers without losing bytes.
- If `ffmpeg` dies early the main thread catches `BrokenPipeError` / `OSError(EPIPE/EINVAL)`, sets `stop_event`, and surfaces the encoder's stderr tail through the final `RuntimeError`.
- Producer exceptions are captured and re-raised on the main thread after `proc.wait()` so the encoder is always cleaned up.

## Progress reporting

The `progress(fraction, message)` callback is **always called from the request thread** — specifically the encoder feed loop in `render_full_video`, never from the producer. The consumer wakes up on `frame_q.get(timeout=_PROGRESS_TICK_SEC)` (250 ms) so the UI bar still ticks during the warmup window before the first frame is queued.

`_CompositorStats.progress_pair()` formats the (fraction, message) pair from the current scalar snapshot:

* **Warmup (no frame encoded yet)** → `Compositing 0/N - <phase> - <layer label>`. Phases the producer transitions through: `initializing`, `initializing GPU shader context`, `preparing kinetic typography (N words)`, `warming up`. So a 30 s SDXL/typography preroll is no longer indistinguishable from a hang.
* **Steady state** → `Compositing K/N (X.X%) - F.FF fps - ETA hh:mm:ss - <layer label>`. `fps` is the cumulative encoded-frame rate (`frames_encoded / (now - render_started)`), which equals overall pipeline throughput because the encoder is the bottleneck whenever NVENC is unavailable. ETA is `(N - K) / fps`. The layer label (`layers=BG+SHADER+TYPO+TITLE+LOGO+PULSE`, or whichever subset is active) is built once and tells users at a glance which stages are costing them time — lyric-heavy renders typically drop to ~1-2 fps because Skia kinetic typography draws ~1 Skia stroke per visible word per frame.

The 250 ms tick cadence (4 progress callbacks/s) is locked by a regression test (`tests/test_compositor_progress_threading.py::TestProgressTickConstant`). Faster spams Gradio's internal websocket queue (visibly laggy at 1080p NVENC ~30 fps); slower feels frozen on long encodes.

The compositor also emits an `INFO` log on start (`Compositor starting: N frames · WxH @ fps · layers=…`) and a throttled ~5 s log line (`Compositor: K/N frames (X.X%) · F fps`) so terminal output stays readable on hour-long renders without hiding stalls. Logs are thread-safe so this stays in the producer.

## ffmpeg

- Command built with the same helper used by the M1 spectrum renderer: `rawvideo` → H.264 encoder selected via `pipeline.ffmpeg_tools.select_video_codec()`. Default preference is `h264_nvenc` (NVENC args `-preset <p5|slow> -rc vbr -cq 19 -b:v 12M`); if the local ffmpeg cannot actually open NVENC (e.g. NVIDIA driver older than what the ffmpeg build requires, or a conda-env-bundled ffmpeg without `--enable-nvenc`), the compositor transparently falls back to `libx264` (`-preset medium -crf 23`).
- **Multi-candidate codec selection.** `select_video_codec()` first tries the highest-priority binary returned by `resolve_ffmpeg()`; on probe failure it sweeps **every** ffmpeg discovered by `_iter_candidates()` (env override → `sys.prefix\Library\bin` (active venv/conda env) → PATH → well-known install dirs) via `_pick_codec_capable_ffmpeg(codec)`, and **promotes** the first codec-capable candidate into `_cache["ffmpeg"]` so subsequent encode commands use the working binary. This catches the Pinokio failure mode where a conda-env-bundled ffmpeg shadows a working winget one — without the multi-candidate sweep, NVENC would fail and we'd fall back to `libx264` even though a perfectly capable ffmpeg existed elsewhere on the system. After codec selection runs, both the renderer and compositor **re-resolve** `ffmpeg_bin` so the encode command and the NVENC-preset probe both target the same (potentially promoted) binary.
- **NVENC preset auto-detection.** `pipeline.ffmpeg_tools.select_nvenc_preset(binary)` probes whether the chosen ffmpeg understands the modern `p1..p7` preset family (FFmpeg ≥ 4.4 / NVENC SDK ≥ 11) by running a 1-frame `lavfi` encode with `-preset p5`. If that fails — the symptom is `Undefined constant or missing '(' in 'p5' / Unable to parse option value "p5"`, common with older conda-forge ffmpegs that get bundled into Pinokio venvs — the helper returns `slow` instead, which is the closest legacy equivalent (~5% encoder-speed difference at 1080p, near-equivalent visual quality). Result is cached per binary. `pipeline.renderer._ffmpeg_video_args("h264_nvenc", ffmpeg_bin=...)` consults this helper rather than hardcoding `p5`. Without this, the compositor would pick a working NVENC ffmpeg via the multi-candidate sweep and then crash mid-render the moment the encoder tried to parse `-preset p5`.
- **Probe diagnostics.** `_probe_encoder()` runs ffmpeg at `-loglevel info` (not `error`) and captures the last 14 stderr lines on failure. At `error` level ffmpeg suppresses NVENC's actual diagnostics ("Cannot load nvEncodeAPI64.dll", "Driver does not support the required nvenc API version", "OpenEncodeSession failed") and only prints its useless wrapper, which historically made the original Pinokio bug undebuggable. The `-loglevel info` choice is locked by a regression test (`tests/test_compositor_progress_threading.py::TestCodecCapableFfmpegPicker`).
- **Startup diagnostics.** `log_ffmpeg_diagnostics()` (called from `app.py` startup) lists every discovered candidate in priority order, the resolved binary's version banner, and the NVIDIA-related `configure:` flags (`--enable-nvenc`, `--enable-cuda-llvm`, `--enable-nvdec`, `--enable-cuvid`). If those flags are missing it warns up front that NVENC will never work with this build — fail loudly rather than burning probe attempts.
- Users can still force a specific codec via `CompositorConfig.video_codec` or `GLITCHFRAME_FFMPEG_VIDEO_CODEC` (legacy `MUSICVIDS_FFMPEG_VIDEO_CODEC`) env var. Override skips both the probe and the multi-candidate sweep — the user's preference wins. See **`docs/technical/spectrum-renderer-ffmpeg.md`** for CI overrides (`GLITCHFRAME_FFMPEG_VIDEO_CODEC` / `GLITCHFRAME_FFMPEG_VIDEO_ARGS`, or legacy `MUSICVIDS_*`).
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
