# Spectrum visualizer encode (M1)

End-to-end **1080p30** render: solid background, **8-band spectrum bars** from `analysis.json`, **raw `bgr24`** on ffmpeg stdin, **`original.wav`** muxed as the second input. Implements PRD §3.7 / milestone M1 without reactive shaders or typography.

## Entry point

- **`pipeline.renderer.render_spectrum_m1(cache_dir, ...)`** — expects `cache/<song_hash>/analysis.json` and `original.wav` unless paths are overridden.
- Returns **`SpectrumRenderResult`**: `run_id`, `output_dir`, `output_mp4`, `frame_count`, `audio_path`, `analysis_path`.

## Run outputs

- **`config.new_run_id(song_hash=...)`** — UTC timestamp plus short song hash (or random hex); used when `run_id` is omitted.
- Video path: **`outputs/<run_id>/output.mp4`** (`OUTPUTS_DIR` configurable via `GLITCHFRAME_OUTPUTS_DIR`, legacy `MUSICVIDS_OUTPUTS_DIR`).

## Spectrum and timing

- Reads **`spectrum.values`** as `(frames, bands)` and **`spectrum.fps`** (falls back to top-level `fps`).
- Per video frame, time is **frame-centered** `(i + 0.5) / video_fps`; bands are **linearly interpolated** along the spectrum time axis.
- Frame count is **`floor(audio_duration × fps)`** from **`soundfile`** on the muxed WAV so the video stream does not exceed audio length; ffmpeg **`-shortest`** trims the mux to the shorter stream.

## ffmpeg

- Default video codec: chosen by `pipeline.ffmpeg_tools.select_video_codec()`. Preference is **`h264_nvenc`** (`-preset p5 -rc vbr -cq 19 -b:v 12M`); when an explicit codec isn't forced, the helper runs a tiny 1-frame `lavfi` encode to confirm NVENC actually opens on the local machine (ffmpeg ≥ 8.x requires NVIDIA driver **≥ 570** — older drivers are auto-detected and we transparently fall back to `libx264 -preset medium -crf 23`). Audio is **AAC 192k**; muxed with **`-shortest`**.
- **Non-NVENC / CI / forced fallback:** set **`GLITCHFRAME_FFMPEG_VIDEO_CODEC=libx264`** (or legacy `MUSICVIDS_FFMPEG_VIDEO_CODEC`, or any other encoder string) to skip the probe entirely. Use **`GLITCHFRAME_FFMPEG_VIDEO_ARGS`** (or legacy `MUSICVIDS_FFMPEG_VIDEO_ARGS`) for extra `-c:v` flags. See `.env.example`.

## Errors

- Requires **`ffmpeg` on PATH**; subprocess stderr is collected to avoid pipe deadlocks; non-zero exit raises **`RuntimeError`** with stderr tail.

## Verification

- **`ffprobe`**: 1920×1080, 30 fps, video + audio streams.
- Optional: compare mux duration to source audio (target low drift; encoder uses `-shortest`).
