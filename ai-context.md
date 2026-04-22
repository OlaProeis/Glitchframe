# MusicVids - AI Context

## Rules (DO NOT UPDATE)

- Never auto-update this file or current-handover-prompt.md — only update when explicitly requested.
- Only do the task specified, do not start the next task, or go over scope.
- Run `uv sync && uv run pytest` after changes to verify code compiles (if the project has no tests yet, use `uv sync && python -m compileall .` until pytest is wired).
- Follow existing code patterns and conventions.
- Document by feature (e.g., `auth-layer.md`), not by task.
- Update `docs/index.md` when adding new documentation.
- Use Context7 MCP tool to fetch library documentation when needed (resolve library ID first, then fetch docs)

## Tech Stack

- **Language:** Python 3.11
- **GPU:** CUDA 12.x, PyTorch 2.x (FP16/BF16)
- **UI:** Gradio 4.x (`app.py` entry)
- **Audio:** librosa, soundfile, BeatNet (madmom fallback), demucs, whisperx
- **Diffusion:** diffusers, SDXL (+ optional AnimateDiff)
- **Graphics:** moderngl (reactive shaders), skia-python (kinetic type), Pillow, numpy
- **Video:** ffmpeg with NVENC (`h264_nvenc`), raw BGR frame pipe from compositor
- **Packaging:** `pyproject.toml` (hatchling, `pipeline` package) + `requirements.txt`; optional `MUSICVIDS_*` paths in `.env` (see `.env.example`)

## Architecture & Data Model

- **Flow:** Gradio UI → orchestrator coordinates audio analysis, lyrics alignment, background generation (`BackgroundSource`: SDXL stills, static Ken Burns, optional AnimateDiff), reactive + typography layers → compositor blends frames → ffmpeg encodes `output.mp4`; thumbnail + `metadata.txt` alongside.
- **Caching:** Per-song hash under `cache/<song_hash>/` — after ingest: `original.wav`, `analysis_mono.wav`; later: `analysis.json`, stems, `lyrics.aligned.json`, `background/`.
- **Outputs:** `outputs/<run_id>/` with `output.mp4`, `thumbnail.png`, `metadata.txt`.
- **Presets:** YAML in `presets/` (prompt, shader stem, typography style, hex palette); `config.py` loads and validates the registry; `pipeline/builtin_shaders.py` lists allowed shader stems without importing OpenGL.

## Conventions

- **Modularity:** One feature per file.
- **Errors:** Strict error handling, no silent failures.

## Where Things Live

| Want to... | Look in... |
|------------|-------------|
| Gradio entry, UI wiring | `app.py` |
| Dependencies, Python version | `pyproject.toml`, `requirements.txt` |
| Setup/feature notes | `docs/technical/project-setup-and-config.md` |
| Defaults, paths, presets registry | `config.py` |
| End-to-end pipeline coordination | `orchestrator.py` |
| Audio ingest (hash, WAV cache, mono preview) | `pipeline/audio_ingest.py` |
| Drop / build-up / band transients (analysis v2 `events` block) | `pipeline/musical_events.py`, `pipeline/beat_pulse.py` (`build_band_pulse_track`), `docs/technical/musical-events.md` |
| M1 spectrum render → ffmpeg (`output.mp4` under `outputs/<run_id>/`) | `pipeline/renderer.py`, `config.new_run_id` |
| Audio / lyrics / backgrounds / layers / encode | `pipeline/*.py` |
| Background modes (factory + caches) | `pipeline/background.py`, `pipeline/background_stills.py`, `pipeline/background_kenburns.py`, `pipeline/background_animatediff.py`, `docs/technical/background-modes.md` |
| Visual style presets | `presets/*.yaml`, `docs/technical/visual-style-presets.md` |
| Bundled reactive shader stems (no GL import) | `pipeline/builtin_shaders.py` |
| Reactive shader, background texture composite, Gradio reactive preview | `pipeline/reactive_shader.py`, `docs/technical/reactive-composite-and-gradio-preview.md` |
| Logo overlay (Pillow + NumPy blend, Gradio Branding) | `pipeline/logo_composite.py`, `docs/technical/logo-composite.md` |
| Logo rim prep (line mask, centroid, halo fallback) | `pipeline/logo_rim_lights.py`, `docs/technical/logo-rim-lights-prep.md` |
| Logo rim light field (traveling wave, halo + inward bleed, multicolour + hue drift, premult RGBA patch) + optional snare/bass `RimAudioModulation` and compositor stepper | `pipeline/logo_rim_lights.py` (`RimLightConfig`, `compute_logo_rim_light_patch`), `docs/technical/logo-rim-lights.md`, `docs/technical/logo-rim-lights-color.md`, `docs/technical/logo-rim-audio-modulation.md` |
| Logo rim + classic neon blended on output frames (`LogoGlowMode`, compositor `logo_rim_enabled`) | `pipeline/logo_composite.py`, `pipeline/compositor.py`, `docs/technical/logo-rim-compositing.md` |
| Logo rim Gradio controls → orchestrator (`OrchestratorInputs`, `resolve_logo_rim_compositor_fields`) | `app.py` (Branding accordion), `orchestrator.py`, `docs/technical/logo-rim-branding-ui.md` |
| Per-frame compositor → ffmpeg NVENC (bg + reactive + typo + logo, bounded queue) | `pipeline/compositor.py`, `docs/technical/frame-compositor.md` |
| Thumbnail PNG (chorus/RMS frame, Skia title, beside `output.mp4`) | `pipeline/thumbnail.py`, `docs/technical/thumbnail-generator.md` |
| Title / lyrics fill + glow color picker (bright-end of preset palette) | `pipeline/preset_colors.py::resolve_text_colors`, consumed by `orchestrator.py` and `pipeline/thumbnail.py::_resolve_title_colors` |
| YouTube `metadata.txt` (title, description, chapters, tags) | `pipeline/metadata.py`, `orchestrator.write_run_metadata`, `docs/technical/metadata-generator.md` |
| Preview 10 s (loudest RMS window), full render, ffprobe A/V sync check | `orchestrator.orchestrate_preview_10s`, `orchestrator.orchestrate_full_render`, `pipeline/preview.py`, `pipeline/av_sync.py`, `docs/technical/preview-and-render.md` |
| GLSL shaders | `assets/shaders/` |
| Fonts | `assets/fonts/` |
| Run outputs | `outputs/<run_id>/` |
| Cached analysis and models | `cache/<song_hash>/` |
