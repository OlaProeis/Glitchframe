## Rules for this file

- This index lists documentation files only, each with a one-line description.
- Do not add project history, task lists, architecture overviews, or narrative context here.
- When you add a new doc under `docs/` or `docs/technical/`, add exactly one bullet with its path and description.

# Documentation Index

## Core Context

- `ai-context.md` - Core project architecture, rules, and conventions.

## Technical Docs

- `docs/technical/project-setup-and-config.md` - Repo layout, pyproject/requirements, config paths and preset registry.
- `docs/technical/gradio-ui.md` - Gradio `app.py` tabbed UI, inputs, queue, placeholder actions, and run log.
- `docs/technical/audio-ingest-and-cache.md` - Upload ingest, `cache/<hash>/` WAV artifacts, and Gradio waveform preview.
- `docs/technical/audio-analyzer.md` - Beat/onset/spectrum/RMS/segment extraction, optional demucs vocal stem, and `analysis.json` cache.
- `docs/technical/musical-events.md` - Schema v2 drop detection, build-up tension series, post-drop afterglow, and low/mid/high band transient envelopes for reactive shaders.
- `docs/technical/spectrum-renderer-ffmpeg.md` - M1 spectrum bars, `bgr24` ffmpeg pipe, NVENC mux with `original.wav`, run outputs and CI encoder overrides.
- `docs/technical/lyrics-aligner.md` - WhisperX large-v3 word timings + Needleman-Wunsch alignment of pasted lyrics; inline `[m:ss]` anchors, typed `--- tag` markers, Silero VAD tighten, vocal-onset snap, and `lyrics.aligned.json` v3 cache (with per-word CTC score).
- `docs/technical/lyrics-timeline-editor.md` - Visual per-word timeline editor (Gradio tab + vanilla-JS waveform / drag handles) that writes `manually_edited: true` to the aligned JSON so re-aligns don't clobber user corrections.
- `docs/technical/pipeline-orchestrator.md` - Root `orchestrator.py`: stage order, cache-key rules, progress mapping, Gradio wiring.
- `docs/technical/reactive-shader-layer.md` - Offscreen moderngl fragment-shader pass, bundled GLSL shaders, `analysis.json`→uniforms mapping (`onset_env`, `bar_phase`, `build_tension` samplers + compositor-injected `bass_hit`/`transient_*`/`drop_hold`), and the per-uniform shader authoring guide.
- `docs/technical/reactive-composite-and-gradio-preview.md` - Reactive pass blended over RGB background in GLSL, strict shader resolution, intensity, and Gradio one-frame preview.
- `docs/technical/visual-style-presets.md` - Preset YAML schema, `config` registry validation, `builtin_shaders` allowlist, and Gradio preset auto-fill.
- `docs/technical/kinetic-typography.md` - Skia per-word typography layer, motion presets keyed by `typo_style`, and `lyrics.aligned.json`→RGBA frames.
- `docs/technical/logo-composite.md` - Optional PNG logo load, resize, corner/center placement, and NumPy alpha blend onto RGB/RGBA frames; Gradio branding preview.
- `docs/technical/logo-rim-lights-prep.md` - `compute_logo_rim_prep`: luma+edge line mask, alpha centroid, and stroke-vs-halo `use_line_features` fallback for future rim lighting.
- `docs/technical/logo-rim-lights.md` - Rim field overview: `compute_logo_rim_light_patch` / `RimLightConfig`, mask fallback, multicolour and audio modulation (cross-ref prep/color/audio/compositing/UI docs).
- `docs/technical/logo-rim-lights-color.md` - Multi-colour rim (`rim_color_layers` 2--3), HSV spread, per-layer phase offsets, `hue_drift_per_sec`, `song_hash` palette seed, and halo-only dual-tone behaviour.
- `docs/technical/logo-rim-audio-modulation.md` - Snare/bass `RimAudioModulation` on `compute_logo_rim_light_patch`, compositor `logo_rim_audio_reactive` stepper and analysis tracks.
- `docs/technical/logo-rim-compositing.md` - `LogoGlowMode`, rim vs classic neon blend order, `composite_logo_onto_frame` kwargs, and compositor `_effective_rim_light_config` / defaults.
- `docs/technical/logo-rim-branding-ui.md` - Gradio Branding accordion → `OrchestratorInputs` → `resolve_logo_rim_compositor_fields` / `CompositorConfig`; cosmetic-only (no song cache impact).
- `docs/technical/logo-rim-beams.md` - Pre-choreographed rim beams on drops + snare lead-ins: schedule algorithm, 10 s group gating, per-frame premultiplied RGBA patch, and `BeamConfig` tuning.
- `docs/technical/background-stills.md` - SDXL FP16 keyframe generator, section-aware prompts, cached PNGs under `cache/<hash>/background/`, and smoothstep-crossfade `background_frame(t)` API.
- `docs/technical/background-modes.md` - `BackgroundSource` factory, Ken Burns + AnimateDiff caches, Gradio/orchestrator mode wiring (cross-ref SDXL stills doc).
- `docs/technical/frame-compositor.md` - Per-frame compositor pipeline: bg + reactive + typography + logo, compositor-scope shader uniforms (`bass_hit`, `transient_lo/mid/hi`, `drop_hold`) built once per render, bounded queue producer/consumer, `bgr24` ffmpeg stdin.
- `docs/technical/thumbnail-generator.md` - Chorus/RMS frame pick, `render_single_frame`, Skia title overlay, and `thumbnail.png` beside `output.mp4`.
- `docs/technical/metadata-generator.md` - `metadata.txt` (title, description, chapters, tags) from analysis, lyrics, and preset; `write_run_metadata` in `orchestrator.py`.
- `docs/technical/preview-and-render.md` - Preview 10 s (loudest RMS window), full render, progress/ETA mapping, and ffprobe A/V sync validation.
- `docs/technical/title-and-beat-pulse.md` - Burned-in `Artist — Title` overlay (9-point grid, 3 sizes) and the logo-branding reactive stack: attack-dominant bass pulse, snare neon, snare squeeze, and RMS-jump impact glitch.
- `docs/technical/gpu-memory.md` - Shared `release_cuda_memory` / `move_to_cpu` helpers used by demucs, WhisperX, SDXL, and AnimateDiff to hand off VRAM cleanly between pipeline stages.
