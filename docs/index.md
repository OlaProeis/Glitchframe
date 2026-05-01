## Rules for this file

- This index lists documentation files only, each with a one-line description.
- Do not add project history, task lists, architecture overviews, or narrative context here.
- When you add a new doc under `docs/` or `docs/technical/`, add exactly one bullet with its path and description.

# Documentation Index

## Core Context

- `ai-context.md` - Core project architecture, rules, and conventions.

## User guides

- `docs/guides/getting-started-windows.md` - Windows: Pinokio search **glitchframe** (easiest); or manual order of installs (Python, optional Git, ffmpeg via winget), PowerShell, ZIP vs git clone, venv, PyTorch, run.

## Technical Docs

- `docs/technical/pinokio-package.md` - Pinokio scripts (`install.js` / `start.js` / `pinokio.js`); discover by searching **glitchframe** in Pinokio; venv `env`, optional `[all]`, `ffmpeg` prereq.
- `docs/technical/pinokio-lyrics-align-windows-handover.md` - **Handover prompt:** Pinokio on Windows, Align lyrics / WhisperX / cuDNN, what we tried, open questions, revert notes.
- `docs/technical/project-setup-and-config.md` - Repo layout, pyproject/requirements, config paths and preset registry.
- `docs/technical/windows-venv-recovery-guide.md` - Windows: step-by-step venv recovery after PyTorch / WhisperX lyrics-align issues (weights_only, cuDNN DLL mismatches).
- `docs/technical/gradio-ui.md` - Gradio `app.py` tabbed UI, inputs, queue, placeholder actions, and run log.
- `docs/technical/audio-ingest-and-cache.md` - Upload ingest, `cache/<hash>/` WAV artifacts, and Gradio waveform preview.
- `docs/technical/audio-analyzer.md` - Beat/onset/spectrum/RMS/segment extraction, optional demucs vocal stem, and `analysis.json` cache.
- `docs/technical/musical-events.md` - Schema v2 drop detection, build-up tension series, post-drop afterglow, and low/mid/high band transient envelopes for reactive shaders.
- `docs/technical/spectrum-renderer-ffmpeg.md` - M1 spectrum bars, `bgr24` ffmpeg pipe, NVENC mux with `original.wav`, run outputs and CI encoder overrides.
- `docs/technical/lyrics-aligner.md` - WhisperX large-v3 word timings + Needleman-Wunsch alignment of pasted lyrics; inline `[m:ss]` anchors, typed `--- tag` markers, Silero VAD tighten, vocal-onset snap, and `lyrics.aligned.json` v3 cache (with per-word CTC score).
- `docs/technical/lyrics-timeline-editor.md` - Visual per-word timeline editor (Gradio tab + vanilla-JS waveform / drag handles) that writes `manually_edited: true` to the aligned JSON so re-aligns don't clobber user corrections; `user-select: none` on word UI and high-contrast help `kbd` styles.
- `docs/technical/waveform-peaks.md` - Shared `compute_peaks` WAV→min/max column downsampling for canvas waveforms (`pipeline/_waveform_peaks.py`).
- `docs/technical/effects-timeline.md` - `EffectKind` / `EffectClip` / `EffectsTimeline`, per-kind settings allowlist, and atomic `effects_timeline.json` load/save under the song cache.
- `docs/technical/effects-editor-backend.md` - `load_editor_state` / `save_edited_timeline` / `bake_auto_schedule` in `pipeline/effects_editor.py` (peaks, ghost auto hints, 20 ms dedupe when baking).
- `docs/technical/effects-timeline-editor.md` - `build_editor_html` in `pipeline/effects_editor.py`: self-contained CSS + markup + inline JS (gr.HTML), 7 colour-coded rows, master reactivity slider, per-clip gear panel bound to `EFFECT_SETTINGS_KEYS`, `window._glitchframe_effects_state` round-trip, number keys 1–7 to add clips at the playhead, `user-select` on clips and help `kbd` contrast.
- `docs/technical/effects-timeline-renderers.md` - Umbrella: frame post-pass renderers vs logo-path kinds; compositor order; links to per-renderer docs.
- `docs/technical/effects-timeline-test-suite.md` - Which `tests/test_effects_*.py` and compositor tests cover the effects-timeline stack; how to run `unittest`.
- `docs/technical/effects-timeline-gradio-tab.md` - Gradio **Effects timeline** tab in `app.py`: handler wiring, audio URL precedence, save `js=` pattern, clear-all semantics, and smoke test; orchestrator hook-up is out of scope here.
- `docs/technical/screen-shake-renderer.md` - `shake_offset(t, clips, song_hash)` for deterministic `SCREEN_SHAKE` pixel offsets from timeline clips (`pipeline/screen_shake.py`).
- `docs/technical/chromatic-aberration-renderer.md` - `apply_chromatic_aberration(frame, t, clips, song_hash)` for full-frame R/B channel split from `CHROMATIC_ABERRATION` clips (`pipeline/chromatic_aberration.py`).
- `docs/technical/scanline-tear-renderer.md` - `apply_scanline_tear(frame, t, clips, song_hash)` for horizontal band shifts from `SCANLINE_TEAR` clips (`pipeline/scanline_tear.py`).
- `docs/technical/color-invert-renderer.md` - `invert_mix(t, clips)` for a [0, 1] lerp weight toward a colour-inverted frame from `COLOR_INVERT` timeline clips (`pipeline/color_invert.py`).
- `docs/technical/zoom-punch-renderer.md` - `zoom_scale(t, clips)` for a whole-frame punch-in scale factor from `ZOOM_PUNCH` timeline clips (`pipeline/zoom_punch.py`).
- `docs/technical/effects-timeline-compositor.md` - Compositor integration of the effects timeline: fixed-order post-stack frame pass, user BEAM / LOGO_GLITCH merging, `auto_reactivity_master` scaling of the auto envelopes.
- `docs/technical/pipeline-orchestrator.md` - Root `orchestrator.py`: stage order, cache-key rules, progress mapping, Gradio wiring.
- `docs/technical/orchestrator-effects-timeline-wiring.md` - Preview/full render: load `effects_timeline.json` into `CompositorConfig`, `OrchestratorInputs` flags, merged auto-reactivity master.
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
- `docs/technical/rim-beams-bloom-handover.md` - Ongoing investigation handover: beam glow cutoff, rim attachment, tried fixes, and suggested next steps for a future session.
- `docs/technical/background-stills.md` - SDXL FP16 keyframe generator, section-aware prompts, cached PNGs under `cache/<hash>/background/`, and smoothstep-crossfade `background_frame(t)` API.
- `docs/technical/background-modes.md` - `BackgroundSource` factory, Ken Burns + AnimateDiff caches, AnimateDiff seeded from SDXL stills (init-latent injection, sequential SDXL→AnimateDiff lifecycle, cross-segment prompt travel), Gradio/orchestrator mode wiring (cross-ref SDXL stills doc).
- `docs/technical/frame-compositor.md` - Per-frame compositor pipeline: bg + reactive + typography + logo, compositor-scope shader uniforms (`bass_hit`, `transient_lo/mid/hi`, `drop_hold`) built once per render, bounded queue producer/consumer, `bgr24` ffmpeg stdin.
- `docs/technical/audio-vignette.md` - Audio-pulsing dark-edge vignette post-pass (between shader composite and typography) — adds baseline SDXL/shader contrast and a subtle bass + drop_hold breath at the corners.
- `docs/technical/thumbnail-generator.md` - Chorus/RMS frame pick, `render_single_frame`, Skia title overlay, and `thumbnail.png` beside `output.mp4`.
- `docs/technical/metadata-generator.md` - `metadata.txt` (title, description, chapters, tags) from analysis, lyrics, and preset; `write_run_metadata` in `orchestrator.py`.
- `docs/technical/preview-and-render.md` - Preview 10 s (loudest RMS window), full render, progress/ETA mapping, and ffprobe A/V sync validation.
- `docs/technical/title-and-beat-pulse.md` - Burned-in `Artist — Title` overlay (9-point grid, 3 sizes) and the logo-branding reactive stack: attack-dominant bass pulse, snare neon, snare squeeze, and RMS-jump impact glitch.
- `docs/technical/gpu-memory.md` - Shared `release_cuda_memory` / `move_to_cpu` helpers used by demucs, WhisperX, SDXL, and AnimateDiff to hand off VRAM cleanly between pipeline stages.
