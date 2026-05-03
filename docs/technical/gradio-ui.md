# Gradio UI (single-page skeleton)

Feature doc for the browser UI entrypoint: tabbed layout, inputs, queued actions, and run log. Pipeline logic is wired in later tasks.

## Entry

- **`app.py`** — `build_ui()` defines `gr.Blocks`; `main()` calls `ensure_runtime_dirs()` from `config.py`, then `queue()` and `launch()`.
- Run locally: `python app.py` (requires Gradio 4.x).

## Layout

- **`gr.Tabs`** include **Audio**, **Metadata**, **Branding**, **Lyrics**, **Lyrics timeline**, **Effects timeline**, **Visual style**, **Output**, **Background keyframes**, **Actions** (HTML/JS editors: lyrics timeline, effects timeline, background keyframes). Below the tabs: **Run log** and **Clear log**.
- **Audio** tab: **`gr.File`** upload triggers **`pipeline/audio_ingest.ingest_audio_file`** — writes `cache/<sha256>/original.wav` (native SR, channels preserved) and `analysis_mono.wav` (44.1 kHz mono). **`gr.Audio`** (`type="filepath"`, interactive) plays the mono file and shows the waveform; **`gr.State`** holds the song hash for later pipeline steps.

## Inputs (layout-level)

- File uploads: audio (common formats), logo PNG, optional static background image.
- Text: metadata fields, lyrics, scene prompt / typography style / comma-separated palette (Visual style), filename prefix.
- **Visual style** tab: **Reactive shader** dropdown (includes **No reactive shader**). Changing the shader pre-fills **Scene prompt**, **Typography style**, and **Color palette** from `pipeline/visual_style.py`; you can edit all of them. Preview / render pass an inline preset dict into `OrchestratorInputs` and set `preset_id` to `style-<shader_stem>` for background caches. Optional `presets/*.yaml` still load via `config.load_preset_registry()` for advanced setups.
- Sliders: logo opacity and reactive intensity (0–100%).
- Choices: logo position, background mode, resolution, FPS.
- **Branding:** logo PNG upload, position dropdown, opacity slider; beat pulse, snare neon, squeeze, and impact glitch sliders; **Traveling rim light (optional)** accordion — rim mode (`off` / classic neon only / traveling rim + neon), travel speed, colour spread, inward bleed %, CW/CCW direction, audio-reactive rim with snare/bass link toggles and modulation strength (wired to `OrchestratorInputs` → `CompositorConfig`, branding-only — no song cache impact). **Preview logo on test frame** runs `pipeline.logo_composite` on a built-in RGB gradient (rim settings apply on full **Preview 10 s** / **Render**, not this static preview). **Preview reactive frame** also applies the same logo overlay when a file is uploaded (placement and opacity match the Branding tab).

## Actions and progress

- Buttons: **Analyze**, **Preview 10 s**, **Render full video**. Each dispatches through `orchestrator.orchestrate_analysis` / `orchestrate_preview_10s` / `orchestrate_full_render` and wraps `gr.Progress` with `_EtaProgress`, which appends elapsed time and an ETA to the stage message while a handler runs.
- **`demo.queue()`** is enabled so long-running runs stream progress updates even under concurrent requests.
- **Preview 10 s** renders the loudest ~10 seconds; **Render full video** writes `output.mp4`, `thumbnail.png`, `metadata.txt` under `outputs/<run_id>/` and runs an ffprobe A/V sync check whose PASS/FAIL message is appended to the run log.

## Background keyframes tab

- **`Background keyframes`** — `app.py` wires **Load / Save timeline**, **Generate SDXL stills**, and proxy controls for **Regenerate / Replace / Crop** that the inline timeline (`pipeline/keyframes_editor.py`) drives via `elem_id` and `js=` preprocessors. Detailed behaviour, disk layout, and limits: [`background-keyframes-editor.md`](background-keyframes-editor.md).

## Related code

- **`config.py`** — `CACHE_DIR`, `OUTPUTS_DIR`, `load_preset_registry()` / optional YAML presets, `ensure_runtime_dirs()`.
