# Gradio UI (single-page skeleton)

Feature doc for the browser UI entrypoint: tabbed layout, inputs, queued actions, and run log. Pipeline logic is wired in later tasks.

## Entry

- **`app.py`** — `build_ui()` defines `gr.Blocks`; `main()` calls `ensure_runtime_dirs()` from `config.py`, then `queue()` and `launch()`.
- Run locally: `python app.py` (requires Gradio 4.x).

## Layout

- **`gr.Tabs`** group PRD-aligned sections: **Audio**, **Metadata**, **Branding**, **Lyrics**, **Visual style**, **Output**, **Actions**.
- Below the tabs: **Run log** (`gr.Textbox`) and **Clear log**.
- **Audio** tab: **`gr.File`** upload triggers **`pipeline/audio_ingest.ingest_audio_file`** — writes `cache/<sha256>/original.wav` (native SR, channels preserved) and `analysis_mono.wav` (44.1 kHz mono). **`gr.Audio`** (`type="filepath"`, interactive) plays the mono file and shows the waveform; **`gr.State`** holds the song hash for later pipeline steps.

## Inputs (layout-level)

- File uploads: audio (common formats), logo PNG, optional static background image.
- Text: metadata fields, lyrics, custom prompt, filename prefix.
- **Preset** dropdown: `get_preset_ids()` from `config.py`; if `presets/` has no YAML yet, the six PRD default IDs are used.
- Sliders: logo opacity and reactive intensity (0–100%).
- Choices: logo position, background mode, resolution, FPS.
- **Branding:** logo PNG upload, position dropdown, opacity slider; **Preview logo on test frame** runs `pipeline.logo_composite` on a built-in RGB gradient. **Preview reactive frame** also applies the same logo overlay when a file is uploaded (placement and opacity match the Branding tab).

## Actions and progress

- Buttons: **Analyze**, **Preview 10 s**, **Render full video**. Each dispatches through `orchestrator.orchestrate_analysis` / `orchestrate_preview_10s` / `orchestrate_full_render` and wraps `gr.Progress` with `_EtaProgress`, which appends elapsed time and an ETA to the stage message while a handler runs.
- **`demo.queue()`** is enabled so long-running runs stream progress updates even under concurrent requests.
- **Preview 10 s** renders the loudest ~10 seconds; **Render full video** writes `output.mp4`, `thumbnail.png`, `metadata.txt` under `outputs/<run_id>/` and runs an ffprobe A/V sync check whose PASS/FAIL message is appended to the run log.

## Related code

- **`config.py`** — `CACHE_DIR`, `OUTPUTS_DIR`, `get_preset_ids()`, `ensure_runtime_dirs()`.
