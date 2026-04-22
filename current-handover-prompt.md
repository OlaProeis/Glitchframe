# Session Handover

## Environment

- **Project:** MusicVids
- **Tech Stack:** Python 3.11, CUDA/PyTorch, Gradio, librosa/demucs/whisperx, diffusers (SDXL / AnimateDiff), moderngl, skia-python, ffmpeg NVENC
- **Context file:** Always read `ai-context.md` first — it contains project rules, architecture, and model selection.
- **Python interpreter:** `.\.venv\Scripts\python.exe` (the Windows `py` launcher does NOT point at the project venv — use the full path).
- **Run tests:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` (pytest is not wired; Task Master may mention pytest — prefer unittest).
- **Branch:** main

## Core Handover Rules

- **NO HISTORY:** Do not include past task details unless they directly impact this specific task.
- **SCOPE:** Focus ONLY on the current task detailed below.

## Current Task

**Task Master #30 — Logo rim lights: tests, `docs/technical`, and `docs/index.md`**

- **ID:** 30
- **Status:** pending
- **Priority:** medium
- **Complexity:** 4 (recommended subtasks: 1)
- **Dependencies:** Task 29 (done)

### Description (from Task Master)

Complete unit/integration coverage for tasks 24–28, add `docs/technical/logo-rim-lights.md` describing behavior and controls, and add one line to `docs/index.md`.

### Details (from Task Master)

Document: mask fallback, multi-color, beat modulation, determinism, and how to A/B against classic glow. Keep prose consistent with other technical docs. Run `uv run pytest` (or project standard). Update `current-handover-prompt` only if your project convention requires; otherwise skip.

### Test / verification (target)

CI green; doc links valid; at least 4 new tests covering masks, phase, determinism, and modulation edge cases.

## Key Files

- `docs/technical/logo-rim-lights.md` — extend or align with existing rim patch doc (cross-ref `logo-rim-lights-prep.md`, `logo-rim-lights-color.md`, `logo-rim-audio-modulation.md`, `logo-rim-compositing.md`, `logo-rim-branding-ui.md`).
- `docs/index.md` — ensure index lines match any new or renamed docs.
- `tests/test_logo_rim_lights.py`, `tests/test_logo_composite_rim.py`, `tests/test_orchestrator_logo_rim_inputs.py` — existing coverage; add tests for gaps (masks, phase, determinism, modulation edges).
- `pipeline/logo_rim_lights.py` — `RimLightConfig`, `compute_logo_rim_light_patch`, prep/mask fallback, audio modulation.

## Context

- `docs/technical/logo-rim-lights.md` and related rim docs already exist; task 30 may mean consolidating “behavior and controls” narrative, closing testing gaps for tasks 24–28, and keeping `docs/index.md` accurate—reconcile with the repo before duplicating large sections.

## Verification (this session, after your edits)

- `.\.venv\Scripts\python.exe -m compileall g:\DEV\MusicVids\pipeline g:\DEV\MusicVids\orchestrator.py g:\DEV\MusicVids\app.py -q`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

## Task Master

- **Current: #30.** Use `get_task` / `next_task` for full JSON and subtask expansion.
