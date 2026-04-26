# Session Handover

## Environment

- **Project:** MusicVids
- **Tech Stack:** Python 3.11, CUDA/PyTorch, Gradio, librosa/demucs/whisperx, diffusers (SDXL / AnimateDiff), moderngl, skia-python, ffmpeg NVENC
- **Context file:** Always read `ai-context.md` first — it contains project rules, architecture, and model selection.
- **Python interpreter:** `.\.venv\Scripts\python.exe` (the Windows `py` launcher does NOT point at the project venv — use the full path).
- **Run tests:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` (pytest is not wired; Task Master may mention `pytest` — prefer `unittest` unless the project wires pytest).
- **Branch:** main

## Core Handover Rules

- **NO HISTORY:** Do not include past task details unless they directly impact this specific task.
- **SCOPE:** Focus ONLY on the task detailed below.

## Current Task

### Task Master #55 — Implement scanline_tear renderer

- **ID:** 55
- **Status:** pending (set to **in-progress** when you start)
- **Priority:** medium
- **Complexity:** 7 (recommendedSubtasks: 2)
- **Dependencies:** #43 (per Task Master graph)

**Description.** Create `pipeline/scanline_tear.py` for horizontal row-offset distortion driven by active `SCANLINE_TEAR` clips.

**Implementation notes (from Task Master).**

- API shape: `apply_scanline_tear(frame: np.ndarray, t: float, clips: list[EffectClip], song_hash: str) -> np.ndarray`
- Find active `SCANLINE_TEAR` clips at `t`; early-return input frame if none (no in-place mutation — compositor frame-queue invariant).
- Per-clip settings: `intensity` (0..1, max horizontal offset as fraction of frame width), `band_count` (int, number of torn horizontal bands; default in 3–6 range when unset), `band_height_px` (optional), `wrap_mode` (`wrap` | `clamp` | `black`, default `wrap`).
- Per-frame seed: `hash(song_hash + clip.id + round(t*1000))` — pick band y-positions and per-band `dx` from a deterministic RNG; apply horizontal shift per band (e.g. `np.roll` along axis=1 per band).
- Overlapping clips: **later clips stack on earlier** (accumulate).
- Bound shifts to **± intensity × W × 0.25** so tears never exceed a quarter-frame displacement.
- Reference: deterministic seeding like `pipeline/logo_rim_beams.py`; ndarray contract like `pipeline/chromatic_aberration.py`.

**Test strategy (from Task Master).** Frame bit-identical to input when no clip active. With one clip and `intensity > 0`: exactly `band_count` bands shifted horizontally; R/G/B move together (no channel split); dtype/shape preserved. Deterministic for fixed `song_hash` + `clip.id` + `t`. `intensity=0` returns input unchanged.

## Key Files

- `pipeline/scanline_tear.py` — **create** (core API and NumPy implementation)
- `pipeline/compositor.py` — wire into `_apply_frame_effects` after chromatic aberration, before colour invert; remove the `SCANLINE_TEAR` placeholder warning once the renderer exists
- `pipeline/effects_timeline.py` — `EffectKind.SCANLINE_TEAR`, `EFFECT_SETTINGS_KEYS` (already defined)
- `pipeline/chromatic_aberration.py` — patterns for active-clip sampling, cap/bounds, non-mutation, tests
- `tests/test_scanline_tear.py` — **add** per test strategy
- `docs/technical/scanline-tear-renderer.md` — **create** (match `chromatic-aberration-renderer.md` / `screen-shake-renderer.md` style)
- `docs/technical/effects-timeline-compositor.md`, `docs/technical/effects-timeline-renderers.md`, `docs/technical/effects-timeline-test-suite.md`, `docs/technical/effects-timeline-editor.md` — drop placeholder-only wording for this kind once implemented
- `docs/index.md` — one bullet for the new renderer doc

## Context

- Post-stack order: zoom → shake → chromatic → **scanline** → invert — keep consistent with `docs/technical/effects-timeline-compositor.md`.
- `_FrameEffectsContext` already carries `scanline_clips`; compositor currently logs a warning when those clips exist.

## Verification

- `.\.venv\Scripts\python.exe -m compileall g:\DEV\MusicVids -q`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

## Task Master

- **Current:** #55. Use `get_task --id=55` / `next_task` MCP tools for the live JSON.

## Checklist (this handover)

- [ ] Task 55: set **in-progress** / **done** in Task Master as appropriate
- [ ] Renderer + compositor wiring + tests + docs as per spec
- [ ] Run compileall + unittest
