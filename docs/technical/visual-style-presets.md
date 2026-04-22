# Visual style presets (YAML + Gradio)

Feature: six bundled visual-style presets as validated YAML, registry loading in `config.py`, and Gradio auto-fill on the **Visual style** tab.

## What was implemented

- **Preset files:** `presets/*.yaml` — one file per stem: `neon-synthwave`, `minimal-mono`, `organic-liquid`, `glitch-vhs`, `cosmic-flow`, `lofi-warm`. Each file defines an SD-oriented `prompt`, reactive `shader` stem, `typo_style` string, and `colors` as a non-empty list of `#RRGGBB` values.
- **Registry API:** `load_preset_registry()` reads every `*.yaml`, validates schema and types, normalizes hex to uppercase, and rejects unknown `shader` values. `get_preset_ids()` returns sorted stems; `get_preset(name)` returns a shallow copy or raises `KeyError`.
- **Shader allowlist:** `pipeline/builtin_shaders.py` defines `BUILTIN_SHADERS` (`spectrum_bars`, `particles`, `geometry_pulse`, `nebula_drift`, `liquid_chrome`, `vhs_tracking`, `synth_grid`, `paper_grain`) so preset validation and the UI do not import `moderngl`. `pipeline/reactive_shader` imports the same tuple for the GL layer.
- **Gradio:** **Preset** dropdown drives a `.change` handler that fills **Custom prompt**, **Reactive shader**, **Typography style**, and **Color palette** (comma-separated hex). If no YAML is present, dropdown labels fall back to PRD ids but fields stay empty until files exist.

## What each preset does

Each preset ships a coherent bundle of prompt + shader + typography + palette. The shader is the per-preset reactive overlay that runs on GPU via `moderngl`; the palette is uploaded to it as `uniform vec3 u_palette[5]` (see `docs/technical/reactive-composite-and-gradio-preview.md`).

| Preset | Shader | Vibe | Palette role |
|---|---|---|---|
| **cosmic-flow** | `nebula_flow` | Deep-space nebula with bar-synced drift, pre-drop / drop dynamics, and bloom | 0–3 ramp dark→bright nebula density, 4 = onset bloom |
| **glitch-vhs** | `vhs_tracking` | Analog VHS: scanlines, RGB chromatic aberration, per-line tracking jitter, onset-triggered noise bands | 0 base dark, 1 noise band color, 2 beat flash, 3 highlight, 4 alt dark |
| **lofi-warm** | `paper_grain` | Cozy golden-hour wash: soft bokeh, animated film grain, warm vignette; deliberately gentle reactivity | 0–3 diagonal palette ramp, 0 bokeh tint, 4 onset bloom |
| **minimal-mono** | `geometry_pulse` | Swiss brutalist: concentric rings on black, minimal motion | 4-color mono palette drives ring tones |
| **neon-synthwave** | `synth_grid` | 80s retrowave: 1/z perspective grid with `fwidth` AA, sliced neon sun, horizon halo | 0 sky low, 1 sky high, 2 sun low, 3 sun high, 4 grid lines |
| **organic-liquid** | `liquid_chrome` | Iridescent ink macro: two-step domain-warped FBM remapped onto palette, onset radial ripples | 0→4 dark→bright ramp, 3 ripple color |

All six shaders are written from scratch in this repo using public-domain techniques (value noise, FBM, domain warp, scanlines, perspective grid, film grain). None are ports from Shadertoy or other licensed shader galleries.

### How palette ordering matters

Each shader encodes semantic meaning into its palette slots (see the table above). When authoring a new preset or reordering an existing one, keep the slot contract in mind — for example, `synth_grid` expects slot 4 to be the grid line color, so putting a bright magenta there vs. a dark navy produces wildly different looks.

### Interaction with background modes

The preset's `prompt` feeds:

- **SDXL stills** — one prompt per keyframe (~1 per 8 s), with structural hints (`scene N of M, t=X.Xs`) appended via `pipeline.background_stills._build_keyframe_prompt`.
- **AnimateDiff loops** — the preset prompt is prepended to a preset-specific **motion flavor** plus a pacing cue (`establishing shot` → `steady motion` → `slower fade-out motion`) via `pipeline.background_animatediff._build_motion_prompt`. Structural scene hints are deliberately **not** appended for AnimateDiff — see `docs/technical/background-modes.md`.
- **Static image + Ken Burns** — the prompt is ignored; the uploaded image drives all visuals, the palette + shader still run on top.

## Usage

- Invalid preset YAML fails fast at `load_preset_registry()` with `ValueError` (missing keys, bad hex, or shader not in `BUILTIN_SHADERS`).
- Smoke: `python -c "from config import load_preset_registry; print(sorted(load_preset_registry().keys()))"`.

## Related files

| File | Role |
|------|------|
| `presets/*.yaml` | Preset content |
| `config.py` | `PRESETS_DIR`, validation, `load_preset_registry`, `get_preset_ids`, `get_preset` |
| `pipeline/builtin_shaders.py` | `BUILTIN_SHADERS` tuple |
| `pipeline/reactive_shader.py` | Re-exports `BUILTIN_SHADERS`; loads GLSL by stem; uploads `u_palette[5]` + `u_palette_size` |
| `pipeline/background_animatediff.py` | `MOTION_FLAVORS`, `_build_motion_prompt`, `_pacing_cue` |
| `app.py` | Visual style tab, `preset_dd.change` → field updates, "What is a preset?" accordion |
| `assets/shaders/` | Fragment shaders matching allowed stems |
