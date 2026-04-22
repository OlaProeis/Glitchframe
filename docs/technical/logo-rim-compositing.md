# Logo rim compositing (full-frame path)

End-to-end wiring from compositor time `t` through premultiplied rim and classic neon behind the logo. Implements the blend stack described in [`logo-composite.md`](logo-composite.md); rim field math remains in [`logo-rim-lights.md`](logo-rim-lights.md) and audio scalers in [`logo-rim-audio-modulation.md`](logo-rim-audio-modulation.md).

## Modules

- **`pipeline/logo_composite.py`**
  - **`LogoGlowMode`** — `AUTO`, `CLASSIC`, `RIM_ONLY`, `STACKED` (see enum docstring for stacking vs mutual exclusivity).
  - **`build_classic_neon_glow_patch`** — single-tint Gaussian alpha halo (legacy neon path).
  - **`build_rim_light_premult_patch`** — wrapper over `compute_logo_rim_light_patch` with the same padded premult contract as classic glow.
  - **`composite_logo_onto_frame`** — optional rim, then optional classic neon, then logo; kwargs `t_sec`, `rim_light_config`, `rim_audio_mod`, `logo_glow_mode`, optional `logo_rim_prep`. On RGB frames only for the glow paths. If `logo_rim_prep` is omitted, prep is recomputed each frame from the **pulse-scaled** logo (single render thread; higher CPU).

- **`pipeline/compositor.py`**
  - **`CompositorConfig`**: `logo_rim_enabled`, `logo_rim_light_config`, `logo_glow_mode`, plus task-27 fields (`logo_rim_audio_reactive`, sync toggles, `logo_rim_mod_strength`).
  - **`_effective_rim_light_config`** — once per render: `RimLightConfig` when `logo_rim_enabled` is true; fills `song_hash` from `analysis` when the override config omits it; base tint via `rim_base_rgb_from_preset(shadow, base)`.
  - **`_render_compositor_frame`** — passes absolute `t`, `resolved_rim_config`, and per-frame `rim_audio_mod` from `_create_rim_modulation_stepper` into `composite_logo_onto_frame`.

## Defaults

Rim is **off** until `logo_rim_enabled=True`. With rim disabled, behaviour matches the pre-rim logo path (classic neon only when snare glow and strength apply).

**Gradio / orchestrator:** `OrchestratorInputs` carries `logo_rim_mode` (`off` | `classic` | `rim`), travel speed, colour spread (degrees → `color_spread_rad`), inward mix, direction (`cw` / `ccw` → sign of `phase_hz`), audio-reactive rim + snare/bass link toggles, and modulation strength. `orchestrator.resolve_logo_rim_compositor_fields` maps these into `CompositorConfig` (see `app.py` Branding accordion and `_render_pipeline`).

## Tests

- **`tests/test_logo_composite_rim.py`** — regression with rim off, pixel delta with rim on, `CLASSIC` / `RIM_ONLY`, cached prep, `_effective_rim_light_config`.
