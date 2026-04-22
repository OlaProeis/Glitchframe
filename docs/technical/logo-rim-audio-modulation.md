# Logo rim audio modulation (snare + bass)

Builds on [`logo-rim-lights.md`](logo-rim-lights.md) and [`title-and-beat-pulse.md`](title-and-beat-pulse.md). Snare- and bass-derived envelopes (same analysis tracks as the neon logo) drive per-frame **multipliers** on the rim field in **absolute** time `t` (seconds), aligned with the compositor’s frame-centred clock `start_sec + (i+0.5)/fps`.

## Module

- **`pipeline/logo_rim_lights.py`**
  - **`RimAudioModulation`** — `glow_strength_mul` (intensity), `phase_offset_rad` (travelling-wave phase nudge), `inward_strength_mul` (bass “breath” on inward bleed).
  - **`RimModulationState`**, **`RimAudioTuning`** — one render holds one state; tuning caps ranges and snare/bass smoothing times.
  - **`rim_modulation_instant()`** — map raw `[0,1]` snare/bass to modulation (tests and direct use).
  - **`advance_rim_audio_modulation()`** — one audio step: asymmetric **attack/release** on snare for phase (reduces 1-frame stutter); light slew on bass for inward. Glow uses the raw snare sample (the mel track is already short-decay shaped).
  - **`compute_logo_rim_light_patch(..., audio_mod=None)`** — applies the three scalers on top of `RimLightConfig`; `None` preserves the unmodulated look.

## Compositor

- **`pipeline/compositor.py`**
  - **`CompositorConfig`**: `logo_rim_audio_reactive` (default `False`), `logo_rim_sync_snare`, `logo_rim_sync_bass`, `logo_rim_mod_strength`. When `logo_rim_audio_reactive` is `False`, behaviour matches pre–modulation; no extra per-frame work.
  - **`_create_rim_modulation_stepper()`** — returns `t → RimAudioModulation` with shared state, or `None`. Snare uses the same prebuilt track as the logo neon / squeeze (`_snare_track_for_logo`). Bass uses `build_logo_bass_pulse_track` with `logo_pulse_sensitivity` when `logo_rim_sync_bass` is on.
  - **`_effective_rim_light_config()`** — once per render: builds `RimLightConfig` when `logo_rim_enabled` (preset tint via `rim_base_rgb_from_preset`, `song_hash` from `analysis` when the config omits it).
  - **`_render_compositor_frame(..., rim_mod_step=..., resolved_rim_config=...)`** — passes absolute `t`, resolved rim config, and `rim_audio_mod` from the stepper (when `logo_rim_audio_reactive`) into **`composite_logo_onto_frame`** so the rim patch is blended behind the classic neon per `logo_glow_mode`.

## Tests

- **`tests/test_logo_rim_lights.py`** — instant mapping bounds, snare 0 vs 1 patch difference, `PulseTrack` mock shapes (constant / spike / sine) with in-range `advance` output, and compositor stepper `None` when the reactive flag is off.
- **`tests/test_logo_composite_rim.py`** — defaults unchanged with rim off; rim on shifts pixels; `CLASSIC` / `RIM_ONLY` modes; cached prep vs inline prep.

## See also

- `pipeline/beat_pulse.py` — `PulseTrack`, `build_snare_glow_track`, `build_logo_bass_pulse_track`.
