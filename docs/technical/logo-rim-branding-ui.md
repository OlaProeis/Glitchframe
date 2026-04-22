# Logo rim: Gradio branding controls and orchestrator mapping

User-facing **traveling rim** settings live in the **Branding** tab (`app.py`) inside the **Traveling rim light (optional)** accordion. They are **cosmetic only**: they do not affect the per-song cache key under `cache/<song_hash>/`.

## Data flow

1. Gradio collects sliders, dropdowns, checkboxes, and radios.
2. `_build_render_inputs` assembles :class:`orchestrator.OrchestratorInputs` (clamps speeds, degrees, percentages).
3. `_render_pipeline` merges :func:`orchestrator.resolve_logo_rim_compositor_fields` into :class:`pipeline.compositor.CompositorConfig`.
4. The compositor resolves tint via `_effective_rim_light_config` (preset base/shadow colours, `song_hash` on the rim config when needed).

## `OrchestratorInputs` fields

| Field | Role |
|--------|------|
| `logo_rim_mode` | `off` \| `classic` \| `rim` (aliases like `new` / `traveling` accepted in resolver). Product default `rim` |
| `logo_rim_travel_speed` | Wave `phase_hz` magnitude in Hz (0–2); sign comes from direction. Dataclass default `0.5` |
| `logo_rim_color_spread_deg` | Hue separation between layers (0–180°); &lt; 0.5° → single-layer tint |
| `logo_rim_inward_mix` | Inward bleed (0–1), maps to `RimLightConfig.inward_mix` |
| `logo_rim_direction` | `cw` or `ccw` → sign of `phase_hz` |
| `logo_rim_audio_reactive` | Enables compositor rim audio stepper (default **on**) |
| `logo_rim_sync_snare` / `logo_rim_sync_bass` | Snare/bass link toggles on the compositor |
| `logo_rim_mod_strength` | Scales audio modulation (0–2) |
| `logo_rim_brightness` | Emissive gain multiplier. Drives `RimLightConfig.intensity` **and** lifts `halo_boost = max(1.0, brightness)` so the halo reads on screen. Gradio sends **%** (0–500, default **300**); orchestrator stores `pct/100` (default **3.0**) |
| `logo_rim_halo_spread_px` | Outer halo falloff distance (patch px, clamped 4–64). Maps to `RimLightConfig.halo_spread_px`. Default `22.0` |
| `logo_rim_wave_shape` | Keyword resolved to `(waves, wave_sharpness)` — see table below. Default `"comet"` |

### Wave-shape preset keywords

`logo_rim_wave_shape` is deliberately a small enumerated keyword so the UI dropdown stays authoritative. Unknown keywords fall back to `"comet"`.

| Keyword | `(waves, wave_sharpness)` | Visual |
|---------|---------------------------|--------|
| `comet` (default) | `(1, 4.0)` | Single tight bright lobe orbiting the logo — most visible motion cue |
| `twin`  | `(2, 4.0)` | Two tight comets on opposite sides |
| `lobes` | `(2, 2.0)` | Two softer lobes |
| `ring`  | `(3, 1.5)` | Near-uniform soft ring (legacy engine default; little perceptible motion) |

### UI → resolver mapping for the new controls

| Gradio slider/dropdown | Range | Orchestrator field | Compositor effect |
|------------------------|-------|--------------------|-------------------|
| **Rim brightness**     | 0–500 % (default 300 %) | `logo_rim_brightness` | `RimLightConfig.intensity`, and `halo_boost = max(1.0, brightness)` |
| **Rim halo size**      | 4–64 px (default 22)    | `logo_rim_halo_spread_px` | `RimLightConfig.halo_spread_px` |
| **Rim wave shape**     | preset keyword (default `comet`) | `logo_rim_wave_shape` | `RimLightConfig.waves`, `wave_sharpness` |

## Mode mapping

- **off** — `logo_rim_enabled=False`, `LogoGlowMode.AUTO` (no traveling rim; classic snare neon still follows existing Branding toggles).
- **classic** — `logo_rim_enabled=False`, `LogoGlowMode.CLASSIC` (explicit classic-neon-only stack in `composite_logo_onto_frame`).
- **rim** — `logo_rim_enabled=True`, `LogoGlowMode.AUTO`, non-`None` `RimLightConfig` from the fields above.

## Resolver

:func:`orchestrator.resolve_logo_rim_compositor_fields` returns a dict of kwargs suitable for `CompositorConfig(...)`. It is covered by unit tests and safe to call without analysis loaded.

## Static vs full render preview

**Preview logo on test frame** still calls `composite_logo_from_path` without rim timing or analysis; rim appearance is validated via **Preview 10 s** or **Render full video**.

## Related docs and code

- [`logo-rim-compositing.md`](logo-rim-compositing.md) — blend order, `LogoGlowMode`, compositor defaults.
- [`gradio-ui.md`](gradio-ui.md) — full UI layout and Branding summary.
- [`logo-rim-lights.md`](logo-rim-lights.md) — `RimLightConfig` and patch math.
- [`logo-rim-audio-modulation.md`](logo-rim-audio-modulation.md) — snare/bass modulation.
- Tests: `tests/test_orchestrator_logo_rim_inputs.py`.
