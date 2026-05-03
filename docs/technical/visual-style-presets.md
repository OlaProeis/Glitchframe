# Visual style (shaders + example prompts)

The **Visual style** Gradio tab is **shader-first**: you pick **Reactive shader** (including **No reactive shader**), then edit the **scene prompt**, **typography style**, and **palette**. Changing the shader pre-fills the other fields with curated defaults so it behaves like former presets without a YAML file per look.

See `pipeline/visual_style.py` for bundles (example prompt, typography, `#RRGGBB` palette, AnimateDiff **motion flavor** lines).

## Active reactive shaders (`BUILTIN_SHADERS`)

| Stem | Notes |
|---|---|
| **none** | No GPU overlay — background + vignette + type + branding only (`pipeline/compositor` passthrough). |
| **void_ascii_bg** | GPU night void + CPU **ASCII** field + drop cat overlay (`pipeline/voidcat_ascii`); orchestrator enables this when shader is `void_ascii_bg`. |
| **spectral_milkdrop** | Feedback-style spectrum visual; peak tint sliders in UI for bright backgrounds (`docs/technical/reactive-composite-and-gradio-preview.md`). |
| **tunnel_flight** | First-person lattice tunnel. |
| **synth_grid** | Perspective neon grid / retrowave skyline look. |

## Cache keys (`style-<stem>`)

`OrchestratorInputs.preset_id` is set to `style-{shader_stem}` (e.g. `style-synth_grid`) for SDXL / AnimateDiff / RIFE manifests. AnimateDiff adds motion text from `motion_flavor_for_style_preset`; legacy YAML preset ids (`neon-synthwave`, …) remain in `MOTION_FLAVORS` for backward compatibility where old caches mention them.

## Optional YAML presets (advanced)

`config.load_preset_registry()` still loads `presets/*.yaml` if present (`get_preset`, `_resolve_preset` for external callers). Validation requires `shader` ∈ `BUILTIN_SHADERS`. The shipping repo may ship with an **empty** `presets/` directory.

## Related files

| File | Role |
|------|------|
| `pipeline/visual_style.py` | Defaults per shader stem + motion flavor for `style-*` ids |
| `pipeline/builtin_shaders.py` | `BUILTIN_SHADERS` allowlist |
| `config.py` | Optional `presets/*.yaml` registry |
| `app.py` | Visual style tab wiring |
| `assets/shaders/*.frag` | GLSL stems (excluding `none`, which skips GL) |
