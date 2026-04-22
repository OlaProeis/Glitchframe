# Logo composite layer

Feature doc for optional branding overlay: load a PNG, place it on the frame, and alpha-blend with user-controlled opacity. Intended for reuse in the full compositor (same transform every frame).

## Module

- **`pipeline/logo_composite.py`** — Pillow load/convert to RGBA, optional downscale to fit the frame (LANCZOS), corner/center placement, NumPy Porter–Duff *over* blend.

## Public API (summary)

| Function | Role |
|----------|------|
| `normalize_logo_position(label)` | Maps UI strings (e.g. `Top-left`) to canonical `top-left`, etc. |
| `load_logo_rgba(path)` | Returns `uint8` `(H, W, 4)`. |
| `prepare_logo_rgba(logo_rgba, frame_h, frame_w)` | Shrinks logo only when larger than the frame; same array if no resize. |
| `composite_logo_onto_frame(frame, logo_rgba, position, opacity_pct)` | Blends onto `uint8` `(H, W, 3)` or `(H, W, 4)`; `opacity_pct` is 0–100 and multiplies the logo’s per-pixel alpha. Optional rim + classic neon (see below). |
| `composite_logo_from_path(frame, logo_path, position, opacity_pct)` | Loads from disk; no-op copy if path is empty. |
| `LogoGlowMode` | `AUTO` / `CLASSIC` / `RIM_ONLY` / `STACKED` — how traveling-wave rim and classic Gaussian neon stack; see enum docstring in `logo_composite.py`. |
| `build_classic_neon_glow_patch(...)` | Classic premult RGBA halo (single-colour alpha blur). |
| `build_rim_light_premult_patch(prep, t_sec=..., config=..., audio_mod=...)` | Thin wrapper over `compute_logo_rim_light_patch` for the same blend contract. |

## Rim vs classic neon

- **Order:** optional **rim** premult patch first, then optional **classic** neon, then the logo (RGB frames only for both glow paths).
- **Defaults:** with `rim_light_config=None`, behaviour matches the pre-rim pipeline (classic neon only when `glow_amount > 0`).
- **Prep:** if `logo_rim_prep` is omitted, it is computed each frame from the **beat-pulse-scaled** logo (correct motion; higher CPU). Pass a cached `LogoRimPrep` only when it matches the scaled RGBA exactly.
- **Compositor:** `CompositorConfig.logo_rim_enabled`, `logo_rim_light_config`, `logo_glow_mode`, plus task-27 audio flags; see [`logo-rim-compositing.md`](logo-rim-compositing.md) and [`logo-rim-audio-modulation.md`](logo-rim-audio-modulation.md).

## Placement

Pixel origin of the logo’s top-left corner: **top-left**, **top-right**, **bottom-left**, **bottom-right**, **center**. Partially out-of-bounds regions are clipped.

## Gradio (`app.py`)

- **Branding** tab: logo file, position dropdown, opacity slider; **Preview logo on test frame** composites onto a fixed 960×540 RGB gradient (no GPU).
- **Preview reactive frame** (Visual style): after the reactive GPU pass, applies the same logo path/position/opacity when a file is uploaded.

## Related docs

- `docs/technical/gradio-ui.md` — tab layout and control references.
