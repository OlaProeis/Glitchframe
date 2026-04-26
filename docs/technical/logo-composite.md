# Logo composite layer

Feature doc for optional branding overlay: load a PNG, place it on the frame, and alpha-blend with user-controlled opacity. Intended for reuse in the full compositor (same transform every frame).

## Module

- **`pipeline/logo_composite.py`** — Pillow load/convert to RGBA, optional downscale to fit the frame (LANCZOS), corner/center placement, NumPy Porter–Duff *over* blend.

## Public API (summary)

| Function | Role |
|----------|------|
| `normalize_logo_position(label)` | Maps UI strings (e.g. `Top-left`) to canonical `top-left`, etc. |
| `load_logo_rgba(path)` | Returns `uint8` `(H, W, 4)`. |
| `prepare_logo_rgba(logo_rgba, frame_h, frame_w, *, max_size_pct=None)` | Shrinks logo only when larger than the frame, and — when `max_size_pct` is set — caps the longest logo edge at `max_size_pct / 100 × min(frame_h, frame_w)`. Same array if no resize is needed. |
| `composite_logo_onto_frame(frame, logo_rgba, position, opacity_pct, *, max_size_pct=None, ...)` | Blends onto `uint8` `(H, W, 3)` or `(H, W, 4)`; `opacity_pct` is 0–100 and multiplies the logo’s per-pixel alpha. `max_size_pct` forwards to `prepare_logo_rgba`. Optional rim + classic neon (see below). |
| `composite_logo_from_path(frame, logo_path, position, opacity_pct, *, max_size_pct=None, ...)` | Loads from disk; no-op copy if path is empty. |
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

## Resolution-independent sizing (`logo_max_size_pct`)

`CompositorConfig.logo_max_size_pct` (default `30.0`) caps the logo's longest edge at that percent of the **shorter** frame edge before any beat-pulse scaling. Using the shorter frame edge as the reference keeps the slider behaving identically across 720p / 1080p / 4K: 30 % on 1080p ≈ 324 px on the longest logo edge; 30 % on 4K ≈ 648 px. Values ≤ 0 disable the cap and fall back to the legacy "fit inside frame" behaviour (`_fit_logo_hw`). The cap is applied once per render in `render_single_frame` and `render_full_video` when the prepared logo is first loaded, so subsequent per-frame pulse scaling / beam geometry operates on the capped logo directly.

## Gradio (`app.py`)

- **Branding** tab: logo file, position dropdown, opacity slider, and **Logo size** slider (5–60 %, default 30 %) that binds `OrchestratorInputs.logo_max_size_pct`; **Preview logo on test frame** composites onto a fixed 960×540 RGB gradient with the same size cap applied (no GPU).
- **Preview reactive frame** (Visual style): after the reactive GPU pass, applies the same logo path/position/opacity/size when a file is uploaded.

## Related docs

- `docs/technical/gradio-ui.md` — tab layout and control references.
