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
| `composite_logo_onto_frame(frame, logo_rgba, position, opacity_pct)` | Blends onto `uint8` `(H, W, 3)` or `(H, W, 4)`; `opacity_pct` is 0–100 and multiplies the logo’s per-pixel alpha. |
| `composite_logo_from_path(frame, logo_path, position, opacity_pct)` | Loads from disk; no-op copy if path is empty. |

## Placement

Pixel origin of the logo’s top-left corner: **top-left**, **top-right**, **bottom-left**, **bottom-right**, **center**. Partially out-of-bounds regions are clipped.

## Gradio (`app.py`)

- **Branding** tab: logo file, position dropdown, opacity slider; **Preview logo on test frame** composites onto a fixed 960×540 RGB gradient (no GPU).
- **Preview reactive frame** (Visual style): after the reactive GPU pass, applies the same logo path/position/opacity when a file is uploaded.

## Related docs

- `docs/technical/gradio-ui.md` — tab layout and control references.
