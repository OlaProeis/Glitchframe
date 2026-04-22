# Logo rim lights — preparation (line mask & centroid)

Feature module for a future **traveling / multi-color** logo rim effect. Task 24 in Taskmaster: derive geometry and stroke energy from a single RGBA logo patch before any per-frame emissive pass.

## Public API

- **Module:** `pipeline/logo_rim_lights.py`
- **`LogoRimPrep`** — `line_mask` (float32, `[0,1]`), `alpha_f`, `centroid_xy` `(cx, cy)` in patch pixels, `line_confidence` (`[0,1]`), `use_line_features` (bool).
- **`compute_logo_rim_prep(logo_rgba, *, min_line_confidence=0.06, luma_bright_q=0.72, edge_vs_luma=0.45, morph_dilate=0)`**

`logo_rgba` must be **uint8** `(H, W, 4)`.

## Behaviour

- **Luma** = `max(R, G, B)` to favour white line art on dark or transparent background.
- **Edge** = Scharr magnitude on `luma * alpha` (scipy `ndimage.convolve`, `mode="nearest"`).
- **Line mask** = normalized blend of weighted luma and edge; adaptive high-luma spread via quantiles on `w_luma` where alpha is present.
- **Centroid** = first moment of the alpha channel (center of the frame when there is no opaque mass).
- **Line confidence** combines spatial support of the mask, mean weighted luma, and a **fill penalty** when most of the glyph is very bright (solid blob, not stroke art).
- **Fallback:** if confidence is below `min_line_confidence` or the bright-fill ratio is too high, `use_line_features` is **False** and `line_mask` is **all zeros** so later stages use **halo / silhouette** only (via `alpha_f`), not stroke-hugging.

Optional **`morph_dilate`** (0–3): light binary dilation on high-luma support to stabilise very thin strokes.

## Tests

`tests/test_logo_rim_lights.py` — synthetic disc + stroke, solid white square, empty alpha, COM centroid, invalid dtype/shape, determinism.

## Upcoming (not in this module)

Travelling phase, multi-color stack, and beat coupling are tasks **25+** and will call into this prep from `logo_composite` / compositor.
