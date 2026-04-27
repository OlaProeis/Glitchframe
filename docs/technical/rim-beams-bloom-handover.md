# Handover: Rim beam glow / cutoff / layering (ongoing)

This document captures the **user-reported visual issues**, **code areas involved**, **changes attempted in development**, and **open problems** so a future session can continue without re-deriving context. It is not a finished spec; the issue is **not fully resolved** as of the last iteration.

## Feature summary

Rim-light **beams** are implemented in `pipeline/logo_rim_beams.py` (`compute_beam_patch`, `ScheduledBeam`, `BeamConfig`, sustain shaping for long timeline `BEAM` clips). The compositor blends a premultiplied RGBA patch onto the frame via `pipeline/compositor.py` (`_draw_beam_patch_onto_frame`, `_build_beam_render_context`). Official product doc: `docs/technical/logo-rim-beams.md`.

## User-visible problems (still reported)

1. **Hard “cutoff” / shelf**  
   A straight edge (often described as **horizontal** in screen space) where the glow **stops abruptly** instead of fading smoothly. Most visible when the beam / bloom is **large** (long sustain, high intensity, wide afterglow). Can look like a **bounding box** or **viewport** clip.

2. **Start point at the logo**  
   The beam **root** can read as **rectangular** or **flat** against the circular logo rather than a soft wrap along the rim; squared-off or linear boundaries where the ray meets the mark.

3. **Bleed “through” or around the logo**  
   Faint glow on the **far side** of the logo (opposite the main ray), or haze that does not feel like light coming **only** from the intended direction.

4. **Beam vs. glow balance (long clips)**  
   On long timeline beams, the **diffuse** afterglow seemed to **dominate** while the **bright core** did not get proportionally stronger; user wanted **core + energy** to ramp with a **modest** halo increase and **longer** hang / tail.

5. **Layering**  
   Beams are intended to read as coming **from behind** the logo. When the glow was drawn **on top** of the composited logo, the mark looked **washed out** or covered by light.

## Root cause hypotheses (partially verified)

- **Ray half-plane model**  
  The beam is built in 2D ray coordinates (`u` along the ray from the rim, `v` across). Luminous terms use `u >= 0` in places; the isocontour `u = 0` is a **straight line** in the image, which can look like a **cut** for wide `v` (halo).

- **Sub-rectangular patch + Gaussian blur**  
  The effect is rendered into a **tight axis-aligned** patch, then `scipy.ndimage.gaussian_filter` is applied with `mode="constant"`, `cval=0`. If padding is **insufficient** for the effective blur sigma, **non-negligible** alpha can remain on the **inner** edge of the buffer; **outside the patch** nothing is composited, producing a **straight** “shelf” (often a horizontal strip at the bottom of the effect region in screen space, i.e. the bottom edge of the AABB).

- **Shared state across beams (ruled out for one bug class)**  
  Per-beam time uses `age = t - t_start`; there is no single global timer. A separate issue was **one global blur** on the **combined** patch, which was fixed by **per-beam** blur.

## What was tried (chronological / thematic)

| Area | Change | Files / notes |
|------|--------|----------------|
| Halo back-bleed along negative `u` | Shifted halo rise along `u` by a few `σ` so bloom extended slightly “behind” the rim in math | `logo_rim_beams.py` `_draw_beam_into` — **reverted** later: read as light **through** the mark to the far side. |
| Per-beam blur | Blur each beam layer with its own `br`, then add; avoid one beam’s `br` smearing all | `compute_beam_patch` |
| Patch padding for sigma | `blur_pad` from `~4σ` → later **`~6.6σ + 1`** (effective `σ = blur_sigma_px × max_br`) | `_gaussian_blur_footprint_pad_px` |
| Full-frame buffer | When `eff_blur_sigma` is high, `blur_pad` is large, or tight bbox area exceeds ~38% of frame, raster the **entire** frame `0,0—W,H` so taper only at **video** edges | `compute_beam_patch` after `_patch_bounds` |
| `gaussian_filter` edge mode | `mode="constant"`, `cval=0.0` to avoid edge reflection artifacts | per-beam loop in `compute_beam_patch` |
| Compositor order | Draw **beams before** `composite_logo_onto_frame` so the logo **occludes** the ray | `compositor.py` `_render_compositor_frame` |
| Long-beam **core** vs halo | `_sustain_energy_mul`: extra scale on `env` when glow phase `g` rises; capped halo-only swell; slightly longer end **fade** in `_sustain_knots` | `logo_rim_beams.py` |
| Rounded rim attachment | 2D Gaussian “endcap” in `(u,v)` plus small core nudge; `u` shifted slightly inward to tuck under the logo | `_draw_beam_into` |
| Unit tests | Various tests in `tests/test_logo_rim_beams.py` (sustain, premult, patch bounds, blur pad) | Still passing after last code state |

## Current status

Users **still** report the **same** cutoff / shelf and rim attachment issues in **real renders** (including large / strong beams), despite the full-frame path and increased padding. Possible reasons to investigate next:

- **Tight** patch path still used when heuristics do not trigger, but bloom is still large enough to shelf.
- **Another** clip / layer (reactive shader, voidcat ASCII, title, or **logo rim** not beam) contributing straight edges; worth **isolating** a frame (beam-only schedule, `rim_beams` off) in debug.
- **Gradio preview** vs **encode** path difference (different size, `beam_ctx` present, etc.).
- **Blend** in `_blend_premult_rgba_patch` or **alpha** accumulation producing a visible step; inspect **per-channel** and **row sums** on a **bad** frame export.
- **Endcap** + half-plane may still **combine** into a **visible** kink; may need a **fully different** root model (e.g. **distance to segment** with **falloff** instead of half-plane, or **explicit** disc annulus at rim only for the root lobe).
- **Performance**: full-frame beam path is costly; if expanded, may need a **ceiling** or downsampled bloom for very large `σ`.

## Key entry points (for the next implementer)

- `pipeline/logo_rim_beams.py` — `compute_beam_patch`, `_draw_beam_into`, `_patch_bounds`, `_gaussian_blur_footprint_pad_px`, `_sustain_*`, `_active_beam_states`
- `pipeline/compositor.py` — `_draw_beam_patch_onto_frame`, order of `composite_logo_onto_frame` vs beam
- `pipeline/logo_composite.py` — `_blend_premult_rgba_patch` (or equivalent) used for the patch
- `tests/test_logo_rim_beams.py` — regression tests; extend with **synthetic** bad cases if a fix is found (e.g. assert no row in the middle of the frame has a **step** in alpha above threshold)

## Suggested next steps

1. Export a **single** 1080p (or current preset) **numpy** / PNG frame with **maximal** problem + save **tight** `patch` + **coordinates** for **numeric** row/column **profiles** to locate **which** buffer edge (or if it is **not** the beam layer).
2. **Binary search** heuristics: **always** use full frame for any active beam, or for `sustain_shaping` only, to see if the shelf **disappears** (confirms AABB).
3. Prototype **segment-distance** or an **analytic** 2D falloff for the **halo** only, replacing the `u` half-plane for a narrow band at the **rim** only.
4. Visual **contrast** pass: **temporarily** render beam **magenta** and everything else **green** in a test flag to see **all** straight edges in one view.

## Related user-facing design doc

- `docs/technical/logo-rim-beams.md` — may need a short “known limitations / tuning” section once this is closed.
