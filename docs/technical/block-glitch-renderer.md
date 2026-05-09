# Block glitch renderer

Full-screen **macroblock displacement** ("JPEG corruption" / datamosh) effect from `EffectKind.BLOCK_GLITCH` clips. Splits the frame into a grid of square blocks and randomly displaces a fraction of those blocks by per-axis offsets — the visual reads as discrete rectangular chunks of the frame jumping around. Distinct from `SCANLINE_TEAR` (single-axis horizontal band slides) and `CHROMATIC_ABERRATION` (sub-pixel R/B drift).

## API

- **Module:** `pipeline/block_glitch.py`
- **`apply_block_glitch(frame, t, clips, song_hash) -> np.ndarray`** — applies every active clip in order. Returns the input array unchanged when no clip is active, `t` is non-finite, or every active clip's settings collapse to zero contribution.

## Active clips

Activity matches the rest of the post-stack renderers: `t_start <= t < t_start + duration_s`. Only `BLOCK_GLITCH` kinds are considered. Multiple active clips compose by running each clip's pass on the running output.

## Settings

Per `EffectClip.settings` (allowlist in `EFFECT_SETTINGS_KEYS[BLOCK_GLITCH]`):

| Key | Role |
|-----|------|
| `intensity` | Fraction of blocks displaced per frame. Clamped to `[0, 1]`. Default **`0.35`**. |
| `block_size_px` | Block edge length in pixels. Coerced to `int`, floored at 2, clamped at `min(h, w)`. Default **`32`**. |
| `displace_frac` | Per-axis displacement cap as a fraction of `block_size_px`. Default **`0.6`** (so blocks can shift up to ~60% of their size in either direction). |

For each picked block the renderer reads from the **un-displaced** source frame and writes onto a fresh copy, so neighbouring picks don't cascade-displace into each other. Destination patches that fall partly off the frame are clipped (no wrap, no fill).

## Determinism

Per-frame seed is `sha256(song_hash + clip.id + round(t * 1000))[:8]`. Re-renders of the same cache stay bit-stable; different `song_hash` values produce different displacement patterns.

## Tests

`tests/test_block_glitch.py` — inactive window passthrough, zero-{intensity,displace} short-circuits, active changes pixels, deterministic per seed, different song hash diverges, non-block clips ignored, non-finite `t`, oversized `block_size_px` clamped, shape guard.
