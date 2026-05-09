# Pixel smear renderer

Full-screen horizontal **pixel-streak** ("datamosh") effect from `EffectKind.PIXEL_SMEAR` clips. Picks a pseudo-random subset of rows and stretches a single sampled column across each row in a random direction, producing the iconic glitch-art smear look — distinct from `SCANLINE_TEAR` (which slides whole horizontal bands by N pixels) and `CHROMATIC_ABERRATION` (which shifts R/B sub-pixel offsets without changing geometry).

## API

- **Module:** `pipeline/pixel_smear.py`
- **`apply_pixel_smear(frame, t, clips, song_hash) -> np.ndarray`** — applies every active clip in order. Returns the input array unchanged when no clip is active, `t` is non-finite, or every active clip's settings collapse to zero contribution.

## Active clips

Activity matches the rest of the post-stack renderers: `t_start <= t < t_start + duration_s`. Only `PIXEL_SMEAR` kinds are considered. Multiple active clips compose by running each clip's pass on the running output.

## Settings

Per `EffectClip.settings` (allowlist in `EFFECT_SETTINGS_KEYS[PIXEL_SMEAR]`, all clamped to `[0, 1]`):

| Key | Role |
|-----|------|
| `intensity` | Lerp weight from the original row pixel to the source-column streak (`0` = no streak, `1` = full smear). Default **`0.6`**. |
| `density` | Fraction of rows that get smeared (sampled without replacement). Default **`0.18`**. |
| `streak_length_frac` | Maximum streak extent as a fraction of frame width. Default **`0.45`**. |

For each smeared row the renderer randomly picks a source column and a direction (left- or right-running), then stretches that column's pixel across `1..streak_length_frac * w` pixels with the given intensity.

## Determinism

Per-frame seed is `sha256(song_hash + clip.id + round(t * 1000))[:8]`. Re-renders of the same cache stay bit-stable; different `song_hash` values produce different smears.

## Tests

`tests/test_pixel_smear.py` — inactive window passthrough, zero-{intensity,density,streak} short-circuits, active changes pixels, deterministic per seed, different song hash diverges, non-smear clips ignored, non-finite `t`, shape guard.
