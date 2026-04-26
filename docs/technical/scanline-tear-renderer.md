# Scanline tear renderer

Horizontal band shifts from **`EffectKind.SCANLINE_TEAR`** clips on the effects timeline. Each active clip tears a configurable number of horizontal strips of the frame, shifting each strip by an integer pixel offset along **x** (all **RGB** channels use the same shift).

## API

- **Module:** `pipeline/scanline_tear.py`
- **`apply_scanline_tear(frame, t, clips, song_hash) -> np.ndarray`** — returns **`(H, W, 3) uint8`** RGB; passes through the input array unchanged (same object) when no clip is active, when **`t`** is non-finite, or when every band’s shift rounds to zero.

## Active clips

A clip counts at time **`t`** iff **`t_start <= t < t_start + duration_s`**. Only **`SCANLINE_TEAR`** kinds are considered; other clips in **`clips`** are ignored.

## Settings

Per **`EffectClip.settings`** (validated against **`EFFECT_SETTINGS_KEYS[SCANLINE_TEAR]`** in `pipeline/effects_timeline.py`):

| Key | Role |
|-----|------|
| **`intensity`** | **`[0, 1]`** scale on the horizontal cap: the absolute shift in pixels is at most **`round(intensity × width × 0.25)`**. **`0`** or non-finite values → no contribution. |
| **`band_count`** | Number of horizontal bands per clip; **`>= 1`**. If missing or invalid, a default in **3…6** is derived from the per-frame seed (see below). |
| **`band_height_px`** | Optional row count per band. If omitted, band height is **`max(1, height // (band_count + 1))`**. Clamped to the frame height. |
| **`wrap_mode`** | **`wrap`** (default) uses **`np.roll`** on each band along **x**; **`clamp`** replicates edge pixels; **`black`** fills vacated columns with **0**. Invalid values fall back to **`wrap`**. |

## Determinism

Per frame, each clip uses a **64-bit seed** from **`SHA-256(song_hash + clip.id + round(t × 1000))`**, then **`numpy.random.Generator(PCG64(seed))`** to pick each band’s top row **`y0`** and integer shift **`dx`**. The same **`song_hash`**, **`clip.id`**, and **`t`** therefore yield the same offsets.

## Overlaps

- **Multiple bands in one clip:** bands are applied in loop order on a working copy; overlapping bands see later writes on top of earlier ones.
- **Multiple active clips:** clips are applied in **timeline order** (the order of clips in **`effects_timeline.json`**), each stage reading the result of the previous — **later clips stack on earlier**.

## Compositor order

Runs in **`_apply_frame_effects`** after **`CHROMATIC_ABERRATION`** and before **`COLOR_INVERT`**. See `docs/technical/effects-timeline-compositor.md`.

## Tests

`tests/test_scanline_tear.py` — inactive window, **`intensity == 0`**, non-finite **`t`**, determinism, **`wrap`** / **`black`**, RGB lockstep on flat colour, order of two clips, shape guard.
