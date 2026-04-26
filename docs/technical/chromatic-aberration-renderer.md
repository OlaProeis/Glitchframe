# Chromatic aberration renderer

Full-frame RGB channel split from **`EffectKind.CHROMATIC_ABERRATION`** clips on the effects timeline.

## API

- **Module:** `pipeline/chromatic_aberration.py`
- **`apply_chromatic_aberration(frame, t, clips, song_hash) -> np.ndarray`** — returns a new **`(H, W, 3) uint8`** RGB frame; passes through the input array unchanged (same object) when no clip is active or the net shift rounds to zero.

## Active clips

A clip counts at time **`t`** iff **`t_start <= t < t_start + duration_s`**. Only **`CHROMATIC_ABERRATION`** kinds are considered; other clips in **`clips`** are ignored.

## Settings

Per **`EffectClip.settings`** (validated against **`EFFECT_SETTINGS_KEYS[CHROMATIC_ABERRATION]`** in `pipeline/effects_timeline.py`):

| Key | Role |
|-----|------|
| **`shift_px`** | Base magnitude (pixels) for the R/B split; **`0`** or missing → no contribution from that clip. |
| **`jitter`** | **`[0, 1]`** per-frame wobble on magnitude: `|shift_px| * (1 + jitter * wobble)` with **`wobble ∈ [-1, 1]`** from a deterministic LCG. |
| **`direction_deg`** | Optional axis angle in degrees (from **+x**); invalid or omitted → deterministic angle from **`song_hash + clip.id`** (SHA-256 seed). |

R is shifted along **`+u`**, B along **`-u`**, where **`u = (cos θ, sin θ)`** in image coordinates (**x** right, **y** down). **G** is never shifted.

The **combined** R offset vector (and thus B) is capped so its length never exceeds **`min(H, W) * 0.1`** pixels. Integer sampling uses **`round`** on the capped float vector.

## Determinism

Per-frame jitter uses **`glitch_seed_for_time(song_hash, t)`** XOR **`adler32(clip.id)`**, matching the **`SCREEN_SHAKE`** / logo-glitch pattern in `pipeline/screen_shake.py` and `pipeline/logo_composite.py`.

## Overlaps

Multiple active clips **sum** their R offset vectors (B is always the negation of the summed R vector). The result is then length-clamped to the cap above.

## Tests

`tests/test_chromatic_aberration.py` — outside interval, **`shift_px == 0`**, G unchanged on a horizontal cue, determinism, non-finite **`t`**, non-chromatic ignored, overlapping sum vs a single clip with doubled **`shift_px`**.
