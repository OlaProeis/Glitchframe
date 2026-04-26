# Screen shake renderer

Deterministic per-frame pixel shift from **`EffectKind.SCREEN_SHAKE`** clips on the effects timeline.

## API

- **Module:** `pipeline/screen_shake.py`
- **`shake_offset(t, clips, song_hash) -> tuple[float, float]`** — returns **`(dx, dy)`** in pixels for compositor use (translation of the full frame or a layer — compositor wiring is separate).

## Active clips

A clip counts at time **`t`** iff **`t_start <= t < t_start + duration_s`**. Only **`SCREEN_SHAKE`** kinds are considered; other clips are ignored.

## Settings

Per **`:class:`~pipeline.effects_timeline.EffectClip`**.`settings`** (validated against **`EFFECT_SETTINGS_KEYS[SCREEN_SHAKE]`** in `pipeline/effects_timeline.py`):

| Key | Role |
|-----|------|
| **`amplitude_px`** | Scale of motion; **`0`** or missing → no contribution from that clip. |
| **`frequency_hz`** | Oscillation rate in Hz; invalid or **`≤ 0`** falls back to an internal default (**4.0** Hz). |

## Determinism

The seed chain matches **`glitch_seed_for_time(song_hash, t)`** (`pipeline/logo_composite.py`) XORed with **`adler32(clip.id)`** so the same song hash and time yield the same offset; overlapping clips do not fight over a single global seed.

## Overlaps

If multiple **`SCREEN_SHAKE`** clips are active at **`t`**, each clip contributes a **`(dx, dy)`**; the result is the **vector sum** of those contributions (documented in the module docstring).

## Tests

`tests/test_screen_shake.py` (outside intervals, inside motion, determinism, amplitude scaling, overlap summing, non-shake ignored).
