# Colour invert renderer

Scalar **mix in [0, 1]** from **`EffectKind.COLOR_INVERT`** clips on the effects timeline, for compositor or shader use (lerp source frame toward its colour inverse).

## API

- **Module:** `pipeline/color_invert.py`
- **`invert_mix(t, clips) -> float`** — lerp weight at time **`t`** (ignores non-`COLOR_INVERT` clips).

## Active clips

A clip counts at time **`t`** iff **`t_start <= t < t_start + duration_s`** (same half-open rule as the screen-shake renderer).

## Settings

Per **`EffectClip.settings`** (validated against **`EFFECT_SETTINGS_KEYS[COLOR_INVERT]`** — keys **`mix`**, **`intensity`**):

| Key | Role |
|-----|------|
| **`mix`** | Factor in **[0, 1]** after clamping; default **1.0** if missing, **`None`**, or non-numeric. |
| **`intensity`** | Factor in **[0, 1]** after clamping; default **1.0** if missing, **`None`**, or non-numeric. |

**Per-clip contribution** is **`mix * intensity`** (each clamped to **[0, 1]** before multiply).

## Overlaps

If multiple **`COLOR_INVERT`** clips are active at **`t`**, each clip contributes **`mix * intensity`**. The **total** is the **sum** of those contributions, **capped** at **1.0** (unlike `shake_offset`, which sums 2D vectors; here the result must stay a scalar in **[0, 1]**).

## Tests

`tests/test_color_invert.py`

## Related

- `docs/technical/effects-timeline.md` — data model and settings allowlist.
