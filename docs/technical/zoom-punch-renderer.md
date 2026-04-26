# Zoom punch renderer

Per-frame **scale factor** from **`EffectKind.ZOOM_PUNCH`** clips on the effects timeline (whole-frame bilinear resize + center crop in the compositor is separate).

## API

- **Module:** `pipeline/zoom_punch.py`
- **`zoom_scale(t, clips) -> float`** — returns a scale ``>= 1.0``; **``1.0``** when no punch applies. Values **``> 1.0``** enlarge the source before crop (punch-in).

## Active clips

Timeline activity matches other effect renderers: a clip counts at time **`t`** iff **`t_start <= t < t_start + duration_s`**. Only **`ZOOM_PUNCH`** kinds are considered.

Within an active clip, the punch envelope runs only over the first **`width_frac * duration_s`** seconds (clamped to the clip length). After that prefix, scale is **`1.0`** until the clip ends.

## Settings

Per **`:class:`~pipeline.effects_timeline.EffectClip`**.`settings`** (allowlist in **`EFFECT_SETTINGS_KEYS[ZOOM_PUNCH]`**):

| Key | Role |
|-----|------|
| **`peak_scale`** | Maximum scale; missing/invalid → **1.12**; **`<= 1`** → no zoom from that clip. |
| **`ease_in_s`** | Duration (seconds) of smoothstep ease from **`1.0`** to **`peak_scale`**. |
| **`ease_out_s`** | Duration (seconds) of smoothstep ease from **`peak_scale`** back to **`1.0`**. |
| **`width_frac`** | Fraction **`(0, 1]`** of clip length for the punch window; invalid/ **`<= 0`** → **1.0**. |

If **`ease_in_s + ease_out_s`** exceeds the punch window, both are scaled down proportionally so the envelope fits. Hold at **`peak_scale`** fills any time between the ramps.

Easing uses a Hermite **smoothstep** (**`C^1`**), consistent with typical bilinear sampling (no velocity jump).

## Overlaps

Multiple active punches: the result is the **maximum** of per-clip scales (each **`>= 1.0`**).

## Tests

`tests/test_zoom_punch.py` — outside intervals, peak in hold, ease start, **`peak_scale <= 1`**, **`width_frac`** prefix, overlap **max**, non-punch clips ignored, non-finite **`t`**, defaults.
