# Fade renderer

Per-frame **fade-to-black** overlay alpha from `EffectKind.FADE` clips on the effects timeline. Fade is the **single timeline lane** that handles both fade-in (reveal from black) and fade-out (cover to black) — the user picks `direction_mode` in the gear panel and the clip's `duration_s` *is* the ramp length.

## API

- **Module:** `pipeline/fade.py`
- **`fade_alpha(t, clips) -> float`** — black-overlay alpha in `[0, 1]`; **`0.0`** when no fade is active. The compositor multiplies the frame by `(1 - alpha)`, so `alpha = 1` is fully black.
- **`apply_fade(frame, alpha) -> np.ndarray`** — blends `frame` (H, W, 3 uint8 RGB) toward black by `alpha`. Returns the input unchanged at `alpha <= 0`, a fresh black frame at `alpha >= 1`.

## Active clips

Activity matches the rest of the post-stack renderers: a clip counts at time `t` iff `t_start <= t < t_start + duration_s`. Only `FADE` kinds are considered. Non-finite `t` returns `0.0` (identity).

## Settings

Per `EffectClip.settings` (allowlist in `EFFECT_SETTINGS_KEYS[FADE]`):

| Key | Role |
|-----|------|
| `direction_mode` | `"in"` (start black, reveal) or `"out"` (start clear, fade to black). Default **`"in"`**. Unknown strings fall back to `"in"`. |
| `peak_alpha` | Maximum black-overlay alpha at the extreme of the ramp; clamped to `[0, 1]`. Default **`1.0`** (fully black). |
| `ease_mode` | `"smoothstep"` (Hermite, default) or `"linear"`. Unknown strings fall back to `"smoothstep"`. |

Within an active clip:

- **Fade-in:** progress runs `1.0 → 0.0` (so at `t_start` the screen is fully black, at `t_start + duration_s` it's clear).
- **Fade-out:** progress runs `0.0 → 1.0` (clear at the start, black at the end).

The chosen `ease_mode` is applied to that progress before multiplying by `peak_alpha`.

## Overlaps

Multiple active fades: the contribution is the **maximum** of per-clip alphas, then clamped to `[0, 1]`. Stacking two fades never pushes past full black.

## Compositor order

Fade is the **last** stage of `_apply_frame_effects` so it sits on top of every other glitch / inversion (see `docs/technical/effects-timeline-compositor.md`).

## Tests

`tests/test_fade.py` — outside-window passthrough, fade-in / fade-out endpoints, smoothstep midpoint, `peak_alpha` cap, overlap max, unknown direction fallback, non-fade clips ignored, non-finite `t`, empty clips, `apply_fade` shape guard.
