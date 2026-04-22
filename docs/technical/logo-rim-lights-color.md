# Logo rim lights — multi-colour & hue drift (task 26)

Builds on `compute_logo_rim_light_patch` in `logo-rim-lights.md` by layering **2–3
coloured emissive terms** on the same task-25 scalar `base` field (halo, inward
bleed, line energy + blur). Each layer multiplies the base by a **phase-shifted
angular wave** and its own sRGB (or HSV-spread) colour; contributions are
summed in **linear sRGB** then encoded back to 8-bit premultiplied RGBA (see
:mod:`pipeline.logo_rim_lights` for exact blending).

## Config knobs (``RimLightConfig``)

- **`rim_color_layers`** — `1` = legacy single-`rim_rgb` path (unchanged
  output contract). `2`–`3` enables multi-layer.
- **`color_spread_rad`** — HSV **hue** delta between adjacent layers, in
  **radians** (divide by `2π` to compare to a ``[0, 1)`` hue fraction). A good
  neon baseline is `2π/3` (~120°) so three layers read as **pink / cyan /
  violet** when spaced from a magenta `rim_rgb`.
- **`layer_phase_offsets`** — extra radians added to
  `2π*phase_hz*t + phase_offset` for each layer’s angular wave. Default when
  empty: evenly distributed on a circle, e.g. for three layers
  `0, 2π/3, 4π/3` so the travelling lobes **interlock** instead of sitting on top
  of one another.
- **`hue_drift_per_sec`** — how many full **HSV hue cycles** per second (same
  scale as a frequency in **Hz** on the 0–1 hue ring). Drives a smooth global
  rotation of every layer’s colour in lockstep; combine with
  `rim_color_layers=2` and `use_line_features=False` for a **dual-tone
  animated halo** (geometry already falls back to silhouette-only; this task
  keeps only **two hue layers** for readability).
- **`song_hash`** — optional str/int/bytes; mixed into a small stable base-hue
  offset so the same track + time reproduces the same frame bytes, while
  different runs can vary the palette.
- **`flicker_amount`** — optional `0…1` gain multiplier tied to a ~11 Hz
  sinusoid for a subtle **neon** shimmer without harsh stepping.

**Preset-anchored defaults:** :func:`pipeline.logo_composite.resolve_logo_glow_rgb`
(`shadow` / `base` hex) is the right source for a primary `rim_rgb` when
wiring; layer hues are then derived with `color_spread_rad` and drift. Optional
helper :func:`pipeline.logo_rim_lights.rim_base_rgb_from_preset` returns that
glow colour for a preset pair.

## Behaviour

- **Halo-only logos** (`use_line_features=False`): if `rim_color_layers ≥ 2`,
  the implementation uses at most **two** hue layers (dual-tone) while the line
  term remains zero; angular wave + hue drift still animate the rim.
- **Determinism:** identical `(prep, t, config)` (including `song_hash`) →
  identical patch bytes; multi-colour is free of unseeded randomness.
- **Integration:** consumed via [`logo-rim-compositing.md`](logo-rim-compositing.md)
  when rim is enabled on the compositor.

## Tests

`tests/test_logo_rim_lights.py` — multi-colour determinism, hue-drift
monotonicity, and per-frame max delta (anti-banding) over a short time sweep.
