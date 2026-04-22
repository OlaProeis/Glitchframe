# Title overlay and audio-reactive logo pulse

Two small but user-visible polish passes sit on top of the
[`frame-compositor`](frame-compositor.md): a persistent ``Artist — Title``
card burned onto every video frame, and an optional audio-reactive pulse
on the logo overlay. The pulse can key off the analyzer's **beat grid** or
off **bass / kick energy** from ``analysis.json`` — the second mode keeps
the logo still during hi-hat-heavy subdivisions and only kicks on actual
low-end hits.

## Title overlay

**Module:** `pipeline/title_overlay.py`.

- The orchestrator composes the display line from `OrchestratorInputs.metadata`
  via `format_title_text(artist, title)`; empty metadata automatically
  disables the overlay regardless of the UI toggle.
- `render_title_rgba()` rasterises the line **once** per render via Skia
  into a premultiplied `(H, W, 4)` RGBA layer the size of the video frame.
  The compositor reuses the same layer for every output frame, so the
  per-frame cost is a single NumPy alpha blend.
- Font size is picked as `CompositorConfig.font_size * multiplier`, where
  `multiplier ∈ {0.55, 0.80, 1.15}` for `small / medium / large`. If the
  line would overflow 92% of the frame width the helper shrinks the font
  iteratively until it fits, mirroring `pipeline/thumbnail.py`.
- Position uses a 9-point grid (`top-left` → `bottom-right`, plus
  `middle-left / center / middle-right`). Helpers `normalize_title_position`
  and `normalize_title_size` accept free-form UI labels (`"Top-center"`,
  `"TOP"`, `"middle"`, etc.) and raise `ValueError` on unknown values.
- Fill / shadow colors come from the active preset's palette (`base_color`,
  `shadow_color` on `CompositorConfig`) so the card matches the preset
  aesthetic without extra configuration.
- The layer is composited **above** the reactive shader and kinetic
  typography but **below** the logo, so branding always sits on top.

### Glyph rendering recipe

`render_title_rgba()` configures the Skia font for display-size text —
`Edging.kAntiAlias`, `setSubpixel(True)`, `FontHinting.kNone` — via
`_configure_font_for_display()`. Skia's default `kNormal` hinting is tuned
for 12–16 px UI text and distorts stroke weights at the sizes we render,
so disabling it gives smoother, evenly-weighted letters.

Each glyph is painted in three layered passes so the card stays legible
over arbitrary backgrounds without looking "stamped twice":

1. **Soft halo** — `MaskFilter.MakeBlur(kNormal_BlurStyle, σ)` with
   `σ ≈ size_px * 0.11`, offset `+size_px * 0.03` on Y only. Acts as a
   grounding glow.
2. **Thin outline** — `Paint.kStroke_Style` with
   `stroke_width ≈ size_px * 0.035` and round joins, in the shadow color
   at 55 % alpha. Provides crisp edge contrast on busy/similar-luminance
   backgrounds.
3. **Fill** — the main glyph pass in `base_color` at the configured
   `title_opacity`.

### Stacking order (full compositor pipeline)

1. Background (`BackgroundSource.background_frame(t)`)
2. Reactive shader (moderngl)
3. Kinetic typography (lyrics)
4. **Title overlay (static premultiplied RGBA)**
5. Logo (with optional beat pulse)

## Audio-reactive logo pulse

**Module:** `pipeline/beat_pulse.py`, consumed by `pipeline/compositor.py`
and `pipeline/logo_composite.py`.

Two envelope shapes share the same visual mapping so switching modes only
swaps the signal driving the pulse, not the way it looks on screen.

### `bass` mode (default)

- The compositor uses `build_logo_bass_pulse_track` for the **logo**: it
  blends (a) **attack** from positive first-differences in the low mel bins
  + short decay, with (b) a **sustain follower** on the same low-bin average
  (fast rise, ~0.68 s release) so long 808s keep the mark expanded while sub
  holds, then ease out slowly. `build_bass_pulse_track` is unchanged for
  reactive-shader `bass_hit` (attack-only).
- Returns `None` when the analyzer lacks a usable spectrum; the compositor
  disables the pulse for that render.

### `beats` mode

- `beat_pulse_envelope(t, beats, bpm=…)` returns a value in `[0, 1]`:
  snaps to `1.0` at every analyzer beat time and decays exponentially
  with a BPM-aware `tau`. The time constant is clamped to
  `max(0.04, min(0.40, (60 / bpm) * tau_fraction))` so extreme BPM values
  from a noisy analyzer still produce watchable pulses.
- Reads `beats` from `analysis["beats"]` and `bpm` from
  `analysis["tempo"]["bpm"]` (with a top-level `analysis["bpm"]`
  fallback). When either is missing the pulse is disabled for this
  render even if the user asked for it.

### Shared visual mapping

- `scale_and_opacity_for_pulse(pulse, strength=…)` maps either envelope
  to two channels (with caps so very strong settings stay usable):
  - **scale:** up to about **+22%** at `pulse=1` and `strength=2`
  - **opacity multiplier:** up to about **+38%** at `pulse=1` and `strength=2`
  Both axes return to rest between hits; the compositor clamps the final
  `opacity_pct` to `[0, 100]`.
- **Snare neon (logo):** `build_snare_glow_track` keys a blurred premultiplied
  halo **behind** the logo from mel bands 3–6 (mid / snare-clap range on the
  default 8-bin grid). Colour comes from the preset’s shadow / second palette
  colour (`CompositorConfig.shadow_color`). Toggles: `logo_snare_glow`,
  `logo_glow_strength`, `logo_glow_sensitivity` on `CompositorConfig` /
  Gradio Branding.
- `pipeline/compositor.py::_build_pulse_fn(cfg, analysis)` builds a
  single `Callable[[float], float]` once per render, encoding the active
  mode. When `cfg.logo_beat_pulse` is `False`, the analyzer output is
  unusable for the selected mode, or `logo_pulse_mode` is unknown, the
  frame loop skips the envelope call entirely so the hot path stays free
  of per-frame float math. Unknown modes raise `ValueError` at setup
  time — strict error handling as per project conventions.
- `composite_logo_onto_frame(…, scale=…)` applies the per-frame scale
  before anchoring via `_origin_for_position`, so a `bottom-right` logo
  still sits flush to the corner while it kicks.
- Resize uses PIL **BILINEAR** instead of LANCZOS: per-frame scale deltas are
  modest and bilinear is ~5–10× faster with no visible quality cost.

## Tuning

- **Track won't pulse / weak kicks.** Raise *Bass sensitivity* first
  (`1.25` → `2.0`). If the mix has no clear low-end at all, switch to
  *Every beat* mode so the grid tracker drives the pulse instead.
- **Logo pulses on every hat / subdivision.** Confirm you're in
  `bass` mode, then drop *Bass sensitivity* (`0.75` → `0.5`). The logo
  envelope also adds **sustained sub** (808 tail); if that feels too
  “sticky”, lower sensitivity or pulse strength.
- **Pulse too subtle / too aggressive visually.** Tune *Pulse strength*;
  it's a pure visual amplitude knob (`scale` / `opacity_mul`) independent
  of the envelope shape.

## UI

The `Branding` tab exposes:

- Logo: file upload, position dropdown, opacity slider.
- **Pulse logo on audio** (checkbox, default on) — master switch.
- **Pulse signal** (radio: *Bass / kick energy* ▸ default, *Every beat*).
- **Bass sensitivity** slider `0.25 → 3.0` (`1.0` default). Only
  meaningful when the signal is *Bass / kick energy*.
- **Pulse strength** slider `0 → 2` (0 = off, 1 = default, 2 = exaggerated).
- **Snare-reactive neon glow** (checkbox + strength %) — halo behind the logo.
- **Burn Artist — Title onto every frame** (checkbox, default on).
- **Title position** (9-point grid dropdown).
- **Title size** (`Small / Medium / Large`).

## Fonts

The title card uses `CompositorConfig.title_font_path` (default: bundled `Inter-SemiBold.ttf` when present). Kinetic lyrics use `CompositorConfig.font_path` (default: `Inter.ttf`). Both should live under `assets/fonts/` and be version-controlled so renders are consistent across machines (see `config.default_title_font_path` / `default_ui_font_path`).

## Orchestrator inputs

Fields on `OrchestratorInputs` (all optional, sensible defaults):

| Field | Default | Effect |
|-------|---------|--------|
| `logo_beat_pulse` | `True` | Master switch for the pulse effect. |
| `logo_pulse_mode` | `"bass"` | `"bass"` (low-frequency energy) or `"beats"` (grid). |
| `logo_pulse_sensitivity` | `1.0` | Scales the bass envelope before clipping. Ignored in `beats` mode. |
| `logo_pulse_strength` | `1.0` | Scales the visual amplitude of each pulse. |
| `logo_snare_glow` | `True` | Mid-band reactive neon behind the logo. |
| `logo_glow_strength` | `1.0` | Multiplier on the snare-glow envelope. |
| `logo_glow_sensitivity` | `1.0` | Scales snare-glow spectral detection. |
| `show_title` | `True` | Master switch for the burned-in card. |
| `title_position` | `"top-center"` | Grid cell for the title card. |
| `title_size` | `"medium"` | `small / medium / large`. |

## Tests

- `tests/test_beat_pulse.py` — beats envelope shape, BPM-derived tau,
  NaN/negative beat filtering, the pulse-to-scale/opacity mapping,
  `PulseTrack` sampling, and `build_bass_pulse_track` behaviour (missing
  spectrum, silence, synthetic kicks normalised to ~1.0, sensitivity
  scaling).
- `tests/test_title_overlay.py` — position aliases, size normalisation,
  `format_title_text` metadata composition, and geometric checks on the
  rasterised RGBA layer (top-position centroid above bottom-position).

## See also

- [`frame-compositor.md`](frame-compositor.md) — where the title RGBA and
  pulsed logo are stacked each frame.
- [`logo-composite.md`](logo-composite.md) — position math and alpha blend
  that the pulse path extends with `scale`.
- [`thumbnail-generator.md`](thumbnail-generator.md) — the separate
  YouTube-cover path that uses the same `Artist — Title` string.
