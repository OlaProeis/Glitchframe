# Title overlay and audio-reactive logo pulse

Two small but user-visible polish passes sit on top of the
[`frame-compositor`](frame-compositor.md): a persistent ``Artist — Title``
card burned onto every video frame, and a stack of optional audio-reactive
behaviours on the logo overlay:

- **Bass pulse** — size + brightness kick on low-end hits (or on every
  analyzer beat), attack-dominant so the logo bounces instead of sitting
  permanently inflated on sustained sub bass.
- **Snare neon glow** — blurred premultiplied halo behind the logo, keyed
  off mid-band spectral hits (~snare / clap range).
- **Snare squeeze** — brief inward scale dip on the same mid-band detector
  as the neon glow (can run independently).
- **Impact glitch** — RGB-split / horizontal tears on loudness jumps (build-
  up → drop), keyed off RMS in ``analysis.json``.

All four drivers feed the same `composite_logo_onto_frame` call, so
stacking order and corner anchoring are shared.

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

Each glyph is painted with a **layered outer-bloom** so the card stays
legible over arbitrary backgrounds without looking "stamped twice":

1. **Wide bloom** — `MaskFilter.MakeBlur(kOuter_BlurStyle, σ)` with
   `σ ≈ size_px * 0.22`, alpha ≈ `0.18`. Far-falloff halo.
2. **Mid halo** — `σ ≈ size_px * 0.11`, alpha ≈ `0.30`. Mid-range glow.
3. **Tight edge lift** — `σ ≈ size_px * 0.045`, alpha ≈ `0.55`. Brightest
   part of the halo, flush against the glyph edge.
4. **Thin outline** — `Paint.kStroke_Style`, `stroke_width ≈ size_px *
   0.010`, round joins, in the shadow color at 26 % alpha. Provides
   crisp edge contrast on busy / similar-luminance backgrounds.
5. **Fill** — the main glyph pass in `base_color` at the configured
   `title_opacity`.

The `kOuter_BlurStyle` variant bleeds outward *only*, so the fill stays
crisp. The three wide-to-narrow passes approximate a Gaussian pyramid —
the result reads as a single smooth neon-style glow rather than the hard
"stamped blob" a single-pass `kNormal` halo produces. Sigmas, styles, and
alphas are looked up defensively so skia-python builds that don't expose
`MaskFilter.MakeBlur` / `kOuter_BlurStyle` fall back to a plain tint
instead of crashing the render.

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
  (fast rise, ~0.68 s release).
- **Attack-dominant by design** (`sustain_weight=0.30` default). On bass-heavy
  tracks the sustain follower normalises close to `1.0` for most of the song;
  if it were weighted equally to the attack the envelope would saturate and
  the logo would lock to the ceiling scale with no dynamic bounce. At `0.30`
  the sustain contributes a modest baseline that stops the logo fully
  collapsing between kicks on 808 tails, while attacks drive the visible
  motion. Callers can pass a higher `sustain_weight` for always-on sub-heavy
  aesthetics; the caller that drives the logo uses the default.
- `build_bass_pulse_track` is unchanged for reactive-shader `bass_hit`
  (attack-only, longer decay).
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
  to two channels. Defaults pick an **attack-dominant** shape so kicks read
  clearly:
  - **scale:** `+12%` at `pulse=1`, `strength=1` (default);
    up to **+22%** ceiling at `pulse=1`, `strength=2`.
  - **opacity multiplier:** `+22%` at `pulse=1`, `strength=1` (default);
    up to **+38%** ceiling at `pulse=1`, `strength=2`.
  Both axes return to rest between hits; the compositor clamps the final
  `opacity_pct` to `[0, 100]`. The `max_scale_cap` / `max_opacity_cap`
  arguments guard against extreme resize artefacts on sub-pixel logos.
- **Snare neon (logo):** `build_snare_glow_track` keys a blurred premultiplied
  halo **behind** the logo from mel bands 3–6 (mid / snare-clap range on the
  default 8-bin grid). Colour comes from the preset’s shadow / second palette
  colour (`CompositorConfig.shadow_color`). Toggles: `logo_snare_glow`,
  `logo_glow_strength`, `logo_glow_sensitivity` on `CompositorConfig` /
  Gradio Branding.
- **Snare squeeze (logo scale):** the same `build_snare_glow_track` envelope
  drives a brief inward scale dip applied on top of the bass-pulse scale:
  `logo_scale *= max(0.68, 1.0 - squeeze_strength * snare * 0.42)`. At the
  default `logo_snare_squeeze_strength=0.40` peak snares contract the logo
  by ~16.8 %. Runs independently of `logo_snare_glow` — the squeeze can be
  on while the halo is off and vice versa. Clamped at `0.68×` so snares can
  never squish the logo below two-thirds its rest size.
- **Impact glitch:** `build_rms_impact_pulse_track` smooths RMS (from
  `analysis["rms"]`), takes positive differences against a short lagged
  copy (build-up energy ramp), and peak-picks through a fast decay — a
  similar shape to the snare detector but keyed off overall level so it
  fires on drops / transitions rather than every percussive hit.
  `_rgb_glitch_logo_rgba` then applies an RGB-split + horizontal tear
  distortion to the logo RGBA at amplitude
  `impact * logo_impact_glitch_strength`; the per-frame seed is
  `glitch_seed_for_time(song_hash, t)` so a given render is reproducible.
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
  `bass` mode, then drop *Bass sensitivity* (`0.75` → `0.5`).
- **Logo sits permanently large / doesn't bounce.** Usually a sign the
  envelope is saturating — confirm `build_logo_bass_pulse_track` is
  running with the default `sustain_weight=0.30` (attack-dominant). A
  higher `sustain_weight` pins bass-heavy tracks near the scale ceiling
  and hides the bounce; `0.78` produced a measured `p50≈0.96`, `0.30`
  produces `p50≈0.44` with a clean peak on every kick.
- **Pulse too subtle / too aggressive visually.** Tune *Pulse strength*;
  it's a pure visual amplitude knob (`scale` / `opacity_mul`) independent
  of the envelope shape. At `strength=2.0` the pulse reaches the
  `max_scale_cap` / `max_opacity_cap` ceilings (~22 % / 38 %).
- **No visible snare contraction.** Two likely causes: (1) *Snare squeeze*
  slider is at `0 %` — raise it; (2) the bass envelope is saturating, so
  the logo baseline itself sits near the scale ceiling and the squeeze
  dip reads as "back to normal" instead of a pop. Fix the bass side
  first (see above), then the squeeze becomes visible at default 40 %.
- **Impact glitch fires too often / not enough.** Use *Impact sensitivity*
  to rescale the envelope before the strength multiplier; the glitch
  **amount** slider is the pure visual knob.

## UI

The `Branding` tab exposes:

- Logo: file upload, position dropdown, opacity slider.
- **Pulse logo on audio** (checkbox, default on) — master switch.
- **Pulse signal** (radio: *Bass / kick energy* ▸ default, *Every beat*).
- **Bass sensitivity** slider `0.25 → 3.0` (`1.0` default). Only
  meaningful when the signal is *Bass / kick energy*.
- **Pulse strength** slider `0 → 2` (0 = off, 1 = default, 2 = exaggerated).
- **Snare-reactive neon glow** (checkbox + strength %) — halo behind the logo.
- **Snare squeeze (logo scale)** slider `0 → 100 %` (`40 %` default) —
  brief inward scale dip on mid-band hits; independent of the neon toggle.
- **Drop / impact glitch** slider `0 → 100 %` (`45 %` default) — RGB-split
  / tear on RMS jumps.
- **Impact sensitivity** slider `0.25 → 3.0` (`1.0` default) — only
  meaningful when glitch > 0 %.
- **Burn Artist — Title onto every frame** (checkbox, default on).
- **Title position** (9-point grid dropdown).
- **Title size** (`Small / Medium / Large`).

## Fonts

The title card uses `CompositorConfig.title_font_path`. Default resolution
order in `config.default_title_font_path()`:

1. `assets/fonts/SpaceGrotesk-SemiBold.ttf` — bundled display face (OFL,
   see `SpaceGrotesk-LICENSE.txt`). Geometric neo-grotesque, reads
   distinctly "modern" at 1080p+ title-card sizes.
2. `assets/fonts/Inter-SemiBold.ttf` — kept as a fallback for machines
   that haven't pulled the new asset. Was the previous default.
3. The body font (`Inter.ttf`) as a last resort.
4. `None` — Skia falls back to a system typeface.

Kinetic lyrics use `CompositorConfig.font_path` (default: `Inter.ttf` via
`config.default_ui_font_path()`). Both should live under `assets/fonts/`
and be version-controlled so renders are consistent across machines.

## Colors

`CompositorConfig.base_color` (title fill) and `shadow_color` (glow) are
resolved from the preset palette by
`pipeline.preset_colors.resolve_text_colors()`. Preset palettes are
ordered dark → bright for the shader's `u_palette[5]` — text needs the
opposite, so the resolver picks the *brightest* entry for the fill and
the most saturated remaining entry for the glow. This keeps titles
readable on dark-theme presets (cosmic-flow, neon-synthwave, organic-liquid,
glitch-vhs) where the first two palette slots are near-black
background colors.

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
| `logo_snare_squeeze_strength` | `0.40` | Max inward scale dip on snares (0 = off; 1.0 = slider max). |
| `logo_impact_glitch_strength` | `0.45` | RGB-split / tear amplitude on RMS jumps (0 = off). |
| `logo_impact_sensitivity` | `1.0` | Scales the impact envelope before clipping. |
| `show_title` | `True` | Master switch for the burned-in card. |
| `title_position` | `"top-center"` | Grid cell for the title card. |
| `title_size` | `"medium"` | `small / medium / large`. |

## Tests

- `tests/test_beat_pulse.py` — beats envelope shape, BPM-derived tau,
  NaN/negative beat filtering, the pulse-to-scale/opacity mapping,
  `PulseTrack` sampling, `build_bass_pulse_track` behaviour (missing
  spectrum, silence, synthetic kicks normalised to ~1.0, sensitivity
  scaling), `build_logo_bass_pulse_track` dynamic range on narrow raw
  bands and sustained-sub behaviour vs attack-only, and
  `build_rms_impact_pulse_track` drop detection.
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
