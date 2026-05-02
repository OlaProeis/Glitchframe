# Title overlay and audio-reactive logo pulse

Two small but user-visible polish passes sit on top of the
[`frame-compositor`](frame-compositor.md): a persistent ``Artist — Title``
card burned onto every video frame, and a stack of optional audio-reactive
behaviours on the logo overlay:

- **Bass pulse** — size + brightness kick on low-end hits (or on every
  analyzer beat), attack-dominant so the logo bounces instead of sitting
  permanently inflated on sustained sub bass.
- **Kick punch** — dedicated scale / opacity bump on clean low-band
  transients (`transient_lo`). Combined with the bass pulse via
  `max` of deltas so one signal never cancels the other; cleanly
  separated kicks produce a visibly bigger bounce than the blended
  sustain-aware pulse alone.
- **Snare neon glow** — blurred premultiplied halo behind the logo, keyed
  off mid-band spectral hits (~snare / clap range).
- **Kick neon glow** — the same halo also pumps on low-band (kick / sub)
  transients, so the rim light reads on tracks with sparse or absent
  snare energy.
- **Snare squeeze** — brief inward scale dip on the same mid-band detector
  as the neon glow (can run independently).
- **Impact glitch** — RGB-split / horizontal tears + ±22° per-event tilt on loudness jumps (build-
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
- Fill / shadow colors come from the resolved visual-style palette (`base_color`,
  `shadow_color` on `CompositorConfig` — propagated from `OrchestratorInputs.presets`) so the card matches the active shader bundle without extra configuration.
- The layer is composited **above** the reactive shader and kinetic
  typography but **below** the logo, so branding always sits on top.

### Glyph rendering recipe

`render_title_rgba()` configures the Skia font for display-size text —
`Edging.kAntiAlias`, `setSubpixel(True)`, `FontHinting.kNone` — via
`_configure_font_for_display()`. Skia's default `kNormal` hinting is tuned
for 12–16 px UI text and distorts stroke weights at the sizes we render,
so disabling it gives smoother, evenly-weighted letters.

Each glyph is painted as a **clean fill only** — no halo, no outer
glow, no `kStroke_Style` outline, no drop shadow. Every variant was
tried and rejected on saturated preset palettes:

- A `kStroke_Style` outline in the glow colour drew as a **crisp colour
  ring** on complementary palettes (e.g. cyan fill vs magenta-accent
  backgrounds, or yellow fill vs cyan-accent backgrounds) — the
  original "text border issue".
- A wide multi-pass blur (wide + mid + tight) bled far past the glyph
  and stacked into a **solid colour cloud** at the rim on those same
  palettes.
- A tuned single-pass soft outer blur at ~14 % alpha still read as a
  visible ring on the thumbnail-size render, which is where the
  treatment is first seen.

Fill-vs-background contrast is instead handled by
`pipeline.preset_colors.resolve_text_colors`, which picks the brightest
palette entry for the fill so it stands out against the dark half of
the palette that the reactive shader renders against. `shadow_hex` /
`shadow_color` are still accepted at the API boundary for signature
parity (so a future design iteration can re-enable a treatment without
an API break), but they are unused in the current render paths in
`pipeline.title_overlay`, `pipeline.kinetic_typography`, and
`pipeline.thumbnail`. Do not reintroduce any halo / stroke / bloom in
any of those three modules without a preset-matrix visual sign-off
first.

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
- **Kick neon (logo):** `_kick_glow_envelope_fn` builds
  `build_bass_pulse_track` (attack-only, ~0.18 s decay) and adds
  `kick * logo_kick_glow_strength` to the halo amplitude **on top of** the
  snare contribution, so every low-end transient flashes the halo even on
  snare-light tracks. Toggles: `logo_kick_glow` (default `True`) and
  `logo_kick_glow_strength` (default `1.6`) on `CompositorConfig`; the
  envelope is master-scaled alongside the other auto reactivity tracks.
- **Kick punch (logo scale / opacity):** `_kick_punch_envelope_fn` builds
  `build_lo_transient_track` — the same shape-gated low-band transient
  envelope the reactive shaders read as `transient_lo` — and drives a
  dedicated, *larger-budget* map via `kick_punch_scale_and_opacity`:
  - **scale:** `+20%` at peak kick, `strength=1` (default); up to the
    `+35%` `max_scale_cap` at `strength=2`.
  - **opacity multiplier:** `+32%` at peak, `strength=1`; up to the
    `+55%` `max_opacity_cap` at `strength=2`.
  Peak kicks therefore punch harder than the sustain-aware bass pulse
  (which caps at `+12 / +22%`), which is the point: cleanly-separated
  kicks read visibly on screen without amplifying the chill-section
  breathing that made the older single-channel path feel blurry on hits.
  The compositor combines the two channels via `max` of deltas
  (`logo_scale = max(pulse_scale, kick_scale)`, likewise for opacity) so
  the pulse still owns sustained bounces and the kick punch only wins
  when it's genuinely bigger. Uses the build-time shape gate from
  `shape_reactive_envelope` (see below) so between-hit wobble is
  pre-silenced — the kick punch helper uses a smaller in-helper
  deadzone (`0.12`) than the logo pulse (`0.22`) to avoid
  double-gating real kicks. Toggle: `logo_kick_punch_strength` on
  `CompositorConfig` (default `1.0`; `0` disables). Master-scaled with
  the rest of the auto stack via `auto_reactivity_master`, and the
  existing `logo_pulse_strength` slider scales the punch too so the
  UI's "Pulse strength" knob stays a single-point-of-control.
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
  `_rgb_glitch_logo_rgba` then applies an RGB-split + horizontal tear +
  **±22° tilt** distortion to the logo RGBA at amplitude
  `impact * logo_impact_glitch_strength`; the per-frame seed
  (`glitch_seed_for_time(song_hash, t)`) drives the RGB jitter / tear
  randomness so a given render is reproducible. The tilt direction,
  however, is picked from a **separate `tilt_seed` that stays stable for
  every frame of one glitch event** — the compositor quantises `t` into
  ~0.4 s buckets (`glitch_seed_for_time(song_hash, floor(t / 0.4) * 0.4)`)
  and passes that bucket seed through `composite_logo_onto_frame`. Without
  this, a typical 0.2 s impact spanning ~6 frames would re-pick a random
  left/right sign on every frame, producing violent jitter (the "super
  shaky" regression we chased). With the stable bucket seed the tilt
  rises with `amount` on attack, holds one direction, and falls cleanly
  on release — one crisp impact tilt, not shake noise. `tilt_deg`
  itself is eased as `direction * _GLITCH_TILT_DEG * amount ** 0.85` so
  the tilt reads immediately at low amounts while still decaying to zero
  at the end of the event. The rotated logo is re-centered in its
  expanded bounding box by `Image.rotate(expand=True)` so the visual
  centre stays anchored to `_origin_for_position`; beams continue to
  emit from the pre-glitch centroid captured once per render in
  `_build_beam_render_context`, which reads as a hard camera shake.
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

## Stability (micro-shake deadzone + smart smoothing)

Quiet / "chill" sections of a song still carry low-amplitude wobble in the
normalised bass envelope (5–15 % of peak). `scale_and_opacity_for_pulse`
used to map those values linearly to a scale delta, which read on screen
as the logo **micro-shaking** even when nothing was really happening.

The stability pipeline has two cooperating stages:

1. **Stateless smoother** (`beat_pulse.stable_pulse_value`) — asymmetric
   low-pass around `pulse_fn(t)` driven by a short look-back window
   (default `60 ms · logo_motion_stability`). On rising edges (current
   value clearly above every recent sample) it passes the raw value
   through so kick attacks still hit in one frame; otherwise it returns
   a 50/50 blend of current and past-average, which kills sub-kick jitter
   in the release tail. Because the logo's origin is computed via
   integer-rounded `(frame_w - logo_w) // 2`, a tiny pulse wiggle of
   `~0.01` can shift the scaled logo by a whole pixel — the smoother
   removes exactly that class of noise before the scale mapping runs.
2. **Soft deadzone** (`apply_pulse_deadzone`) — collapses anything below
   a configurable noise floor to zero and smoothsteps the shoulder; see
   below. Runs *after* the smoother so the floor operates on clean data.

`pipeline/beat_pulse.py::apply_pulse_deadzone(pulse, *, deadzone, soft_width)`
handles the hard cut-off:

- Inputs `≤ deadzone` (default `0.22`) collapse to `0.0` — the logo is
  perfectly still. The default was raised from the original `0.12` after
  renders showed the logo still shaking on soft bass passes; real kicks
  land at `~1.0` so they're unaffected.
- Inputs in `(deadzone, deadzone + soft_width]` (default `0.14`-wide
  shoulder) are remapped through a classic smoothstep
  (`x*x*(3 - 2x)`) so motion eases in rather than snapping on.
- Inputs above the shoulder pass through untouched, so real kicks still
  reach `1.0` and the existing scale / opacity caps still apply.

`scale_and_opacity_for_pulse` calls the helper before the `strength`
multiplier, so the gate sits on the *envelope* rather than the visual
amplitude. Setting `deadzone=0.0` reproduces the legacy linear
behaviour byte-for-byte.

### Compositor hook-up

`CompositorConfig.logo_motion_stability: float = 1.0` is exposed through
`OrchestratorInputs.logo_motion_stability` and the Branding accordion's
**Logo stability (ignore micro-shake)** slider. It linearly scales both
smart controls in three places:

1. The bass-pulse scale/opacity mapping (via `stable_pulse_value` →
   `scale_and_opacity_for_pulse(..., deadzone=...)`).
2. The snare value read (`stable_pulse_value` on `snare_fn`) that drives
   the snare-squeeze gate in `_render_compositor_frame`.
3. The soft deadzone applied to the smoothed snare before the
   `max(0.68, 1.0 - ...)` dip, so a quiet mid-band ripple no longer
   squishes the logo.

Tuning intent:

- `0.00` - legacy behaviour; no smoothing, no deadzone. Every envelope
  sample moves the logo (shakiest).
- `1.00` - default; ~60 ms release smoothing + ~22 % deadzone hides
  chill-section wobble while leaving kicks untouched.
- up to `2.00` - aggressive; ~120 ms smoothing + ~44 % deadzone. Useful
  for lo-fi / ambient renders where you want a mostly-static logo that
  only reacts to genuine peaks.

The slider is percentage-based in the UI (`0-200 %`, default `100 %`)
and maps to the raw `0.0-2.0` scalar in `_build_render_inputs`.

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
- **Drop / impact glitch** slider `0 → 100 %` (`45 %` default) — RGB-split / ±22° per-event tilt
  / tear on RMS jumps.
- **Impact sensitivity** slider `0.25 → 3.0` (`1.0` default) — only
  meaningful when glitch > 0 %.
- **Rim beams on drops & snare rolls** (checkbox, default on) — clean
  radial rays from the rim on drops + snare lead-ins, rate-limited to
  ~1 burst per 10 s. See [`logo-rim-beams.md`](logo-rim-beams.md).
- **Logo stability (ignore micro-shake)** slider `0 → 200 %` (`100 %`
  default) — scales the soft deadzone on the bass pulse + snare squeeze
  so quiet sections don't micro-shake the logo.
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
`pipeline.preset_colors.resolve_text_colors()`. Style palettes are
ordered dark → bright for the shader's `u_palette[5]` — text needs the
opposite, so the resolver picks the *brightest* entry for the fill and
the most saturated remaining entry for the glow. This keeps titles
readable on **dark-first** bundles from `pipeline/visual_style.py` where
the first palette slots are near-black background tones.

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
| `logo_impact_glitch_strength` | `0.45` | RGB-split / tear / ±22° per-event tilt amplitude on RMS jumps (0 = off). |
| `logo_impact_sensitivity` | `1.0` | Scales the impact envelope before clipping. |
| `logo_motion_stability` | `1.0` | Multiplier on the pulse/snare soft deadzone (0 = legacy, 1 = default, 2 = extra stable). |
| `rim_beams_enabled` | `True` | Master switch for rim-light beams on drops + snare lead-ins; see [`logo-rim-beams.md`](logo-rim-beams.md). |
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
