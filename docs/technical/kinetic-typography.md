# Kinetic typography layer

Feature: render per-word kinetic typography with `skia-python` using
`cache/<song_hash>/lyrics.aligned.json`, producing transparent RGBA frames
that the compositor stacks on top of the reactive shader pass.

## Flow

1. Caller loads word timings with `load_aligned_words(lyrics_aligned_json)`
   → `(lines, words)`. The schema matches
   `docs/technical/lyrics-aligner.md` (schema version 1).
2. Caller instantiates `KineticTypographyLayer(aligned_words, motion=…,
   font_path=…, width=…, height=…)`. The layer precomputes per-line layout
   (word widths, centered horizontal positions, baseline at
   `baseline_y_ratio * height`) once up front.
3. Each frame: `layer.render_frame(t, uniforms=…)` returns an
   `(H, W, 4) uint8` RGBA numpy array. Uniforms are the same dict produced by
   `pipeline.reactive_shader.uniforms_at_time(...)` (`beat_phase`, `rms`,
   `onset_pulse`, `intensity`), so the typography pulses in sync with the
   shader pass without any extra plumbing.
4. Previous line fades out over `line_fade_seconds` after its last word ends
   so consecutive lyric lines cross-fade instead of flashing.

## Motion presets

Motion presets are keyed by the preset YAML's `typo_style` string. Every
preset string used in `presets/*.yaml` has a matching pure function in
`MOTION_PRESETS`:

| `typo_style` | Behaviour |
|--------------|-----------|
| `pop-in` | Alpha fade-in + scale overshoot `0.2 → 1.15 → 1.0` at `t_start`, linear fade-out over `outro_seconds`. |
| `slide` | Word slides up from `+48 px` below the baseline during `intro_seconds`; drifts `-24 px` up while fading out. |
| `flicker` | Deterministic per-word alpha jitter driven by `(t, word_index)` — no RNG state so renders are reproducible. |
| `scale-pulse` | Scale reacts to `uniforms["rms"]` (+`onset_pulse`) so the line breathes with the track. |
| `beat-shake` | Horizontal/vertical offset driven by `uniforms["beat_phase"]` and `intensity`; amplitude tapers between beats. |

Each motion function returns a `WordMotion(alpha, scale, dx, dy)` applied
around the word's layout position (translate → scale → draw). Scale pivots
around the glyph centre so words pop without shifting horizontally.

## Layout

- Lower-third placement (`baseline_y_ratio=0.75`).
- Whole active line is pre-measured and centered around `width / 2`; each
  word gets a fixed anchor X at its glyph centre.
- `word_spacing_px` is added between word boxes.
- Current line + previous line are both drawn during the cross-fade window;
  everything else is skipped.

## Fonts

- Optional `font_path` under `assets/fonts/`. Pass `None` to use the Skia
  default typeface so the layer still renders in headless / CI hosts with no
  bundled fonts.
- Font loading tries `skia.Typeface.MakeFromFile` first and falls back to
  `skia.FontMgr.OneFontMgr` (or `New_Custom_Empty`) for newer skia-python
  builds. A missing path, or a skia-python that exposes neither loader,
  raises `FileNotFoundError` / `RuntimeError` rather than silently rendering
  blank text.

## Colors

- `base_color` (hex `#RRGGBB`) is blended with the per-word alpha envelope.
- `shadow_color` (optional) is drawn as a three-pass outer-blur halo
  (sigma ≈ `font_size * {0.22, 0.11, 0.045}` with alpha
  `{0.18, 0.32, 0.55}`) so lyrics stay legible over busy shader
  backgrounds without the hard-stamped look of a fixed-offset drop shadow.
  Mirrors the recipe in `pipeline.title_overlay.render_title_rgba`.
- Both colors come from `pipeline.preset_colors.resolve_text_colors()`
  applied to the preset palette, so the *brightest* palette entry drives
  the fill and the most saturated mid-tone drives the glow — preset
  palettes are ordered dark→bright for the shader's `u_palette[5]`, so
  taking `colors[0]` / `colors[1]` directly would paint lyrics in the
  darkest background tones and render them unreadable.

## Output

- `render_frame` returns `(H, W, 4) uint8` RGBA in top-left origin.
- Skia's native `kN32` layout is little-endian BGRA, so the layer swaps
  channels to RGBA before returning — compositor / ffmpeg stages can treat
  the array as plain RGBA without platform checks.
- The returned array is a copy; the internal backing buffer is reused to
  avoid per-frame allocations.

## Fallback behaviour (no silent failures)

- Missing aligned JSON → `FileNotFoundError`.
- Malformed JSON / missing `lines` / `words` → `ValueError`.
- Unknown `motion` preset → `ValueError` listing `SUPPORTED_MOTIONS`.
- Missing font file → `FileNotFoundError`.
- skia-python without any typeface loader → `RuntimeError`.
- `render_frame` after `close()` → `RuntimeError`.
- Negative `t` → `ValueError`.

## Code

| Piece | Location |
|-------|----------|
| `KineticTypographyLayer`, `AlignedWord`, `WordMotion` | `pipeline/kinetic_typography.py` |
| Motion preset registry (`MOTION_PRESETS`, `SUPPORTED_MOTIONS`) | `pipeline/kinetic_typography.py` |
| Aligned JSON reader (`load_aligned_words`) | `pipeline/kinetic_typography.py` |
| Preset `typo_style` values feeding the motion name | `presets/*.yaml` |
| Word timings consumed by the layer | `cache/<song_hash>/lyrics.aligned.json` |

## Related

- Aligned lyrics schema and aligner: `docs/technical/lyrics-aligner.md`
- Reactive uniforms consumed by `scale-pulse` / `beat-shake`:
  `docs/technical/reactive-shader-layer.md`
- Preset `typo_style` field: `docs/technical/visual-style-presets.md`
