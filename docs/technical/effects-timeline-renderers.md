# Effects timeline — renderer passes (umbrella)

This page ties together the **V1 post-stack renderers** that consume `EffectClip` rows from `effects_timeline.json`, and how they connect to the compositor. For the JSON model and per-kind `settings` allowlists, see `docs/technical/effects-timeline.md`. For end-to-end preview/full wiring (orchestrator → `CompositorConfig`), see `docs/technical/orchestrator-effects-timeline-wiring.md` and `docs/technical/pipeline-orchestrator.md`.

## What runs today (frame stack)

| Kind | Module | Public function | Standalone doc |
|------|--------|-----------------|----------------|
| `ZOOM_PUNCH` | `pipeline/zoom_punch.py` | `zoom_scale(t, clips) → ≥ 1.0` | `docs/technical/zoom-punch-renderer.md` |
| `SCREEN_SHAKE` | `pipeline/screen_shake.py` | `shake_offset(t, clips, song_hash) → (dx, dy)` | `docs/technical/screen-shake-renderer.md` |
| `CHROMATIC_ABERRATION` | `pipeline/chromatic_aberration.py` | `apply_chromatic_aberration(frame, t, clips, song_hash) → ndarray` | `docs/technical/chromatic-aberration-renderer.md` |
| `SCANLINE_TEAR` | `pipeline/scanline_tear.py` | `apply_scanline_tear(frame, t, clips, song_hash) → ndarray` | `docs/technical/scanline-tear-renderer.md` |
| `COLOR_INVERT` | `pipeline/color_invert.py` | `invert_mix(t, clips) → [0, 1]` | `docs/technical/color-invert-renderer.md` |

These are the **full-frame post-pass**; they do not draw logo-only effects.

## Logo path (not the frame stack)

- **`BEAM`** — scheduled with analyser-driven rim beams; user clips merge in without the 10 s group gate. See `docs/technical/effects-timeline-compositor.md` and `docs/technical/logo-rim-beams.md`.
- **`LOGO_GLITCH`** — adds to the impact / glitch envelope that feeds logo compositing, not full-frame colour math. Same compositor doc.

## Compositor: fixed order

After reactive + typography + logo, the compositor may run `pipeline.compositor._apply_frame_effects` when `_build_frame_effects_context` returns a non-`None` cache. The order is documented in `docs/technical/effects-timeline-compositor.md`:

`ZOOM_PUNCH` → `SCREEN_SHAKE` → `CHROMATIC_ABERRATION` → `SCANLINE_TEAR` → `COLOR_INVERT`.

Each step **short-circuits** when the corresponding clip set is inactive at time `t` or the effect is neutral (e.g. zoom scale `≈ 1`, shake offset `≈ 0`, invert mix `≈ 0`), so an empty or irrelevant timeline stays a no-op (regression guard).

## Determinism and hashing

- **Screen shake**, **chromatic aberration**, and **scanline tear** use `song_hash` (and `clip.id` where applicable) so repeated renders with the same cache stay bit-stable.
- **Zoom punch** and **colour invert** are deterministic from clip geometry and `settings` alone.

## Cross-references

- **Compositor integration (merge rules, `auto_reactivity_master`, post-pass detail):** `docs/technical/effects-timeline-compositor.md`
- **Editor HTML/JS (seven rows, gear panel, `window._musicvids_effects_state`):** `docs/technical/effects-timeline-editor.md`
- **Backend load/save/bake:** `docs/technical/effects-editor-backend.md`
- **Shared waveform for the editor canvas:** `docs/technical/waveform-peaks.md`
