# Effects timeline in the compositor

Per-frame wiring of the **effects timeline** (`pipeline.effects_timeline`) into
the full-video compositor: user `EffectClip` rows merge into the existing auto
paths (rim beams, impact glitch) and drive a fixed-order post-stack frame
effects pass (`ZOOM_PUNCH → SCREEN_SHAKE → CHROMATIC_ABERRATION → SCANLINE_TEAR
→ COLOR_INVERT`). A single `auto_reactivity_master` scalar damps **only** the
auto envelopes, leaving user clip contributions unchanged.

## Config entry points (`CompositorConfig`)

- **`effects_timeline: EffectsTimeline | None`** — when `None`, the compositor
  produces byte-identical output vs pre-Task-49 `main` (regression bar).
- **`auto_reactivity_master: float = 1.0`** — `[0, 2]` multiplier threaded in
  from `EffectsTimeline.auto_reactivity_master` by the orchestrator; clamped
  to non-negative finite values (NaN / negatives collapse to `1.0`).

## Auto reactivity scaling

`auto_reactivity_master` scales these closures **once per render** via
`_scaled_pulse_fn` (one scalar multiply per frame, no new allocations):

- **Logo bass / beats pulse** — `_build_pulse_fn` → logo scale / opacity.
- **Snare envelope** — `_snare_envelope_fn` → snare glow + squeeze.
- **Auto impact envelope** — `_impact_envelope_fn` → logo RGB-split glitch
  (user `LOGO_GLITCH` contributions are added **after** the scale).
- **Rim audio modulation** — `RimAudioTuning.global_strength` is multiplied
  by the master inside `_create_rim_modulation_stepper`.

Shader uniforms (`bass_hit`, `transient_lo/mid/hi`, `drop_hold`) are
intentionally **not** scaled: they feed GLSL shaders, not the compositor's
auto reactivity stack. Reserved for a later task if the PRD is extended.

## Additive user cues

- **`BEAM` clips** (`pipeline/logo_rim_beams.py`) are converted in
  `_user_beam_schedule` into `ScheduledBeam` rows and appended to whatever
  `schedule_rim_beams` produces. User beams skip the 10 s group gate that
  throttles the analyser path, and the schedule also survives when
  `auto_enabled[BEAM] = False` (auto off, user still fires). Angles and
  colour layer indices derive deterministically from `clip.id + song_hash`.
  If the clip is **longer** than the default `BeamConfig.duration_sec` (hit
  length), the beam uses **sustain shaping**: strength ramps, a short
  hold, then halo size and afterglow (blur) grow toward the clip end; very
  short timeline ticks still use the floored hit envelope (unchanged).
- **`LOGO_GLITCH` clips** turn into a per-clip rectangular envelope in
  `_user_glitch_envelope_fn` (sum of active strengths, clamped to `[0, 1]`).
  `_combined_glitch_fn` adds the **scaled** auto impact envelope and the
  user envelope together, still clamped to `[0, 1]` before it reaches
  `composite_logo_onto_frame`.

## Post-stack frame effects pass

`_build_frame_effects_context(cfg, analysis)` pre-indexes clips per kind and
captures `song_hash`. It returns `None` (fast path, byte-identical to main)
when the timeline is absent, empty, or contains only `BEAM` / `LOGO_GLITCH`
clips (those live in the logo path, not the frame-effects pass).

`_apply_frame_effects(frame, t, fx)` runs **only when `fx` is non-None** and
walks the clip sets in fixed order, short-circuiting each step when its
contribution is neutral:

1. **`ZOOM_PUNCH`** — `pipeline.zoom_punch.zoom_scale(t, clips)` returns a
   scale `>= 1.0`; when strictly above `1`, `apply_zoom_scale` bilinear-
   resamples the frame and center-crops back to the configured resolution.
2. **`SCREEN_SHAKE`** — `pipeline.screen_shake.shake_offset(t, clips,
   song_hash)` returns a `(dx, dy)` offset; `apply_shake_offset` shifts the
   frame and fills the vacated border black (integer-rounded offsets).
3. **`CHROMATIC_ABERRATION`** — `pipeline.chromatic_aberration.apply_chromatic_aberration`
   shifts R and B in opposite directions along each clip's axis; G is fixed.
4. **`SCANLINE_TEAR`** — `pipeline.scanline_tear.apply_scanline_tear` applies
   per-clip horizontal band shifts (see `docs/technical/scanline-tear-renderer.md`).
5. **`COLOR_INVERT`** — `pipeline.color_invert.invert_mix(t, clips)`
   returns a `[0, 1]` weight; `apply_invert_mix` lerps the frame toward
   `255 - frame` in float32 (or hands back the cached invert at `mix ≈ 1`).

## Regression bar

An **empty** `effects_timeline` must produce the same frames as pre-Task-49
`main`. The pass is exercised by `tests/test_compositor_effects_timeline.py`:

- `_build_frame_effects_context` returns `None` for `None` / empty / beam-
  only timelines.
- `_apply_frame_effects(frame, t, None)` returns the input array unchanged
  (identity).
- A `SCANLINE_TEAR` clip is applied in the post-stack pass; `CHROMATIC_ABERRATION`
  does not log warnings in this path.
- A user `BEAM` clip surfaces in `_build_beam_render_context` even when the
  analyser schedule is empty or `auto_enabled[BEAM]` is `False`.
- Three user beams spaced 100 ms apart all survive the merge (no 10 s gate).
- `auto_reactivity_master = 0` collapses scaled auto envelopes to `0`.

Per-renderer contracts continue to live in `tests/test_zoom_punch.py`,
`tests/test_screen_shake.py`, `tests/test_chromatic_aberration.py`, `tests/test_scanline_tear.py`, and
`tests/test_color_invert.py`.

## Related

- `docs/technical/frame-compositor.md` — surrounding compositor pipeline.
- `docs/technical/effects-timeline.md` — on-disk data model.
- `docs/technical/effects-editor-backend.md` — editor load / save / bake.
- `docs/technical/zoom-punch-renderer.md` — ZOOM_PUNCH scale maths.
- `docs/technical/screen-shake-renderer.md` — SCREEN_SHAKE offset maths.
- `docs/technical/chromatic-aberration-renderer.md` — CHROMATIC_ABERRATION R/B split.
- `docs/technical/scanline-tear-renderer.md` — SCANLINE_TEAR band shifts.
- `docs/technical/color-invert-renderer.md` — COLOR_INVERT mix maths.
- `docs/technical/logo-rim-beams.md` — auto beam schedule + 10 s gate.
- `.taskmaster/docs/prd-effects-timeline.txt` — design + acceptance criteria.
