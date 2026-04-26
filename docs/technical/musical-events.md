# Musical events: drops, build-up tension, and band transients

Analysis-time enrichment that gives the reactive shader layer more musical
signal than the original four uniforms (`beat_phase`, `rms`, `onset_pulse`,
`band_energies[8]`). Implemented in:

- `pipeline/musical_events.py` — drop detector + build-up tension series +
  post-drop `drop_hold` sampler; output persisted under `analysis.json` →
  `events`.
- `pipeline/beat_pulse.py` — generalised `build_band_pulse_track` plus
  low / mid / high transient envelope builders for the shader layer.
- `pipeline/audio_analyzer.py` — `ANALYSIS_SCHEMA_VERSION` bumped to **2**;
  older caches re-analyze once and then persist the new block.

The Phase 2 plumbing is now complete: `uniforms_at_time` emits `bar_phase`,
`onset_env`, and `build_tension` (tasks 33 / 35 / 36); the compositor
injects `bass_hit`, `transient_lo/mid/hi`, and `drop_hold` once per render
via `_shader_bass_track_for_analysis` / `_shader_transient_tracks_for_analysis`
/ `_drop_hold_fn_for_analysis` in `pipeline/compositor.py` (task 37); and
every bundled fragment shader declares the new uniforms (task 38). The only
remaining work is the per-shader consumption pass in task 42 — shaders
currently **read** `bass_hit` (plus the classic `beat_phase` / `rms` /
`onset_pulse` / `band_energies` set) and ignore the new envelopes until
that pass lands.

## Schema v2 (`analysis.json`)

The two structural additions:

```json
{
  "schema_version": 2,
  "events": {
    "drops": [ { "t": 48.733, "confidence": 0.91 }, ... ],
    "build_tension": { "fps": 30, "frames": 6480, "values": [...] },
    "build_window_sec": 6.0,
    "drop_hold_decay_sec": 2.0
  }
}
```

On load, `analyze_song` compares `schema_version` to the module constant and
triggers a full re-analyze if they differ (see
`pipeline/audio_analyzer.py::analyze_song`). Demucs vocal separation still
upgrades cache-in-place on a v2 hit — behaviour unchanged from v1.

## Drop detector (`detect_drops`)

Deterministic, no ML. Operates on `analysis.rms`, `analysis.spectrum`, and
`analysis.segments` — all of which the analyzer already produces.

1. Smooth `rms` over a **0.30 s** moving average (`DEFAULT_RMS_SMOOTH_SEC`).
2. Average the first two mel bands into a *bass* series and smooth with the
   same window.
3. Over a **0.60 s** lag window (`DEFAULT_LAG_SEC`), take the half-wave
   rectified step of each: `drms = max(0, rms[i] - rms[i - lag])` and
   `dbass = max(0, bass[i] - bass[i - lag])`.
4. Score = `drms * dbass` — this gates out mid-only transients (snare rolls,
   vocal shouts) that would otherwise fire the detector on build-ups.
5. Normalise by the **99th percentile** of the score series, then peak-pick
   strict local maxima above an absolute floor (`DEFAULT_SCORE_FLOOR = 0.25`)
   with a minimum spacing of **5 s** (`DEFAULT_MIN_DROP_INTERVAL_SEC`).
6. Each candidate must have a structural segment boundary within **±1.0 s**
   (`DEFAULT_SEGMENT_SNAP_SEC`), OR the normalised score must exceed
   `DEFAULT_SEGMENT_OVERRIDE_SCORE = 0.55` — very strong jumps are kept
   even when segmentation missed the boundary.
7. Cap at `DEFAULT_MAX_DROPS = 12` per song (keep the strongest, preserve
   time order).

Output: a list of `{"t": float, "confidence": float in [0, 1]}` dicts.
Missing / malformed inputs return `[]` rather than raising, so every
downstream consumer can treat drop detection as best-effort enrichment.

### Tuning knobs

Every constant above is exported from the module. In practice the tuning
trade-off is: **more permissive** (lower `SCORE_FLOOR`, larger snap window,
lower `MIN_DROP_INTERVAL_SEC`) gives more drops per song at the cost of
occasional false positives on build-ups; **stricter** settings miss subtle
entries on low-dynamic-range mixes (lo-fi, rock mastered flat).

## Build-up tension series (`compute_build_tension_series`)

For each drop `{t_d, c}`, a smoothstep ramp painted over the
`[t_d - build_window_sec, t_d]` range:

- At `t = t_d - build_window_sec` the value is `0`.
- At `t = t_d` the value is `c` (the drop's confidence).
- Immediately after `t_d`, the value **snaps back to 0** — the post-drop
  "release" is a different envelope (`drop_hold`, below) with a different
  time scale.
- Overlapping build windows take the `max` per frame.

Output shape mirrors `analysis.rms`: `{fps, frames, values}`, so shader
consumers can use the existing `_interp_scalar_series` helper in
`pipeline/reactive_shader.py` once Phase 2 lands.

Default `build_window_sec = 6.0` — long enough to "lock down" several bars of
motion before impact, short enough that the ramp is never confused for
steady-state.

## Post-drop afterglow (`sample_drop_hold`)

Called per frame, not a pre-rendered series:

```
drop_hold(t) = c * exp(-(t - t_d) / decay_sec)    for the most recent drop t_d <= t
             = 0                                   when no drop has fired yet
```

Default `decay_sec = 2.0` (`DEFAULT_DROP_HOLD_DECAY_SEC`) — roughly an
8-bar phrase at 120 BPM. Same shape as `onset_pulse` but at a longer
time scale: `onset_pulse` decays in ~0.5 s for per-transient flashes,
`drop_hold` decays in ~2.0 s for "still riding the drop" effects.

## Band transient envelopes (`pipeline/beat_pulse.py`)

`build_band_pulse_track(*, band_lo, band_hi, decay_sec, sensitivity,
norm_percentile)` is the single shared core that shapes a
`PulseTrack` from any contiguous mel-band slice:

1. Mean of `spectrum[:, band_lo:band_hi]` per frame.
2. Half-wave rectified first difference → only rising energy (attacks)
   survives.
3. One-pole exponential decay with `decay_sec` time constant.
4. Normalise by `norm_percentile` (default 90) with a half-max floor to
   guard near-silent tracks.

`build_bass_pulse_track` is now a thin wrapper over this core — the logo
pulse semantics are byte-identical (confirmed by
`tests/test_beat_pulse.py`). Three new convenience builders expose
shader-oriented presets:

| Builder | Bands | Decay | Purpose |
|---|---|---|---|
| `build_lo_transient_track` | `[0, 2)` | **0.34 s** | Kick / sub; deliberately longer decay than the logo bass pulse so backgrounds breathe instead of flickering |
| `build_mid_transient_track` | `[3, 6)` | **0.12 s** | Snare / clap / body |
| `build_hi_transient_track` | `[6, 8)` | **0.06 s** | Hats / cymbals / air; snaps fast for sparkle response |

All three are cheap (~ms for a 4-min song) and therefore built once per
render, not persisted. The compositor injects them into the per-frame
uniform dict alongside `bass_hit` — see
`pipeline/compositor.py::_shader_transient_tracks_for_analysis` and the
`transient_lo/mid/hi` entries in
`docs/technical/reactive-shader-layer.md`.

**Shape gate.** `build_lo/mid/hi_transient_track` default to `shape=True`
which post-processes the returned envelope with
`shape_reactive_envelope(deadzone=0.18, soft_width=0.12, gamma=1.3)`:

1. Values `<= 0.18` → hard zero (kills the chill-section leakage floor
   that otherwise constantly wobbles into shader motion).
2. Values in `(0.18, 0.30)` → smoothstep shoulder so motion eases in
   instead of snapping on.
3. Values above the knee → rescaled to span `[0, 1]` then compressed by
   `x ** 1.3` so mids drop more than peaks.

Real hits still peak near `1.0` because the top of the curve is the
identity — the gate only calms the between-hit amplitude. Pass
`shape=False` (or supply custom `shape_deadzone / shape_soft_width /
shape_gamma` kwargs) for legacy output / A/B debugging; the logo bass
pulse (`build_bass_pulse_track` / `build_logo_bass_pulse_track`) is
unaffected and still uses the linear envelope it always has.

**Logo kick punch.** The compositor also builds its own dedicated
`build_lo_transient_track` instance inside `_kick_punch_envelope_fn`
(separate from the shader-facing one so it can be scaled by
`logo_pulse_sensitivity` instead of `shader_transient_sensitivity`).
It feeds `kick_punch_scale_and_opacity` — a larger-budget variant of
`scale_and_opacity_for_pulse` — so clean kicks produce a visibly bigger
logo bump than the sustain-aware bass pulse can on its own. See
`docs/technical/title-and-beat-pulse.md` for the combining rules.

## Verification

- `tests/test_beat_pulse.py` — 30 tests, all pass after the refactor,
  confirms `build_bass_pulse_track` behaviour is preserved.
- Smoke test (see task 34 implementation notes): synthetic 30 s song with
  a sharp drop at 15 s → detector fires at 15.07 s, `build_tension` hits
  0.999 one frame before the drop and snaps to 0 after, `drop_hold`
  decays from 0.995 at the drop to 0.135 four seconds later, low / high
  transient tracks correctly split kicks from hats.

## Related

- `docs/technical/audio-analyzer.md` — base analysis pipeline (beats,
  onsets, mel spectrum, RMS, segments, demucs vocals) and the v2
  `events` block schema that persists the detector output.
- `docs/technical/reactive-shader-layer.md` — how the `onset_env`,
  `transient_lo/mid/hi`, `build_tension`, `drop_hold`, and `bar_phase`
  uniforms are mapped in, plus the shader authoring guide that says
  what each signal should drive visually.
- `docs/technical/frame-compositor.md` — compositor-scope injection of
  `bass_hit` / `transient_lo/mid/hi` / `drop_hold` (build-once-per-render
  helpers keep the per-frame path to scalar lookups).
- `docs/technical/title-and-beat-pulse.md` — logo / title pulse consumer
  of `build_bass_pulse_track` (unchanged by the refactor).
