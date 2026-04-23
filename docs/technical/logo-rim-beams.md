# Rim-light beams on drops & snare lead-ins

`pipeline/logo_rim_beams.py` adds short, straight light beams that shoot outward from the logo rim on high-impact musical moments. The feature sits on top of the existing travelling-wave rim field (`pipeline/logo_rim_lights.py`) and reuses the same colour pipeline so beams visually spring from the rim they sit on.

Goals
-----

- **Earned, not spammy.** A global gate enforces ~1 burst per 10 s so beams read as reward beats, not ambient decoration.
- **Musically choreographed.** When a drop is preceded by a classic 2-3 snare roll the scheduler fires one beam per snare and a thicker beam on the drop itself (up to 4 beams per group). Drops without a roll get a single beam.
- **Deterministic.** Angles, colour-layer cycling, and jitter are all seeded by `song_hash + event index` so the same render produces the same choreography.

## Scheduling (`schedule_rim_beams`)

`schedule_rim_beams(analysis, *, snare_track, impact_track, cfg, song_hash, n_color_layers)` returns a flat, time-sorted list of `ScheduledBeam` records. It runs **once per render** inside `_build_beam_render_context` (see `pipeline/compositor.py`).

Algorithm:

1. Pull drop times from `analysis["events"]["drops"]` (see `docs/technical/musical-events.md`).
2. For each drop `T` peak-pick the **snare envelope** (`build_snare_glow_track`) inside `[T - lead_in_window_sec, T)` using strict local maxima ≥ `lead_in_snare_threshold` (default `0.55`) with a minimum spacing of `lead_in_min_spacing_sec` (default `0.12 s`). Keep the top `lead_in_max_beams` (default `3`) by strength.
3. Append one **drop beam** at `T` with thicker/longer geometry and full intensity.
4. Separately peak-pick the **RMS impact track** (`build_rms_impact_pulse_track`) for standalone impacts above `standalone_impact_threshold` (default `0.70`). Drop any peak within `standalone_impact_exclusion_sec` of an already-scheduled drop.
5. Apply the group-interval gate: drop any group whose **start** is less than `min_group_interval_sec` (default `10 s`) after the previous accepted group's **end**.

Angles are seeded per-drop from `song_hash`: a random base angle, ~120° between successive beams in the same group, each jittered by `± angle_jitter_rad` (≈26° by default). Colour-layer indices cycle through `0..n_color_layers - 1` so beams track the active rim palette.

`BeamConfig` exposes every knob; `BeamConfig(enabled=False)` returns an empty schedule so the compositor can short-circuit.

## Per-frame patch (`compute_beam_patch`)

`compute_beam_patch((H, W), *, centroid_xy, t, scheduled, rim_rgb, cfg, color_spread_rad, song_hash, hue_drift_per_sec, n_color_layers)` returns a padded premultiplied-RGBA patch (or `None` when no beam is active at `t`). Key steps:

1. Filter the schedule to beams where `0 <= t - t_start <= duration_s`; empty → return `None`.
2. Compute the patch bounding box around the active beams + the logo centroid (axis-aligned, clipped to the frame).
3. For each active beam, evaluate an envelope `linear-attack(0.04 s) · exponential-decay` and paint the beam as:
   - along-axis profile: fast rise from the centroid (~10 % of length), then tail into the tip,
   - across-axis profile: Gaussian sigma = `thickness_px`.
4. Accumulate contributions as straight sRGB `alpha · tint` + per-pixel max alpha so the premultiplied invariant (`rgb <= alpha`) holds by construction.
5. A light 1.8 px Gaussian blur softens pixel-grid aliasing.
6. Return `BeamPatchResult(patch, x0, y0)` ready for `_blend_premult_rgba_patch` (the same helper the rim patch uses).

Colours come from `_layer_srgb_tints` in `pipeline/logo_rim_lights.py`, so beams automatically inherit `color_spread_rad` + `hue_drift_per_sec` from the resolved `RimLightConfig`. When the travelling rim is disabled but beams are on, the scheduler still fires; the tint falls back to the preset's shadow/base hex.

## Compositor wiring

`CompositorConfig`:

- `rim_beams_enabled: bool = True` - master switch.
- `rim_beams_config: BeamConfig = BeamConfig()` - fine-grained tuning; rarely edited outside tests.

`_build_beam_render_context` (called once per render in both `render_single_frame` and `render_full_video`):

- Skips when beams are disabled, no logo is loaded, or the schedule is empty.
- Captures the rim colour context (`rim_rgb`, `color_spread_rad`, `n_color_layers`, `hue_drift_per_sec`) from the resolved rim config (or preset defaults when rim is off).
- Runs `compute_logo_rim_prep` once to get the logo centroid in prepared-logo pixel space.

`_render_compositor_frame` then projects the prepared centroid through the per-frame logo scale + `_origin_for_position` to find the output-space centroid, calls `compute_beam_patch`, and blends the patch in place **after** the logo composite so beams always sit on top of the branding element they seem to emit from.

## UI

The Branding accordion in `app.py` exposes a single checkbox:

- **Rim beams on drops & snare rolls** (`OrchestratorInputs.rim_beams_enabled`, default on). No per-beam tuning is surfaced - advanced users can edit `BeamConfig` in code; the defaults are tuned for a centered 30 %-of-shorter-edge logo on a 1080p frame.

## Tests

`tests/test_logo_rim_beams.py` covers:

- Drop-with-snare-lead-in produces a 4-beam group (3 snares + drop).
- Drop without snare track still fires a single drop beam.
- Group-interval gate drops a 5 s-separated second group but preserves a 12 s-separated one.
- Schedule is bit-exact deterministic under a repeated run with the same `song_hash`.
- Standalone impact within 1 s of a drop is consumed by the drop group.
- `compute_beam_patch` returns `None` before `t_start` / after `t_start + duration_s`, produces non-zero alpha on active beams, and satisfies the premultiplied invariant.

`tests/test_compositor_reactive_tracks.py::TestBuildBeamRenderContext` asserts the compositor short-circuits when beams are disabled, the logo is missing, or the schedule is empty.

## Out of scope / future work

- **Lightning / zig-zag variant.** The `ScheduledBeam` record is renderer-agnostic; a future module can reuse the schedule with a different `compute_*_patch`.
- **Non-drop-non-snare triggers.** Vocal screams, bar downbeats, pitched transients - all possible once the analyser publishes them; plug them into `schedule_rim_beams` the same way `impact_track` does today.
