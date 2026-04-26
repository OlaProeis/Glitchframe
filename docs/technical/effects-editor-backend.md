# Effects editor backend (Python)

Module: `pipeline/effects_editor.py` — state loading, JSON save validation, and **bake** (auto events → :class:`EffectClip` rows) for a future Gradio/JS effects timeline. No HTML/JS in this file (that is task 50+).

## API

- **`load_editor_state(cache_dir, *, target_peak_width=…, compositor_config=…)`**  
  Returns a JSON-friendly `dict` with: `song_hash` (from the cache folder name), `schema_version`, `auto_reactivity_master`, `auto_enabled` (per-kind name → bool), `clips` (serialised like `effects_timeline.json`), `peaks` / `duration` / `sample_rate` (from :func:`compute_peaks` on `analysis_mono.wav` or `original.wav`), and **`ghost_events`**.

- **`save_edited_timeline(cache_dir, json_payload, song_hash_from_dir=None)`**  
  Accepts a JSON string, bytes, or dict. Strips UI-only fields (`peaks`, `ghost_events`, etc.) and persists only the timeline file keys. The canonical `song_hash` is always `Path(cache_dir).name`; if the payload or `song_hash_from_dir` disagrees, :class:`ValueError` is raised. Writes `effects_timeline.json` via :func:`save` in `effects_timeline.py`.

- **`bake_auto_schedule(cache_dir, *, compositor_config=…)`**  
  Appends auto-sourced clips when the per-kind `auto_enabled` flag is on:
  - **BEAM** from :func:`schedule_rim_beams` (same inputs as the compositor ghost path).
  - **LOGO_GLITCH** from RMS-impact peak picking.
  - **ZOOM_PUNCH** from `analysis["events"]["drops"]` times.
  - **SCREEN_SHAKE** from low-band (kick / sub) transient peaks via `build_lo_transient_track` — `amplitude_px` scales with peak strength, `frequency_hz` fixed at `6.0` so the burst reads as a single camera jolt.
  - **CHROMATIC_ABERRATION** from high-band (hat / cymbal) transient peaks via `build_hi_transient_track` — `shift_px` scales with peak strength, `jitter=0.35`, `direction_deg=0.0`.
  Skips a candidate when an existing clip of the **same** :class:`EffectKind` has `t_start` within **`DEDUPE_TOL_S` (0.02 s = 20 ms)**.

- **`build_ghost_events(analysis, *, song_hash, compositor_config=…)`**  
  Produces the `ghost_events` list used by `load_editor_state` (and callable on its own for tests). Emits ghost rows for every auto source the baker can fire (BEAM, LOGO_GLITCH, ZOOM_PUNCH on drops, SCREEN_SHAKE on kick peaks, CHROMATIC_ABERRATION on hat peaks) so the editor can show analyser hints on the matching row.

## Related

- Data model: `docs/technical/effects-timeline.md`, `pipeline/effects_timeline.py`
- Waveform downsampling: `docs/technical/waveform-peaks.md`, `pipeline/_waveform_peaks.py`
- PRD (UX + ghost markers): `.taskmaster/docs/prd-effects-timeline.txt`
- Reference pattern: `pipeline/lyrics_editor.py` (lyrics state load/save)
