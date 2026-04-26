# Effects timeline data model

V1 data structures for a future **Effects timeline** editor: manual and baked effect clips, persisted with the per-song cache next to the lyrics alignment.

## Location and file

- **Module:** `pipeline/effects_timeline.py`
- **On disk:** `cache/<song_hash>/effects_timeline.json` (field `EFFECTS_TIMELINE_JSON`)

## Core types

- **`EffectKind`** — Seven clip kinds: `BEAM`, `LOGO_GLITCH`, `SCREEN_SHAKE`, `COLOR_INVERT`, `CHROMATIC_ABERRATION`, `SCANLINE_TEAR`, `ZOOM_PUNCH` (string-valued enum; JSON uses the same names).
- **`EffectClip`** — `id`, `kind`, `t_start`, `duration_s` (finite; duration > 0), per-kind `settings` dict, `auto_source` (baked-from-auto vs user).
- **`EffectsTimeline`** — ordered `clips`, per-kind `auto_enabled` (all kinds required), and `auto_reactivity_master` in **\[0, 2\]** (0–100% to 200% of analyser-driven reactivity; reserved for the compositor auto path only).

## Settings validation

`EFFECT_SETTINGS_KEYS` defines allowed keys per `EffectKind`. Unknown keys raise **`ValueError`**; values must be JSON-scalar (`str | int | float | bool | None`).

## Persistence

- **Schema:** `schema_version: 1` in JSON.
- **Atomic save:** write `effects_timeline.json.tmp`, then `os.replace` to `effects_timeline.json` (same pattern as the lyrics editor).
- **`load(cache_dir)`** — returns a default empty timeline if the file is missing; always validates the payload when present.

## Tests

- `tests/test_effects_timeline.py` — round-trip, bad settings, stale `.tmp` not corrupting a good file, etc.

## Related

- Design goals and editor UX: `.taskmaster/docs/prd-effects-timeline.txt`
- Downstream: compositor integration and per-effect renderers (future tasks).
