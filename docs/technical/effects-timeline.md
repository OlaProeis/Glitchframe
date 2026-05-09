# Effects timeline data model

V1 data structures for a future **Effects timeline** editor: manual and baked effect clips, persisted with the per-song cache next to the lyrics alignment.

## Location and file

- **Module:** `pipeline/effects_timeline.py`
- **On disk:** `cache/<song_hash>/effects_timeline.json` (field `EFFECTS_TIMELINE_JSON`)

## Core types

- **`EffectKind`** — Nine clip kinds: `BEAM`, `LOGO_GLITCH`, `SCREEN_SHAKE`, `COLOR_INVERT`, `CHROMATIC_ABERRATION`, `SCANLINE_TEAR`, `FADE`, `PIXEL_SMEAR`, `BLOCK_GLITCH` (string-valued enum; JSON uses the same names).
- **`EffectClip`** — `id`, `kind`, `t_start`, `duration_s` (finite; duration > 0), per-kind `settings` dict, `auto_source` (baked-from-auto vs user).
- **`EffectsTimeline`** — ordered `clips`, per-kind `auto_enabled` (all kinds required), `auto_reactivity_master` in **\[0, 2\]** (0–100% to 200% of analyser-driven reactivity; reserved for the compositor auto path only), and **`ken_burns_rms_automation`**: sorted `{"t", "v"}` knots with `v` in **\[0, 2\]** — piecewise-linear envelope for SDXL Ken Burns RMS drive (empty → neutral `1.0` at all times).

## Settings validation

`EFFECT_SETTINGS_KEYS` defines allowed keys per `EffectKind`. Unknown keys raise **`ValueError`**; values must be JSON-scalar (`str | int | float | bool | None`).

Notable per-kind settings:

- **`FADE`** — `direction_mode` (`"out"` for fade-to-black, `"in"` for reveal-from-black; default `"out"`), `peak_alpha` in `[0, 1]` (default `1.0`, fully black), `ease_mode` (`"smoothstep"` default, or `"linear"`). Duration is the **clip length** — the longer the clip, the longer the fade.
- **`PIXEL_SMEAR`** — `intensity` in `[0, 1]`, `density` in `[0, 1]` (fraction of rows smeared per frame), `streak_length_frac` in `[0, 1]` (streak length relative to frame width).
- **`BLOCK_GLITCH`** — `intensity` in `[0, 1]` (fraction of blocks displaced), `block_size_px` (≥ 2, clamped to frame size), `displace_frac` (per-axis displacement cap as a fraction of `block_size_px`).

## Persistence

- **Schema:** `schema_version: 1` in JSON.
- **Atomic save:** write `effects_timeline.json.tmp`, then `os.replace` to `effects_timeline.json` (same pattern as the lyrics editor).
- **Fields:** besides `clips` / `auto_enabled` / `auto_reactivity_master`, **`ken_burns_rms_automation`** is an array of `{"t": number, "v": number}` with `v` in \[0, 2\].
- **`load(cache_dir)`** — returns a default empty timeline if the file is missing; always validates the payload when present.

## Backward compatibility

- **Deprecated kinds:** clips whose `kind` is in `_DEPRECATED_KIND_NAMES` (currently `{"ZOOM_PUNCH"}`) are silently dropped during load, so older `effects_timeline.json` files keep working after `ZOOM_PUNCH` was replaced by `FADE`.
- **`auto_enabled` migration:** any kind missing from a legacy `auto_enabled` map defaults to `True`, matching the default for new timelines.

## Tests

- `tests/test_effects_timeline.py` — round-trip, bad settings, stale `.tmp` not corrupting a good file, etc.

## Related

- Design goals and editor UX: `.taskmaster/docs/prd-effects-timeline.txt`
- Downstream: compositor integration and per-effect renderers (future tasks).
