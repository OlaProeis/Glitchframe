# Effects timeline editor (HTML / JS)

Per-clip visual editor for `cache/<hash>/effects_timeline.json`. Mirrors the lyrics timeline editor (`docs/technical/lyrics-timeline-editor.md`): a single self-contained `gr.HTML` blob with inline vanilla JS; state round-trips through `window._glitchframe_effects_state`.

This doc covers only the **HTML factory** (`build_editor_html`). The data model lives in `docs/technical/effects-timeline.md`; state load / save / bake in `docs/technical/effects-editor-backend.md`. The **Gradio tab** that injects the HTML is **`Effects timeline` in `app.py`**, directly after **Lyrics timeline**: `Load timeline` / `Save edits` (single-step `js=` payload, same pattern as the lyrics save) / `Bake auto events` / `Clear all`. Audio uses `analysis_mono.wav` with fallback to `original.wav` via `/_file=` URLs under `allowed_paths` (the song cache).

## API

```python
from pipeline.effects_editor import build_editor_html, load_editor_state

html = build_editor_html(
    load_editor_state(cache_dir),
    audio_url="/file=/abs/path/to/analysis_mono.wav",
    container_id="mv_fx_root",
    state_js_var="_glitchframe_effects_state",  # default
)
```

Mirrors `pipeline.lyrics_editor.build_editor_html` exactly so the Gradio tab can reuse the same injection pattern. The returned string contains CSS, markup, an inline `<audio>` element, and an inert `<script type="text/plain">` pre-seeded with the JS. An `<img src="x" onerror="…">` re-runs the JS every time Gradio replaces `innerHTML` (which does *not* execute `<script>` tags on its own — the same pattern the lyrics editor uses).

## Layout

```
┌ toolbar ─────────────────────────────────────────────────────────────┐
│ [▶ Play/Pause]  [+] [−] [Fit]   [+ Beam] [+ Glitch] … [+ Zoom]       │
│                                                          info: hash …│
├ Master reactivity ───────────────────────────────────────────────────┤
│ 0 ───────█────── 200%     100%                                       │
├ Labels ─┬─ scroller (horizontal) ────────────────────────────────────┤
│ wave    │ waveform canvas (from state.peaks)                         │
│ Beam  ☑ │ ░░░░ BEAM clips + ghost ticks                              │
│ Glitch☑ │ ░░░░ LOGO_GLITCH clips + ghost ticks                       │
│ Shake ☑ │ ░░░░ SCREEN_SHAKE clips                                    │
│ Invert☑ │ ░░░░ COLOR_INVERT clips                                    │
│ Chrom.☑ │ ░░░░ CHROMATIC_ABERRATION clips                            │
│ Scan. ☑ │ ░░░░ SCANLINE_TEAR clips                                   │
│ Zoom  ☑ │ ░░░░ ZOOM_PUNCH clips + ghost ticks                        │
├ <audio controls> (src = audio_url) ──────────────────────────────────┤
│ help line                                                            │
└──────────────────────────────────────────────────────────────────────┘
```

Seven rows are always present, one per `EffectKind`, including `SCANLINE_TEAR`. Row label column is sticky — only the waveform + clip rows scroll horizontally.

## Interactions

| Input | Action |
|-------|--------|
| Toolbar `+ Beam` / `+ Glitch` / … | Create a clip of that kind at the playhead with defaults |
| Click inside a clip | Seek audio to `t_start` |
| Drag clip body | Move in time (group-move if multi-selected, shared clamped delta) |
| Drag left / right edge handle | Resize `t_start` / `t_end` |
| Click ⚙ gear | Open floating settings panel (closes on outside-click) |
| Click empty row / waveform | Seek audio |
| Click-drag empty area | Rubber-band select (Shift/Ctrl held = additive) |
| `Shift`/`Ctrl`+click | Toggle clip in selection |
| `Del` / `Backspace` | Delete selected |
| `Esc` | Clear selection, close settings panel |
| `Ctrl`/`⌘`+`A` | Select all |
| `Space` | Play / pause |
| `+` / `−` | Zoom in / out (`pxPerSec` clamped `[4, 600]`) |
| `1`–`7` | Add a clip for the *n*th effect row (Beam, Glitch, Shake, Invert, Chromatic, Scanline, Zoom — top to bottom) at the current playhead, same as the matching toolbar **+** button; for live “punch in” while audio plays. No modifiers. Ignored when focus is in an `input`, `textarea`, or `select`. |
| Per-row "auto" checkbox | Toggle `auto_enabled[kind]`; row gets hatched pattern when off |
| Master reactivity slider | Writes `auto_reactivity_master` in `[0, 2]` |

**Styling** — Clip elements use `user-select: none` so dragging does not
select the label text. The help line’s `<kbd>` elements use a light
foreground on a dark key-cap background (Gradio can render the block on
a light page; previously the key text could match the background).

## Settings panel

The floating panel is built dynamically from `EFFECT_SETTINGS_KEYS[kind]` (see `pipeline/effects_timeline.py`). Keys ending in `_hex` render as a colour picker, keys ending in `_mode` as a text input, all others as a number input with `step=0.01`. Unknown / legacy keys already on a clip are preserved — the server-side `validate_settings_for_kind` remains authoritative.

`t_start` and `duration_s` are always shown first so the user can nudge numbers when dragging is too coarse. NaN / empty inputs are ignored until the user types a parseable value (stale values fail server-side validation otherwise).

## Ghost markers

Faint vertical ticks on each row at the corresponding `state.ghost_events` times (generated server-side from `schedule_rim_beams`, RMS-impact peak picks, `analysis["events"]["drops"]`, low-band kick transient peaks for `SCREEN_SHAKE`, and high-band hat transient peaks for `CHROMATIC_ABERRATION`). Non-interactive; they're there so the user can tell what the analyser would fire without having to hit Bake.

## Waveform

The header canvas renders `state.peaks` (mono min/max pairs, default `DEFAULT_PEAK_WIDTH=6000`) as a filled min→max band with a top and bottom outline so individual kicks and snares are readable at the default zoom. When the stage is wider than `peaks.length` the draw routine picks the extreme of every bucket that falls inside the destination pixel, avoiding alias beating at deep zoom; when narrower it samples the nearest bucket so the fill stays continuous.

## State round-trip

`window[state_js_var]` (default: `_glitchframe_effects_state`) holds the full in-memory state, not just the persisted subset. On every mutation (add / move / resize / delete / settings edit / auto toggle / master slider), the JS updates the same object in place. The Gradio Save button serialises it with `JSON.stringify(window._glitchframe_effects_state)` and posts to `save_edited_timeline`, which strips UI-only fields (`peaks`, `ghost_events`, `duration`, `sample_rate`, `kind_*`, `settings_keys`) and keeps only `schema_version`, `song_hash`, `auto_reactivity_master`, `auto_enabled`, `clips`.

`song_hash` in the payload is advisory; the save handler always takes the canonical hash from the cache directory name.

## Defaults for toolbar-created clips

Kept in `_KIND_DEFAULTS` in `pipeline/effects_editor.py`. Where a renderer exposes a private `_DEFAULT_*` constant (zoom punch, screen shake, colour invert) the UI defaults match those values; the rest are sensible starting points only. Server-side validation is still the authority, so out-of-range values get caught on save.

## Where the HTML / JS lives

| Piece | Location |
|-------|----------|
| CSS | `_EFFECTS_CSS` in `pipeline/effects_editor.py` |
| Markup + factory | `build_editor_html` |
| JS | `_EFFECTS_JS` (placeholder-substituted at build time) |

Placeholders (`__MV_CONTAINER_ID__`, `__MV_STATE_JS_VAR__`, `__MV_AUDIO_ELEMENT_ID__`, `__MV_PAYLOAD_JSON__`, `__MV_PIXELS_PER_SECOND__`) are rewritten via `str.replace` so literal `%` operators inside the JS don't need Python printf escaping.

## Tests

- `tests/test_effects_editor.py::TestEffectsEditor::test_build_editor_html_smoke` — asserts the returned HTML contains the container id, the inert `<script type="text/plain">` tag, all seven row / toolbar / auto checkboxes, and the master slider hook.
- `tests/test_effects_editor.py::TestEffectsEditor::test_build_editor_html_custom_state_var` — confirms the `state_js_var` kwarg makes it into the emitted JS.
- The same module covers `load_editor_state`, `save_edited_timeline` (including omitted `song_hash`), and `bake_auto_schedule` (zoom dedupe, RMS-impact glitch bake, and related cases).

## Related

- Data model: `docs/technical/effects-timeline.md`, `pipeline/effects_timeline.py`
- Backend state load / save / bake: `docs/technical/effects-editor-backend.md`, `pipeline/effects_editor.py`
- Renderer stack (post-pass order, per-kind modules): `docs/technical/effects-timeline-renderers.md`
- Reference pattern (lyrics editor): `docs/technical/lyrics-timeline-editor.md`, `pipeline/lyrics_editor.py`
- Compositor integration: `docs/technical/effects-timeline-compositor.md`
- PRD (full UX, Gradio tab, orchestrator wiring): `.taskmaster/docs/prd-effects-timeline.txt`
