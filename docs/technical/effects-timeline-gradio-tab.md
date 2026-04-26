# Effects timeline Gradio tab

The **Effects timeline** tab in `app.py` (directly after **Lyrics timeline**) embeds the self-contained effects editor from `build_editor_html` / `load_editor_state` in `pipeline/effects_editor.py`. See `docs/technical/effects-timeline-editor.md` for the HTML/JS surface and `docs/technical/effects-editor-backend.md` for load/save/bake.

## Behaviour

- **State:** The editor’s JS object is `window._musicvids_effects_state` (constant `_EFFECTS_EDITOR_STATE_JS_VAR` in `app.py`). A hidden `gr.Textbox` is only used as a Gradio input slot; **Save** uses the single-step `js=` pattern so the payload is the third tuple element without a race (same fix as the lyrics **Save** button).
- **Audio URL:** Full-song preview uses `cache/<hash>/analysis_mono.wav` if present, else `original.wav`, resolved by `_resolve_wav_path_for_effects_editor` and served via `/_file=` and `allowed_paths` (see `app.main` → `CACHE_DIR`).
- **Handlers:** `_load_effects_editor`, `_save_effects_editor` (save then re-render; returns both `html` and `run_log` — no extra `.then` reload), `_bake_effects_editor` (`bake_auto_schedule` then reload), `_clear_effects_editor` (`load` → clear `clips` list → `save` so per-kind **auto** toggles and `auto_reactivity_master` stay).
- **Tests:** `tests/test_app_effects_tab.py` smoke-imports the handler symbols.

Renders and preview do **not** yet read these inputs from `OrchestratorInputs` — that wiring is a separate orchestrator task.
