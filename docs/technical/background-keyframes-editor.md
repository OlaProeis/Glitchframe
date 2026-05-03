# Background keyframes editor (Gradio timeline)

Visual editor for **SDXL background stills**: one waveform timeline with draggable clips, per-clip prompts, preview at the playhead, and actions wired to Gradio (regenerate one slot, replace/crop uploads). Persisted data lives next to the existing stills cache under `cache/<hash>/`.

## Scope

- **Gradio tab:** **Background keyframes** in `app.py` (after ingest + **Analyze**, use **Load timeline**).
- **HTML/JS:** `pipeline/keyframes_editor.py` — `build_keyframes_editor_html`, shared peaks via `compute_peaks`, in-browser state on `window._glitchframe_keyframes_state`.
- **Disk:** `cache/<hash>/keyframes_timeline.json` (ordered entries: id, time, prompt, source), `background/manifest.json` + `keyframe_*.png` (see [`background-stills.md`](background-stills.md)). Optional **`background/upload_staging/<entry_id>.png`** before **Save timeline** commits uploads to the numbered stills.
- **Validation / persistence helpers:** `pipeline/keyframes_timeline.py` (`validate_timeline_entries`, `persist_timeline_and_manifest`, `refresh_manifest_from_timeline`, staging merge on save).

## User flow

1. **Load timeline** — needs `analysis.json` + audio in cache; builds editor state from disk timeline or plans from analysis; may write an initial `keyframes_timeline.json`.
2. **Save timeline** — parses the editor JSON from the browser (`JSON.stringify(window._glitchframe_keyframes_state)`), writes timeline + manifest (same path as `save_keyframes_editor_payload`); **clears RIFE morph cache** when manifest changes.
3. **Generate SDXL stills** — fills missing PNGs for the current manifest order; optional “regenerate all SDXL slots”.
4. **Per-clip (inline):** edit prompt, **Regenerate (SDXL)** for the selected slot, **Replace with image…** / **Crop keyframe** (upload + cropper + **Apply crop**). Inline buttons trigger hidden Gradio controls so the server sees the same handlers as the tab.
5. **Selection** — click a clip to select it; `state.selected_target_id` is kept in sync for the payload so targeting works even when the filterable dropdown lags for new or custom ids. Regenerate / generate / apply-crop paths **auto-sync** in-memory edits to disk (`save_keyframes_editor_payload`) before resolving slot indices so prompts and timing match what you see.

### Crop tool (`build_crop_canvas_html`)

- The red selection rectangle is **locked to the Output resolution** aspect ratio (`Output resolution` dropdown in the Visual style tab → parsed width×height passed into the cropper). Dragging resizes with that fixed ratio so **Apply crop** resizes into `out_width×out_height` **without non-uniform stretch**.
- The initial crop after load is the **largest centered** rectangle with that aspect ratio that fits the displayed image (not necessarily full-bleed when the image aspect differs).

### Windows: saving PNGs (`*.tmp` → final path)

- **Apply crop** and **SDXL keyframe writes** use a temp file then rename/replace into `upload_staging/*.png` or `background/keyframe_*.png`. On Windows, replacement fails if another handle still has the destination open (e.g. **same path opened for read** in PIL, or the **browser / Gradio** preview holding a read lock on a `/file=…` URL).
- The pipeline **closes** the source `Image` before replacing when the source and destination can be the same file, and uses **short retries + `gc.collect()`** on access denied when swapping the temp file over the final name. If errors persist, retry after a moment or avoid leaving Explorer preview or other viewers open on those PNGs.

## Limits (current product)

- **Keyframe count** is driven by the analyzer/plan (roughly one still per ~8 s of audio). The UI **does not** offer “add keyframe” — extra clips would desync `manifest.json` / `keyframe_*.png` numbering until the pipeline fully supports growing the stack.
- **RIFE / render** still assume a coherent manifest after **Save timeline**; staged uploads are visible in the editor preview from `upload_staging/` but are fully committed only on save (see timeline module).

## Implementation notes

- **Gradio bridges:** prompt + proxy buttons use `elem_id` (`mv_kf_gr_prompt`, `mv_kf_btn_regen`, …) and stay **mounted in the DOM** (off-screen CSS) so timeline JS can sync the prompt and call `click()` on the real buttons; `visible=False` Gradio components are omitted from the tree and broke inline actions.
- **Windows / asyncio:** `app.py` sets `WindowsSelectorEventLoopPolicy` where possible and filters noisy `ConnectionResetError` / proactor `connection_lost` logs from local Gradio/WebSocket churn.

## Related files

| File | Role |
|------|------|
| `app.py` | Tab layout, `_load_keyframes_editor`, `_save_keyframes_editor`, SDXL regen/generate, crop handlers, bridge `js=` preprocessors |
| `pipeline/keyframes_editor.py` | Editor HTML/CSS/JS, `load_keyframes_editor_state`, `save_keyframes_editor_payload`, `generate_sdxl_keyframes_for_cache`, upload crop → staging |
| `pipeline/keyframes_timeline.py` | Timeline model, validation, manifest refresh, staging merge |
| `pipeline/background_stills.py` | `BackgroundStills`, `keyframe_*.png` layout |

See also: [`background-stills.md`](background-stills.md), [`background-modes.md`](background-modes.md), [`rife-morph-background.md`](rife-morph-background.md).
