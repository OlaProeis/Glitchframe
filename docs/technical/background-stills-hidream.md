# HiDream-O1-Image background stills (alternative to SDXL)

Optional, opt-in image-generation backend for AI background stills, sitting
behind the same `BackgroundSource` API as `pipeline/background_stills.py` so
RIFE morph, Ken Burns, the keyframes editor, and the manifest cache all
work unchanged. SDXL stays the default; users pick HiDream from the
**Image generator** radio in the *Background keyframes* tab.

## Why an out-of-process worker

HiDream-O1-Image ([HiDream-ai/HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image))
ships with its own heavy CUDA / `flash-attn` / `transformers` pin set that
conflicts with our main app's `torch == 2.2.2+cu121` Windows path
(`pyproject.toml`). Importing HiDream into Glitchframe's venv would force
upgrades that break WhisperX / Demucs / SDXL.

Instead, we spawn HiDream's *own* venv (Pinokio installs one at
`<pinokio>/api/github-com-cocktailpeanut-hidream-o1/HiDream-O1-Image/env/`)
as a subprocess and stream JSONL jobs over stdin/stdout — see
`pipeline/background_stills_hidream_worker.py`. The worker loads the 8 B
HiDream pipeline once and processes every keyframe in a batch, so we do not
pay a 30-90 s reload per image.

## Configuration

Set these in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `GLITCHFRAME_HIDREAM_PYTHON` | Path to HiDream venv's `python.exe` |
| `GLITCHFRAME_HIDREAM_REPO` | Path to a checkout of `HiDream-ai/HiDream-O1-Image` |
| `GLITCHFRAME_HIDREAM_MODEL_PATH` | Path to weights directory (e.g. drbaph FP8 dev) |
| `GLITCHFRAME_HIDREAM_MODEL_TYPE` | `dev` (28 steps, ~16 GB on FP8) or `full` (50 steps) |
| `GLITCHFRAME_HIDREAM_GEN_WIDTH` / `_GEN_HEIGHT` | Generation resolution (default 1280×720) |
| `GLITCHFRAME_HIDREAM_PIPELINE_IMPORT` | `auto` (default), or `module:Class` for a fork |

If any of the first three is unset, switching the radio to **HiDream**
raises a clear error at generation time. SDXL keeps working regardless.

## Cache namespacing

The on-disk layout matches the SDXL path:

```
cache/<song>/background/
  manifest.json
  keyframe_0000.png
  keyframe_0001.png
  ...
```

The `manifest.json` `model_id` field is the cache namespace:

| Backend | `model_id` |
|---|---|
| SDXL  | `stabilityai/stable-diffusion-xl-base-1.0` |
| HiDream | `hidream:<model_type>:<weights_path_hash>` (12-char SHA-256 prefix) |

`BackgroundManifest.matches_key` compares `model_id` strictly, so switching
the radio between SDXL and HiDream invalidates the cache for that song and
forces regeneration with the chosen backend. Switching back later
(without changing the weights path) restores the previous backend's cache.

## Wire protocol

`pipeline/background_stills_hidream.py::_HiDreamWorker` writes JSONL jobs
to the worker subprocess's stdin and reads JSONL events back:

```
→ {"index": 0, "prompt": "...", "output_path": ".../keyframe_0000.png", "seed": 12345}
← {"event": "ready"}
← {"event": "step", "index": 0, "step": 1, "steps_total": 28}
← ...
← {"event": "saved", "index": 0, "path": "..."}
← {"event": "error", "index": 0, "message": "..."}
```

Worker stderr is forwarded to the Glitchframe Gradio log so HiDream warnings
and tracebacks remain visible without corrupting the JSONL channel.

## Resume + RIFE compatibility

`BackgroundStillsHiDream` only overrides `_generate_and_persist`. Every
other behaviour — partial-resume of half-baked PNGs, RIFE morph between
keyframes, Ken Burns on stills, `background_frame(t)` interpolation, the
keyframes editor's per-clip preview — is inherited from `BackgroundStills`.
A render that mixes a few HiDream keyframes with later RIFE-densified
frames behaves identically to the SDXL path.

## Test plan (manual, since the worker needs HiDream installed)

1. Install HiDream-O1-Image (Pinokio one-click app or manual `pip install -r
   requirements.txt`); confirm `python inference.py --model_type dev …`
   succeeds for one prompt.
2. Set the four `GLITCHFRAME_HIDREAM_*` env vars in `.env`.
3. Ingest + analyse a song; open the **Background keyframes** tab; switch
   **Image generator** to *HiDream-O1-Image*; click **Generate stills**.
4. Verify `cache/<song>/background/manifest.json` shows
   `"model_id": "hidream:dev:..."` and the PNGs are visibly higher
   prompt-fidelity than the SDXL bake.
5. Toggle the radio back to *SDXL*, re-click **Generate stills**; cache
   regenerates with `model_id` `stabilityai/...`.
6. Switch back to *HiDream* a third time; the manifest match should
   short-circuit and reuse the previously generated HiDream PNGs.
