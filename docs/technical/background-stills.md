# SDXL background stills

Feature: AI-generated background track for the compositor — `N = ceil(duration / 8 s)`
SDXL FP16 keyframes driven by the active preset's `prompt` + per-section modifiers
from `analysis.json`, persisted under `cache/<hash>/background/`, with a
crossfade-interpolated `background_frame(t)` callable so the compositor can sample
any wall-clock time.

## Flow

1. Read `cache/<hash>/analysis.json` — requires `duration_sec` and a non-empty
   `segments` array (from `pipeline/audio_analyzer.py`). Missing / malformed JSON
   raises.
2. Plan keyframes: `N = max(1, ceil(duration_sec / DEFAULT_KEYFRAME_INTERVAL))`.
   Times are evenly spaced so `t_0 = 0` and `t_{N-1} = duration_sec`; `N == 1`
   drops a single keyframe at the song midpoint. Each plan entry looks up the
   containing segment and builds a deterministic prompt
   `"<preset prompt>, scene {i+1} of {N}, song section {label+1} of {S}, t={sec}s"`.
3. Compute the cache manifest key
   `(preset_id, prompt_hash, section_count, num_keyframes, model_id, width, height)`.
   `prompt_hash` is SHA-256 over the preset prompt, model id, generation resolution,
   and the full list of per-keyframe prompts. When a matching manifest + all
   `keyframe_{i:04d}.png` files are on disk, the renderer reuses them.
4. On miss (or `force=True`) load `StableDiffusionXLPipeline.from_pretrained(
   model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
   cache_dir=config.MODEL_CACHE_DIR, add_watermarker=False)` and move to `cuda:0`.
   The pipeline is configured with VAE slicing/tiling and xformers memory-
   efficient attention (falling back to attention slicing) so peak VRAM stays
   bounded on 12–24 GB cards.
5. Keyframes are generated **one at a time** (batch size = 1) in a Python loop
   with `callback_on_step_end` reporting per-denoising-step progress to the UI.
   Batched generation (list of prompts in a single pipe call) scales activation
   memory with `len(prompts)` and silently spills into shared system RAM on
   24 GB GPUs, making a 17-keyframe song take 20+ minutes. A serial loop keeps
   VRAM flat and finishes in ~3 minutes on an RTX 3090. Per-keyframe seeds are
   derived from `(seed + keyframe_index)` so resumed runs reproduce the same
   images as a fresh run.
6. Images are written atomically (`keyframe_{i}.png.tmp` → `.replace()`) at native
   generation resolution immediately after each one is generated, Lanczos-resized
   to the output size in memory, and the manifest is written atomically as
   `background/manifest.json` only after every keyframe finishes. A crash or
   cancel mid-run leaves valid PNGs on disk; the next call resumes by skipping
   any keyframe whose PNG is already readable and only invoking SDXL for the
   missing indices.
6. `background_frame(t) -> np.ndarray (H, W, 3) uint8` clamps `t` to
   `[t_0, t_{N-1}]`, finds the bracketing pair, and returns a crossfade blend
   weighted by `smoothstep((t - t_i) / (t_{i+1} - t_i))`.

## Cache layout

```
cache/<hash>/background/
  manifest.json           # see BackgroundManifest schema
  keyframe_0000.png       # native SDXL generation resolution (1344x768 by default)
  keyframe_0001.png
  ...
```

`manifest.json` schema (v1):

```json
{
  "schema_version": 1,
  "preset_id": "neon-synthwave",
  "prompt_hash": "<sha256>",
  "section_count": 8,
  "num_keyframes": 27,
  "duration_sec": 212.43,
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "width": 1344,
  "height": 768,
  "keyframe_times": [0.0, 8.17, 16.34, ...],
  "prompts": ["<preset prompt>, scene 1 of 27, song section 1 of 8, t=0.0s", ...]
}
```

## Usage

```python
from pipeline.background_stills import BackgroundStills

with BackgroundStills(
    cache_dir=cache / song_hash,
    preset_id="neon-synthwave",
    preset_prompt=preset["prompt"],
) as bg:
    bg.ensure_keyframes()               # generate or reuse cache
    frame = bg.background_frame(42.0)   # (1080, 1920, 3) uint8 RGB
```

`ensure_keyframes(force=True)` forces regeneration even on a matching manifest;
`progress` is an optional `(p_in_[0,1], message)` callback compatible with the
rest of the pipeline.

## Strict error policy

- `analysis.json` missing / corrupt → `FileNotFoundError` / `ValueError`.
- Duration non-positive, segments empty, preset id empty, or bad gen resolution →
  `ValueError`.
- No CUDA device available → `RuntimeError` at generation time.
- `torch.cuda.OutOfMemoryError` → `RuntimeError` with guidance to lower gen
  resolution or batch size.
- Diffusers or torch import failure → `RuntimeError`.

## Dependencies

- Always used: `diffusers`, `torch`, `pillow`, `numpy` (already in core deps).
- Model weights download on first run under `MODEL_CACHE_DIR` (env overridable
  via `GLITCHFRAME_MODEL_CACHE` (legacy `MUSICVIDS_MODEL_CACHE`); HF also respects `HF_HOME` / `TORCH_HOME`).

## Related files

| File | Role |
|------|------|
| `pipeline/background_stills.py` | `BackgroundStills`, `BackgroundManifest`, `plan_keyframes`, `prompt_hash` |
| `pipeline/audio_analyzer.py` | Writes `analysis.json` with `duration_sec` + `segments` |
| `config.py` | `MODEL_CACHE_DIR`, per-song `song_cache_dir` helper |
| `presets/*.yaml` | Source of the `prompt` field fed into each keyframe |
| `cache/<hash>/background/` | Persisted keyframe PNGs + `manifest.json` |
