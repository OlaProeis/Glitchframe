# Background modes (SDXL stills, Ken Burns, AnimateDiff)

Feature: selectable full-frame backgrounds for the compositor via a shared
`BackgroundSource` API (`ensure()` / `background_frame(t)` / `close()`).

See also `docs/technical/background-stills.md` for the SDXL keyframe cache and
interpolation details.

## Modes

| Canonical ID | Module / class | Notes |
|--------------|----------------|-------|
| `sdxl-stills` | `pipeline.background_stills.BackgroundStills` | Default; `manifest.json` + `keyframe_*.png`. |
| `static-kenburns` | `pipeline.background_kenburns.StaticKenBurnsBackground` | `manifest_static_kenburns.json`, `source.<ext>`; RMS from `analysis.json` drives zoom/pan/tilt. |
| `animatediff` | `pipeline.background_animatediff.AnimateDiffBackground` | `manifest_animatediff.json` (schema v2), `anim_{seg}_{f}.png`; requires CUDA + diffusers with `AnimateDiffSDXLPipeline`. |

## Factory and orchestration

- **Source modules:** `pipeline/background.py` (protocol + factory), `background_stills.py`, `background_kenburns.py`, `background_animatediff.py`; UI/inputs: `app.py`, `orchestrator.py` (`background_mode`, `static_background_image`).
- `pipeline.background.create_background_source(mode, cache_dir, preset_id=..., preset_prompt=..., static_image_path=...)` returns a concrete source.
- `pipeline.background.normalize_background_mode` maps UI labels to canonical IDs (raises on unknown).
- `OrchestratorInputs.background_mode` and `OrchestratorInputs.static_background_image` carry user choices into full render / preview (non-cache-key metadata).

## AnimateDiff prompt construction

AnimateDiff uses a **dedicated motion-prompt builder** (`_build_motion_prompt` in
`pipeline/background_animatediff.py`), separate from the SDXL stills keyframe
builder (`_build_keyframe_prompt` in `background_stills.py`):

- **Stills** append structural hints (`scene N of M, song section K of C,
  t=X.Xs`) to diversify still keyframes; harmless for image diffusion.
- **AnimateDiff** skips structural hints — the motion adapter treats them as
  content and drifts off-topic. Instead every loop gets:
  1. The preset's scene prompt (from YAML `prompt`).
  2. A preset-specific **motion flavor** from `MOTION_FLAVORS` (e.g. "slow
     cosmic drift, subtle parallax between dust layers" for `cosmic`). Unknown
     preset ids fall back to `DEFAULT_MOTION_FLAVOR`.
  3. A **pacing cue** that varies by song position: `establishing shot, slow
     motion` in the first quartile, `steady motion` through the middle,
     `slower fade-out motion` in the last quartile (`_pacing_cue`).
  4. A short quality tail (`cinematic, high detail, coherent frames`).

### Inference & negative prompt

- `DEFAULT_NUM_INFERENCE_STEPS = 35` (bumped from the stills default of 28).
  The SDXL-beta motion adapter needs more denoising steps before temporal
  attention settles; below ~32 frames often look soft or ghosted.
- `ANIMATEDIFF_NEGATIVE_PROMPT` extends the stills `DEFAULT_NEGATIVE_PROMPT`
  with motion-specific failure terms: `static frame, frozen motion, stutter,
  duplicate frames, jerky camera, hard cut, scene cut, flickering, morphing
  shapes, distorted proportions, rolling shutter`.

### Cache invalidation

Any change to prompts, motion flavors, inference steps, negative prompt,
model ids, or resolution bumps the hash or schema and invalidates existing
`manifest_animatediff.json` files. Phase 4 ships **schema v2**; v1 caches are
ignored by `matches_key` and regenerated on first run.

## Errors

No silent fallback between modes: missing uploads, missing `analysis.json`, missing
CUDA (AnimateDiff / SDXL), missing diffusers motion stack, or OOM all surface as
explicit exceptions per project policy.
