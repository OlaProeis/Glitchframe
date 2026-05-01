# Background modes (SDXL stills, Ken Burns, AnimateDiff)

Feature: selectable full-frame backgrounds for the compositor via a shared
`BackgroundSource` API (`ensure()` / `background_frame(t)` / `close()`).

See also `docs/technical/background-stills.md` for the SDXL keyframe cache and
interpolation details.

> **Status — AnimateDiff black-frame regression fixed.** The SDXL stock VAE
> silently NaNs in fp16 once the (16-frame × latent) tensor passes through
> decode, which surfaced as all-black PNGs. Fixed by loading
> [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
> (Apache-2.0, ~330 MB, drop-in replacement) and assigning it to `pipe.vae`
> after pipeline construction in `_load_pipe`. The default constructor uses
> `DEFAULT_FP16_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"`; override via the
> `vae_id=` kwarg. Old (v2 / no-`vae_id`) caches automatically invalidate
> because `vae_id` participates in `_prompt_hash_segments`.

## Modes

| Canonical ID | Module / class | Notes |
|--------------|----------------|-------|
| `sdxl-stills` | `pipeline.background_stills.BackgroundStills` | **Default.** `manifest.json` + `keyframe_*.png`. |
| `static-kenburns` | `pipeline.background_kenburns.StaticKenBurnsBackground` | `manifest_static_kenburns.json`, `source.<ext>`; RMS from `analysis.json` drives zoom/pan/tilt. |
| `animatediff` | `pipeline.background_animatediff.AnimateDiffBackground` | **AnimateDiff seeded by SDXL stills.** Wraps a `BackgroundStills` instance as `init_image_source` and uses each closest SDXL keyframe as the **init latent** for the matching AnimateDiff segment — the output IS the SDXL still, animated. No sample-time blending / overlay. `manifest_animatediff.json` (schema v2), `anim_{seg}_{f}.png`; requires CUDA + diffusers with `AnimateDiffSDXLPipeline`. Loads `madebyollin/sdxl-vae-fp16-fix` for fp16-safe decode. |

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
     cosmic drift, subtle parallax between dust layers" for `cosmic-flow`). Unknown
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
model ids, **VAE id**, init-image keyframe times, or resolution bumps the
hash or schema and invalidates existing `manifest_animatediff.json` files.
Phase 4 ships **schema v2**; v1 caches are ignored by `matches_key` and
regenerated on first run. The VAE-fix patch adds `vae_id` to
`_prompt_hash_segments`, so any cache produced before the fp16-safe VAE
swap (which would have contained black frames) automatically regenerates.
The cross-segment morph patch hashes each segment as the pair
`"<prompt>|||<prompt_2>"` so toggling prompt travel also invalidates the
cache. The init-image path hashes ``init_key`` as
``img2img-v1|s=<strength>|t=<comma-separated keyframe times>`` (or
``init=none`` when no stills cache), so changing img2img strength, keyframe
timing, or disabling stills invalidates cleanly.

## AnimateDiff seeded from SDXL stills (default for `animatediff` mode)

`AnimateDiffBackground` accepts an optional
`init_image_source: BackgroundSource`. The factory wires a `BackgroundStills`
instance as the source whenever `mode == "animatediff"`, so the user-facing
behaviour matches the spec:
**load SDXL → make stills → dump SDXL → load AnimateDiff → animate the stills.**

There is no sample-time blending or overlay. The AnimateDiff frames that go
to disk *are* the output; the SDXL still seeds img2img-style temporal
diffusion (see render step 5 below). **Segment count** follows `analysis.json`
musical sections (`_segments_from_analysis`), not SDXL keyframe count — you may
see ~8 AnimateDiff clips on a long song because the beat/spectrogram segmenter
produced eight sections; each gets one 16-frame loop seeded from the *nearest*
SDXL keyframe in time.

### Render flow

1. `ensure(progress)` runs the **init-image source** first with progress
   mapped to `[0.0, 0.4]`. This generates `keyframe_*.png` + `manifest.json`
   under `cache/<song_hash>/background/`.
2. The init source is **closed inline** (so its SDXL pipeline releases its
   ~5 GB of VRAM) and its `keyframe_times` are snapshotted to disk lookup.
3. AnimateDiff loads its pipeline (also fp16, also gets the fp16-safe VAE
   swap) with progress mapped to `[0.4, 1.0]`.
4. For each musical section, `_init_image_for_segment` picks the SDXL
   keyframe PNG whose time is closest to the segment start.
5. **Img2img init (critical):** Passing raw VAE latents as ``pipe(latents=…)``
   does *not* preserve the still. ``AnimateDiffSDXLPipeline.prepare_latents``
   assumes Gaussian noise scaled by ``init_noise_sigma``, then runs the **full**
   timestep schedule — clean latents land in the wrong diffusion state and the
   UNet hallucinates unrelated shapes (random smears / colors unrelated to the
   SDXL PNG). We instead mirror ``StableDiffusionXLImg2ImgPipeline`` inside
   ``_animatediff_img2img_generate``: slice ``scheduler.timesteps`` by
   ``init_image_strength``, run ``scheduler.add_noise`` on the VAE-encoded still
   at the first kept timestep (latent tiled across ``num_frames``), align latent
   scale with ``init_noise_sigma`` (``1.0`` on DDIM), then run only the suffix of
   the schedule in the same UNet loop as upstream AnimateDiff SDXL.
6. ``init_image_strength`` defaults to ``DEFAULT_INIT_IMAGE_STRENGTH`` (0.38).
   Higher ⇒ more denoising steps / more motion / freer deviation from the
   still; lower ⇒ sticks closer to the keyframe. Allowed range on the class:
   ``[0.05, 1.0]``.
7. If the SDXL cache is missing, `_init_image_for_segment` returns `None` and
   the segment falls back to plain text-to-video (logged).

### Cross-segment prompt morph

Each AnimateDiff segment also passes the **next** segment's motion prompt
to SDXL's second text encoder via `pipe(prompt=..., prompt_2=...)`. The
dual-encoder design conditions the loop on a blend of the two, softening
section boundaries on top of the init-image seed. The mapping lives in
`_prompt_2_for_index` and participates in the manifest hash. The last
segment uses its own prompt as `prompt_2` (no morph past the song end).

### Tuning

- Drop ``init_image_source`` from the factory for classic text-to-video
  AnimateDiff (no SDXL pre-pass).
- Adjust ``init_image_strength`` on ``AnimateDiffBackground`` when constructing
  the source (not exposed in Gradio yet); it participates in ``init_key`` so
  caches regenerate when it changes.
- ``init_image_source`` is not held after ``ensure()`` returns; keyframes are
  read from disk per segment.

## Errors

No silent fallback between modes: missing uploads, missing `analysis.json`, missing
CUDA (AnimateDiff / SDXL), missing diffusers motion stack, or OOM all surface as
explicit exceptions per project policy.
