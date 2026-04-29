# Audio-pulsing dark vignette

A subtle radial darkening of the frame edges that breathes with the music, applied between the reactive shader composite (background + reactive shader + voidcat ASCII) and the typography / title / logo passes.

## Why

Two problems were rolled into one feature:

1. **Contrast vs. SDXL backgrounds.** Once the SDXL stills returned to the renderer, the reactive shaders sometimes read as a thin haze on top of busy backdrops. A whisper of edge darkening pulls the eye toward the centre of the frame and gives every preset a baseline of contrast without per-shader tuning.
2. **A "smooth" pulse layer.** Add a globally-applied dark frame that breathes on bass + post-drop afterglow so the whole picture feels held by the music, even on shaders whose own reactivity is intentionally calm.

Because the pass runs **before** typography / title / logo, lyrics and branding stay clean — only the picture (bg + shader + ascii) gets the breathing dark frame.

## Math (`pipeline/audio_vignette.py`)

`build_audio_vignette_context(width, height, *, strength=1.0)` precomputes a `float32` mask shaped `(H, W)` that is `0.0` at the centre and ramps to `1.0` at the corners via a smoothstep between `0.55` and `1.05` (in aspect-corrected unit-radius coordinates). The mask is computed once per render and reused every frame.

`apply_audio_vignette(frame, uniforms, ctx)` is the per-frame hot path:

```text
audio   = 0.55 * bass_hit + 0.30 * drop_hold + 0.18 * rms                # 0..1
darken  = strength * (0.18 + 0.12 * audio)                              # 0..0.30
out     = frame * (1 - mask * darken)
```

The numbers are deliberately small: corners drop by ~18 % at rest and up to ~30 % on a peak bass hit + drop afterglow stack. The centre of the frame is untouched.

## Wiring (`pipeline/compositor.py`)

* **`CompositorConfig.audio_vignette_enabled: bool = True`** — feature flag; `False` reverts to the legacy no-op path.
* **`CompositorConfig.audio_vignette_strength: float = 1.0`** — scales every contribution; `0.0` is equivalent to disabling the pass.
* `_build_audio_vignette_context(cfg)` builds the precomputed mask once per render or returns `None` (fast path).
* `_render_compositor_frame` accepts the optional `audio_vignette_ctx` and calls `apply_audio_vignette(...)` between the voidcat ASCII pass and the typography pass.
* The pass is `O(W * H)` with a single `float32` multiply + `clip` per frame; on 1920×1080 that's a few ms on the producer thread, well below the per-frame budget dominated by Skia kinetic typography.

## Errors

`apply_audio_vignette` rejects non-`(H, W, 3) uint8` arrays and mismatched mask shapes — these are programmer errors that would silently corrupt frames otherwise.

## See also

* `docs/technical/frame-compositor.md` — per-frame pipeline that hosts this pass.
* `docs/technical/reactive-shader-layer.md` — alpha contract for the shader pass that runs immediately before the vignette.
