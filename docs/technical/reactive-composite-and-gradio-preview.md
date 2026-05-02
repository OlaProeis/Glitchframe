# Reactive background composite and Gradio preview

Feature: blend the reactive moderngl pass over a full-frame **RGB** background inside the fragment shader, expose a **strict** shader stem resolver, and drive **intensity** plus shader choice from the Visual style tab with a one-frame GPU preview.

## GPU compositing

- Bundled fragment shaders (`assets/shaders/*.frag`) declare `sampler2D u_background` and `float u_comp_background`. When `u_comp_background` is **1**, each fragment computes premultiplied overlay color as before, then outputs **opaque** RGB: `overlay.rgb + bg * (1.0 - overlay.a)`. When **0**, behavior is unchanged (premultiplied RGBA overlay only).
- `ReactiveShader` allocates an RGB `u1` texture matching the FBO size. `write_background_rgb()` uploads `(H, W, 3)` `uint8` with rows flipped for OpenGL. `render_frame_composited_rgb(uniforms, background_rgb)` uploads the background, sets `u_comp_background = 1`, renders, and returns `(H, W, 3)` `uint8` for downstream layers (typography, logo, compositor).

## Validation and CPU blend helper

- `resolve_builtin_shader_stem(stem)` checks `BUILTIN_SHADERS`. The stem `none` is valid and skips the fragment file check (compositor uses a CPU passthrough). Other stems require `{stem}.frag` under `SHADERS_DIR`. Custom `ReactiveShader(..., shaders_dir=tmpdir)` used in tests resolves stems from that directory without the allowlist.
- `composite_premultiplied_rgba_over_rgb(rgba, rgb)` applies the same premultiplied-over-RGB math in NumPy (tests or offline use).

## Gradio

- **Reactive intensity** (0–100) maps to `uniforms_at_time(..., intensity=value/100)`.
- **Preview reactive frame (t = 0 s)** requires an ingested song and `cache/<song_hash>/analysis.json` (run **Analyze** first). It uses the **Reactive shader** dropdown; **No reactive shader** shows the test gradient only (optional logo overlay). Other stems render at **960×540** over the gradient. Failures are logged; GL and missing-analysis errors are not swallowed.

## See also

- `docs/technical/reactive-shader-layer.md` — uniforms, `uniforms_at_time`, FBO lifecycle.
- `docs/technical/visual-style-presets.md` — shader list, example prompts, and optional YAML presets.
