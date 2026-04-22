# Reactive shader layer (M2)

GPU fragment-shader pass via **moderngl** running in a **standalone OpenGL 3.3+ context** on an offscreen **RGBA FBO** at the target resolution (default **1920×1080**). Implements PRD §3.5: each frame is driven by analysis-derived uniforms (`time`, `beat_phase`, `band_energies[8]`, `rms`, `onset_pulse`, `intensity`) that turn into reactive overlays later alpha-blended onto the background.

## Entry points (`pipeline/reactive_shader.py`)

- **`ReactiveShader(shader_name, *, width, height, num_bands, shaders_dir, vertex_shader)`** — lifecycle wrapper. Creates the context, compiles the GLSL pair, allocates the FBO. Context-manager safe; `close()` is idempotent.
- **`ReactiveShader.render_frame(uniforms) -> np.ndarray`** — returns `(H, W, 4)` `uint8` RGBA with top-left origin (framebuffer is flipped after `fbo.read`). When `u_comp_background` is `0` (default), fragments output premultiplied RGBA overlay only.
- **`ReactiveShader.render_frame_composited_rgb(uniforms, background_rgb) -> np.ndarray`** — uploads `(H, W, 3)` `uint8` RGB to `u_background`, sets `u_comp_background = 1`, and returns opaque `(H, W, 3)` `uint8` RGB (reactive layer alpha-blended over the background in the fragment shader).
- **`ReactiveShader.write_background_rgb(background_rgb)`** — uploads a full-frame RGB image to `u_background` (row order converted for OpenGL).
- **`uniforms_at_time(analysis, t, *, num_bands, intensity, onset_decay)`** — translates an `analysis.json` dict into the uniform dict for wall-clock time `t` in seconds.
- **`resolve_builtin_shader_stem(stem)`** — validates the stem against `BUILTIN_SHADERS` and the presence of `{stem}.frag` under `SHADERS_DIR`; raises on unknown or missing files (no fallback).
- **`composite_premultiplied_rgba_over_rgb(rgba, rgb)`** — CPU alpha-blend matching the GPU premultiplied-over-RGB math (for tests or offline use).
- **`ShaderUniforms`** — optional typed uniform bag (`.as_dict()` is what `render_frame` expects).
- **`BUILTIN_SHADERS`** — tuple of bundled fragment-shader stems.

## Shader library (`assets/shaders/`)

- **`passthrough.vert`** — shared fullscreen-quad vertex shader; feeds a `[0, 1]` `v_uv` to every fragment shader.
- **`spectrum_bars.frag`** — 8 reactive bars, color shifts with `beat_phase`, glow pulses on `onset_pulse`.
- **`particles.frag`** — cellular noise field; per-cell radius reacts to the mapped spectrum band; drift speed scales with `rms`.
- **`geometry_pulse.frag`** — concentric rings pulsing on onsets; ring tone mixes on `rms`, spacing modulates with `time`.

Every fragment shader expects the same uniform contract and writes premultiplied RGBA so downstream compositing can alpha-blend without further math.

## Uniform mapping (`analysis.json` → shader)

- **`beat_phase`** — position in `[0, 1)` inside the current beat interval (bisect on `analysis.beats`; falls back to `60 / tempo.bpm` outside the grid).
- **`band_energies[N]`** — linear interpolation along the time axis of `spectrum.values` (`(frames, bands)`), using `spectrum.fps` (or top-level `fps`). Output clamped to `[0, 1]` and padded/truncated to `num_bands`.
- **`rms`** — linear interpolation of `rms.values` at `rms.fps`.
- **`onset_pulse`** — `exp(-ONSET_DECAY_PER_SEC * (t - last_peak))` from `onsets.peaks`; `0.0` before the first peak.
- **`bass_hit`** — optional `0…1` kick-shaped envelope from the compositor (`build_bass_pulse_track` with a longer decay than the logo); fragment shaders use it for smooth low-end motion instead of raw RMS/onsets alone. Gradio reactive preview merges the same track at `t=0`.
- **`time`, `resolution`, `intensity`** — forwarded from callers; `resolution` is pinned to the FBO size regardless of input.
- **`u_background`**, **`u_comp_background`** — `sampler2D` RGB background at full FBO resolution and `0`/`1` flag; when `u_comp_background` is `1`, each fragment blends the reactive premultiplied RGBA over `texture(u_background, v_uv)` and outputs opaque RGB (`alpha = 1`). The Gradio **Reactive intensity** slider should scale `intensity` as `percent / 100` and pass it into `uniforms_at_time(..., intensity=...)` (or merge into the uniform dict) so it multiplies the overlay strength inside the shaders.

Missing or malformed analysis sections degrade to zero-valued uniforms rather than raising, so the renderer stays usable with synthetic mocks.

## Errors

- Context creation wraps driver failures in `RuntimeError` (OpenGL 3.3+ required).
- Shader compile/link errors surface with the shader name in the message.
- Per-frame `ctx.error` is checked after draw; non-`GL_NO_ERROR` aborts the frame.
- Unknown uniforms are silently skipped so a single uniform dict works across all bundled shaders.

## Verification

- `python -m compileall .` — syntax / import check.
- Runtime smoke (GPU host with OpenGL): render a ~10 s synthetic sequence via `uniforms_at_time` on a mocked analysis dict and confirm `render_frame` returns a non-zero RGBA array with no GL errors.
