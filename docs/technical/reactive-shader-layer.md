# Reactive shader layer (M2)

GPU fragment-shader pass via **moderngl** running in a **standalone OpenGL 3.3+ context** on an offscreen **RGBA FBO** at the target resolution (default **1920×1080**). Implements PRD §3.5: each frame is driven by analysis-derived uniforms (`time`, `beat_phase`, `bar_phase`, `band_energies[8]`, `rms`, `onset_pulse`, `onset_env`, `build_tension`, `intensity`) that turn into reactive overlays later alpha-blended onto the background. Compositor-injected signals (`bass_hit`, `transient_lo/mid/hi`, `drop_hold`) share the same uniform dict — see `docs/technical/frame-compositor.md`.

## Entry points (`pipeline/reactive_shader.py`)

- **`ReactiveShader(shader_name, *, width, height, num_bands, shaders_dir, vertex_shader, palette, feedback_enabled)`** — lifecycle wrapper. Creates the context, compiles the GLSL pair, allocates the FBO. Context-manager safe; `close()` is idempotent. `feedback_enabled=None` (the default) auto-detects Milkdrop-style ping-pong feedback by checking whether the fragment shader declares `u_prev_frame`; explicit `True`/`False` forces it.
- **`ReactiveShader.render_frame(uniforms) -> np.ndarray`** — returns `(H, W, 4)` `uint8` RGBA with top-left origin (framebuffer is flipped after `fbo.read`). When `u_comp_background` is `0` (default), fragments output premultiplied RGBA overlay only.
- **`ReactiveShader.render_frame_composited_rgb(uniforms, background_rgb) -> np.ndarray`** — uploads `(H, W, 3)` `uint8` RGB to `u_background`, sets `u_comp_background = 1`, and returns opaque `(H, W, 3)` `uint8` RGB (reactive layer alpha-blended over the background in the fragment shader).
- **`ReactiveShader.write_background_rgb(background_rgb)`** — uploads a full-frame RGB image to `u_background` (row order converted for OpenGL).
- **`ReactiveShader.reset_feedback()`** — zero the previous-frame texture and force `u_has_prev = 0` on the next draw. No-op when feedback wasn't enabled. Call this between independent render sessions (e.g. each Gradio preview click) so trails don't bleed across runs.
- **`ReactiveShader.feedback_enabled`** / **`ReactiveShader.has_prev_frame`** — read-only view of the ping-pong state.
- **`uniforms_at_time(analysis, t, *, num_bands, intensity, onset_decay)`** — translates an `analysis.json` dict into the uniform dict for wall-clock time `t` in seconds.
- **`resolve_builtin_shader_stem(stem)`** — validates non-`none` stems against `BUILTIN_SHADERS` and `{stem}.frag` under `SHADERS_DIR`; raises on unknown or missing files. The stem **`none`** is allowlisted and skips the fragment file check (full frames use a CPU passthrough in `pipeline.compositor`, not `ReactiveShader`). **`ReactiveShader(..., shaders_dir=tmpdir)`** (tests) resolves arbitrary stems from that directory without the allowlist.
- **`composite_premultiplied_rgba_over_rgb(rgba, rgb)`** — CPU alpha-blend matching the GPU premultiplied-over-RGB math (for tests or offline use).
- **`ShaderUniforms`** — optional typed uniform bag (`.as_dict()` is what `render_frame` expects).
- **`BUILTIN_SHADERS`** — tuple of bundled fragment-shader stems.

## Shader library (`assets/shaders/`)

Active stems match `pipeline.builtin_shaders.BUILTIN_SHADERS`. When the user picks **No reactive shader** (`none`), the compositor skips `ReactiveShader` and passes the background through unchanged (then typography, logo, effects, etc.).

- **`passthrough.vert`** — shared fullscreen-quad vertex shader; feeds a `[0, 1]` `v_uv` to every fragment shader.
- **`void_ascii_bg.frag`** — GPU void / grain field; the orchestrator also enables the CPU **ASCII** overlay in `pipeline/voidcat_ascii` for this stem.
- **`tunnel_flight.frag`** — first-person wireframe tunnel (spiral lattice, corner nodes, dual-layer line glow); scroll and palette cross-fade with `rms` / `bass_hit`, sparks on high-band transients.
- **`spectral_milkdrop.frag`** — Milkdrop-style hybrid: curl-noise *translation* feedback, polar-wedge kaleidoscope, domain-warped FBM, Lissajous waveform trace over `band_energies`, hue-cycling palette. Uses optional peak-tint uniforms for bright backgrounds.
- **`synth_grid.frag`** — retrowave perspective grid and horizon; **`void main`** documents the intentional opaque ground plane.

Every fragment shader expects the same uniform contract and writes premultiplied RGBA so downstream compositing can alpha-blend without further math (except **`none`**, where no fragment pass runs).

## Feedback / warp framebuffer (Milkdrop-style trails)

Opt-in ping-pong FBO on `ReactiveShader`. When the fragment shader declares a `sampler2D u_prev_frame` uniform (plus the paired `float u_has_prev` flag), construction auto-allocates a second RGBA8 framebuffer at the output resolution and swaps roles after every `render_frame`. The next draw sees the previous frame's RGBA as `u_prev_frame` on texture unit **1** (unit 0 is the `u_background` sampler) — this is the primitive that unlocks trails, tunnels and liquid-smear effects.

- **Opt-in cost.** Zero for shaders that don't declare the sampler: auto-detection is a single `program["u_prev_frame"]` lookup at link time, the extra FBO isn't allocated, and the hot render path skips the ping-pong swap. Explicit `feedback_enabled=True`/`False` overrides the auto-detect result.
- **VRAM.** The feedback texture is RGBA8 at the full FBO resolution — **~8 MB per frame at 1920×1080**, so a feedback-enabled instance costs ~16 MB total (primary colour attachment + the second texture). Disabled instances stay at the ~8 MB baseline.
- **First-frame contract.** `u_has_prev = 0.0` on the very first draw and `1.0` on every subsequent draw until `reset_feedback()` is called. The texture is initialised to opaque-zero, so a shader that reads `u_prev_frame` unconditionally still samples well-defined pixels on frame 0 — but the recommended pattern is `mix(fresh, warped_prev, u_has_prev * trail_weight)` so the first frame has no "ghost" of a stale buffer.
- **Reset semantics.** `reset_feedback()` zeros the previous-frame texture and clears the has-prev flag. Call it at the start of any independent render sequence (Gradio preview reload, new song render) so trails never bleed across runs. No-op when feedback wasn't enabled for the instance.
- **Sampler binding.** `u_prev_frame` is wired to texture unit 1 once at construction — the per-frame bind just calls `tex.use(location=1)`. The compositor's `u_background` sampler stays on unit 0 unaffected, so a feedback shader can still accept a composited background.
- **Ownership.** `render_frame` is the only method that advances the ping-pong — `render_frame_composited_rgb` delegates to it, so the compositor path gets feedback for free.

Authoring sketch:

```glsl
uniform sampler2D u_prev_frame;
uniform float u_has_prev;
// ... rest of the usual uniforms ...
void main() {
    vec4 fresh = /* base pattern for this frame */;
    vec2 warp  = v_uv + 0.003 * vec2(cos(time), sin(time));  // radial warp
    vec4 trail = texture(u_prev_frame, warp) * 0.94;         // decay per frame
    out_color  = mix(fresh, max(fresh, trail), u_has_prev);
}
```

## Uniform mapping (`analysis.json` → shader)

- **`beat_phase`** — position in `[0, 1)` inside the current beat interval (bisect on `analysis.beats`; falls back to `60 / tempo.bpm` outside the grid).
- **`bar_phase`** — position in `[0, 1)` inside the current 4-beat bar. Prefers `analysis.downbeats` when present, else groups `analysis.beats` into runs of `beats_per_bar=4`, else falls back to `60 / tempo.bpm * beats_per_bar` with bar 0 anchored at `t=0`. Uses the median span to extrapolate outside the grid.
- **`band_energies[N]`** — linear interpolation along the time axis of `spectrum.values` (`(frames, bands)`), using `spectrum.fps` (or top-level `fps`). Output clamped to `[0, 1]` and padded/truncated to `num_bands`.
- **`rms`** — linear interpolation of `rms.values` at `rms.fps`.
- **`onset_pulse`** — `exp(-ONSET_DECAY_PER_SEC * (t - last_peak))` from `onsets.peaks`; `0.0` before the first peak. Good for flash-on-hit behaviour.
- **`onset_env`** — continuous transient envelope sampled from `onsets.strength` at `onsets.frame_rate_hz` (onsets use a different hop than the mel spectrum, so this is **not** `analysis.fps`). Normalised once per load by the 95th percentile (`ONSET_ENV_NORM_PERCENTILE`) so a single outlier spike doesn't crush the rest of the track; the normalised array is memoised on `analysis['_onset_env_cache']` keyed by `id(strength)`. Good for sustained texture / shimmer response — pairs with `onset_pulse` for flash-on-hit.
- **`build_tension`** — `[0, 1]` pre-drop smoothstep ramp interpolated from `events.build_tension.values` (`fps` + `values` shape identical to `rms`). Zero when the `events` block is absent. Snaps to zero immediately after each drop — the post-drop "release" is the compositor-injected `drop_hold`.
- **`bass_hit`, `transient_lo/mid/hi`, `drop_hold`** — optional `0…1` envelopes injected by the compositor (not by `uniforms_at_time`). `bass_hit` is a kick-shaped `build_bass_pulse_track`; the three `transient_*` tracks come from `build_lo/mid/hi_transient_track` in `pipeline/beat_pulse.py` (low / mid / high mel-band slices with 0.34 / 0.12 / 0.06 s decays); `drop_hold` is an exponential release off the most recent drop in `events.drops`. **The three `transient_*` tracks are shape-gated at build time via `shape_reactive_envelope(deadzone=0.18, soft_width=0.12, gamma=1.3)`**: below the deadzone the envelope reads as hard zero (kills the chill-section noise floor that produced constant low-amplitude shader wobble), the smoothstep shoulder easing the ramp in, and the gamma pass compressing mid-amplitude values more than peaks so cleanly-separated hits still peak near `1.0`. Shaders consequently see a much cleaner "silent between hits, sharp on hits" profile without any per-frame CPU cost. Opt out per-band via `build_*_transient_track(..., shape=False)` for debugging. The Gradio reactive preview merges all four at `t=0` so the preview never reads zero for these signals.
- **`time`, `resolution`, `intensity`** — forwarded from callers; `resolution` is pinned to the FBO size regardless of input.
- **`u_background`**, **`u_comp_background`** — `sampler2D` RGB background at full FBO resolution and `0`/`1` flag; when `u_comp_background` is `1`, each fragment blends the reactive premultiplied RGBA over `texture(u_background, v_uv)` and outputs opaque RGB (`alpha = 1`). The Gradio **Reactive intensity** slider should scale `intensity` as `percent / 100` and pass it into `uniforms_at_time(..., intensity=...)` (or merge into the uniform dict) so it multiplies the overlay strength inside the shaders.
- **`u_shader_tint`, `u_shader_tint_strength`** — optional `vec3` peak-tint and `[0, 1]` mix weight set once per render. Parsed from `ReactiveShader(shader_tint="#RRGGBB", shader_tint_strength=0..1, ...)` (and the matching `CompositorConfig.shader_tint*` / `OrchestratorInputs.shader_tint*` fields). Defaults are `white` + `0`, which means the historical white peak blow-out is preserved exactly for any caller that doesn't opt in. Shaders are expected to use the strength as a `mix()` weight over the white target so they degrade to a no-op at `strength = 0`. Currently consumed by `spectral_milkdrop.frag` (replaces `vec3(1.6)` / `vec3(1.0)` peak targets); other shaders can pick up the same contract trivially because `_set_uniform` silently skips uniforms they don't declare. **UI**: the *Visual style* tab exposes a `gr.ColorPicker` plus a `0..100` strength slider inside the **Shader peak tint** accordion; the orchestrator forwards them verbatim to `CompositorConfig`, and the per-song cache key is unaffected (cosmetic-only render input). Use this to keep peaks visible against bright SDXL stills where pure white reads poorly.

Missing or malformed analysis sections degrade to zero-valued uniforms rather than raising, so the renderer stays usable with synthetic mocks.

## Alpha contract — content-driven, no floors

Every bundled shader writes a **premultiplied RGBA overlay** that the
compositor (or the in-shader composite when ``u_comp_background = 1``)
alpha-blends over the background:

```glsl
vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
```

To keep the background (SDXL stills, AnimateDiff, static Ken Burns)
visible, **the alpha must be content-driven**: it should track the
brightness of the rendered RGB, optionally lifted by audio terms on
hits. Hard floors (``clamp(alpha, 0.30, …)``) and constant bases
(``alpha = 0.55 + …``) paint every pixel of every frame at ≥ floor
opacity and hide the background outright — that's a bug, not a
feature. The canonical pattern is:

```glsl
float content    = clamp(dot(col, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
float audio_lift = 0.18 * bass_hit + 0.14 * hold;  // shader-specific
float alpha      = clamp((content + audio_lift) * intensity, 0.0, 1.0);
```

Reference implementations in-tree: ``void_ascii_bg.frag``, ``tunnel_flight.frag``,
``spectral_milkdrop.frag`` (luma-driven with a small audio lift), ``synth_grid.frag``.
The one deliberate exception is ``synth_grid.frag``, where the synthwave ground plane is
explicitly opaque (``local_alpha = 1.0``) and only the sky bleeds
through — the comment in the shader documents why.

## Shader authoring guide — how to consume each signal

All new uniforms are declared in every bundled `.frag` file today
(declaration-only sweep landed with task 38) but most shaders still ignore
them pending the per-shader pass in task 42. Treat the matrix below as the
canonical "what is this signal *for*" when picking which knobs to wire up.

| Signal | Time scale | Typical use |
|---|---|---|
| `beat_phase` | 1 beat | Per-beat LFOs, color shift between two palette tones, anything that should tick over on every kick. |
| `bar_phase` | 4 beats | Bar-scale LFOs that must not drift off the downbeat: camera drifts, long colour sweeps, macro motion envelopes. Phase resets to `0` on every downbeat by construction. |
| `onset_pulse` | ~0.5 s decay | Flash-on-hit: stabs, additive highlights, quick colour punches on discrete onsets. |
| `onset_env` | continuous | Sustained texture / shimmer response — noise amplitude, grain density, secondary modulation on top of `onset_pulse`. Pairs well with `onset_pulse` (flash vs. texture). |
| `rms` | continuous | Global loudness gate. Multiply motion speed / overall brightness by `rms` so quiet intros stay calm. |
| `bass_hit` | ~0.34 s decay | Smooth kick envelope matching the logo pulse. Bounce / scale reactive elements, drive low-frequency bloom. |
| `transient_lo` | 0.34 s decay | Low-end bloom, thick sub rumble, chromatic aberration on the kick. Slightly different shaping than `bass_hit` (band rectified, not sustain-tracked) so they complement rather than duplicate. |
| `transient_mid` | 0.12 s decay | Snare / clap hits — short color flashes, body-band sparkle, camera micro-kicks. |
| `transient_hi` | 0.06 s decay | Hats / cymbals — fine grain sparkle, hi-frequency noise bursts, fast specular flashes. |
| `build_tension` | rises over 6 s, snaps to 0 after drop | Pre-drop **dampening**: desaturate, slow motion, pull focus, tighten a vignette, compress palette. Inverse of a "release" signal; explicitly snaps to `0` after the drop so you don't get a second hit. |
| `drop_hold` | ~2 s decay (8 bars @ 120 bpm) | Post-drop afterglow: bloom boost, camera kick, saturation surge. Starts at the drop's confidence and exponentially decays — layer on top of the rest of the reactive stack. |

General rules:

* Treat every signal as `[0, 1]` — clamp before using them as a `mix()`
  weight or a multiplier. Shaders that want more headroom should scale
  explicitly (e.g. `1.0 + 0.4 * transient_hi`).
* Unused uniforms cost nothing: `ReactiveShader._set_uniform` silently
  skips any uniform the shader does not declare, and the same `_apply_uniforms`
  defaults zero out every signal for shaders that declare but don't read one.
* Prefer `mix(base, hot, clamp(signal, 0, 1))` over additive saturation so
  brightness stays bounded on dense transient stacks (kick + snare + hat
  landing on the same frame).
* The shader transient sensitivity default (`CompositorConfig.shader_transient_sensitivity`)
  is `0.75`, not `1.0`, to keep the three bands in the same rough budget as
  `shader_bass_sensitivity=0.72`. Combined with the build-time shape gate,
  this tames the additive signal pile-up (most shaders read 4–6 reactive
  envelopes at once) that otherwise manifested as constant low-amplitude
  wobble during non-hit sections. Shaders that genuinely need more
  transient punch should scale inside the shader (`1.0 + 0.6 * transient_lo`)
  rather than lift the global sensitivity.

The `pilot_shader` effort (task 40) and the per-shader propagation pass
(task 42) will land concrete examples — this table is the reference they
all follow.

## Errors

- Context creation wraps driver failures in `RuntimeError` (OpenGL 3.3+ required).
- Shader compile/link errors surface with the shader name in the message.
- Per-frame `ctx.error` is checked after draw; non-`GL_NO_ERROR` aborts the frame.
- Unknown uniforms are silently skipped so a single uniform dict works across all bundled shaders.

## Verification

- `python -m compileall .` — syntax / import check.
- Runtime smoke (GPU host with OpenGL): render a ~10 s synthetic sequence via `uniforms_at_time` on a mocked analysis dict and confirm `render_frame` returns a non-zero RGBA array with no GL errors.
