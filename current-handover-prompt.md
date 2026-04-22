# Session Handover

## Environment

- **Project:** MusicVids
- **Tech Stack:** Python 3.11, CUDA/PyTorch, Gradio, librosa/demucs/whisperx, diffusers (SDXL / AnimateDiff), moderngl, skia-python, ffmpeg NVENC
- **Context file:** Always read `ai-context.md` first — it contains project rules, architecture, and model selection.
- **Python interpreter:** `.\.venv\Scripts\python.exe` (the Windows `py` launcher does NOT point at the project venv — use the full path).
- **Run tests:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` (pytest is not wired; Task Master may mention pytest — prefer unittest).
- **Branch:** main

## Core Handover Rules

- **NO HISTORY:** Do not include past task details unless they directly impact this specific task.
- **SCOPE:** Focus ONLY on the task detailed below.

## Current Task

### Task Master #42 — Propagate new signals to remaining 7 shaders

- **ID:** 42
- **Status:** in-progress
- **Priority:** low
- **Complexity:** 5 (recommendedSubtasks: 5 — but the task body names 7 shaders; treat one shader = one commit)
- **Dependencies:** 40, 41 (both **done** — `nebula_flow` pilot proved out the authoring pattern; `ReactiveShader` now supports opt-in ping-pong feedback, though task 42 does NOT use the feedback FBO — keep it as opt-in for future "flow" variants only).

**Description.** Revisit each of the remaining bundled shaders and use `transient_hi`, `transient_mid`, `transient_lo`, `build_tension`, `drop_hold`, and `bar_phase` where each makes musical sense. Keep visual continuity — this is a tasteful upgrade to the existing `.frag` bodies, **not** a rewrite. Every `.frag` already declares the full uniform set (task 38) and `nebula_flow.frag` is the canonical example of how to consume them. `nebula_drift.frag` stays untouched as the A/B peer.

**Shaders to upgrade (one per commit/PR):**

1. `spectrum_bars.frag`
2. `particles.frag`
3. `geometry_pulse.frag`
4. `liquid_chrome.frag`
5. `vhs_tracking.frag`
6. `paper_grain.frag`
7. `synth_grid.frag`

**Per-shader recipe.**

1. Add a short **design note** as a top-of-file comment: which signals drive which effect in this shader. Keep it aligned with the vocabulary table in `docs/technical/reactive-shader-layer.md` (Shader authoring guide section).
2. Modify `main()` only. Do **not** touch the uniform declaration block — they already match across all shaders and grep-ability depends on that.
3. Stay inside the 5-color palette contract (`u_palette[5]` + `u_palette_size`, `palette_pick(idx)` wraps). No new uniforms, no new sampler bindings, no feedback FBO.
4. Preserve the existing `u_comp_background`/`u_background` tail block byte-for-byte (the compositor relies on the dual-mode output).
5. Clamp every reactive signal to `[0, 1]` defensively; prefer `mix(base, hot, clamp(signal, 0, 1))` over unbounded additive saturation so dense transient stacks (kick + snare + hat + drop_hold) stay bounded.
6. Compile target: OpenGL 3.3 Core (`#version 330`). No GL4 features.

**Signal → visual intent (reference — adapt per shader aesthetic).**

- `transient_lo` → low-end bloom / thick chrome warp / kick bounce (0.34 s decay; different shaping than `bass_hit`, they complement).
- `transient_mid` → snare/clap micro-flashes, short color punches (0.12 s decay).
- `transient_hi` → hats/air fine grain — sparkle density, specular flashes (0.06 s decay).
- `build_tension` → pre-drop **dampening**: desaturate, slow motion, tighten vignette, compress dynamic range. Snaps to 0 after the drop — do NOT treat as post-drop energy.
- `drop_hold` → post-drop afterglow: bloom boost via `palette[4]`, saturation surge, ~2 s decay.
- `bar_phase` → bar-scale LFO that resets on every downbeat; use `sin(bar_phase * TAU)` / `cos(bar_phase * TAU)` for anything that should not feel locked to raw `time`.
- `onset_env` → continuous shimmer layered on top of discrete `onset_pulse` flashes.

**Per-shader design-note suggestions (not prescriptive — adjust to taste).**

- **`spectrum_bars`** — `transient_hi` amps top-of-bar sparkle, `build_tension` desaturates and compresses bar-height range pre-drop, `drop_hold` punches an extra bloom from `palette[4]`, `bar_phase` slowly sweeps the cool/hot palette offset across the 4-beat bar.
- **`particles`** — `transient_hi` raises per-cell density / twinkle, `build_tension` dampens drift speed, `drop_hold` boosts cell radius, `bar_phase` rotates drift direction.
- **`geometry_pulse`** — `transient_mid` snaps rings on snares, `build_tension` tightens ring spacing (compressive feel), `drop_hold` blooms the outermost ring via `palette[4]`, `bar_phase` rotates ring orientation.
- **`liquid_chrome`** — `transient_lo` warps the chrome sheet on kicks, `build_tension` cools the drift and desaturates, `drop_hold` punches highlight bloom, `bar_phase` rotates the flow basis.
- **`vhs_tracking`** — `transient_hi` triggers tracking bursts, `build_tension` compresses scanlines and mutes chroma, `drop_hold` punches chroma bleed, `bar_phase` drives a slow vertical roll.
- **`paper_grain`** — `transient_hi` ups grain density, `build_tension` desaturates and tightens fibre, `drop_hold` warms a palette[4] wash, `bar_phase` drifts fibre angle.
- **`synth_grid`** — `transient_mid` flashes grid lines, `build_tension` tightens perspective and cools the horizon, `drop_hold` blooms the horizon, `bar_phase` slowly tilts grid scroll angle.

**Implementation order.** Tackle one shader per commit. After each edit, re-run the full unittest suite (below) to catch any palette/preset-loading regressions. GLSL compile/link errors only surface at GPU runtime — if no GPU host is available, flag that in the session log.

## Key Files

- `assets/shaders/nebula_flow.frag` — **reference only**. Canonical worked example of the Phase-2 signal vocabulary. Mirror its structure (clamped signals, `TAU`-based bar rotation, `mix()`-biased palette shifts, `drop_hold`-weighted bloom via `palette[4]`) when upgrading each target shader.
- `assets/shaders/nebula_drift.frag` — **do not modify** (A/B peer to `nebula_flow` from task 40).
- `assets/shaders/spectrum_bars.frag`, `particles.frag`, `geometry_pulse.frag`, `liquid_chrome.frag`, `vhs_tracking.frag`, `paper_grain.frag`, `synth_grid.frag` — the 7 targets. Uniform declaration block is already identical across all of them (task 38); only `main()` + the top-of-file comment change.
- `docs/technical/reactive-shader-layer.md` — **Shader authoring guide** table is the canonical time-scale / intent reference. Re-read the table rows for each signal before authoring `main()` changes.
- `pipeline/compositor.py::_render_compositor_frame`, `pipeline/reactive_shader.py::_apply_uniforms` — **no code change**. Already inject / default every signal.
- `tests/test_reactive_palette.py`, `tests/test_nebula_flow_preset.py` — existing palette/preset sanity tests that keep passing. No new tests required by the task, but add one if a shader upgrade exposes a gap.

## Context

- Compositor + shader uniform plumbing is complete: every signal (`bass_hit`, `transient_lo/mid/hi`, `drop_hold`, `bar_phase`, `onset_env`, `build_tension`) is injected per frame, and `ReactiveShader._apply_uniforms` silently skips any uniform a shader doesn't declare — so adding a new signal consumer never crashes at link time.
- **5-color palette contract:** `u_palette[5]` is always populated (padded by repeating the last color when the preset ships fewer), `u_palette_size` reports the effective count, `palette_pick(idx)` wraps via `((idx % n) + n) % n`. Stay inside it — no new uniforms.
- `build_tension` is explicitly the **inverse of release**: rises to 1 over 6 s before each drop, **snaps to 0** immediately after. Never treat it as a post-drop signal — that's `drop_hold`'s job. Using `build_tension` as dampener + `drop_hold` as energiser is the intended visual story on every shader.
- `nebula_drift.frag` stays byte-for-byte identical so `cosmic` vs `cosmic-flow` remains a pure-math A/B. Task 42 does the same spirit on the other shaders but without a new `-flow` preset — the upgrade happens in-place.
- **Feedback FBO (task 41) is out of scope here.** Ping-pong trails are opt-in for shaders that declare `u_prev_frame`; task 42 deliberately does NOT add that declaration to any of the 7 targets. Future "flow" variants can opt in.
- Docs: the authoring guide table in `docs/technical/reactive-shader-layer.md` is authoritative; no new docs required by this task. If a per-shader upgrade needs more than a one-line design-note comment, extend the existing shader doc rather than creating a new one.

## Verification (per shader, after each edit)

- `.\.venv\Scripts\python.exe -m compileall g:\DEV\MusicVids\pipeline\builtin_shaders.py g:\DEV\MusicVids\config.py -q` — Python syntax check (GLSL is not covered).
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` — full unittest suite (expect 351+ passing; no new tests required unless an upgrade exposes a gap).
- `.\.venv\Scripts\python.exe -c "from config import load_preset_registry; reg = load_preset_registry(); print(sorted(reg))"` — preset registry sanity.
- **GPU host (if available):** run the Gradio reactive preview with the preset that uses the upgraded shader, with a song whose `analysis.json` has a detected drop in its `events` block. Confirm no GL link/compile error and confirm the intended signal → visual mapping reads musically (dampen on build, bloom on drop, bar-scale drift, etc.).
- `python -m compileall` won't catch GLSL errors — a smoke render is the only way to confirm a shader still links after the edit. Call this out in the session log if no GPU is available.

## Task Master

- **Current: #42 (in-progress).** Use `get_task --id=42` / `next_task` MCP tools for the live JSON.
- Recommended approach: one shader = one commit = one session log entry, in the order listed above. Stop and re-run tests after every shader so a regression is bisectable.
- After task 42: the Phase-2 rollout is effectively complete. Next phase will branch into opt-in `*-flow` presets that use the feedback FBO (task 41) for trails/tunnels.
