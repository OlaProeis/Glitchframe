# Session Handover — RIFE morph "skip / still image" at keyframe boundaries (Glitchframe)

**Purpose:** Bring a **new agent/session** up to speed on a recurring perceptual issue in the rendered SDXL+RIFE background where the morph appears to "snap onto a still image" or "skip entire frames" at every internal SDXL keyframe boundary. **Multiple iterations have not fixed it and the most recent attempt may have made it worse — read the "Why we are stuck" section before writing any code.**

**Always read [`ai-context.md`](ai-context.md) first** for project rules and architecture. **Lyrics / WhisperX / Pinokio align historic** details live in [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) — **not** duplicated here.

**Branch:** All RIFE morph commits below are on **`dev`**. Confirm with `git branch` before editing.

---

## TL;DR for the next agent

The previous chats kept iterating on a single mental model — "the boundary samples in the dense RIFE timeline cluster too close to the shared keyframe, fix the timeline" — and shipped four schema bumps in a row (v3 → v4 → v5 → v6). The user reports each iteration **either did not help or made the perceived issue worse** ("seems as we are skipping entire frames now", "better on some transitions than others", "gotten worse since I first started getting a smoother transition").

**Do not write more code first.** Open this with diagnostic / measurement work to verify what the user is actually seeing. Strong odds we have the wrong root-cause model. The user explicitly said: *"I think we are looking at it the wrong way."*

The single most useful next step is probably to get the user to share a short rendered MP4 (or a frame-extract) for one keyframe boundary so we can stop guessing.

---

## What the user actually reports (verbatim, in order)

These are the user's own words across the chat history. Re-read them as one block before forming hypotheses — the descriptions have shifted as the timeline changed under them, and the *current* symptom may be different from the original one.

1. **Original symptom.** *"On the video output, when we do the RIFE transitions… its smooth most of the time, except when it hits the original still image. then it seems like it stops there, like it holds on the original image too long, it should be only one frame i think or short atleast… every transition is smooth, and sometimes it looks like the transition speeds up right before stopping on the original image."*
2. **After v4** (inset-warped centered sampling). *"It got better, but there is still a very noticable 'skip' now from the last RIFE to a original image, looks like there is missing some frames? or something, it morphs very nice, and right before it hits the last RIFE, goes to the original still its like a skip."*
3. **After v5** (velocity-matched IFNet bookends). *"There is still a very noticable skip from the rife to the original next frame, this did not make it better, and i suspect we are missing something crutial, this have been tried to be fixed in several other chats also."*
4. **Clarification mid-v5.** *"Its not jumpint back to anything, the Skip is between the last frame of the RIFE and the next still image, like smooth transition then skip then a still image then starts transitioning."*
5. **After v6** (cross-pair bridges + inset 0.25). *"It did not get any better sadly. seems as we are skipping entire frames now, and im not shure but i think it is better on some transitions than other. im not shure what is happening here, it seems to have gotten worse since i first started getting a smoother transition."*

The shifts to flag for the next agent:

- The user originally said the morph **"holds on the original image too long"** (a *long* still moment).
- After v4/v5/v6 the description shifted to **"skip"** and **"skipping entire frames"** (a *short* discontinuity).
- v6 introduced **"better on some transitions than others"** — a regularity that none of the previous attempts predicted. None of v3–v6 should have produced per-boundary variability for a uniform-keyframe-spacing song.

The "different on some transitions than others" plus "skipping entire frames" hint that we may have introduced a **rendering-layer** (compositor / FFmpeg / Ken Burns) issue, not the perceptual-blend issue we have been trying to fix.

---

## Why we are stuck (read carefully)

Every previous attempt assumed the same root cause:

> "The RIFE dense timeline contains samples that are visually too close to the shared keyframe `kf_{i+1}`. The compositor's pixel-space linear blend across the seam therefore looks like a static near-`kf_{i+1}` plateau (perceived as a 'still image moment'). Fix the timeline so the seam samples are visually distinct from `kf_{i+1}`."

That model produced v3 → v4 → v5 → v6, none of which fully fixed the issue and the latest of which apparently made things worse. Three reasons it may be the wrong model:

1. **Frame-skipping symptom.** "Skipping entire frames" is not what the perceptual-plateau model predicts. A pixel-space crossfade between two near-identical images produces a *static-looking* moment, not a *jump*. Frame skips suggest the compositor is dropping samples, the FFmpeg encoder is duplicating/dropping frames, or the cache is half-baked.
2. **Per-transition variability.** "Better on some transitions than others" with uniform keyframe spacing is inconsistent with all of v3–v6, which treat every internal boundary identically. This suggests a content-dependent or timing-dependent factor we have not modeled — e.g. SDXL keyframes that are visually very different from each other, non-uniform keyframe times in this particular song, or a Ken Burns / encoder beat that lines up differently per boundary.
3. **The "skip" appeared as we removed the still.** v3 internal boundaries already had no exact still — yet the original symptom was "stops on the original image". Either the user's v3 cache still had stills somehow, or the user has been calling something a "still image" that is *not* the SDXL keyframe (it might be the boundary blend zone itself, or it might be that RIFE is **off** in their config and they are seeing the smoothstep crossfade between sparse SDXL keyframes — which has zero velocity at every keyframe time).

**Stop iterating on the timeline until we have actual evidence of what the user sees in the rendered video.**

---

## Cache invalidation reality check

**Do this before anything else.** The user runs through Pinokio. There is a non-zero chance previous iterations did not actually take effect for them, or that some caches stuck. Verify before forming new hypotheses.

1. Confirm the user is on **commit `d0f68ef`** (`fix(rife): cross-pair boundary bridges + 0.25 inset (v6 schema)`) or later — `git log -1 --format='%H %s'` from inside the Pinokio install dir.
2. Confirm `RIFE_MANIFEST_SCHEMA_VERSION == 6` in their on-disk `pipeline/background_stills.py` (Pinokio sometimes lags syncing).
3. Confirm their `cache/<hash>/background/manifest_rife.json` was actually re-baked since the v6 push (mtime later than `d0f68ef` commit timestamp).
4. Confirm RIFE is **on** for the render in question — `OrchestratorInputs.sdxl_rife_morph` true, `rife_exp` >= 2. The "still image at every transition" symptom is what an **RIFE-off** render with smoothstep crossfades looks like (smoothstep velocity is zero at every keyframe time, so the morph slows to a stop on each `kf_i`). If RIFE is off, every single fix above is irrelevant.
5. Manually inspect a few PNGs from `cache/<hash>/background/rife_timeline/` around an internal keyframe time. They should be IFNet predictions (look like blurry / motion-warped composites), not sharp SDXL stills. If any look exactly like an SDXL still, the bake silently degraded.

If any of (1)–(5) fails, fix that first; don't write new pipeline code yet.

---

## What v6 actually does (current state on `dev`)

The dense morph timeline emitted by [`pipeline/rife_runtime.py::rife_build_morph_timeline`](pipeline/rife_runtime.py) for `N` keyframes at `rife_exp = exp` (`n = 2**exp` body samples per segment) is:

| index | wall-clock `t` | content |
|---|---|---|
| 0 | `keyframe_times[0]` | velocity-matched start bookend = `IFNet(kf_0, kf_1, s = inset)` |
| body of seg 0 | `t_0 + T_0 * (j+0.5)/n` for `j=0..n-1` | `IFNet(kf_0, kf_1, s = inset + (1-2*inset)*(j+0.5)/n)` |
| bridge after seg 0 | `keyframe_times[1]` | `IFNet(kf_0, kf_2, s = 0.5)` (cross-pair) |
| body of seg 1 | `t_1 + T_1 * (j+0.5)/n` | `IFNet(kf_1, kf_2, …)` |
| bridge after seg 1 | `keyframe_times[2]` | `IFNet(kf_1, kf_3, s = 0.5)` |
| … | … | … |
| body of seg N-2 | (last segment, no bridge after) | `IFNet(kf_{N-2}, kf_{N-1}, …)` |
| last | `keyframe_times[-1]` | velocity-matched end bookend = `IFNet(kf_{N-2}, kf_{N-1}, s = 1 - inset)` |

- **Default body inset:** `DEFAULT_IFNET_TIMESTEP_INSET = 0.25` (was 0.12 pre-v6, was 0 pre-v4). Override with `GLITCHFRAME_RIFE_IFNET_INSET` (clamped `[0, 0.45]`).
- **Frame count formula:** `frame_count = total_segs * n_steps + (total_segs - 1) + 2`.
- **Schema:** `RIFE_MANIFEST_SCHEMA_VERSION = 6` in [`pipeline/background_stills.py`](pipeline/background_stills.py).
- **Compositor side:** [`pipeline/background_stills.py::BackgroundStills.background_frame`](pipeline/background_stills.py) calls [`_interpolate_frame`](pipeline/background_stills.py) which **linearly blends** in pixel space between bracketing dense samples (`smooth_blend = not rife_dense`, set to `False` for RIFE since commit `a51debf`).

There are **no exact SDXL stills anywhere in the v6 dense timeline.** The user's "still image at every keyframe" perception, if it persists at v6, must come from one of:

- the linear-blend zone between two near-`kf_{i+1}` samples reading visually as a frozen `kf_{i+1}` (the model we have been chasing),
- the compositor or encoder dropping/duplicating frames,
- the user not actually being on v6,
- RIFE not being enabled at all for that render.

---

## Timeline of fixes attempted

All on branch `dev`. Each schema bump silently re-bakes earlier caches.

| schema | commit shape | what changed | user reaction |
|---|---|---|---|
| v1 | initial | exact SDXL stills at every keyframe time, dense IFNet between — produced "stops on the still" |
| v2 | `2896ce4` ish | centered sampling `(j+0.5)/n` — removed exact stills at internal boundaries, kept exact `kf_0` / `kf_N` at song bookends | original report came in here |
| v3 | `04f6d78` | added `keyframes_content_hash` so editing keyframe PNGs invalidates the cache; sampling unchanged from v2 | same complaint |
| v4 | (this chat) | `DEFAULT_IFNET_TIMESTEP_INSET = 0.12`; body samples warped inward to `s ∈ [inset, 1-inset]` | "got better, but there is still a very noticable 'skip'" |
| v5 | `5d30709` | velocity-matched IFNet bookends at `s = inset` / `s = 1 - inset` (replaced the legacy exact `kf_0` / `kf_N` stills); inset still 0.12 | "did not make it better… still a very noticable skip" |
| v6 | `d0f68ef` | inset bump 0.12 → 0.25; cross-pair IFNet bridge `IFNet(kf_i, kf_{i+2}, s=0.5)` inserted at every internal keyframe time | "did not get any better… seems as we are skipping entire frames now… better on some transitions than other" |

If you intend to add v7, **first** justify why v3 → v6 didn't fix it (with diagnostic evidence, not theory).

---

## Hypotheses we have NOT investigated

Numbered in rough "test cheapest first" order. Most of these need observation, not code changes.

1. **RIFE actually off in the user's config.** The classic "smooth → pause on still → smooth" pattern is exactly what an RIFE-off render produces (smoothstep crossfade between sparse SDXL keyframes — velocity is zero at every `kf_i` because smoothstep's derivative is zero at the endpoints). Verify `OrchestratorInputs.sdxl_rife_morph` is true for the render in question and the manifest_rife.json is being consumed by the compositor (`background_stills.py` falls back to the sparse SDXL timeline when no RIFE manifest is present).
2. **Stale Pinokio sync.** Pinokio installs sometimes lag the dev branch. Confirm `git log -1` inside the Pinokio MusicVids dir matches `d0f68ef`.
3. **Half-baked RIFE timeline.** If the bake crashed midway, the streaming writer may have left some PNG slots empty; the compositor's lazy `_RifeFrameSource` would silently substitute the nearest neighbor. Look for missing or zero-byte files in `cache/<hash>/background/rife_timeline/`. Also check whether the `est_total` count matches the actual PNG count after a fresh bake.
4. **FFmpeg encoder duplicating/dropping frames.** [`pipeline/renderer.py::_build_ffmpeg_args`](pipeline/renderer.py) writes raw bgr24 at 30 fps and lets FFmpeg encode at `-r 30` with default vsync. Run `ffprobe -show_frames -select_streams v` on the rendered MP4 and check for non-uniform `pkt_duration` or `repeat_pict`. The "skip entire frames" symptom is exactly what a vsync mismatch produces.
5. **Ken Burns pixel jitter.** [`pipeline/background_kenburns.py::_ken_burns_transform`](pipeline/background_kenburns.py) casts crop-box coords to `int()`, so a continuously varying box can jump by 1 pixel when crossing integer boundaries. Test by rendering with Ken Burns disabled (`OrchestratorInputs.sdxl_ken_burns = False`). If the "skip" disappears, the issue is Ken Burns, not RIFE.
6. **Audio-driven Ken Burns beat.** The KB transform uses `rms_n` from analysis, so its motion couples to the audio. Could produce per-transition variability if the keyframe times happen to land on different RMS phases. The "better on some transitions than others" report fits this.
7. **Non-uniform keyframe times in this song.** [`pipeline/keyframes_timeline.py`](pipeline/keyframes_timeline.py) lets the user (or a beat-aligned planner) place keyframes at non-uniform `t_sec`. With non-uniform spacing, the IFNet velocity changes per segment, and the v6 cross-pair bridge (which always uses `s = 0.5`) is no longer at the visual midpoint between flanking body samples — it could feel timing-uneven across boundaries. Print `keyframe_times` for a problem render and check.
8. **High visual variance between consecutive keyframes.** If consecutive SDXL keyframes were sampled with very different prompts or seeds, IFNet has no real flow to estimate; its output near `s = 0.5` may look closer to a discrete crossfade than a morph. The cross-pair bridge `IFNet(kf_i, kf_{i+2})` would then look completely different from the flanking samples and read as a *jump*. This would make the v6 "skip" worse than v5 specifically when consecutive keyframes are visually distant.
9. **Compositor frame timing off-by-one.** [`pipeline/compositor.py::render_full_video`](pipeline/compositor.py) generates frame timestamps as `start_sec + (i + 0.5) / fps`. Combined with the v6 timeline that has a sample at exactly `t_kf` (the bridge), a video frame at `(i + 0.5) / 30` may land between two timeline samples in a way that produces a discrete step. Trace what alphas the compositor computes for the 8 frames around an internal keyframe and confirm they look continuous.
10. **`_interpolate_frame` boundary handling.** With v6 the timeline has both a body sample and a bridge sample very close in wall-clock around each internal keyframe. Confirm `_interpolate_frame` correctly bisects to the right pair when the query `t` is exactly equal to a sample timestamp, and that no off-by-one in the binary search ever returns the wrong neighbor pair.

---

## Concrete diagnostic plan for the next session

Before any code changes:

1. **Get a sample.** Ask the user for a short rendered MP4 (10–20 s spanning at least one internal keyframe) plus the matching `cache/<hash>/background/manifest_rife.json` and a handful of `rife_timeline/*.png` around the boundary. Without these we are guessing.
2. **Confirm config.** Have them paste the relevant `OrchestratorInputs` for the render: `sdxl_rife_morph`, `rife_exp`, `sdxl_ken_burns`, the `keyframes_timeline` entries, and the resolved `GLITCHFRAME_RIFE_IFNET_INSET` env.
3. **Frame-by-frame.** Extract every frame around an internal keyframe at the rendered fps (`ffmpeg -ss … -t 1.0 -i out.mp4 frames/%05d.png`). Look at the actual visual progression. Is there a duplicated frame? A jump? A static plateau?
4. **A/B with Ken Burns off.** Single biggest disambiguator. If "skip" disappears, the fix lives in `background_kenburns.py`, not RIFE.
5. **A/B with RIFE off.** Confirms that RIFE is genuinely the active path (and rules out the smoothstep-velocity-zero hypothesis).
6. **Trace the timeline in compositor.** Add a debug log that, for a small window around `t = keyframe_times[1]`, logs `(i, t, t0, t1, alpha, bracketing sample indices)` for each video frame. Confirm the compositor is sampling what `rife_build_morph_timeline` actually emitted.
7. **Verify cache integrity.** `python -c` script that loads `manifest_rife.json` and asserts every referenced PNG exists, has size > 0, and decodes to the expected resolution.

---

## Things to NOT try again without good reason

- Bumping `DEFAULT_IFNET_TIMESTEP_INSET` higher than 0.25 — it makes the morph "muted" (never reaches the keyframes visually) and didn't fix v6.
- Adding more samples around the boundary on the same flanking pairs — already tried, that's what v4/v5/v6 incrementally did.
- Switching from linear to smoothstep blend in the compositor for the RIFE path — was tried and reverted in commit `a51debf` because smoothstep made the "stops on still" worse (zero velocity at every sample boundary).

---

## Key files

| File | Role |
|------|------|
| [`pipeline/rife_runtime.py`](pipeline/rife_runtime.py) | `rife_build_morph_timeline`, `rife_exp_interpolate_pair`, `rife_ifnet_at_timestep`, `DEFAULT_IFNET_TIMESTEP_INSET`, `_resolve_ifnet_timestep_inset` |
| [`pipeline/background_stills.py`](pipeline/background_stills.py) | `RIFE_MANIFEST_SCHEMA_VERSION = 6`, `_apply_rife_morph_if_needed`, `_RifeFrameSource`, `BackgroundStills.background_frame`, `_interpolate_frame` (linear blend; smoothstep only for sparse SDXL-only path) |
| [`pipeline/background_kenburns.py`](pipeline/background_kenburns.py) | `_ken_burns_transform` (int() box-coord cast — possible jitter source), `apply_ken_burns_to_rgb_array` |
| [`pipeline/compositor.py`](pipeline/compositor.py) | `render_full_video` — frame timing `t = start_sec + (i + 0.5) / fps`, calls `background.background_frame(t)` |
| [`pipeline/renderer.py`](pipeline/renderer.py) | `_build_ffmpeg_args` — bgr24 raw pipe, `-r 30`, default vsync |
| [`pipeline/keyframes_timeline.py`](pipeline/keyframes_timeline.py) | `KeyframeTimelineEntry`, `entries_to_keyframe_plans` — non-uniform `t_sec` allowed |
| [`pipeline/rife_vendor/ifnet_hdv3.py`](pipeline/rife_vendor/ifnet_hdv3.py) | `IFNet.forward` — flow displacement collapses near `s=0` / `s=1` |
| [`docs/technical/rife-morph-background.md`](docs/technical/rife-morph-background.md) | Long-form spec of the v6 timeline, schema-bump history, cache layout |
| [`tests/test_rife_morph_timeline.py`](tests/test_rife_morph_timeline.py) | v6 contracts: bridge per internal boundary, cross-pair math, bookend velocity, env override |

---

## Quick verification

Tests for the timeline math (do **not** require CUDA / RIFE weights — everything is mocked):

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rife_morph_timeline.py -v
```

If `torch` is not in the active venv (e.g. on a CPU-only dev box), the import will fail before any test runs. The fakes in `test_rife_morph_timeline.py` mock the IFNet calls but still import `pipeline.rife_runtime`, which imports `torch` at module load. Run from a venv that has `torch` installed.

---

## Checklist for the next agent

- [ ] Read `ai-context.md` and this file fully, including the verbatim user reports and the "Why we are stuck" section.
- [ ] Resist the urge to ship v7 immediately. The user explicitly asked us to step back.
- [ ] Run the **Cache invalidation reality check** (5 items) before forming hypotheses.
- [ ] Ask the user for the **diagnostic plan** items in priority order: rendered sample, config dump, KB-off A/B.
- [ ] Only after the diagnostic data is in hand, choose between (a) a Ken Burns / encoder fix, (b) a compositor blend-aware fix, (c) a content-conditional fix for high-variance keyframe sequences, (d) something we haven't thought of.
- [ ] When committing, update `RIFE_MANIFEST_SCHEMA_VERSION` only if the dense timeline structure actually changes; do **not** bump it for compositor-side or renderer-side fixes.

---

# Secondary blocker — HiDream background keyframes (Float8 vs BF16)

**Status:** Active but parallel to the RIFE morph issue above. Keep this section if the next agent ends up touching HiDream; otherwise it can be ignored for the RIFE work.

**Goal:** Optional **SDXL alternative** in *Background keyframes*: [`pipeline/background_stills_hidream.py`](pipeline/background_stills_hidream.py) spawns [`pipeline/background_stills_hidream_worker.py`](pipeline/background_stills_hidream_worker.py) in a **separate Python** so HiDream's CUDA / `flash-attn` stack does not mix with Glitchframe's venv.

**Doc:** [`docs/technical/background-stills-hidream.md`](docs/technical/background-stills-hidream.md) · **Env samples:** [`.env.example`](.env.example)

### Symptom

First keyframe fails with:

```text
RuntimeError: Promotion for Float8 Types is not supported, attempted to promote BFloat16 and Float8_e4m3fn
```

**Cause (high level):** Upstream `HiDream-O1-Image` `models/pipeline.py` `generate_image` uses `dtype = torch.bfloat16` and `torch.autocast(..., dtype=dtype)`. **FP8** checkpoints (default **dev** download `drbaph/HiDream-O1-Image-FP8`) leave `Float8_e4m3fn` in the graph; PyTorch does not promote that with BF16 the way this path expects.

### Directions for HiDream

| Direction | Notes |
|-----------|-------|
| **Official BF16 dev repo** | Set `GLITCHFRAME_HIDREAM_HF_REPO_ID=HiDream-ai/HiDream-O1-Image-Dev` (may need new cache dir / clear weights). |
| **Dedicated HiDream venv** | Set `GLITCHFRAME_HIDREAM_PYTHON` to Pinokio's HiDream app `python.exe`. |
| **Patch `generate_image`** | Monkeypatch or fork: `dtype = torch.float32` + compatible autocast when FP8 detected. |
| **Verify after `from_pretrained`** | Log dtypes of parameters/buffers; confirm whether `torch_dtype=float32` actually removes Float8 everywhere. |

### Key HiDream files

| File | Role |
|------|------|
| `pipeline/background_stills_hidream.py` | Parent: `load_hidream_config`, `_HiDreamWorker`, `BackgroundStillsHiDream` |
| `pipeline/background_stills_hidream_worker.py` | Subprocess: JSONL, UTF-8, stdout hijack, `_native_weights_torch_dtype` |
| `pipeline/background.py` | `create_background_source`, `hidream_config=` for tests |
| `tests/test_background_hidream.py` | Config/factory only (no GPU worker) |
