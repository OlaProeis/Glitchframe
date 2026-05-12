# RIFE morph between SDXL keyframes

Optional **optical-flow** interpolation (Practical-RIFE–style IFNet) between consecutive **SDXL stills** so the full-frame background **warps** from keyframe to keyframe instead of relying only on smoothstep **crossfades**.

## When it applies

- **Background mode** must be **`sdxl-stills`**. RIFE is **not** used for static Ken Burns uploads or for AnimateDiff (AnimateDiff’s init stills are generated with RIFE disabled so seed PNGs stay raw).
- **Defaults (UI):** **Ken Burns on SDXL stills** and **Morph keyframes (RIFE)** are **on** unless the user turns them off. Programmatic defaults match via `OrchestratorInputs.sdxl_ken_burns` / `sdxl_rife_morph`.

## User-facing controls (`app.py`)

| Control | Effect |
|--------|--------|
| **Morph keyframes (RIFE)** | After SDXL keyframes exist, runs RIFE between each adjacent pair and replaces the sampling timeline with a denser control set. |
| **RIFE subdivisions (2^N steps between keyframes)** | Integer **N** in **[2, 8]**; **2^N** centered IFNet samples per segment (see “Centered sampling” below). Higher = smoother morphs and **longer** RIFE bake. Default **4** → 16 IFNet samples per segment; **N=8** → 256 (heavy). |

Ken Burns (if enabled) is applied **after** the background sample at time `t`, unchanged from pre-RIFE behaviour.

**Sampling:** `background_frame(t)` uses **linear** blending between neighbouring RIFE samples (dense optical-flow timeline). Sparse SDXL-only keyframes still use **smoothstep** crossfades—double-easing dense RIFE was slowing motion toward every sample and stretching the blend from the last approximate IFNet frame to the exact endpoint still.

### Inset-warped centered sampling (no near-keyframe visual cluster)

Plain centered sampling ``(j + 0.5)/n`` (the v2 / v3 timeline) avoided emitting *exact* SDXL stills at internal keyframe boundaries, but it still placed the boundary IFNet samples of every segment at ``s ≈ 0.5/n`` and ``s ≈ (n-0.5)/n``  (≈ 0.031 / 0.969 at the default ``N = 4``). At those near-endpoint timesteps IFNet's flow displacement collapses toward zero, so the predicted frames are visually near-identical to the SDXL keyframe pixel-wise. The compositor then renders an extended ~``T_seg/n`` window of "near-keyframe" pixels around every internal keyframe time, which the viewer perceives as the morph **pausing on the original still**, even though no exact still is in the timeline.

The dense morph timeline therefore samples IFNet at **inset-warped** centered timesteps. For each segment ``[kf_i, kf_{i+1}]`` of duration ``T_seg``:

* Sample IFNet at ``s_j = inset + (1 - 2*inset) * (j + 0.5)/n`` for ``j ∈ [0, n)`` (where ``n = 2**N`` and ``inset`` defaults to **0.25** at v6, was **0.12** in v4 / v5; override via ``GLITCHFRAME_RIFE_IFNET_INSET``, clamped to ``[0, 0.45]``). The closest-to-boundary samples now sit at ``s ≈ inset`` and ``s ≈ 1 - inset`` instead of the legacy ``≈ 0.5/n`` / ``≈ 1 - 0.5/n``, so every dense sample carries visible flow displacement and is visually distinct from both bracketing keyframes. The v6 default ``0.25`` puts each seam at ~73 % keyframe-dominant blends versus v4 / v5's ~86 %, halving the perceived "pull" toward the shared keyframe at every boundary.
* Wall-clock placement of body samples is **decoupled** from the IFNet timestep and stays uniform centered: ``t_j = t_kf_i + T_seg * (j + 0.5)/n``. Apparent motion velocity is therefore constant — only the visual *content* of each frame is shifted away from the keyframe.

### Cross-pair boundary bridges (no static plateau at internal keyframes)

v4 / v5 left a subtler perceptual issue at every *internal* keyframe time even after the inset moved boundary samples away from ``s ≈ 0`` / ``s ≈ 1``: the last body sample of segment ``i`` and the first body sample of segment ``i + 1`` are taken from **different pairs** but both visually pulled toward the **same shared keyframe** ``kf_{i+1}``. The compositor's pixel-space linear blend between them therefore crossfades two near-``kf_{i+1}`` images for the entire ``T_seg/n`` boundary window — and a crossfade between two visually-similar images is, in pixel space, indistinguishable from holding that image static for the duration of the blend. The viewer reads it as **smooth → skip → still image → smooth**: the morph approaches ``kf_{i+1}`` smoothly, then snaps onto the keyframe-looking pixel content for ~``T_seg/n`` seconds, then resumes — exactly the symptom users hit even with v5 inset and bookend fixes in place.

v6 inserts a **cross-pair bridge** sample at every internal keyframe time. For each internal boundary at ``t = t_{i+1}`` (between segment ``i`` and segment ``i + 1``):

* Compute one IFNet inference on the **cross pair** ``(kf_i, kf_{i+2})`` at ``s = 0.5`` (i.e. on the keyframes immediately *before* and *after* the seam keyframe, skipping ``kf_{i+1}``).
* Place that sample at exactly ``t = t_{i+1}`` (one frame, on the keyframe time).

The bridge is visually still in the ``kf_{i+1}`` neighbourhood for coherent SDXL sequences, but its **flow direction** points *across* the keyframe (``kf_i → kf_{i+2}``) instead of *into* it (``kf_i → kf_{i+1}``) on one side and *out of* it (``kf_{i+1} → kf_{i+2}``) on the other. The two flanking body samples and the bridge therefore carry three visibly different ghost states, and the compositor now blends across two ``T_seg/(2n)`` windows (last body → bridge → first body) rather than one ``T_seg/n`` window — three distinct ghost states traversed in the same wall-clock duration as before, which breaks the static plateau without removing the visual pass through ``kf_{i+1}``.

Cost: one extra IFNet inference per internal keyframe (``N - 2`` extra calls per song, where ``N`` is the keyframe count). At a typical 30-keyframe song that's ~28 extra calls, ~1.5 s of bake on a 3090-class card — negligible against the existing per-segment budget.

### Velocity-matched IFNet bookends (no skip at song open / close)

The v4 sampling eliminated the **internal** keyframe pause but introduced a velocity discontinuity at the song's bookends: keeping the *exact* SDXL still at ``t = 0`` (``s = 0``) and ``t = duration`` (``s = 1``) leaves an IFNet jump of ``inset + span/(2n)`` between the bookend and the first body sample over only ``T_seg/(2n)`` wall-clock seconds. At the legacy ``inset = 0.12`` and ``N = 4`` (``n = 16``) that was a ``0.144`` IFNet-timestep jump in ``0.25 s`` — roughly **6×** the body's pace ``span/T_seg``. The compositor renders ~7–8 video frames flying through 14 % of the morph in that window, which the viewer reads as a brief "skip" right before the song's closing still (and a symmetric one at the open).

v5 fixes this by sampling the **bookends** at IFNet timesteps that match the body velocity:

* **Start bookend** at ``t = keyframe_times[0]``: IFNet at ``s = inset`` on the first keyframe pair ``(kf_0, kf_1)`` — IFNet jump to the first body sample becomes ``span/(2n)`` (half a body step), which divided by ``T_seg/(2n)`` wall-clock equals ``span/T_seg`` (the body velocity).
* **End bookend** at ``t = keyframe_times[-1]``: symmetrically, IFNet at ``s = 1 - inset`` on the last keyframe pair ``(kf_{N-1}, kf_N)``.

The visual cost is small: IFNet at ``s = inset`` is within ~``inset`` flow displacement of ``kf_0`` (and symmetrically for ``kf_N``), so the song still visibly opens and closes on the generated background image — just with a hint of motion instead of a freeze that snaps. The eye reads the resulting uniform velocity as a smooth open/close instead of the previous skip. Setting ``GLITCHFRAME_RIFE_IFNET_INSET=0`` collapses the bookends to ``s = 0`` / ``s = 1`` (≈ exact stills under IFNet), which is also the legacy uniform-velocity case (no skip at ``inset = 0`` either); cross-pair bridges still apply at ``inset = 0`` (they're independent of the body inset).

**Spacing:** within a segment the wall-clock spacing of body samples is uniform ``T_seg / n`` (e.g. with ``T_seg = 8 s`` and ``N = 4`` that's ``0.5 s``). Across every internal keyframe boundary the spacing is halved to ``T_seg / (2n)`` per side because the v6 cross-pair bridge sits exactly on ``t_kf``. The only other spacing irregularity is the short ``T_seg/(2n)`` gap between the velocity-matched bookend and its neighbouring body sample, where IFNet velocity now matches the body — visually smooth.

**Frame count formula:** ``frame_count = total_segs * n_steps + (total_segs - 1) + 2`` — body samples + cross-pair bridges + two velocity-matched bookends. For ``N = 31`` keyframes (30 segments) at ``rife_exp = 4`` (``n_steps = 16``): ``30 * 16 + 29 + 2 = 511`` frames (vs. 482 in v5 and earlier).

**Cache invalidation:** ``RIFE_MANIFEST_SCHEMA_VERSION`` is **6**. v1 → centered (v2) introduced no exact internal stills; v2 → v3 added ``keyframes_content_hash`` (SHA-256 over keyframe RGB payloads in order) so **replacing or editing keyframe PNGs** invalidates the RIFE cache even when text prompts and layout are unchanged; v3 → **v4** introduced inset-warped centered sampling; v4 → **v5** replaced the exact-still bookends with velocity-matched IFNet bookends (eliminating the skip at song open / close); v5 → **v6** raised the body inset default ``0.12 → 0.25`` and inserted a cross-pair IFNet bridge at every internal keyframe (eliminating the static plateau at internal keyframe boundaries). Older manifests are silently re-baked the next time RIFE runs.

## Implementation sketch

- **Vendor code:** `pipeline/rife_vendor/` — warper + `IFNet` (MIT, derived from [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)).
- **Runtime:** `pipeline/rife_runtime.py` — downloads weights via `huggingface_hub`, builds the dense timeline with `rife_build_morph_timeline`.
- **Integration:** `pipeline/background_stills.py` — after SDXL `ensure`, `_apply_rife_morph_if_needed` either loads cache or runs RIFE; `background_frame(t)` uses `RifeMorphManifest.times` when RIFE is active (linear interpolation within segments), else `BackgroundManifest.keyframe_times` with smoothstep.
- **Factory / orchestrator:** `create_background_source(..., sdxl_rife_morph=..., rife_exp=...)`; `OrchestratorInputs` carries the same flags.

## Weights and licensing

- Checkpoint: **Hugging Face** [`MonsterMMORPG/RIFE_4_26`](https://huggingface.co/MonsterMMORPG/RIFE_4_26) (`train_log/flownet.pkl`, ~24 MB). First use downloads into `MODEL_CACHE_DIR` under `rife_practical/…` (same Windows symlink-safe hub behaviour as WhisperX / SDXL caches).
- **Code** follows upstream Practical-RIFE **MIT** terms; **verify** checkpoint license/host terms for your redistribution scenario.

## Cache layout

```
cache/<hash>/background/
  manifest.json              # SDXL stills (unchanged)
  keyframe_*.png
  manifest_rife.json         # RIFE cache key + per-frame times
  rife_timeline/
    rife_000000.png
    ...
```

Invalidates when the **SDXL** manifest key changes (prompt / model / resolution / keyframe count), when **RIFE** settings change (`rife_exp`, output width/height, repo id), when the manifest **schema version** changes (e.g. v1 → v2 centered sampling, v2 → v3 pixel fingerprint), when **keyframe image bytes** change (v3 `keyframes_content_hash`), when timeline PNGs are missing, or on a fresh bake (old `rife_timeline/*.png` and `manifest_rife.json` are removed before writing new frames).

## Requirements

- **CUDA** for the RIFE bake (same machine class as SDXL). If RIFE is on and CUDA is unavailable, the pipeline raises a clear error.
- Extra **GPU time** after SDXL: one forward pass per generated intermediate between pairs (scales with subdivision **N** and number of keyframe intervals).

## GPU inference settings

`rife_runtime.py` mirrors what upstream Practical-RIFE does to reach its quoted 3090-class throughput, none of which were enabled previously:

- **`torch.inference_mode()` around every IFNet forward pass.** Disables autograd graph construction (we never call `.backward()`); cuts inference latency ~15–20 % and roughly halves transient VRAM versus `.eval()` alone.
- **`torch.backends.cudnn.benchmark = True` while baking.** IFNet runs the same input shape thousands of times per timeline, so cudnn's algorithm-search cache pays off after the first call. The flag is saved/restored around the bake so other CUDA users (SDXL, AnimateDiff) keep their previous setting.
- **FP16 (auto-on for CUDA).** `load_rife_model` calls `net.half()` on CUDA and `_prepare_pair_tensors` matches the model's dtype. The 3090's tensor cores roughly halve forward-pass time and also halve IFNet's VRAM footprint. Outputs are quantised to uint8 immediately, so any FP16 round-off is lost in the cast — visually byte-equivalent in spot checks. `pipeline/rife_vendor/warplayer.py` keys its `_backwarp_grid` cache on `dtype` so the same `F.grid_sample` call works in either precision (PyTorch ≥ 1.10 requires `input` and `grid` dtypes to match). Disable with `GLITCHFRAME_RIFE_FP16=0` if you ever need byte-exact reproducibility against an FP32 baseline.

Combined, the GPU phase at **N=8** on a 3090-class card drops from ~30+ minutes to roughly 5–10 minutes per typical SDXL keyframe set — small enough that the save phase (covered above) becomes the next thing to watch.

## Save phase (cache writes) and RAM behaviour

The dense morph timeline is persisted as one PNG per frame under `cache/<hash>/background/rife_timeline/`. PIL's PNG encoder is single-threaded and CPU-bound (zlib), so on high-`rife_exp` runs (e.g. **N=8** → 256 frames per segment, often 5 000+ frames at 1080p) the save phase used to dominate wall-clock far more than the actual GPU bake. NVMe disk bandwidth was never the bottleneck — the encoder was, and the in-RAM accumulation behind it.

`_apply_rife_morph_if_needed` now uses a streaming bake plus a lazy on-disk source for both fresh bakes and cache hits:

- **Streaming write during the bake.** `rife_build_morph_timeline` accepts an `on_frame(idx, rgb, t)` callback (`keep_frames=False` skips the in-RAM list). The bake fans frames into a small `ThreadPoolExecutor` (`max(2, min(8, os.cpu_count()))`) of PNG writers running with `compress_level=1` (lossless; ~3–5× faster encode at the cost of ~30–50 % larger cache files; user-visible keyframes still use the default `6`). A small in-flight queue caps RAM at roughly `workers × frame_size`. The PNG manifest is written **last**, after every writer drains, so a crashed bake leaves an inconsistent state that fails `_rife_timeline_pngs_complete` on retry rather than presenting a half-written manifest.
- **Lazy in-RAM source for sampling.** `_RifeFrameSource` is a thread-safe disk-backed `Sequence[np.ndarray]` with a small LRU cache (default 16 frames). `_interpolate_frame` only ever reads two adjacent frames per call, and the compositor walks time monotonically, so this size is enough to keep every read warm. Both the cache-hit path and the post-bake handoff install this source instead of decoding all PNGs into a `list[np.ndarray]`.

Combined, peak RAM for the RIFE timeline drops from `frame_count × frame_size` (≈ 36 GB at 1080p × 5 889 frames) to roughly `workers × frame_size` during the bake and `cache_size × frame_size` during render — typically well under 200 MB. The save phase wall-clock collapses from hours to minutes on typical hardware.

## Tests

- `tests/test_rife_manifest.py` — `RifeMorphManifest` JSON round-trip and `matches_key`.

## See also

- `docs/technical/background-stills.md` — SDXL keyframe planning and crossfade baseline.
- `docs/technical/background-modes.md` — `BackgroundSource` factory and modes table.
