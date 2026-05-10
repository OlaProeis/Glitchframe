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

* Sample IFNet at ``s_j = inset + (1 - 2*inset) * (j + 0.5)/n`` for ``j ∈ [0, n)`` (where ``n = 2**N`` and ``inset`` defaults to **0.12**, override via ``GLITCHFRAME_RIFE_IFNET_INSET``, clamped to ``[0, 0.45]``). The closest-to-boundary samples now sit at ``s ≈ inset`` and ``s ≈ 1 - inset`` instead of the legacy ``≈ 0.5/n`` / ``≈ 1 - 0.5/n``, so every dense sample carries visible flow displacement and is visually distinct from both bracketing keyframes.
* Wall-clock placement is **decoupled** from the IFNet timestep and stays uniform centered: ``t_j = t_kf_i + T_seg * (j + 0.5)/n``. Apparent motion velocity is therefore constant — only the visual *content* of each frame is shifted away from the keyframe.
* The exact SDXL still is emitted **only** as the very first frame (``t = 0``) and very last frame (``t = duration``) of the timeline — never at internal keyframe times. The bookend stills now blend into a meaningfully-displaced first IFNet sample (``s ≈ inset``, not ``s ≈ 0``), so the song's open and close are smooth ease-ins rather than the previous near-static intros.

Setting ``GLITCHFRAME_RIFE_IFNET_INSET=0`` recovers the legacy plain centered sampling for byte-exact reproducibility against an older bake.

**Spacing:** within a segment and across every internal keyframe boundary the wall-clock spacing is uniform ``T_seg / n`` (e.g. with ``T_seg = 8 s`` and ``N = 4`` that's ``0.5 s``). The only spacing irregularity is the short ``T_seg/(2n)`` gap between the first/last exact-still bracket and its neighbouring IFNet sample.

**Cache invalidation:** ``RIFE_MANIFEST_SCHEMA_VERSION`` is **4**. v1 → centered (v2) introduced no exact internal stills; v2 → v3 added ``keyframes_content_hash`` (SHA-256 over keyframe RGB payloads in order) so **replacing or editing keyframe PNGs** invalidates the RIFE cache even when text prompts and layout are unchanged; v3 → **v4** introduced inset-warped centered sampling. Older manifests are silently re-baked the next time RIFE runs.

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
