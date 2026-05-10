# Session Handover — HiDream background keyframes (Glitchframe)

**Purpose:** Bring a **new agent/session** up to speed on the **HiDream-O1-Image** path for the *Background keyframes* tab: what’s built, what’s broken, and what to try next—without rereading the whole chat.

**Always read [`ai-context.md`](ai-context.md) first** for project rules and architecture. **Lyrics / WhisperX / Pinokio align historic** details live in [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md)—**not** duplicated here.

**Branch:** HiDream integration commits are on **`dev`** (merge to `main` when stable). Confirm with `git branch` before editing.

---

## HiDream background keyframes — blocked on Float8 vs BF16 {#hidream-background-keyframes--blocked-on-float8-vs-bf16}

**Goal:** Optional **SDXL alternative** in *Background keyframes*: [`pipeline/background_stills_hidream.py`](pipeline/background_stills_hidream.py) spawns [`pipeline/background_stills_hidream_worker.py`](pipeline/background_stills_hidream_worker.py) in a **separate Python** so HiDream’s CUDA / `flash-attn` stack does not mix with Glitchframe’s venv.

**Doc:** [`docs/technical/background-stills-hidream.md`](docs/technical/background-stills-hidream.md) · **Env samples:** [`.env.example`](.env.example)

### Symptom (current blocker)

First keyframe fails with:

```text
RuntimeError: Promotion for Float8 Types is not supported, attempted to promote BFloat16 and Float8_e4m3fn
```

**Cause (high level):** Upstream `HiDream-O1-Image` [`models/pipeline.py`](https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/pipeline.py) `generate_image` uses **`dtype = torch.bfloat16`** and **`torch.autocast(..., dtype=dtype)`**. **FP8** checkpoints (default **dev** download **`drbaph/HiDream-O1-Image-FP8`**) leave **`Float8_e4m3fn`** in the graph; PyTorch does not promote that with BF16 the way this path expects.

### What we implemented (on `dev`)

1. **Auto-setup** — With no `.env` paths, `load_hidream_config(allow_fetch=True)` clones **`HiDream-ai/HiDream-O1-Image`** under **`GLITCHFRAME_MODEL_CACHE/hidream/`** and `snapshot_download`s weights (**dev** → drbaph FP8; **full** → `HiDream-ai/HiDream-O1-Image`). Manifest/UI peek uses `allow_fetch=False` / `strict_env=False` where appropriate.

2. **Upstream API** — No `HiDreamImagePipeline`; worker **`--pipeline-import auto`** uses **`generate_image`** + **`Qwen3VLForConditionalGeneration`** (same as upstream `inference.py`), with fallback if an old fork still exposes a pipeline class.

3. **Windows worker robustness**
   - **`_reconfigure_stdio_utf8()`** — avoids **`UnicodeEncodeError`** when `transformers` prints emoji during import on cp1252 consoles.
   - **`_wire_jsonl_only_stdout()`** — parent reads **only JSON lines** from the real stdout pipe; **`_emit`** writes to saved **`_JSONL_STDOUT`**; **`sys.stdout = sys.stderr`** so library `print()` does not corrupt the protocol. Opt-out: **`GLITCHFRAME_HIDREAM_WORKER_ALLOW_LIB_STDOUT=1`**.

4. **Float8 mitigation attempt** — **`_native_weights_torch_dtype()`** uses **`torch.float32`** for `from_pretrained` when path or **`GLITCHFRAME_HIDREAM_HF_REPO_ID`** suggests FP8, else **`bfloat16`**. Override **`GLITCHFRAME_HIDREAM_NATIVE_WEIGHTS_DTYPE`**. **User still hit Float8 promotion** afterward → likely some params/buffers stay Float8, or worker/env mismatch; **needs dtype audit** in the worker after load.

### Not solved

- Reliable **drbaph FP8** + upstream **`generate_image`** in this subprocess without BF16/Float8 clashes.
- No vendored patch to **`generate_image`** to force **float32** end-to-end (upstream hardcodes bf16 ~line 124).

### Directions for the next session

| Direction | Notes |
|-----------|--------|
| **Official BF16 dev repo** | Set **`GLITCHFRAME_HIDREAM_HF_REPO_ID=HiDream-ai/HiDream-O1-Image-Dev`** (may need new cache dir / clear weights). Heavier VRAM/download; avoids FP8 weights if the artifact is BF16/FP32. |
| **Dedicated HiDream venv** | Set **`GLITCHFRAME_HIDREAM_PYTHON`** to Pinokio’s HiDream app `python.exe`; still may not fix dtype if recipe is unchanged. |
| **Patch `generate_image`** | e.g. monkeypatch or forked line `dtype = torch.float32` + compatible autocast when FP8 checkpoint detected. |
| **Verify after `from_pretrained`** | Log dtypes of parameters/buffers; confirm whether **`torch_dtype=float32`** actually removes Float8 everywhere. |

### Key files

| File | Role |
|------|------|
| `pipeline/background_stills_hidream.py` | Parent: `load_hidream_config`, `_HiDreamWorker`, `BackgroundStillsHiDream` |
| `pipeline/background_stills_hidream_worker.py` | Subprocess: JSONL, UTF-8, stdout hijack, native load, `_native_weights_torch_dtype` |
| `pipeline/background.py` | `create_background_source`, `hidream_config=` for tests |
| `tests/test_background_hidream.py` | Config/factory only (no GPU worker) |

### Quick verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_background_hidream.py -v
```

---

## Checklist for the next agent

- [ ] Read **`ai-context.md`** and this file.
- [ ] On **`dev`** (or equivalent) with HiDream commits; reproduce under user’s Pinokio paths if possible.
- [ ] Before large changes: confirm **`background_stills_hidream_worker.py`** on disk matches `dev` (Pinokio sometimes syncs lagging).
