# Session Handover — Glitchframe / Windows Pinokio “Align lyrics” + related work

**Purpose:** Bring a **new agent/session** up to speed on **Align lyrics**, **WhisperX / faster-whisper / CTranslate2**, **Pinokio**, and **dependency pinning** attempts—without rereading the whole chat.

**Always read [`ai-context.md`](ai-context.md) first** for project rules and architecture.

---

## Environment

- **Project:** Glitchframe (music video generator, Gradio UI).
- **Python:** 3.11+ (`requires-python`); **WhisperX 3.3.0 declares `<3.13`**—the **pinned Windows “Track A”** stack only applies to **Python 3.11 / 3.12** on Windows.
- **Interpreter (local dev):** `.\.venv\Scripts\python.exe` — Windows `py` launcher may **not** point at this venv.
- **Tests:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`
- **Branch:** `main`

---

## The core problem

On **Windows**, **Align lyrics** (WhisperX → faster-whisper → **CTranslate2**) often fails with native DLL errors, commonly:

- `Could not load library cudnn_ops_infer64_8.dll`
- **`Error code 1920`** (“The file cannot be accessed by the system”—can be AV/path/ACL, not only missing DLL)

Logs often show **Silero VAD first**; Silero itself is **CPU** in WhisperX. The failure is usually **CTranslate2 / CUDA / cuDNN** when the Whisper encoder runs—not Pyannote VAD.

**Pinokio vs local `.venv`:** Same `git` tree does **not** mean identical behavior. Different **pip solve**, **`C:\pinokio\...` paths**, antivirus, etc. Comparison needs **`pip freeze` from both envs** (runbook lives in **`docs/technical/pinokio-lyrics-align-windows-handover.md`** → § *Runbook*).

---

## What we implemented in-repo (committed)

These are **not** hypothetical—they are on `main` (see recent commits around Pinokio / Windows lyrics / Gradio fixes).

### 1. Windows dependency “Track A” (Python 3.11–3.12 on Windows only)

 **`pyproject.toml`** optional extras **`all`**, **`lyrics`**, **`analysis`**, **`vocals`** (torchaudio clause) use **PEP 508**:

- **`platform_system == "Windows"` and `python_version < "3.13"`** → pin:

  | Package | Pin | Note |
  |---------|-----|------|
  | PyTorch CUDA | `torch==2.2.2+cu121`, `torchvision==0.17.2+cu121`, `torchaudio==2.2.2+cu121` | CUDA **12.1** wheel index—not cu124 |
  | WhisperX | **`==3.3.0`** | **Important:** WhisperX **3.3.6+** declares **`torch>=2.5.1`** and conflicts with 2.2.2—we deliberately use **3.3.0**, which declares `torch>=2`, `ctranslate2<4.5`, `faster-whisper==1.1.0`. |
  | faster-whisper | `==1.1.0` | Matches WhisperX 3.3.0 metadata |
  | ctranslate2 | `==4.4.0` | Aligns with “cuDNN8-style naming” narrative for older stacks |

- **`platform_system == "Windows"` and `python_version >= "3.13"`** → **`Track B`:** loose **`whisperx>=3.1`**, **`ctranslate2>=4.5,<5`**, **`torchaudio>=2`** (cu124-era flow)—because **WhisperX 3.3.0 does not support Python 3.13**.

### 2. Core dependency bounds (Gradio 4.x safety)

 **`requirements.txt`** and **`pyproject.toml`** now constrain:

- **`markupsafe>=2.0,<3`**
- **`pillow>=10,<11`** ( `<11` because Gradio 4.x rejects Pillow 12+ )

Gradio pulls these transitively but **explicit pins** reduce drift after torch reinstalls.

### 3. Pinokio **`install.js`**

Order (simplified):

1. `ensurepip`, `pip -U pip`
2. Install **torch trio** from **`https://download.pytorch.org/whl/cu121`** with exact Track A versions
3. `pip install -r requirements.txt`
4. `pip install -e .`
5. `pip install -e ".[all]"`
6. **Re-pin torch trio:** `pip install --force-reinstall **`--no-deps`** torch==… torchvision==… torchaudio==… --index-url cu121`  
   - **`--no-deps` is critical:** a full **`--force-reinstall`** **without** `--no-deps` was upgrading **MarkupSafe → 3.x** and **Pillow → 12.x**, **breaking Gradio** (UI / ingest errors—including weird lazy-import failures).
7. **`pip install "markupsafe>=2.0,<3" "pillow>=10,<11"`** to restore Gradio-compatible versions
8. Re-pin WhisperX stack: **`whisperx==3.3.0`**, **`faster-whisper==1.1.0`**, **`ctranslate2==4.4.0`**
9. Optional **`nvidia-cudnn-cu12`**
10. **`python scripts/windows_provision_cudnn_next_to_ctranslate2.py`** (best-effort copy of CUDNN-related DLLs next to **`ctranslate2`** package dir)

### 4. **`pipeline/lyrics_aligner.py` — `_pick_device`**

- Respect **`GLITCHFRAME_WHISPERX_DEVICE`** = **`cpu`** or **`cuda`** (with CUDA availability check for `cuda`).
- If unset: **`sys.platform == "win32"`** → **default `cpu`** for Align lyrics so Windows works without trusting Pinokio env (older docs assumed Pinokio always set vars).

### 5. **`start.js`** (Pinokio)

- Sets **`GLITCHFRAME_WHISPERX_VAD_METHOD=silero`**
- Sets **`GLITCHFRAME_WHISPERX_DEVICE=cpu`** (explicit safe default—user can remove for GPU trial)
- Sets **`HF_HUB_DISABLE_SYMLINKS=1`** + **`HF_HUB_DISABLE_SYMLINKS_WARNING=1`** so `huggingface_hub >= 0.36` copies blobs instead of symlinking (avoids `WinError 1314` when running as non-admin / Developer Mode off — Bug F in the handover doc)

### 6. **`pipeline/win_cuda_path.py`** + **`app.py`**

- **`ensure_windows_cuda_dll_paths()`** before importing WhisperX / CTranslate2: `os.add_dll_directory` for **`torch\lib`**, **`nvidia\*\bin`**, **`ctranslate2`**, plus PATH prepend (once).

- **`app.py`** startup probe logs torch; imports whisperx for diagnostics; **ctranslate2 version logging** distinguishes **Track A** (expects **4.4.x** on Win/py&lt;313) vs **Track B** (expects **≥4.5** on Win/py≥313)—**does not** tell Track A users to “upgrade past 4.5” blindly anymore.

### 7. **`scripts/windows_provision_cudnn_next_to_ctranslate2.py`**

Copies plausible **`cudnn*_8.dll`** names from **`torch\lib`** and **`nvidia\cudnn\bin`** into **`ctranslate2`** package folder. Does **not** download NVIDIA’s standalone cuDNN 8.9.7 MSI.

### 8. Documentation (where to look; README is deliberately thin on Pinokio internals)

| Doc | Role |
|-----|------|
| [`docs/technical/pinokio-package.md`](docs/technical/pinokio-package.md) | What each Pinokio file does; **`no interactive shell`** in Pinokio—**Update + Install/Reinstall**; optional **File Explorer → address bar → `cmd` → `env\Scripts\activate.bat`** for manual pip |
| [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) | Investigation history, **`Runbook`** (traceback + `pip freeze`), Track A/B summary |
| [`docs/technical/windows-venv-recovery-guide.md`](docs/technical/windows-venv-recovery-guide.md) | Track A vs Track B reinstall steps |
| [`docs/technical/lyrics-aligner.md`](docs/technical/lyrics-aligner.md) | Brief “platform deps” paragraph |
| **README** | High-level Pinokio blurb + troubleshooting bullets; **not** long Pinokio-terminal UX |

---

## What user reports indicated (post-changes)

- **Pins + install.js changes did not universally “fix Pinokio”**—owner still saw **similar Align lyrics failures** in some Pinokio runs (Dll / cuDNN class).
- **Gradio breakage** surfaced after **`--force-reinstall`** torch pulled incompatible **MarkupSafe / Pillow**—addressed with **`--no-deps`** + explicit reinstall of those pins (see § committed above).
- **Owner cannot use an obvious Pinokio-integrated shell**—documentation now states **Repair = Update + Reinstall** in Pinokio; **manual pip** requires **Explorer + `cmd`** outside Pinokio. **Removed** long “no shell” copy from README (kept technical doc only)—see **`d33c236`** era README commits.

---

### 9. Compat shims for newer libraries on Track A torch 2.2.2

Three monkey-patches applied at `app.py` startup so newer
`diffusers` / `transformers` / `peft` / `accelerate` / `huggingface_hub`
import cleanly against pinned PyTorch 2.2.2 + Windows non-admin:

- **`pipeline/_speechbrain_compat.patch_speechbrain_lazy_module()`** —
  fixes the `inspect.py` separator bug in speechbrain's `LazyModule`
  guard so audio ingest doesn't force-import every speechbrain
  integration (k2/flair). See handover § Bug A.
- **`pipeline/_torch_xpu_compat.patch_all()`** — installs permissive
  `torch.xpu` and `torch.distributed.device_mesh` stubs (PyTorch 2.3
  / 2.4 surfaces) so import-time `RANDOM_FUNCS` / `device_mesh` probes
  don't crash on torch 2.2.2. Stubs are class-shaped to support PEP
  604 unions (`DeviceMesh | None`) in type annotations. Also wired
  into `pipeline/background_animatediff._import_animatediff_sdxl`
  belt-and-braces. See handover § Bug E.
- **`pipeline/_huggingface_symlink_compat.patch_huggingface_disable_symlinks()`** —
  rebinds `huggingface_hub.are_symlinks_supported` to `False` on
  Windows so non-admin users (Pinokio default) don't crash with
  `WinError 1314` while writing snapshot dirs. Also wired into
  `pipeline/lyrics_aligner._import_whisperx`. See handover § Bug F.

### 10. CPU retry for cuDNN-class GPU alignment failures (DONE)

`pipeline/lyrics_aligner.align_lyrics` wraps `_run_whisperx_forced`
with a single retry on `device="cpu"` / `compute_type="int8"` when
the GPU attempt's exception (or any `__cause__` / `__context__`)
matches `_is_cudnn_class_error`. SDXL / Demucs / render stay on CUDA.
Tests in `tests/test_pinokio_windows_resilience.py` pin the matcher
patterns and the no-retry-when-already-CPU behaviour.

### 11. Multi-candidate ffmpeg discovery + UI progress fix (DONE)

`pipeline/ffmpeg_tools.py` now enumerates every candidate ffmpeg
(env override → active venv/conda env's `bin` → PATH → well-known)
and `select_video_codec()` sweeps all candidates probing for
`h264_nvenc` before falling back to `libx264`. The first
codec-capable binary is promoted into `_cache` so subsequent encode
commands use it. Probes run at `-loglevel info` and capture the
last 14 stderr lines on failure (fixes the case where Pinokio's
NVENC failed silently because real diagnostics were suppressed at
`-loglevel error`).

`pipeline/compositor.py` decouples per-frame progress from the
producer thread: the producer writes scalar counters to
`_CompositorStats` and the consumer (encoder feed loop, on the
request thread) polls every 250 ms and forwards to `gr.Progress`.
Fixes the symptom where the UI bar parked at "Compositing video..."
for the entire render even though the terminal log was updating.
New message format includes live fps + ETA + active-layer label.
After a render, `app._summarise_render` appends a one-line summary
(`compositor: N frames in Xm Ys · avg Z.ZZ fps · encoder=...`) to
the in-app run log so fork users on read-only Pinokio terminals can
confirm whether NVENC actually engaged.

See handover § Bug G for full details.

## What we have **not** implemented

Nothing currently outstanding from this thread. CPU retry, compat
shims, ffmpeg discovery, and progress threading are all on `main`.

---

## Other backlog (outside this thread)

- **Task Master #55** — `scanline_tear` renderer (pending in TM); **not** part of lyrics/Pinokio work unless explicitly picked up.

---

## Verification commands (after edits)

```powershell
.\.venv\Scripts\python.exe -m compileall . -q
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

---

## Checklist for the next agent

- [ ] Read **`ai-context.md`** and this file end-to-end
- [ ] Read [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) Bugs A–G for the full investigation history
- [ ] Pinokio regressions: confirm **`install.js`** still uses **`--no-deps`** + **markupsafe/pillow** lines after torch trio
- [ ] If touching `pipeline/compositor.py` progress code: keep `progress(...)` calls on the request thread; the producer must only write `_CompositorStats` scalars
- [ ] If touching `pipeline/ffmpeg_tools.py`: don't fall back to a single-binary `_resolve()` — the multi-candidate sweep is what catches the conda-env-shadowed NVENC ffmpeg bug
- [ ] If adding new compat shims (newer `diffusers` / `transformers` / etc. probing missing torch APIs): add to `pipeline/_torch_xpu_compat.py`'s permissive stub family rather than patching one attribute at a time
- [ ] Do **not** re-expand README with Pinokio-terminal essays—use **`pinokio-package.md`**
