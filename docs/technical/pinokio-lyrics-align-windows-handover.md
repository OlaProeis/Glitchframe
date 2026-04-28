# Handover: Pinokio + Windows “Align lyrics” (WhisperX / cuDNN / Pyannote)

Use this as the **prompt for a fresh chat** when continuing this work. The owner reports **Align lyrics** still failing under **Pinokio on Windows** with **`cudnn_ops_infer64_8.dll` / error 1920** (or the same class of load failure), after multiple mitigations. They are concerned about **regressing the local (non-Pinokio) app**; prefer **small, testable changes** and consider **reverting** experimental pipeline edits if they do not help Pinokio.

---

## Goal

Run **Glitchframe** from **Pinokio** (`install.js` / `start.js` / …) on **Windows**, with **Align lyrics** (WhisperX + optional extras) working. **Local** `python -m app` outside Pinokio was reported **working** before this effort; changes should not unnecessarily alter local behaviour.

---

## Symptoms (Pinokio, Windows)

- Failure during WhisperX / VAD path, often after downloads succeed.
- Log line: `>>Performing voice activity detection using Pyannote...`
- Then: **`Could not load library cudnn_ops_infer64_8.dll`** (user also saw **error code 1920**).
- PyTorch **2.6.x+cu124**; Pyannote / Lightning warnings about checkpoint age are usually **not** the direct crash; the hard failure is **native DLL load** (cuDNN naming / faster-whisper / pyannote stack on Windows).

---

## What was added / changed (inventory for the next session)

### Pinokio packaging (repo root; should not affect local `python -m app` unless those files are used)

- `install.js` — `ensurepip`, then `pip`: torch **cu124** index, `requirements.txt`, `pip install -e .`, `pip install -e ".[all]"`. **No** `script.start` → `torch.js` (was hanging with no terminal output).
- `start.js` — `python -m app`, Gradio URL detection; **`env.GLITCHFRAME_WHISPERX_VAD_METHOD: "silero"`** (see pipeline section).
- `reset.js`, `update.js`, `pinokio.js`, `icon.png`, `pinokio_meta.json`
- `docs/technical/pinokio-package.md`, README **Pinokio** section

### Dependencies

- `pyproject.toml` — Windows-only (PEP 508) pin: **`ctranslate2==4.4.0`** on extras **`all`**, **`lyrics`**, **`analysis`** (mitigate faster-whisper / DLL name issues per README / whisperX#899). **No** effect on non-Windows installs of those extras.

### Pipeline (shared codebase — affects any run that uses these code paths / env)

- `pipeline/lyrics_aligner.py` — `whisperx.load_model` may receive **`vad_method`** when **`GLITCHFRAME_WHISPERX_VAD_METHOD`** is `pyannote` or `silero`; **if unset**, **do not** pass `vad_method` (WhisperX default, i.e. Pyannote for VAD). **Retry / TypeError** fallbacks preserved for older WhisperX.
- **Pinokio** is supposed to set `GLITCHFRAME_WHISPERX_VAD_METHOD` via `start.js` so **only Pinokio-launched** processes get **`silero`** unless the user sets `.env` locally.

### Docs / env examples

- `README.md` Troubleshooting (cudnn, Pinokio, ctranslate2, Silero)
- `docs/technical/windows-venv-recovery-guide.md` (section on cudnn)
- `.env.example` — documents `GLITCHFRAME_WHISPERX_VAD_METHOD`

---

## What we tried (chronological)

1. **Pinokio `install.js` used `script.start` → `torch.js` + `venv_python: 3.11`** — appeared **stuck** on step 1/3, empty terminal. **Replaced** with **explicit** `shell.run` + `pip` (torch from **cu124** index, etc.) for visible progress.
2. **Missing `pip` in venv** — some Pinokio venvs had no pip. **Added** **`python -m ensurepip --upgrade`** at the start of `install.js`.
3. **“One click”** — **`pip install -e ".[all]"`** added to `install.js` (Demucs, WhisperX, etc.); post-install text clarified that user must **click Start** (Pinokio does not auto-launch the server). **`default: true`** on **Start** in `pinokio.js` when `env` exists.
4. **`cudnn_ops_infer64_8.dll` after ctranslate2 path** — **Pinned `ctranslate2==4.4.0`** in Windows optional extras in **`pyproject.toml`**. **Did not** fix; log still showed **Pyannote** VAD.
5. **Pyannote VAD vs faster-whisper** — assumed Pyannote was the layer still loading bad DLLs. **Implemented** optional **`vad_method`** in `load_model` (Silero), then **scoped** so:
   - default local: **no** `vad_method` (unchanged from pre-feature behaviour for unset env);
   - Pinokio: **`start.js` sets `GLITCHFRAME_WHISPERX_VAD_METHOD=silero`**.
6. **User reports:** issue **still reproduces in Pinokio** (“same issue basically”). **Afraid the app is broken** and wants a **handover to a new chat**.

---

## What is still unknown / to verify in the next session

- Whether **`vad_method` is actually supported** and applied by the **installed WhisperX** version in the Pinokio venv (TypeError would log a warning and fall back).
- Whether the **Gradio / orchestrator process** in Pinokio **inherits** `start.js`’s `env` for **long-running** work (suspect yes, but confirm).
- Whether **Silero** is used when env is set (log should **not** say “using Pyannote” for that step, if the stack respects `vad_method`).
- Whether the failure is **entirely** in **Pyannote** or also in **faster-whisper/ctranslate2** on another line — **separate** stack traces help.
- **Full traceback** and **`pip list` / `pip show whisperx ctranslate2 pyannote-audio torch`** from **`C:\pinokio\api\Glitchframe.git\env`**.

---

## Suggested next steps (for the new chat)

- Collect **one clean log** from Pinokio: from click **Align lyrics** through crash, plus **`GLITCHFRAME_WHISPERX_VAD_METHOD`** value inside the process (or add temporary logging in `lyrics_aligner` at `load_model` time).
- Confirm **`start.js` `env` is passed** to the shell that runs `python -m app` in the user’s Pinokio version (Pinokio docs / minimal repro).
- If the goal is **Pinokio-only** fixes: prefer **`install.js` / `start.js` / `pinokio.js` + documented `.env`**, and **avoid** broad pipeline defaults; revert or feature-flag pipeline edits if they do not fix Pinokio.
- Revisit **upstream** WhisperX issues (e.g. **#899**, **Silero VAD** PRs) for **Windows + cu124 + torch 2.6** combinations.
- Last resort (documented in README): **NVIDIA cuDNN 8.9** DLL copy into venv `torch\lib` or `ctranslate2` package dir — user **dislikes** manual DLL surgery; treat as **documented opt-in**.

---

## Reverting / “get back to known-good local” (owner actions)

- **Local:** do **not** set `GLITCHFRAME_WHISPERX_VAD_METHOD` in `.env` unless needed; `lyrics_aligner` then matches **pre-`vad_method` behaviour** for unset env.
- **Full revert of pipeline + extras:** use `git log` on `main`/`master` and revert commits touching `pipeline/lyrics_aligner.py`, `pyproject.toml` (ctranslate2), and optionally Pinokio files — only if the team decides the experiment is not worth keeping.

---

## Key file references

| Area | Path |
|------|------|
| Align lyrics / WhisperX | `pipeline/lyrics_aligner.py` (`_run_whisperx_forced`, `load_model`) |
| Torch load shim | `pipeline/torch_checkpoint_compat.py` |
| Pinokio | `install.js`, `start.js`, `pinokio.js` |
| Windows recovery | `docs/technical/windows-venv-recovery-guide.md` |
| Package extras | `pyproject.toml` `[project.optional-dependencies]` |
| User env sample | `.env.example` |

---

## 2026-04-27 (confirmed in Pinokio)

- **Console** showed ``Performing voice activity detection using Silero...`` then  
  ``Could not load library cudnn_ops_infer64_8.dll. Error code 1920``. So **``GLITCHFRAME_WHISPERX_VAD_METHOD`` / Silero is working**; the crash is **not** Pyannote VAD. It is the **faster-whisper / CTranslate2** path, which can still look for **cuDNN8**-named DLLs if **ctranslate2 4.4.x** is installed, while **PyTorch cu124** may only ship **cuDNN9**-style names. WhisperX 3.8+ declares ``ctranslate2>=4.5.0``; a Windows pin on **4.4.0** in `pyproject.toml` was therefore wrong and was **replaced** with ``ctranslate2>=4.5.0``; ``install.js`` was updated to run ``pip install -U "ctranslate2>=4.5.0,<5"`` after ``.[all]``. The app also registers ``torch\lib`` (and common layout paths) with ``os.add_dll_directory`` before importing WhisperX (`pipeline/win_cuda_path.py`).

- **If it still fails after 4.5+ and PATH/DLL fixes:** set ``GLITCHFRAME_WHISPERX_DEVICE=cpu`` (Pinokio: uncomment in ``start.js``) so WhisperX ASR/align runs on **CPU** only — slow, but avoids **ctranslate2 GPU** cuDNN load on broken Windows stacks. ``install.js`` also runs ``pip install nvidia-cudnn-cu12`` to place extra cuDNN DLLs under ``site-packages`` for some setups.

### Deeper read (why it’s so brittle on Windows)

- The log … **Silero …** then **`cudnn_ops_infer64_8.dll`** is **misleading**: Silero in WhisperX uses **`onnx=False`** (PyTorch JIT) and stays **CPU** for inference; Snakers docs say the model is meant for CPU. **Silero itself is not requesting that cuDNN DLL.**
- The fatal load is almost certainly **CTranslate2** (inside **faster-whisper**) on the **first CUDA op** for the Whisper **encoder** — it fires **after** the Silero log lines during ``transcribe``, when the loader can’t satisfy **cuDNN** next to PyTorch **cu124** (e.g. **cuDNN 9**-style names shipped with torch vs **`cudnn_ops_infer64_8.dll`** expectations, **`add_dll_directory` / PATH** ordering, or a **missing dependency DLL** pulled in transitively). The printed **error code 1920** is **`The file cannot be accessed by the system`** in the usual Win32 wording — overlap with ACLs, virtualization, antivirus locks, broken paths, Store-Python quirks, **not only** “file not found.”
- **Fix that actually stops the churn:** ``pipeline/lyrics_aligner.py`` **``_pick_device``** defaults **CPU** on ``win32`` whenever ``GLITCHFRAME_WHISPERX_DEVICE`` is unset, so Align lyrics works **without** relying on Pinokio ``start.js`` (old clones / ``Glitchframe2.git`` / missing env). ``GLITCHFRAME_WHISPERX_DEVICE=cuda`` opts into GPU. Pinokio **`start.js`** still sets the env for documentation parity.

*Created for session handover; update only when the investigation changes materially.*

---

## Runbook: capture traceback + pip freeze (Pinokio vs local)

Do this once and send the artefacts to whoever is debugging — no guessing about “same” installs.

### Pinokio does not give you a shell — use Windows

Pinokio often has **no** interactive terminal for the venv (only install log output). That is **not** a mistake on your side.

- **To fix deps from a new commit:** in Pinokio use **Update** then **Install** / **Reinstall** — no commands.
- **To run `pip` or `pip freeze` yourself:** use **File Explorer → open folder `…\Glitchframe.git`** → address bar → type **`cmd`** → Enter → `env\Scripts\activate.bat` (see [`pinokio-package.md`](pinokio-package.md) § *No interactive terminal*).

### A) Full traceback (exact failure line)

Goal: preserve the **entire** exception, not only the last DLL name.

**Option 1 — Pinokio terminal / log pane**

1. Click **Align lyrics** once and wait until it errors (or start the app and reproduce).
2. In Pinokio, open wherever **stdout/stderr** from `python -m app` appears (often the run panel below **Start**, or **Logs** depending on Pinokio version).
3. **Scroll up** until you see either `Traceback (most recent call last):` or the **first** line mentioning `whisperx` / `ctranslate2` / `cudnn` / `Error`.
4. Select from `Traceback` (or first relevant line) through the **final** exception line (`...Error:` / `Could not load library...`).
5. Copy and paste into a plain text file, or paste directly into the chat/issue.

**Option 2 — Run Glitchframe from a normal terminal (same Pinokio venv)**

This often produces easier copy/paste than the embedded Pinokio UI.

1. Find the Pinokio project folder — usually under something like:
   `C:\pinokio\api\<YourGitFolderName>`  
   Inside it you should see an `env` folder (the Pinokio venv) plus the repo files.
2. Open **PowerShell** (Win+X → Windows Terminal / PowerShell).
3. `cd` to that folder, e.g.  
   `cd C:\pinokio\api\Glitchframe.git`  
   (replace with **your** path; GitHub ZIP clones may differ.)
4. Activate the venv:
   ```powershell
   .\env\Scripts\Activate.ps1
   ```
5. Run the app and reproduce:
   ```powershell
   python -m app
   ```
6. In the browser, trigger **Align lyrics** until it fails.
7. In that same terminal window, select **all traceback output**, copy it.

Include **20–40 lines minimum** ending with `Error:` or DLL message. If Windows shows **`Error code 1920`**, include that line too.

### B) `pip freeze` from the Pinokio `env`

Use the **same** `python.exe` that Pinokio uses for Install/Start (`env`), not global Python. Pinokio usually has **no** built-in terminal — open **cmd** via Explorer (see § *Pinokio does not give you a shell* above), then **`env\Scripts\activate.bat`** once so `python` and `pip` point at the Pinokio venv.

**Command Prompt (`cmd`)** — after activating:

```bat
python -c "import sys; print(sys.executable)"
pip freeze > "%USERPROFILE%\Desktop\pinokio-glitchframe-freeze.txt"
```

**PowerShell** — from the same folder (`env\` present):

```powershell
.\env\Scripts\python.exe -c "import sys; print(sys.executable)"
.\env\Scripts\pip.exe freeze > $HOME\Desktop\pinokio-glitchframe-freeze.txt
```

Open `pinokio-glitchframe-freeze.txt` — it should list **hundreds** of lines (`torch==`, `ctranslate2==`, etc.). Attach that file or paste **all** lines (not a screenshot).

### C) `pip freeze` from local (working or broken) `.venv`

1. Open **PowerShell**.
2. `cd` into your Git clone directory (e.g. `G:\DEV\MusicVids`).
3. Activate your local venv:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Confirm interpreter:

   ```powershell
   python -c "import sys; print(sys.executable)"
   ```

   Expect `...\MusicVids\.venv\Scripts\python.exe` (not `Program Files\Python`).

5. Write freeze:

   ```powershell
   pip freeze > $HOME\Desktop\local-glitchframe-freeze.txt
   ```

### D) Minimal “diff-focused” excerpt (optional but helpful)

If full freezes are huge, grep these names in **both** files so we see versions side by side:

In **PowerShell** (after each freeze exists, or paste into grep):

Filters that matter most for Align lyrics/WINDOWS DLL issues:

- `torch`
- `torchvision`
- `torchaudio`
- `ctranslate2`
- `faster-whisper`
- `whisperx`
- `cuda` wheel packages (anything `nvidia-`)

Manual check: search inside each freeze file for lines starting with:

`torch==`, `ctranslate2==`, `whisperx==`, `faster-whisper==`, `cuda==` (there may be no bare `cuda`).

### E) Checklist — what to send

| Item | Done? |
|------|--------|
| Full traceback text (multi-line), not only the cudnn DLL filename | □ |
| `pinokio-glitchframe-freeze.txt` from `env\Scripts\pip.exe freeze` | □ |
| `local-glitchframe-freeze.txt` from `.venv` `pip freeze` | □ |
| Optional: Pinokio’s `sys.executable` line from step B3 | □ |

With (1)+(2)+(3), the next step is **line-by-line** version comparison — not another blind pin bump.

---

### 2026 — Pinned Windows (3.11/3.12) stack in-repo

`pyproject.toml` optional extras (`all` / `lyrics` / `analysis`) now define **Track A**: **torch 2.2.2+cu121**, **whisperx 3.3.0**, **faster-whisper 1.1.0**, **ctranslate2 4.4.0** (PEP 508: Windows + `python_version < "3.13"`). **Python 3.13** on Windows uses **Track B** (cu124 + `ctranslate2>=4.5`) because WhisperX 3.3.0 does not support 3.13. Pinokio `install.js` installs Track A explicitly; `scripts/windows_provision_cudnn_next_to_ctranslate2.py` copies CUDNN DLLs next to `ctranslate2` after `nvidia-cudnn-cu12`.

---

## 2026-04-28 — Root cause #2 found: it was NEVER just cuDNN

A clean Pinokio traceback finally surfaced what fork users actually hit. The cuDNN
DLL story was real but **was not blocking alignment** for most users — the
*audio ingest* step was crashing first, and a numpy ABI mismatch was silently
corrupting torch. Three concrete bugs were fixed:

### Bug A — Speechbrain `LazyModule` Windows path-separator bug

Pinokio trace from a healthy Track A install (torch 2.2.2+cu121, ctranslate2
4.4.0 — all good per the diagnostic block):

```
File "...\pipeline\audio_ingest.py", line 97, in ingest_audio_file
    y, sr = librosa.load(src_for_librosa, sr=None, mono=False)
File "...\librosa\core\audio.py", line 32, in <module>
    samplerate = lazy.load("samplerate")
File "...\lazy_loader\__init__.py", line 227, in load
    parent = inspect.stack()[1]
File "...\inspect.py", line 988, in getmodule
    if ismodule(module) and hasattr(module, '__file__'):
File "...\speechbrain\utils\importutils.py", line 112, in __getattr__
    return getattr(self.ensure_module(1), attr)
File "...\speechbrain\utils\importutils.py", line 103, in ensure_module
    raise ImportError(f"Lazy import of {repr(self)} failed") from e
ModuleNotFoundError: No module named 'k2'      # or: No module named 'flair'
```

**Mechanism.** `whisperx → pyannote-audio → speechbrain>=1.0,<1.1` registers
many `LazyModule` objects (`speechbrain.integrations.k2_fsa`,
`...integrations.nlp.flair_embeddings`, plus several `deprecated_redirect`
shims). Speechbrain's `LazyModule.__getattr__` force-imports the target on
**any** attribute access, including `hasattr(mod, "__file__")`. CPython's
`inspect.getmodule` walks `sys.modules` and probes `__file__` on every entry;
`librosa`'s `lazy_loader` calls `inspect.stack()` when loading `samplerate`.

Speechbrain v1.0.x **does** have a guard intended to short-circuit
`inspect.py` probes — `LazyModule.ensure_module` raises `AttributeError` (so
`hasattr` returns `False`) when the calling frame is `inspect.py`. The
problem is the literal:

```python
if importer_frame is not None and importer_frame.filename.endswith(
    "/inspect.py"
):
    raise AttributeError()
```

Hard-coded forward slash. On Windows the path is `…\Lib\inspect.py` —
backslash. The guard is a **silent no-op on Windows**. Result: every audio
upload force-imports every speechbrain integration; whichever integration
has a missing optional dep (`k2` — no Windows wheel; `flair` — heavy NLP
package, not used by Glitchframe) crashes the ingest. We hit `k2` first,
fixed it with a `sys.modules` stub, then immediately hit `flair` — pure
whack-a-mole until the underlying separator bug is cured.

Tracked upstream as speechbrain issue [#2995].

**Fix.** `pipeline/_speechbrain_compat.py` ships a `patch_speechbrain_lazy_module()`
function that monkey-patches `LazyModule.ensure_module` with a separator-aware
version (`os.path.basename(filename) == "inspect.py"`). `app.py` calls this
patch from `_log_runtime_python_and_optional_deps()` immediately after
`import whisperx` succeeds (so speechbrain is already in `sys.modules` and we
can reach into the class). The patch is idempotent and a silent no-op when
speechbrain is absent. As belt-and-braces, `app.py` also pre-stubs `k2` in
`sys.modules` so the lazy import succeeds even if the patch ever fails to
apply (e.g. a future speechbrain rename). One-line upstream fix tracked at
[#2995][upstream-issue].

[#2995]: https://github.com/speechbrain/speechbrain/issues/2995
[upstream-issue]: https://github.com/speechbrain/speechbrain/issues/2995

### Bug B — NumPy 2.x silently corrupted torch interop

Same trace prelude:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.3 ...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

**Mechanism.** Track A pins `torch==2.2.2+cu121`, whose wheel was built against
NumPy 1.x. `requirements.txt` / `pyproject.toml` only said `numpy>=1.26.0`,
which lets pip resolve to the latest 2.4.x. Torch keeps running but its
`tensor_numpy.cpp` bridge logs `_ARRAY_API not found` and any tensor↔numpy
conversion (audio ingest, demucs, whisperx) is undefined behaviour.

**Fix.** Pin `numpy>=1.26.0,<2.0` in **both** `requirements.txt` and
`pyproject.toml` `[project]` dependencies. Add an explicit
`pip install --force-reinstall --no-deps "numpy>=1.26.0,<2.0"` step to
`install.js` after the extras install (same pattern used for markupsafe/pillow
to survive `--force-reinstall` torch reinstalls).

### Bug C — Standalone `nvidia-cudnn-cu12` wheels are toxic on Track A

This took two iterations to resolve:

* **Iteration 1.** `install.js` originally had `pip install nvidia-cudnn-cu12`
  with no version. PyPI's latest is **cuDNN 9** (DLL names `cudnn_ops64_9.dll`,
  no `infer/train` split) which **does not** satisfy the cuDNN 8 lookups inside
  ctranslate2 4.4.0 — Align lyrics fails on the GPU path with
  `Could not load library cudnn_ops_infer64_8.dll`.
* **Iteration 2.** Pinning `nvidia-cudnn-cu12==8.9.7.29` (the latest cuDNN 8.9
  wheel on PyPI) instead caused **`WinError 127: The specified procedure could
  not be found`** during `import whisperx`. Mechanism: ctranslate2 4.4.0's
  resolved import table (built against torch 2.2.2's bundled cuDNN minor
  ~8.9.2) references symbols that are not exported in the same way by
  8.9.7.29's standalone DLLs. Windows finds the DLL, fails to bind a function,
  refuses to load the module.

**Final policy: do NOT install `nvidia-cudnn-cu12` at all.** Torch 2.2.2+cu121
ships cuDNN 8.9.x in `torch\lib`; that's the build ctranslate2 4.4.0 was
released against. `scripts/windows_provision_cudnn_next_to_ctranslate2.py`
copies those DLLs next to the `ctranslate2` package so `LoadLibrary` finds
them regardless of PATH ordering quirks. `install.js` also runs
`pip uninstall -y nvidia-cudnn-cu12` to purge wheels left over from prior
install attempts. The resilience test
(`tests/test_pinokio_windows_resilience.py`) pins this policy so a future
helpful refactor doesn't re-add the install step.

### Bug D — GPU alignment had no automatic fallback

Even with all of the above fixed, individual machines can still have a broken
GPU CUDA/cuDNN stack (driver mismatch, AV blocking, ACL on `Program Files\NVIDIA`,
old GPU compute capability, Error 1920, …). `align_lyrics` now wraps
`_run_whisperx_forced` with a **single CPU retry** for cuDNN-class errors only:

* Matched via `_is_cudnn_class_error(exc)` (substring match on
  `cudnn`, `could not load library`, `loadlibrary`, `ctranslate2`, `cublas`,
  `cudart`, `error code 1920`, `cuda error`, …; also walks `__cause__` /
  `__context__`).
* Retry is `device="cpu"` / `compute_type="int8"` — exactly what
  `_default_compute_type("cpu")` returns.
* SDXL / Demucs / render are **not** touched — they stay on CUDA.
* If the user already started on CPU (`GLITCHFRAME_WHISPERX_DEVICE=cpu`),
  there is no second attempt — the original error surfaces.

This is the safety net for fork users whose cuDNN install is unrepairable.
Performance hit: ~5–10× slower alignment, no other downstream impact.

### Why the "ctranslate2 4.4.0 matches the Windows pinned lyrics stack" log was misleading

The startup probe was correct — Track A pins WERE active. But the user's clicks
weren't reaching `_run_whisperx_forced` at all because `librosa.load` (audio
ingest) crashed first. Anyone debugging only the **alignment** path missed the
**ingest** crash above it. Future investigations: scroll above the alignment
log and scan for `ImportError: Lazy import of LazyModule(...)`.

### Bug E — Newer `diffusers` references `torch.xpu` at import time

After Bugs A–D were fixed, lyrics aligned successfully but **rendering**
failed with:

```
RuntimeError: AnimateDiff SDXL requires a recent diffusers install with
AnimateDiffSDXLPipeline and DDIMScheduler ... Import failed: module 'torch'
has no attribute 'xpu'
```

**Mechanism.** PyTorch added the `torch.xpu` submodule (Intel discrete GPU
support) in **2.3.0**. Track A pins `torch==2.2.2+cu121` so cuDNN-era DLL
names match what ctranslate2 4.4.0 expects — there's no `torch.xpu` there.
Newer `diffusers` releases (≈ 0.30+) reference `torch.xpu` at import time
without guarding the access with `hasattr(torch, "xpu")`. So the very first
`from diffusers import AnimateDiffSDXLPipeline` in `_import_animatediff_sdxl`
explodes before any model code runs. Same upstream pattern as the
speechbrain bug — **library code probes a possibly-absent attribute** —
just on a different module. The HuggingFace transformers project hit the
same issue ([transformers#37838][hf-37838]) and fixed it with the
`hasattr` guard; diffusers hasn't (yet).

[hf-37838]: https://github.com/huggingface/transformers/issues/37838

**Fix.** `pipeline/_torch_xpu_compat.py` ships `patch_torch_xpu()` which
installs a tiny `torch.xpu` stub (a `ModuleType` with `is_available()` →
`False`, `device_count()` → `0`, plus `is_initialized` / `is_bf16_supported`
/ `synchronize` / `empty_cache` / nested `amp` so any reasonable probe
returns the "no XPU" answer). The stub is also registered in
`sys.modules["torch.xpu"]` so `import torch.xpu` works. Newer torch
(Track B, ≥ 2.4) has the real submodule already — `patch_torch_xpu` checks
`getattr(torch, "xpu", None)` first and never overwrites it.

The patch runs in two places (idempotent — safe to call twice):

* `app.py` `_log_runtime_python_and_optional_deps()` — right after the
  diagnostic `import torch` succeeds, so any subsequent diffusers import
  (model load, gradio worker, anything) finds a `torch.xpu` to probe.
* `pipeline/background_animatediff._import_animatediff_sdxl()` — belt-and-
  braces, in case some code path imports diffusers before app startup
  completes.

Render runs on CUDA exactly as before; the stub only exists so the
**import-time** XPU probe doesn't crash on Track A.

### Files changed in this fix

| File | What |
|------|------|
| `pipeline/_speechbrain_compat.py` | **NEW** — monkey-patches `LazyModule.ensure_module` so its `inspect.py` guard works on Windows (root-cause fix; cures k2/flair/etc. in one shot) |
| `pipeline/_torch_xpu_compat.py` | **NEW** — installs a `torch.xpu` stub when PyTorch < 2.3 lacks it; cures `diffusers` AnimateDiff SDXL import on Track A |
| `app.py` | Calls `patch_speechbrain_lazy_module()` after `import whisperx`; calls `patch_torch_xpu()` after diagnostic `import torch`; pre-stubs `sys.modules['k2']` as belt-and-braces |
| `pipeline/background_animatediff.py` | `_import_animatediff_sdxl` calls `patch_torch_xpu()` immediately before `from diffusers import AnimateDiffSDXLPipeline` |
| `pipeline/lyrics_aligner.py` | `_is_cudnn_class_error` + single CPU retry around `_run_whisperx_forced` |
| `pyproject.toml` | `numpy>=1.26.0,<2.0` in `[project]` dependencies |
| `requirements.txt` | `numpy>=1.26.0,<2.0` |
| `install.js` | Explicit `numpy<2` force-reinstall step + `pip uninstall -y nvidia-cudnn-cu12` (no standalone cuDNN wheel) |
| `tests/test_pinokio_windows_resilience.py` | Pins all six behaviours (LazyModule patch, k2 stub, NumPy ABI, cuDNN policy, CPU retry, torch.xpu stub) so a future refactor cannot quietly regress |
