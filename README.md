# Glitchframe

Local, GPU-accelerated **music video** generator: upload a track, analyze it, align lyrics, generate stylized backgrounds (**SDXL** keyframe stills by default — with optional **RIFE** morph and **Ken Burns** on those stills enabled by default), composite reactive shaders and kinetic type, and encode with **ffmpeg** (NVENC on NVIDIA GPUs by default).

**Examples (progress log, newest = current state):** [voidcat on YouTube](https://www.youtube.com/@voidcatalog)

- **Easiest install:** [Pinokio](https://pinokio.co/) — install the Pinokio app, search **glitchframe**, install Glitchframe from the listing, then run **Install** and **Start** in the sidebar. You still need **ffmpeg** on your `PATH` and a capable **NVIDIA GPU** for the intended experience — see [Requirements](#requirements) and [Pinokio](#pinokio).
- **UI:** [Gradio](https://www.gradio.app/) — run `python -m app` and open the URL shown (default [http://127.0.0.1:7860](http://127.0.0.1:7860)). **Pinokio** / **`requirements.txt`** tracks **Gradio 5.x**; `pyproject.toml` core metadata may still list Gradio **4.x** bounds — install **`requirements.txt`** before **`pip install -e .`** for parity (see [Install](#install)).
- **Manual setup (Windows CLI):** **[Getting started on Windows](docs/guides/getting-started-windows.md)** — order of installs: Python, optional Git, ffmpeg with winget, venv, PyTorch; or use Pinokio above instead.
- **Deep dive:** [`docs/index.md`](docs/index.md) (includes [`background-keyframes-editor`](docs/technical/background-keyframes-editor.md), [`background-stills`](docs/technical/background-stills.md), effects/lyrics editors) and [`docs/technical/project-setup-and-config.md`](docs/technical/project-setup-and-config.md).

**License:** [MIT](LICENSE) · **Repository:** [github.com/OlaProeis/Glitchframe](https://github.com/OlaProeis/Glitchframe)

## Features (overview)

- **Ingest and analysis:** Per-song cache, waveform preview, beat/onset/spectrum features, optional **Demucs** vocal stem, segment/chapter hints.
- **Lyrics:** **WhisperX** word timings plus alignment to pasted lyrics; visual per-word **timeline editor** for fixes (saved to cache so re-runs do not clobber your edits); optional **Export .srt** aligned to the same per-word timings.
- **Backgrounds:** **Background keyframes** tab — waveform timeline to edit SDXL still **timing** and **per-clip prompts**, **Generate SDXL stills** / **Regenerate** per slot, **Replace** / **Crop** with image upload (staging until **Save timeline**; crop selection is **locked to the output resolution** aspect ratio). Still **count** is fixed from analysis (~one per ~8 s); see [`docs/technical/background-keyframes-editor.md`](docs/technical/background-keyframes-editor.md). **SDXL + optional RIFE morph** is the supported AI background path (smooth optical-flow interpolation between keyframes; **Ken Burns on SDXL stills** stays **on by default**). **Static image + Ken Burns** uses your uploaded plate only (no SDXL). In **Visual style**, pick a **reactive shader** (or **none**); that pre-fills an example **scene prompt**, **typography style**, and **palette**. Use **Scene prompt** for defaults applied when generating stills; refine individual clips on the **Background keyframes** tab.
- **Look and motion:** Optional GLSL **reactive shaders** (curated list + **no shader** for a clean background plate), **Skia** kinetic typography, title/thumbnail text, optional **logo** placement with rim glow, beams, and branding-driven effects.
- **Effects timeline:** Per-clip post effects (e.g. screen shake, chromatic aberration, colour invert, zoom punch, scanline tear) with an in-UI editor and baked JSON under the song cache.
- **Output:** Full-length render and **10 s preview** (loudest window), `output.mp4` + `thumbnail.png` + YouTube-oriented **`metadata.txt`**, with NVENC by default when available.

## Screenshots

**Lyrics timeline** (per-word alignment and editing on the vocal waveform):

![Glitchframe lyrics timeline editor](screenshots/vocal-timeline.png)

**Effects timeline** (clip-based post effects with rows, playhead, and per-clip controls):

![Glitchframe effects timeline editor](screenshots/effect-timeline.png)

**Background keyframes** matches the same interaction pattern (waveform, draggable clips, playhead, per-slot regenerate/replace/crop). See [`docs/technical/background-keyframes-editor.md`](docs/technical/background-keyframes-editor.md).

## Known limitations (read before you depend on it)

- **Background keyframes** count is **fixed** from the analysis/plan (not open-ended in the UI). Edit timing and prompts for existing clips; **Save timeline** writes `keyframes_timeline.json` + `manifest.json` (and clears the RIFE morph cache when the manifest changes). [`docs/technical/background-keyframes-editor.md`](docs/technical/background-keyframes-editor.md)
- **Vocal / lyrics matching** can be **unreliable** in places. Treat alignment as a draft: use the lyrics timeline and listen back **before** you commit time to a full render. Improving this area is a priority; do not assume perfect lip-sync or line timing yet.
- **Rendering is effectively single-threaded** for the heavy pipeline. Full videos often take **on the order of 1–2+ hours** (sometimes more), depending on chosen shader (or none), scene complexity, length, resolution, GPU, and whether **RIFE** morph bakes extra frames after SDXL. Plan batch work accordingly.
- **RIFE morph (optional, default-on in the UI)** runs a **CUDA** bake after SDXL keyframes; it downloads **~24 MB** of weights from Hugging Face on first use (see [`docs/technical/rife-morph-background.md`](docs/technical/rife-morph-background.md)). GPU time scales with keyframe count and the **subdivisions** slider.
- The app is under active development; UI labels and edge cases are still being hardened.

## Future (from project backlog)

The following is a short, user-facing summary of work **not yet done** (also tracked in Taskmaster as `pending` / `deferred` in [`.taskmaster/tasks/tasks.json`](.taskmaster/tasks/tasks.json)):

- **Unify “auto” effects with the timeline** — one control surface: analyser-driven glitch, beams, and related FX should not stack with the Effects timeline in confusing ways; timeline becomes authoritative where intended (*pending*).
- **Faster preview backgrounds** — generate SDXL (and optional RIFE) / Ken Burns assets only for the 10 s preview window (plus padding), then fill the rest on full render, with clearer cache keys so preview is much cheaper than today (*deferred*).
- **Bass-driven logo pulse** — optional mode where logo motion follows low-frequency energy / kicks instead of a generic beat grid, with tunable sensitivity (*deferred*).
- **Overnight / multi-song queue** — batch several full renders (CLI or Gradio) with stable paths and isolated failures (*deferred*).
- **Single primary “export” affordance** — one obvious control that runs the full pipeline, while keeping optional Analyze/Align as precache steps (*deferred*).
- **Timestamped section headers in lyrics** — lines like `[Verse 1 0:12]` or `[Chorus 1:00]` that set both a section break and a coarse time anchor, to reduce manual `[m:ss]` busywork (*deferred*).

## Requirements

- **Python** 3.11+ (3.12/3.13 may work; optional deps like `madmom` are pickier on newer Python)
- **ffmpeg** on your `PATH` (encode/mux). On **Windows**, install with **winget** (see [Getting started on Windows](docs/guides/getting-started-windows.md)); on other systems use your package manager or [ffmpeg.org](https://ffmpeg.org/download.html) if needed
- **NVIDIA GPU + CUDA** recommended for diffusers, analysis, and NVENC; **Pinokio** installs **PyTorch** builds from **`torch.js`** (e.g. **cu128** on Windows/Linux NVIDIA — see `torch.js`). CPU-only is possible for lighter paths but not the main focus
- **Disk:** model and song caches under `.cache/` and `cache/` (large downloads on first use)

## Install

**Recommended:** Use [Pinokio](https://pinokio.co/): install Pinokio, search **glitchframe**, install Glitchframe from the listing, then use **Install** / **Start** in the app. That flow runs the repo’s Pinokio scripts (`install.js`, **`torch.js`**, `start.js`, …). Prerequisites: **ffmpeg** on your `PATH`, a suitable **NVIDIA** GPU (for the full GPU path), and recent drivers — see [Pinokio](#pinokio). **Catalog installs** clone the repo’s **default Git branch**; to test **`dev`** before it is merged, `git checkout dev` in the Pinokio app folder then **Reinstall** (see [`docs/technical/pinokio-package.md`](docs/technical/pinokio-package.md)).

**Windows, full walkthrough (command line):** [docs/guides/getting-started-windows.md](docs/guides/getting-started-windows.md).

The steps below are a **manual** install (clone or ZIP + venv + pip); they match that guide. On Windows prefer **`py -3.11`** if `python` is not on your `PATH`.

### 1. Get the project and create a virtualenv

**With git:**

```bash
git clone https://github.com/OlaProeis/Glitchframe.git
cd Glitchframe
python -m venv .venv
```

**Without git:** from the [GitHub](https://github.com/OlaProeis/Glitchframe) repo, **Code → Download ZIP**, extract, `cd` into the folder, then `python -m venv .venv` (or `py -3.11 -m venv .venv` on Windows).

Then activate the venv:

- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. PyTorch with CUDA (recommended)

**Path A — match Pinokio / `torch.js` (good default for NVIDIA on Windows):** after creating the venv and upgrading `pip`, install project deps (§3) **or** at minimum `requirements.txt`, then install **`[all]`**, then run the **same** `uv pip` / `pip` line as in `torch.js` for your platform. **NVIDIA + Windows** (from `torch.js`):

```bash
python -m pip install -U uv
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

(Use the **`cpu`**, **DirectML**, **ROCm**, or **macOS** blocks inside `torch.js` for other platforms.) **Order:** Pinokio runs **`torch.js` last**, after `requirements.txt`, `.[all]`, **madmom**, and **beatnet** — mirror that if you hit resolver issues.

**Path B — legacy Windows “Align lyrics” stack (Python 3.11–3.12, cu121):** optional extras **`all` / `lyrics` / `analysis`** in `pyproject.toml` still document a coherent **PyTorch 2.2.2+cu121** + **WhisperX 3.3.0** + **ctranslate2 4.4.0** set for DLL alignment. Install the **CUDA 12.1** index **before** the rest if you use this path:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then continue with §3–§4. **`scripts/windows_provision_cudnn_next_to_ctranslate2.py`** and **`windows-venv-recovery-guide.md`** apply mainly to this **cu121** + **ctranslate2 4.4** layout.

**Path C — generic CUDA 12.4 wheels (`cu124`)** for setups not using Path A or B (e.g. some **Python 3.13** flows):

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

On Windows, if `python` is not on `PATH`, use the launcher: `py -3.11 -m pip ...`.

### 3. Project dependencies

**Install `requirements.txt` before** `pip install -e .` so you get the **Gradio 5.x** stack and the pins tested with Pinokio (`pyproject.toml` `[project]` dependencies still target Gradio **4.x** unless you pull everything from `requirements.txt` first):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` pins Gradio **5.x**, `fastapi`, `huggingface_hub`, and related packages; see the comments at the top of that file if `pip` conflicts.

### 4. Optional: full analysis + lyrics stack

For vocal stem separation (Demucs) and lyrics alignment (WhisperX + Silero VAD):

```bash
python -m pip install -e ".[all]"
```

If after installing `whisperx` CUDA **disappears** from PyTorch, reinstall the CUDA wheels from the **same** index you chose in [§2](#2-pytorch-with-cuda-recommended), then re-run **`pip install -r requirements.txt`** if `pip` upgraded MarkupSafe / Pillow past Gradio’s expectations — recovery patterns are in [`requirements.txt`](requirements.txt) comments and [`docs/technical/windows-venv-recovery-guide.md`](docs/technical/windows-venv-recovery-guide.md).

**Optional beat detectors:** Pinokio runs **`madmom`** and **`beatnet`** with `--no-build-isolation` after **`.[all]`**. For a manual venv, mirror Pinokio:

```bash
python -m pip install madmom --no-build-isolation
python -m pip install beatnet --no-build-isolation --no-deps
```

Or use **`pip install -e ".[beats]"`** (can be finicky to build); the analyzer falls back to librosa without them.

### 5. Optional: environment overrides

```bash
copy .env.example .env
```

On Unix: `cp .env.example .env` — then edit `.env` if you need custom **`GLITCHFRAME_*`** paths or ffmpeg codec overrides. Legacy **`MUSICVIDS_*`** names are still read for the same settings. The sample file also lists optional API keys for **Taskmaster** / dev tooling, not for core Glitchframe.

## Run

```bash
python -m app
```

Open the local URL printed in the console (default port **7860**).

## Pinokio

**Easiest path:** Install [Pinokio](https://pinokio.co/), open it, search **glitchframe**, and install Glitchframe from the listing — no need to paste a URL. **Alternative:** **Download from URL** and paste `https://github.com/OlaProeis/Glitchframe.git`.

This repository includes Pinokio scripts (`install.js`, **`torch.js`**, `start.js`, `reset.js`, `update.js`, `pinokio.js` **`version` 3.7**, `icon.png`). **`install.js`** (Python **3.11** venv **`env`**): **`uv pip install -r requirements.txt`**, **`uv pip install -e ".[all]"`**, **`madmom`** / **`beatnet`**, then **`script.start` → `torch.js`** for **platform-specific PyTorch** (**NVIDIA** Windows/Linux: **cu128** **`torch==2.7.0`** trio by default — see `torch.js`). **`start.js`** sets `GLITCHFRAME_WHISPERX_VAD_METHOD=silero`, **`GLITCHFRAME_WHISPERX_DEVICE=cpu`**, and **`HF_HUB_DISABLE_SYMLINKS=1`** (+ warning silencer) for reliable defaults on Windows; change or remove **`GLITCHFRAME_WHISPERX_DEVICE`** in **`start.js`** or `.env` to try **GPU** Align lyrics. **No extra Pinokio step** is required for **RIFE** (~24 MB on first morph). **Click Start** after install. **ffmpeg** must be on **`PATH`**. Technical details: [`docs/technical/pinokio-package.md`](docs/technical/pinokio-package.md). Repo topic **`pinokio`** for GitHub discovery.

**Stale or custom clones:** **Update** runs `git pull` on the **current** branch — use **`master`** vs **`dev`** intentionally (see pinokio-package doc).

## Troubleshooting

- **Step-by-step (Windows, after PyTorch / lyrics issues):** [docs/technical/windows-venv-recovery-guide.md](docs/technical/windows-venv-recovery-guide.md) — `git pull`, clean `torch`/`torchvision`/`torchaudio` reinstall, extras, test **Align lyrics**.
- **Align lyrics** fails with `Weights only load failed` / `omegaconf` / `ListConfig`: PyTorch **2.6+** defaults `torch.load` to a stricter mode that breaks some WhisperX/pyannote checkpoints. **Prefer updating Glitchframe** to a revision that includes `pipeline/torch_checkpoint_compat.py` and keeping a **current** `torch` / `torchvision` / `torchaudio` trio from the same CUDA index ([Install §2](#2-pytorch-with-cuda-recommended)). Downgrading only `torch` to “fix” this often causes the cuDNN mismatch below.
- **`Could not load library cudnn_ops_infer64_8.dll` / `Could not locate cudnn_ops_infer64_8.dll` / error `1920` (often **after** `Performing voice activity detection using Silero`):** the **VAD** line is misleading — Silero runs first; the crash is usually **faster-whisper / CTranslate2** loading **cuDNN** DLLs. **Windows + Python 3.11/3.12:** use one coherent stack — **PyTorch 2.2.2+cu121** + **WhisperX 3.3.0** + **ctranslate2 4.4.0** + **faster-whisper 1.1.0** (see [Install §2](#2-pytorch-with-cuda-recommended) and ``pyproject.toml`` extras). Run ``python scripts/windows_provision_cudnn_next_to_ctranslate2.py`` after ``pip install nvidia-cudnn-cu12``. The app calls ``os.add_dll_directory`` for ``torch\lib`` before WhisperX (`pipeline/win_cuda_path.py`). **Windows + Python 3.13** or **Linux/macOS:** prefer **PyTorch cu124** with **ctranslate2 4.5+** and current WhisperX (see ``pyproject.toml`` markers). **If it still fails:** reinstall the **torch + torchvision + torchaudio** trio from **one** CUDA index in a single command, then ``pip install -e ".[all]"`` again; try ``GLITCHFRAME_WHISPERX_DEVICE=cpu``; last resort — manual cuDNN copy per [NVIDIA cuDNN for CUDA 12](https://developer.nvidia.com/cudnn).

- **Pinokio / Windows:** **`install.js`** installs **`requirements.txt`** + **`.[all]`** + **madmom/beatnet**, then **`torch.js`** (typically **cu128** PyTorch on NVIDIA). **`start.js`** may still default **CPU** WhisperX — remove **`GLITCHFRAME_WHISPERX_DEVICE`** or set **cuda** to try GPU alignment; Align lyrics can still fall back to CPU on DLL/load errors.

- **Align lyrics fails with `[WinError 1314] A required privilege is not held by the client` while writing into `cache\HF_HOME\hub\models--…\snapshots\…`:** `huggingface_hub` tries to **symlink** model snapshot files to the content-addressed `blobs/` directory, but Windows requires admin rights or **Developer Mode** to create symlinks. Glitchframe patches `huggingface_hub.are_symlinks_supported` at startup so it copies blobs instead — **Update** + **Start** in Pinokio (no full reinstall) is enough to pick up the fix. If the cache ended up half-populated before updating, delete `C:\pinokio\api\Glitchframe.git\cache\HF_HOME\` from Explorer and the next Align run will re-download cleanly. Details: [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) § Bug F.

- **Render is unexpectedly slow / "compositing" stuck around 40 %:** the orchestrator label parks on its outer status during compositing because individual frames are reported by an inner Gradio progress callback. Recent builds replaced the per-frame UI message with a richer one (`Compositing 1843/4923 (37.4%) - 1.23 fps - ETA 41m42s - layers=BG+SHADER+TYPO`) and update it from the request thread, so the bar now ticks live during long renders. After a render finishes, the run log appends a one-liner like `compositor: 4923 frames in 41m23s · avg 1.98 fps · encoder=h264_nvenc` — if `encoder=libx264` and you have an NVIDIA GPU, NVENC fell back to CPU; check the startup log for the `ffmpeg` candidate list and any `Cannot load nvEncodeAPI64.dll` / `Driver does not support the required nvenc API version` lines from the probe stderr. Multi-candidate ffmpeg discovery (env override → active env's `bin` → PATH → well-known dirs) means a working NVENC ffmpeg anywhere on your system will be picked up automatically, even if Pinokio's bundled one shadows it.

- **Render fails with `Undefined constant or missing '(' in 'p5'` / `Unable to parse option value "p5"` after background generation:** your local ffmpeg is older than 4.4 (or NVENC SDK < 11) and doesn't understand the modern `p1..p7` preset family. Glitchframe now probes the chosen ffmpeg once and falls back to the legacy `slow` preset automatically, so **Update + Start** in Pinokio is enough to pick up the fix — no reinstall. Visual quality is essentially unchanged (`slow` is the closest legacy equivalent of `p5`). To get back to the modern preset family, install a recent ffmpeg (`winget install ffmpeg` on Windows, or `conda update -c conda-forge ffmpeg` inside Pinokio's env). Details: [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) § Bug H.

- **Pinokio / Gradio or dependency conflicts after updates:** **Update** (`git pull`) then **Reinstall** so **`install.js`** re-runs **`uv pip`** against the current **`requirements.txt`**. If you mix **`pip install -e .`** without **`requirements.txt`**, you can drift back to **`pyproject.toml`** Gradio **4.x** bounds — see [Install §3](#3-project-dependencies). Advanced: [`docs/technical/pinokio-package.md`](docs/technical/pinokio-package.md).

- **Apply crop / Regenerate keyframe: `PermissionError` / `WinError 5` on `*.tmp` → `.png`:** Windows blocks replacing a PNG if it is still open (Gradio or the browser preview of the same file, Explorer thumbnails, antivirus scan). Current builds close PIL readers before overwrite and retry atomic replace briefly; if it still fails, wait a second and retry, or close anything previewing `cache/<hash>/background/` (see [`docs/technical/background-keyframes-editor.md`](docs/technical/background-keyframes-editor.md) § *Windows: saving PNGs*).

## Development

- Smoke test config (paths + optional preset YAML count): `python config.py`
- Tests (after `pip install -e ".[dev]"`): `pytest`
- In this repo, `uv sync` / `uv run pytest` is also used; see `ai-context.md` for maintainer notes.

**AI-assisted development:** Much of this codebase was built with AI coding assistants and planning tools (the same day-to-day workflow as the [Ferrite](https://github.com/OlaProeis/Ferrite) project). For a concrete write-up of that process—context files, handover notes, and how tasks and reviews are organized—see Ferrite’s [AI development workflow](https://github.com/OlaProeis/Ferrite/blob/master/docs/ai-workflow/ai-development-workflow.md).

## Contributing

Issues and pull requests are welcome. Please keep changes focused; match existing style in the files you touch.

## Legal

This project is licensed under the [MIT License](LICENSE). Third-party assets (e.g. fonts under `assets/fonts/`) carry their own license files where applicable. **RIFE** inference code is derived from [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT); runtime weights are downloaded from Hugging Face (`MonsterMMORPG/RIFE_4_26` — confirm license/terms for your use case). SDXL and other diffusion models have their own licenses (e.g. Open RAIL-M).
