# Glitchframe

Local, GPU-accelerated **music video** generator: upload a track, analyze it, align lyrics, generate stylized backgrounds (**SDXL** keyframe stills by default — with optional **RIFE** morph and **Ken Burns** on those stills enabled by default), composite reactive shaders and kinetic type, and encode with **ffmpeg** (NVENC on NVIDIA GPUs by default).

**Examples (progress log, newest = current state):** [voidcat on YouTube](https://www.youtube.com/@voidcatalog)

- **Easiest install:** [Pinokio](https://pinokio.co/) — install the Pinokio app, search **glitchframe**, install Glitchframe from the listing, then run **Install** and **Start** in the sidebar. You still need **ffmpeg** on your `PATH` and a capable **NVIDIA GPU** for the intended experience — see [Requirements](#requirements) and [Pinokio](#pinokio).
- **UI:** [Gradio](https://www.gradio.app/) — run `python -m app` and open the URL shown (default [http://127.0.0.1:7860](http://127.0.0.1:7860)).
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
- **NVIDIA GPU + CUDA 12.x** recommended for diffusers, analysis, and NVENC; CPU-only is possible for lighter paths but not the main focus
- **Disk:** model and song caches under `.cache/` and `cache/` (large downloads on first use)

## Install

**Recommended:** Use [Pinokio](https://pinokio.co/): install Pinokio, search **glitchframe**, install Glitchframe from the listing, then use **Install** / **Start** in the app. That flow runs the repo’s Pinokio scripts (`install.js`, `start.js`, …). Prerequisites: **ffmpeg** on your `PATH`, a suitable **NVIDIA** GPU, and recent drivers — see [Pinokio](#pinokio).

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

**Windows (Python 3.11 or 3.12) — stable GPU “Align lyrics” stack:** the optional extras `all` / `lyrics` / `analysis` pin **PyTorch 2.2.2+cu121**, **WhisperX 3.3.0**, **faster-whisper 1.1.0**, and **ctranslate2 4.4.0** so CTranslate2 and PyTorch ship a **matching** cuDNN layout (avoids common `cudnn_ops_infer64_8.dll` issues next to newer cu124 wheels). Install the **CUDA 12.1** index **before** `requirements.txt` / `pip install -e .` / `pip install -e ".[all]"`:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then continue with §3–§4. Optionally run `python scripts/windows_provision_cudnn_next_to_ctranslate2.py` after `pip install nvidia-cudnn-cu12` to copy CUDNN DLLs next to `ctranslate2` (see `docs/technical/windows-venv-recovery-guide.md`).

**All other platforms (and Windows on Python 3.13+):** install PyTorch **first** from the official **CUDA 12.4** wheel index so you get a GPU build (adjust if you use a different CUDA index from [pytorch.org](https://pytorch.org/)):

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

On Windows, if `python` is not on `PATH`, use the launcher: `py -3.11 -m pip ...`.

### 3. Project dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` pins a few Gradio-related packages; see the comments at the top of that file if `pip` pulls incompatible versions.

### 4. Optional: full analysis + lyrics stack

For vocal stem separation (Demucs) and lyrics alignment (WhisperX + Silero VAD):

```bash
python -m pip install -e ".[all]"
```

If after installing `whisperx` CUDA **disappears** from PyTorch, reinstall the CUDA wheels from the same `cu124` index as in step 2, then re-pin Gradio’s friends if needed — recovery commands are in [`requirements.txt`](requirements.txt) comments.

**Optional beat detectors** (BeatNet + madmom) are available via `pip install -e ".[beats]"` but can be finicky to build; the analyzer falls back to librosa without them.

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

This repository includes Pinokio scripts (`install.js`, `start.js`, `reset.js`, `update.js`, `pinokio.js`, `icon.png`). The installer uses Python **3.11**, installs PyTorch **2.2.2+cu121** (CUDA **12.1** index) and the pinned WhisperX / ctranslate2 set (see `install.js`), then `requirements.txt`, `pip install -e .`, and the **`[all]`** extra — plus optional **`nvidia-cudnn-cu12`** and `scripts/windows_provision_cudnn_next_to_ctranslate2.py`. **No extra Pinokio step** is required for **RIFE** (weights ~24 MB download on first morph via Hugging Face, same hub + symlink behaviour as other models). **Click Start** in Pinokio&rsquo;s sidebar after install. You still need **ffmpeg** on your `PATH` and a capable **NVIDIA** GPU. **On Windows**, `start.js` may still set ``GLITCHFRAME_WHISPERX_DEVICE=cpu`` as a safe default; remove it or set **cuda** in `.env` to try **GPU** alignment once the **cu121** stack is installed. Analyze/render still use the GPU when applicable. Technical details: [`docs/technical/pinokio-package.md`](docs/technical/pinokio-package.md). For GitHub discovery, the repo uses the **`pinokio`** topic.

## Troubleshooting

- **Step-by-step (Windows, after PyTorch / lyrics issues):** [docs/technical/windows-venv-recovery-guide.md](docs/technical/windows-venv-recovery-guide.md) — `git pull`, clean `torch`/`torchvision`/`torchaudio` reinstall, extras, test **Align lyrics**.
- **Align lyrics** fails with `Weights only load failed` / `omegaconf` / `ListConfig`: PyTorch **2.6+** defaults `torch.load` to a stricter mode that breaks some WhisperX/pyannote checkpoints. **Prefer updating Glitchframe** to a revision that includes `pipeline/torch_checkpoint_compat.py` and keeping a **current** `torch` / `torchvision` / `torchaudio` trio from the same CUDA index ([Install §2](#2-pytorch-with-cuda-recommended)). Downgrading only `torch` to “fix” this often causes the cuDNN mismatch below.
- **`Could not load library cudnn_ops_infer64_8.dll` / `Could not locate cudnn_ops_infer64_8.dll` / error `1920` (often **after** `Performing voice activity detection using Silero`):** the **VAD** line is misleading — Silero runs first; the crash is usually **faster-whisper / CTranslate2** loading **cuDNN** DLLs. **Windows + Python 3.11/3.12:** use one coherent stack — **PyTorch 2.2.2+cu121** + **WhisperX 3.3.0** + **ctranslate2 4.4.0** + **faster-whisper 1.1.0** (see [Install §2](#2-pytorch-with-cuda-recommended) and ``pyproject.toml`` extras). Run ``python scripts/windows_provision_cudnn_next_to_ctranslate2.py`` after ``pip install nvidia-cudnn-cu12``. The app calls ``os.add_dll_directory`` for ``torch\lib`` before WhisperX (`pipeline/win_cuda_path.py`). **Windows + Python 3.13** or **Linux/macOS:** prefer **PyTorch cu124** with **ctranslate2 4.5+** and current WhisperX (see ``pyproject.toml`` markers). **If it still fails:** reinstall the **torch + torchvision + torchaudio** trio from **one** CUDA index in a single command, then ``pip install -e ".[all]"`` again; try ``GLITCHFRAME_WHISPERX_DEVICE=cpu``; last resort — manual cuDNN copy per [NVIDIA cuDNN for CUDA 12](https://developer.nvidia.com/cudnn).

- **Pinokio / Windows:** ``install.js`` installs the **cu121** stack above; ``start.js`` may still default **CPU** WhisperX — remove ``GLITCHFRAME_WHISPERX_DEVICE`` or set **cuda** to try GPU alignment after install. Even with GPU enabled, Align lyrics auto-retries once on **CPU** if it hits any cuDNN/CTranslate2 load error, so a broken cuDNN install no longer blocks the render path.

- **Align lyrics fails with `[WinError 1314] A required privilege is not held by the client` while writing into `cache\HF_HOME\hub\models--…\snapshots\…`:** `huggingface_hub` tries to **symlink** model snapshot files to the content-addressed `blobs/` directory, but Windows requires admin rights or **Developer Mode** to create symlinks. Glitchframe patches `huggingface_hub.are_symlinks_supported` at startup so it copies blobs instead — **Update** + **Start** in Pinokio (no full reinstall) is enough to pick up the fix. If the cache ended up half-populated before updating, delete `C:\pinokio\api\Glitchframe.git\cache\HF_HOME\` from Explorer and the next Align run will re-download cleanly. Details: [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) § Bug F.

- **Render is unexpectedly slow / "compositing" stuck around 40 %:** the orchestrator label parks on its outer status during compositing because individual frames are reported by an inner Gradio progress callback. Recent builds replaced the per-frame UI message with a richer one (`Compositing 1843/4923 (37.4%) - 1.23 fps - ETA 41m42s - layers=BG+SHADER+TYPO`) and update it from the request thread, so the bar now ticks live during long renders. After a render finishes, the run log appends a one-liner like `compositor: 4923 frames in 41m23s · avg 1.98 fps · encoder=h264_nvenc` — if `encoder=libx264` and you have an NVIDIA GPU, NVENC fell back to CPU; check the startup log for the `ffmpeg` candidate list and any `Cannot load nvEncodeAPI64.dll` / `Driver does not support the required nvenc API version` lines from the probe stderr. Multi-candidate ffmpeg discovery (env override → active env's `bin` → PATH → well-known dirs) means a working NVENC ffmpeg anywhere on your system will be picked up automatically, even if Pinokio's bundled one shadows it.

- **Render fails with `Undefined constant or missing '(' in 'p5'` / `Unable to parse option value "p5"` after background generation:** your local ffmpeg is older than 4.4 (or NVENC SDK < 11) and doesn't understand the modern `p1..p7` preset family. Glitchframe now probes the chosen ffmpeg once and falls back to the legacy `slow` preset automatically, so **Update + Start** in Pinokio is enough to pick up the fix — no reinstall. Visual quality is essentially unchanged (`slow` is the closest legacy equivalent of `p5`). To get back to the modern preset family, install a recent ffmpeg (`winget install ffmpeg` on Windows, or `conda update -c conda-forge ffmpeg` inside Pinokio's env). Details: [`docs/technical/pinokio-lyrics-align-windows-handover.md`](docs/technical/pinokio-lyrics-align-windows-handover.md) § Bug H.

- **Pinokio / Gradio conflicts after torch (`markupsafe` / `pillow`):** older install flows could break Gradio by upgrading those packages; current ``install.js`` avoids that. **Pull latest** and **Reinstall** (or **Install**) in Pinokio so deps are corrected without manual steps. Advanced: [`docs/technical/pinokio-package.md`](docs/technical/pinokio-package.md).

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
