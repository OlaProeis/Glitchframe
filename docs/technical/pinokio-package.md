# Pinokio package (Glitchframe)

[Pinokio](https://pinokio.co/) runs apps from a public Git URL using small scripts in the repo root. **Discovery:** In the Pinokio app, search **glitchframe** and install Glitchframe from the listing (easiest). You can also use **Download from URL** with this repo’s Git URL (`https://github.com/OlaProeis/Glitchframe.git`).

**Default Git branch on GitHub** is usually **`master`**. **Pinokio’s catalog install** clones that default branch. If you need **`dev`** (pre-release integration) **before** it lands on the default branch: open the Pinokio app folder in Explorer (the directory that contains `install.js` next to `env/`), run **`git fetch`**, **`git checkout dev`**, **`git pull`**, then in Pinokio run **Reinstall** (or **Install** again).

This project ships:

| File | Role |
|------|------|
| `install.js` | Declares Pinokio **`requires.bundle: ai`**. **First** `shell.run`: venv **`env`**, **`venv_python: "3.11"`**, `ensurepip`, `pip -U pip`, then **`uv pip install -r requirements.txt`**, **`uv pip install -e ".[all]"`**, **`uv pip install madmom --no-build-isolation`**, **`uv pip install beatnet --no-build-isolation --no-deps`**. **Then** `script.start` → **`torch.js`** (platform-specific PyTorch; e.g. NVIDIA on Windows: **CUDA 12.8** wheels **`torch==2.7.0`** + matching `torchvision` / `torchaudio` from PyTorch’s index). Ends with a **notify** completion panel. |
| `torch.js` | Pinokio script: **`uv pip install`** for **PyTorch** by **GPU + OS** (`nvidia` Win/Linux cu128, AMD Win DirectML, AMD Linux ROCm, macOS CPU, fallback CPU). Invoked from `install.js` after Python deps. |
| `start.js` | Daemon: `python -m app` with `GLITCHFRAME_WHISPERX_VAD_METHOD=silero`, **`GLITCHFRAME_WHISPERX_DEVICE=cpu`** (safe default for faster-whisper / cuDNN), and **`HF_HUB_DISABLE_SYMLINKS=1`** + **`HF_HUB_DISABLE_SYMLINKS_WARNING=1`** (Windows symlink / privilege workaround; see lyrics handover § Bug F). Remove **`GLITCHFRAME_WHISPERX_DEVICE`** or set **`cuda`** to try **GPU** Align lyrics when your stack supports it; the aligner can fall back to CPU on load errors. First `http://…` → **Open Web UI**. |
| `reset.js` | Delete folder `env` (factory reset; reinstall via Install) |
| `update.js` | `git pull` at repo root (**updates whatever branch is checked out** — use **`dev`** or **`master`** intentionally) |
| `pinokio.js` | Package metadata **`version: "3.7"`**; sidebar menu (installing / start / update / reset states, **Reset** confirm, **Start** default when idle). |
| `icon.png` | Launcher icon (derived from a UI screenshot) |
| `pinokio_meta.json` | Optional name / description / homepage for listings |

**RIFE morph:** First use of **Morph keyframes (RIFE)** downloads **~24 MB** from Hugging Face into `MODEL_CACHE_DIR` (same `HF_HUB_DISABLE_SYMLINKS` behaviour as WhisperX on Windows).

**Launch:** After install, the user must click **Start** in Pinokio; the app does not auto-run. **ffmpeg** must be on the user&rsquo;s `PATH` (Pinokio cannot install system encoders for you). **`requirements.txt`** pins **Gradio 5.x** and related UI deps; manual installs should run **`pip`/`uv pip install -r requirements.txt`** before **`pip install -e .`** if you want the same stack as Pinokio (see README). If `.[all]` fails on a given machine, use **Factory reset** or open an issue with logs.

**Tracebacks and `...\uv\python\...` paths:** If an error mentions `collections` or another stdlib module under `%AppData%\Roaming\uv\python\` (or similar), that is usually the **standard library prefix for the interpreter that ran the failing import**, not proof that Pinokio started a different “global Python” while your app logged the `env` launcher. **`uv`** can install the base interpreter there; **venv still uses `env\Scripts\python.exe`** as `sys.executable`. Same process, library files rooted under that store path.

### No interactive terminal inside Pinokio?

That is normal. Pinokio usually only shows **install output** (streaming log), not an ongoing shell where you type `pip` — you are **not** missing an obvious button.

**Apply repo fixes without typing any command:**

1. In Pinokio, use **Update** (`git pull` at repo root) so you have the latest `install.js`.
2. Run **Install** or **Reinstall** so Pinokio re-executes the install script (`shell.run`). That reinstalls/fixes deps from the repo — **no manual `pip` needed**.

**If you truly need one-off commands** (`pip freeze`, manual repair scripts): use **Windows**, not Pinokio:

1. Open **File Explorer** and go to the Pinokio app folder — often `C:\pinokio\api\Glitchframe.git` (yours may differ; find the folder that contains an **`env`** subfolder next to `install.js`).
2. Click the address bar once, type **`cmd`**, press **Enter**. (Windows opens **Command Prompt** already `cd`'d into that folder — no need to memorize paths.)
3. Run **`env\Scripts\activate.bat`** then **`pip`** / **`python`** as needed.

Alternatively: **Shift+right-click** in an empty spot in that folder → **Open in Terminal** / **Open PowerShell window here** (wording varies by Windows version).

**Discoverability:** Users can find Glitchframe inside Pinokio by searching **glitchframe**. The repo also uses the GitHub topic **`pinokio`** for listing metadata.

Running **outside Pinokio** (``python -m app`` from a normal shell) does **not** set that env var unless you add it to ``.env``; behaviour matches a stock local install.
