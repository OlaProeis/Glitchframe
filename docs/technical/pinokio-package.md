# Pinokio package (Glitchframe)

[Pinokio](https://pinokio.co/) runs apps from a public Git URL using small scripts in the repo root. This project ships:

| File | Role |
|------|------|
| `install.js` | One `shell.run`: venv `env`, `venv_python` **3.11**, then `ensurepip`, then PyTorch **2.2.2+cu121** (CUDA **12.1** wheel index), `requirements.txt`, `pip install -e .`, `pip install -e ".[all]"`, re-pin torch trio, `whisperx==3.3.0` / `faster-whisper==1.1.0` / `ctranslate2==4.4.0`, `nvidia-cudnn-cu12`, and `scripts/windows_provision_cudnn_next_to_ctranslate2.py` |
| `start.js` | Daemon: `python -m app` with `GLITCHFRAME_WHISPERX_VAD_METHOD=silero` and **`GLITCHFRAME_WHISPERX_DEVICE=cpu`** (safe default). After the **cu121** install stack, you may remove **`GLITCHFRAME_WHISPERX_DEVICE`** or set **`cuda`** to try **GPU** Align lyrics. First `http://…` → **Open Web UI** |
| `reset.js` | Delete folder `env` (factory reset; reinstall via Install) |
| `update.js` | `git pull` at repo root |
| `pinokio.js` | Sidebar: Install / Start / Update / Reinstall / Reset; prerequisite hint for **ffmpeg** |
| `icon.png` | Launcher icon (derived from a UI screenshot) |
| `pinokio_meta.json` | Optional name / description / homepage for listings |

**Launch:** After install, the user must click **Start** in Pinokio; the app does not auto-run. **ffmpeg** must be on the user&rsquo;s `PATH` (Pinokio cannot install system encoders for you). **Windows + Align lyrics:** the installer pins a **coherent cu121** stack (see `pyproject.toml` optional deps). ``start.js`` still defaults **CPU** WhisperX so broken setups complete; opt into **GPU** alignment by clearing or overriding ``GLITCHFRAME_WHISPERX_DEVICE`` per README. If `.[all]` fails on a given machine, use **Factory reset** or trim `install.js` locally (see README).

**Discoverability:** Add the GitHub topic `pinokio` so the app can appear on Pinokio&rsquo;s discover page.

Running **outside Pinokio** (``python -m app`` from a normal shell) does **not** set that env var unless you add it to ``.env``; behaviour matches a stock local install.
