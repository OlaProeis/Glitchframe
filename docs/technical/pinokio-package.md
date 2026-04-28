# Pinokio package (Glitchframe)

[Pinokio](https://pinokio.co/) runs apps from a public Git URL using small scripts in the repo root. This project ships:

| File | Role |
|------|------|
| `install.js` | One `shell.run`: venv `env`, `venv_python` **3.11**, then `ensurepip` (some Pinokio venvs ship without `pip`), then `pip`: PyTorch **cu124** index, `requirements.txt`, `pip install -e .`, `pip install -e ".[all]"` |
| `start.js` | Daemon: `python -m app` with `GLITCHFRAME_WHISPERX_VAD_METHOD=silero` and **`GLITCHFRAME_WHISPERX_DEVICE=cpu`** so **Align lyrics** (faster-whisper/CTranslate2) avoids Windows cuDNN load failures beside PyTorch cu124 — slower transcription, GPU still used for Analyze/render; remove env line to try CUDA align locally. First `http://…` → **Open Web UI** |
| `reset.js` | Delete folder `env` (factory reset; reinstall via Install) |
| `update.js` | `git pull` at repo root |
| `pinokio.js` | Sidebar: Install / Start / Update / Reinstall / Reset; prerequisite hint for **ffmpeg** |
| `icon.png` | Launcher icon (derived from a UI screenshot) |
| `pinokio_meta.json` | Optional name / description / homepage for listings |

**Launch:** After install, the user must click **Start** in Pinokio; the app does not auto-run. **ffmpeg** must be on the user&rsquo;s `PATH` (Pinokio cannot install system encoders for you). **Windows + Align lyrics:** ``start.js`` defaults **CPU** for WhisperX only (`GLITCHFRAME_WHISPERX_DEVICE`) so faster-whisper/CTranslate2 does not hit `cudnn_ops_infer64_8.dll`-style failures; pull latest `install.js`/`pyproject.toml`, **Install** again if needed; see [README](../../README.md) troubleshooting to opt into GPU align. If `.[all]` fails on a given machine, use **Factory reset** or trim `install.js` locally (see README).

**Discoverability:** Add the GitHub topic `pinokio` so the app can appear on Pinokio&rsquo;s discover page.

Running **outside Pinokio** (``python -m app`` from a normal shell) does **not** set that env var unless you add it to ``.env``; behaviour matches a stock local install.
