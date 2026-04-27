# Pinokio package (Glitchframe)

[Pinokio](https://pinokio.co/) runs apps from a public Git URL using small scripts in the repo root. This project ships:

| File | Role |
|------|------|
| `install.js` | One `shell.run`: venv `env`, `venv_python` **3.11**, then `ensurepip` (some Pinokio venvs ship without `pip`), then `pip`: PyTorch **cu124** index, `requirements.txt`, `pip install -e .`, `pip install -e ".[all]"` |
| `start.js` | Daemon: `python -m app` with `env.GLITCHFRAME_WHISPERX_VAD_METHOD=silero` (WhisperX VAD; avoids Pyannote+cuDNN issues on some Windows installs). First `http://…` → **Open Web UI** |
| `reset.js` | Delete folder `env` (factory reset; reinstall via Install) |
| `update.js` | `git pull` at repo root |
| `pinokio.js` | Sidebar: Install / Start / Update / Reinstall / Reset; prerequisite hint for **ffmpeg** |
| `icon.png` | Launcher icon (derived from a UI screenshot) |
| `pinokio_meta.json` | Optional name / description / homepage for listings |

**Launch:** After install, the user must click **Start** in Pinokio; the app does not auto-run. **ffmpeg** must be on the user&rsquo;s `PATH` (Pinokio cannot install system encoders for you). **Windows + Align lyrics:** extras in `pyproject.toml` pin `ctranslate2==4.4.0` to reduce `cudnn_ops_infer64_8.dll` load failures; run `git pull` and **`python -m pip install -e ".[all]"`** again in the app&rsquo;s `env` if the venv predates that pin. If `.[all]` fails on a given machine, use **Factory reset**, trim lines from `install.js` locally, and install extras manually (see main [README](../../README.md)).

**Discoverability:** Add the GitHub topic `pinokio` so the app can appear on Pinokio&rsquo;s discover page.

Running **outside Pinokio** (``python -m app`` from a normal shell) does **not** set that env var unless you add it to ``.env``; behaviour matches a stock local install.
