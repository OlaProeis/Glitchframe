# Pinokio package (Glitchframe)

[Pinokio](https://pinokio.co/) runs apps from a public Git URL using small scripts in the repo root. This project ships:

| File | Role |
|------|------|
| `install.js` | One `shell.run`: venv `env`, `venv_python` **3.11**, then `pip` in README order: PyTorch **cu124** index, `requirements.txt`, `pip install -e .` (no `torch.js`, so the terminal shows real `pip` progress) |
| `start.js` | Daemon: `python -m app`, capture first `http://…` for **Open Web UI** |
| `reset.js` | Delete folder `env` (factory reset; reinstall via Install) |
| `update.js` | `git pull` at repo root |
| `pinokio.js` | Sidebar: Install / Start / Update / Reinstall / Reset; prerequisite hint for **ffmpeg** |
| `icon.png` | Launcher icon (derived from a UI screenshot) |
| `pinokio_meta.json` | Optional name / description / homepage for listings |

**Not automatic:** `pip install -e ".[all]"` (Demucs, WhisperX, etc.) is left manual after core install; see the install notify text and the main [README](../../README.md). **ffmpeg** must be on the user&rsquo;s `PATH` (Pinokio cannot replace system encode tools).

**Discoverability:** Add the GitHub topic `pinokio` so the app can appear on Pinokio&rsquo;s discover page.

These files are independent of the Python package and Gradio app; they do not change runtime behavior when you run `python -m app` outside Pinokio.
