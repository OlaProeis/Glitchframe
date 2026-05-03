# Project setup and configuration

Feature: repository layout, dependency metadata, and central path configuration for Glitchframe (optional `presets/*.yaml` registry for advanced use; the default UI is shader-first — see `docs/technical/visual-style-presets.md`).

## What was implemented

- **Directory layout:** `pipeline/` (package root for the processing stack), `presets/` (optional YAML; may be empty), `assets/shaders/`, `assets/fonts/`, `cache/`, `outputs/` — empty dirs kept in git via `.gitkeep` where needed. **Fonts:** keep `assets/fonts/Inter.ttf` (body / lyrics), `Inter-SemiBold.ttf` (fallback title), and `SpaceGrotesk-SemiBold.ttf` (preferred display face for the burned-in title + thumbnail) in the repo. All three are SIL Open Font License 1.1 — see `Inter-LICENSE.txt` and `SpaceGrotesk-LICENSE.txt`. If all three are missing, Skia falls back to system Arial-style faces and typography looks generic. See `config.default_title_font_path` / `default_ui_font_path` for the resolution order.
- **Dependencies:** `pyproject.toml` (PEP 621, `requires-python >= 3.11`, hatchling wheel over `pipeline`) and `requirements.txt` (same runtime deps; PyTorch install note for CUDA index).
- **Configuration module:** `config.py` defines `PROJECT_ROOT`, cache/output/preset/asset paths, optional `GLITCHFRAME_*` environment overrides (legacy `MUSICVIDS_*` still supported), `MODEL_CACHE_DIR`, `ensure_runtime_dirs()`, and optional strict preset loading (`load_preset_registry()`, `get_preset_ids()`, `get_preset()`) when `presets/*.yaml` files exist (see `docs/technical/visual-style-presets.md`).
- **Git:** `.gitignore` ignores `cache/*` and `outputs/*` except `.gitkeep`, and `.cache/` for local model cache; `.env.example` documents optional `GLITCHFRAME_*` path variables.

## Usage

- Smoke test: `python config.py` (prints resolved paths and number of YAML presets found, often `0`; requires `pyyaml`).
- Editable install: `pip install -e .` (requires Python 3.11+ per `pyproject.toml`).
- **One command for all optional Python deps** (demucs vocal stem + WhisperX lyrics):  
  `pip install -e ".[all]"` — same as `pyproject.toml` extras `all` / `analysis` (not `beats`; BeatNet/madmom is separate and often flaky). Use your venv’s Python on Windows, e.g. `.venv\Scripts\python.exe -m pip install -e ".[all]"`.
- **Outside pip:** `ffmpeg` must be on `PATH` for encode/ mux (see `docs/technical/spectrum-renderer-ffmpeg.md`). PyTorch CUDA wheels are pinned separately in `requirements.txt` comments if you need GPU.
- **WhisperX vs CUDA:** installing `whisperx` (or `.[all]`) can replace your CUDA PyTorch with a **CPU** build from PyPI, which breaks AnimateDiff until you reinstall `torch`/`torchvision`/`torchaudio` from the **cu124** index; see the recovery commands in `requirements.txt`.

## Related files

| File | Role |
|------|------|
| `config.py` | Paths, env overrides, preset registry loading |
| `pyproject.toml` | Project name, Python version, dependencies, packaging |
| `requirements.txt` | pip-friendly list + PyTorch CUDA install hint |
| `pipeline/__init__.py` | Package marker for hatchling wheel |
