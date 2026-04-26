# MusicVids

Local, GPU-accelerated **music video** generator: upload a track, analyze audio, align lyrics, style backgrounds (SDXL stills, Ken Burns, optional AnimateDiff), composite reactive shaders and kinetic type, and encode with **ffmpeg** (NVENC on NVIDIA GPUs by default).

- **UI:** [Gradio](https://www.gradio.app/) — run `python -m app` and open the URL shown (default [http://127.0.0.1:7860](http://127.0.0.1:7860)).
- **Docs:** see [`docs/index.md`](docs/index.md) and [`docs/technical/project-setup-and-config.md`](docs/technical/project-setup-and-config.md) for layout, config, and optional features.

**License:** [MIT](LICENSE)

## Requirements

- **Python** 3.11+ (3.12/3.13 may work; optional deps like `madmom` are pickier on newer Python)
- **ffmpeg** on your `PATH` (encode/mux; see the [ffmpeg download page](https://ffmpeg.org/download.html))
- **NVIDIA GPU + CUDA 12.x** recommended for diffusers, analysis, and NVENC; CPU-only is possible for lighter paths but not the main focus
- **Disk:** model and song caches under `.cache/` and `cache/` (large downloads on first use)

## Install

### 1. Clone and virtualenv

```bash
git clone https://github.com/YOUR_ORG/MusicVids.git
cd MusicVids
python -m venv .venv
```

Use your real GitHub URL after you create the repository. Then activate the venv:

- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. PyTorch with CUDA (recommended)

Install PyTorch **first** from the official CUDA 12.4 wheel index so you get a GPU build (adjust if you use a different CUDA index from [pytorch.org](https://pytorch.org/)):

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

On Unix: `cp .env.example .env` — then edit `.env` if you need custom `MUSICVIDS_*` paths or ffmpeg codec overrides. The sample file also lists optional API keys for **Taskmaster** / dev tooling, not for core MusicVids.

## Run

```bash
python -m app
```

Open the local URL printed in the console (default port **7860**).

## Development

- Smoke test config/presets: `python config.py`
- Tests (after `pip install -e ".[dev]"`): `pytest`
- In this repo, `uv sync` / `uv run pytest` is also used; see `ai-context.md` for maintainer notes.

## Contributing

Issues and pull requests are welcome. Please keep changes focused; match existing style in the files you touch.

## Legal

This project is licensed under the [MIT License](LICENSE). Third-party assets (e.g. fonts under `assets/fonts/`) carry their own license files where applicable.
