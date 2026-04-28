# Windows: recover Glitchframe after lyrics / PyTorch issues

Step-by-step instructions if **Align lyrics** failed with **OmegaConf / `ListConfig` / weights_only**, or after a **`torch` downgrade** you see **`Could not locate cudnn_ops_infer64_8.dll`**. Follow in order; use the **same** virtual environment you always use to run the app.

## What you are fixing

1. **PyTorch 2.6+** loads model checkpoints more strictly. WhisperX / pyannote checkpoints need a small compatibility shim shipped in the repo (`pipeline/torch_checkpoint_compat.py`).
2. **Downgrading only `torch`** to avoid that error often breaks **cuDNN DLLs** on Windows (faster-whisper looks for `cudnn_ops_infer64_8.dll` while your venv may only have newer names). The fix is a **clean reinstall** of **torch + torchvision + torchaudio** together from the official CUDA wheel index, **not** a lone `torch` pin.

---

## Before you start

- Close Glitchframe if it is running (close the terminal where `python -m app` is running, or press **Ctrl+C** in that window).
- Know your project folder. Examples below use `C:\glitchframe\glitchframe` — replace with your actual path if different.
- You need **Git** installed and a network connection for `git pull` and `pip`.

---

## Step 1 — Open PowerShell in the project folder

1. Press **Win**, type **PowerShell**, open **Windows PowerShell** (not required as Administrator).
2. Go to your clone:

```powershell
cd C:\glitchframe\glitchframe
```

(Adjust the path if your repo lives elsewhere.)

---

## Step 2 — Activate the virtual environment

If you use a `.venv` inside the repo (recommended):

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by policy, run once **in the same PowerShell window**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try `Activate.ps1` again.

You should see `(.venv)` at the start of your prompt.

---

## Step 3 — Confirm you are using the venv’s Python

This must print a path **inside** `.venv\Scripts\python.exe`:

```powershell
python -c "import sys; print(sys.executable)"
```

Example (good):

`C:\glitchframe\glitchframe\.venv\Scripts\python.exe`

If it points to `Program Files\Python...` instead, you are **not** in the venv — go back to Step 2.

---

## Step 4 — Update the code from GitHub

```powershell
git pull origin master
```

If your default branch is `main`, use:

```powershell
git pull origin main
```

If Git reports conflicts, stop and message the maintainer with the full `git status` output.

**Optional check:** the compatibility file should exist after a successful pull:

```powershell
Test-Path .\pipeline\torch_checkpoint_compat.py
```

It should print `True`.

---

## Step 5 — Reinstall PyTorch as a matched set (pick **one** track)

This avoids mixed cuDNN / partial upgrades.

### Track A — Windows, Python 3.11 or 3.12 (pinned “Align lyrics” GPU stack)

Use **CUDA 12.1** wheels and the versions in `pyproject.toml` **all** / **lyrics** extras (**torch 2.2.2+cu121**, **WhisperX 3.3.0**, **ctranslate2 4.4.0**, **faster-whisper 1.1.0**):

```powershell
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then reinstall extras (Step 6): `python -m pip install -e ".[all]"` so markers apply. Optionally: `pip install nvidia-cudnn-cu12` then `python scripts/windows_provision_cudnn_next_to_ctranslate2.py`.

### Track B — Windows (Python 3.13+), Linux, macOS, or any install using **cu124**

Use the **CUDA 12.4** index (replace **cu124** with **cu121** in both uninstall/install commands **only** if you deliberately standardized on Track A everywhere):

```powershell
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Wait until it finishes without errors.

**Check versions:**

```powershell
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

You want `cuda: True` if you use an NVIDIA GPU.

---

## Step 6 — Reinstall project + lyrics / analysis extras

From the repo root, with venv still active:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ".[all]"
```

If `[all]` fails, try at minimum:

```powershell
python -m pip install -e ".[lyrics]"
```

---

## Step 7 — Run the app

```powershell
python -m app
```

Open the URL shown (usually **http://127.0.0.1:7860**).

---

## Step 8 — Test Align lyrics

1. Ingest / analyze a song as you normally do (so `vocals.wav` exists in the song cache).
2. Paste lyrics and click **Align lyrics** (or your UI equivalent).
3. If it works, you should get timings and no traceback in the terminal.

You may still see a **PyTorch warning** about `weights_only=False` and pickle security when pyannote/lightning loads checkpoints — that is a **notice**, not necessarily a failure.

---

## If something still fails

### A) `Weights only load failed` / `ListConfig` / `omegaconf`

- Confirm `git pull` brought in `pipeline\torch_checkpoint_compat.py`.
- Confirm you restarted the app **after** the pull (Step 7 again).

### B) `Could not locate` / `Could not load library` `cudnn_ops_infer64_8.dll` (or error `1920`) — including **after** Silero VAD runs

1. **If you use Track A (Windows, Python 3.11/3.12, cu121 + WhisperX 3.3.0):** keep **ctranslate2==4.4.0** and the PyTorch **2.2.2+cu121** trio — **do not** upgrade ctranslate2 to 4.5+ unless you also move to **cu124** and newer WhisperX (see `pyproject.toml` or README §2 Track B). Run `python scripts/windows_provision_cudnn_next_to_ctranslate2.py` after `pip install nvidia-cudnn-cu12` if DLL resolution still fails.
2. **If you use Track B (cu124 + Python 3.13 or Linux/macOS):** a mismatched **4.4.x** ctranslate2 next to **PyTorch cu124** can cause this — upgrade:  
   `python -m pip install -U "ctranslate2>=4.5.0,<5"`, then `python -m pip install -e ".[all]"` (or the extra you use).
3. Repeat **Step 5** for your chosen **track** (uninstall all three, reinstall all three from the **same** index), then reinstall extras.
4. The app also registers `torch\lib` for DLL search on Windows before loading WhisperX (`pipeline/win_cuda_path.py`).

5. Last resort: see the **Troubleshooting** section in the repo **README.md** (optional `nvidia-cudnn-cu12`, PATH, or manual cuDNN copy from NVIDIA).

### C) CUDA disappeared after `pip install whisperx`

Re-run **Step 5** (PyTorch trio from `cu124`), then **Step 6**, same as README recovery notes.

---

## Quick reference — copy/paste block

After `cd` + venv activate:

```powershell
git pull origin master
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ".[all]"
python -m app
```

---

## Where to get help

- Repo **README.md** → **Troubleshooting**
- Upstream context: [whisperX#899](https://github.com/m-bain/whisperX/issues/899) (cuDNN DLL on Windows)
