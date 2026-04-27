# Getting started on Windows (step-by-step)

This guide is for people who are **new to the command line** or want a clear **order of installation**. Glitchframe runs locally in your browser; you install dependencies once, then start the app from PowerShell.

**Related:** [README](../../README.md) (overview, features, limitations), [Project setup and configuration](../technical/project-setup-and-config.md) (paths and config), [Windows venv recovery](../technical/windows-venv-recovery-guide.md) (if PyTorch or lyrics tools break after updates).

## What to expect

- A **NVIDIA GPU** with recent drivers is **strongly recommended** (diffusion, analysis, and video encoding). CPU-only use is possible for lighter settings but is not the main focus.
- First runs **download large models** to `.cache/` and `cache/`. Keep **dozens of GB** free on the disk you use for the project.
- Full renders can take **a long time** (often on the order of 1–2+ hours for a full track, depending on settings and hardware). See *Known limitations* in the [README](../../README.md).

## Terms (quick)

| Term | Meaning |
|------|--------|
| **PowerShell** | Windows’ text-based terminal. You type commands; we show them in `monospace` blocks to copy. |
| **Folder / directory** | A location on disk (e.g. `C:\Users\You\Glitchframe`). |
| **PATH** | A list of folders Windows searches when you type a program name (e.g. `ffmpeg`). Installers that “add to PATH” let you run those tools from any folder. |
| **Virtual environment (venv)** | A private Python install for this project so packages do not clash with other software. |
| **Repository (repo)** | The project’s source code, usually from GitHub. |

## Order: what to install first

Install in this order so each step can be verified before moving on.

1. **Python 3.11** (required)  
2. **Git** (optional — only if you want `git clone` and easy updates; you can use a ZIP download instead)  
3. **ffmpeg** (required for encoding) — install with **winget** in PowerShell (see below)  
4. **NVIDIA driver** (recommended if you have an NVIDIA GPU) — use GeForce Experience or [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)  
5. **Glitchframe** project files — git clone *or* GitHub **Download ZIP**  
6. Inside the project: **venv → PyTorch (CUDA) → project packages → run**

**PATH and reboots:** If you install **Python** (from the website **or** with `winget` in PowerShell), Windows often does **not** put `py` / `python` on your `PATH` until you **restart the computer**. A new PowerShell window is **not** always enough—especially if Python was installed from PowerShell. **Reboot at least once before** you create the venv (section 6) or run the app, and use [Reboot before you continue](#reboot-before-you-continue) after **ffmpeg** if you have not rebooted since Python.

You do **not** need to install CUDA Toolkit separately for typical use; the PyTorch pip wheels bring the CUDA **runtime** libraries the app needs.

---

### 1. Python 3.11

Install Python in either of these ways (pick one):

- **Installer (typical):** open [Python releases for Windows](https://www.python.org/downloads/windows/) and install **Python 3.11.x** (64-bit). On the first screen, turn on **“Add python.exe to PATH”**, then **Install Now**.
- **PowerShell (winget):** e.g. `winget install -e --id Python.Python.3.11` (or run `winget search Python 3.11` for the current package id). Installing Python this way almost always needs a **full reboot** before `py` works in PowerShell (see [Reboot before you continue](#reboot-before-you-continue)).

You can install **Git** and **ffmpeg** (sections 2–3) in the same session; they do not need `py`. **Before section 6** (virtual environment), you **must** have rebooted at least once since installing Python if `py` did not work in a new PowerShell window before the reboot.

**In a new PowerShell after that reboot**, run:

```powershell
py -3.11 --version
```

You should see a line like `Python 3.11.x`. If `py` is missing, try:

```powershell
python --version
```

If neither works, confirm the installer had **“Add python.exe to PATH”** (repair the install if needed) and **reboot again**, then retry.

---

### 2. Git (optional)

**You need Git only if** you want to use `git clone` and `git pull` to update. Otherwise skip to ffmpeg.

- Download: [Git for Windows](https://git-scm.com/download/win) and run the installer (default options are fine for most users).
- After install, new PowerShell windows can use:

```powershell
git --version
```

**Without Git:** on the [Glitchframe GitHub](https://github.com/OlaProeis/Glitchframe) page, use **Code → Download ZIP**, extract the ZIP to a folder of your choice (e.g. `C:\Users\You\Glitchframe`), and use that folder in the steps below instead of `git clone`.

---

### 3. ffmpeg (winget, recommended)

[winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) is built into current Windows 10/11. It installs `ffmpeg` and usually sets up your PATH.

1. Open **PowerShell** (not necessarily admin): press **Win**, type `PowerShell`, open **Windows PowerShell**.
2. Install a full `ffmpeg` build (common choice on winget):

```powershell
winget install -e --id Gyan.FFmpeg
```

If that package id is not found, search and pick a **ffmpeg** “full” / non-minimal build:

```powershell
winget search ffmpeg
```

3. **Close PowerShell and open a new window** (so PATH updates apply).
4. Check:

```powershell
ffmpeg -version
```

The first line should mention `ffmpeg version`. If you see *not recognized*, **restart your computer**, then try again in a new PowerShell window.

---

### Reboot before you continue

**Restart Windows now** (full reboot, not only “Sign out”) if **any** of these are true:

- You **just installed or re-upgraded Python** (installer or `winget`), and you have **not** rebooted since.
- `py -3.11` or `ffmpeg` still said *not recognized* after a new PowerShell window.

After reboot, open a **new** PowerShell and run `py -3.11 --version` and `ffmpeg -version` again. Do **not** create the venv (section 6) or run the app until those commands work.

---

### 4. NVIDIA driver (if you have an NVIDIA GPU)

Install the **latest driver** for your card from NVIDIA. You do not need a separate “CUDA installer” for Glitchframe’s pip-based setup in most cases.

To confirm the GPU is visible to the system after the driver is installed (optional):

```powershell
nvidia-smi
```

A table with your GPU name and driver version means the system sees the card.

---

### 5. Get the Glitchframe project files

**Option A — Git (updates with `git pull`):**

```powershell
cd $HOME\Documents
git clone https://github.com/OlaProeis/Glitchframe.git
cd Glitchframe
```

(Use any parent folder you prefer instead of `Documents`.)

**Option B — ZIP (no Git):**

1. Download the ZIP from GitHub and extract it, e.g. to `C:\Users\You\Glitchframe`.
2. In PowerShell, go to that folder (adjust the path to match your machine):

```powershell
cd C:\Users\You\Glitchframe
```

---

### 6. Create and activate a virtual environment

Still in the project root (folder that contains `README.md` and `pyproject.toml`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, you may need to allow scripts for the current user (run PowerShell **once**):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try `.\.venv\Scripts\Activate.ps1` again.

When the venv is active, your prompt often starts with `(.venv)`.

---

### 7. PyTorch with CUDA (recommended on NVIDIA)

Install **GPU** PyTorch **before** the rest of the project packages:

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Use the **same** `python` that belongs to the venv (after activation, `python` should point to `.venv\Scripts\python.exe`).

---

### 8. Project dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

### 9. Optional: full analysis + lyrics (Demucs, WhisperX, VAD)

```powershell
python -m pip install -e ".[all]"
```

If CUDA stops working or WhisperX causes issues, see the recovery steps in [requirements.txt](../../requirements.txt) (comments at the top) and [windows-venv-recovery-guide.md](../technical/windows-venv-recovery-guide.md).

Optional beat detectors (can be finicky to build):

```powershell
python -m pip install -e ".[beats]"
```

---

### 10. Optional: environment file

```powershell
copy .env.example .env
```

Edit `.env` only if you need custom `GLITCHFRAME_*` paths or encoder overrides. Optional API keys in the sample are for **Taskmaster / dev tooling**, not for core Glitchframe.

---

## Run the app

**Before the first start:** if you installed Python (or reinstalled it) in this session and have **not** rebooted since, **restart Windows** now, then open a new PowerShell, `cd` to the project folder, run `.\.venv\Scripts\Activate.ps1`, and continue below. The app must see the same `python` on `PATH` that you used to create the venv.

With the venv activated, from the project root:

```powershell
python -m app
```

In the output, open the local URL (default [http://127.0.0.1:7860](http://127.0.0.1:7860)) in your browser.

To stop the server, focus the PowerShell window and press **Ctrl+C**.

---

## If something goes wrong

- **PyTorch, WhisperX, or cuDNN errors on Windows:** [windows-venv-recovery-guide.md](../technical/windows-venv-recovery-guide.md)
- **Align lyrics / `Weights only load failed` / cuDNN DLL messages:** [README — Troubleshooting](../../README.md#troubleshooting)

---

## Updating Glitchframe

- **If you used Git:** from the project folder, `git pull`, then reactivate the venv and repeat pip steps only when [README](../../README.md) or release notes say dependencies changed.
- **If you used ZIP:** download a fresh ZIP from GitHub and replace the project folder, or use Git for updates next time.
