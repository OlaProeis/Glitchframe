# Handover: Pinokio + Windows “Align lyrics” (WhisperX / cuDNN / Pyannote)

Use this as the **prompt for a fresh chat** when continuing this work. The owner reports **Align lyrics** still failing under **Pinokio on Windows** with **`cudnn_ops_infer64_8.dll` / error 1920** (or the same class of load failure), after multiple mitigations. They are concerned about **regressing the local (non-Pinokio) app**; prefer **small, testable changes** and consider **reverting** experimental pipeline edits if they do not help Pinokio.

---

## Goal

Run **Glitchframe** from **Pinokio** (`install.js` / `start.js` / …) on **Windows**, with **Align lyrics** (WhisperX + optional extras) working. **Local** `python -m app` outside Pinokio was reported **working** before this effort; changes should not unnecessarily alter local behaviour.

---

## Symptoms (Pinokio, Windows)

- Failure during WhisperX / VAD path, often after downloads succeed.
- Log line: `>>Performing voice activity detection using Pyannote...`
- Then: **`Could not load library cudnn_ops_infer64_8.dll`** (user also saw **error code 1920**).
- PyTorch **2.6.x+cu124**; Pyannote / Lightning warnings about checkpoint age are usually **not** the direct crash; the hard failure is **native DLL load** (cuDNN naming / faster-whisper / pyannote stack on Windows).

---

## What was added / changed (inventory for the next session)

### Pinokio packaging (repo root; should not affect local `python -m app` unless those files are used)

- `install.js` — `ensurepip`, then `pip`: torch **cu124** index, `requirements.txt`, `pip install -e .`, `pip install -e ".[all]"`. **No** `script.start` → `torch.js` (was hanging with no terminal output).
- `start.js` — `python -m app`, Gradio URL detection; **`env.GLITCHFRAME_WHISPERX_VAD_METHOD: "silero"`** (see pipeline section).
- `reset.js`, `update.js`, `pinokio.js`, `icon.png`, `pinokio_meta.json`
- `docs/technical/pinokio-package.md`, README **Pinokio** section

### Dependencies

- `pyproject.toml` — Windows-only (PEP 508) pin: **`ctranslate2==4.4.0`** on extras **`all`**, **`lyrics`**, **`analysis`** (mitigate faster-whisper / DLL name issues per README / whisperX#899). **No** effect on non-Windows installs of those extras.

### Pipeline (shared codebase — affects any run that uses these code paths / env)

- `pipeline/lyrics_aligner.py` — `whisperx.load_model` may receive **`vad_method`** when **`GLITCHFRAME_WHISPERX_VAD_METHOD`** is `pyannote` or `silero`; **if unset**, **do not** pass `vad_method` (WhisperX default, i.e. Pyannote for VAD). **Retry / TypeError** fallbacks preserved for older WhisperX.
- **Pinokio** is supposed to set `GLITCHFRAME_WHISPERX_VAD_METHOD` via `start.js` so **only Pinokio-launched** processes get **`silero`** unless the user sets `.env` locally.

### Docs / env examples

- `README.md` Troubleshooting (cudnn, Pinokio, ctranslate2, Silero)
- `docs/technical/windows-venv-recovery-guide.md` (section on cudnn)
- `.env.example` — documents `GLITCHFRAME_WHISPERX_VAD_METHOD`

---

## What we tried (chronological)

1. **Pinokio `install.js` used `script.start` → `torch.js` + `venv_python: 3.11`** — appeared **stuck** on step 1/3, empty terminal. **Replaced** with **explicit** `shell.run` + `pip` (torch from **cu124** index, etc.) for visible progress.
2. **Missing `pip` in venv** — some Pinokio venvs had no pip. **Added** **`python -m ensurepip --upgrade`** at the start of `install.js`.
3. **“One click”** — **`pip install -e ".[all]"`** added to `install.js` (Demucs, WhisperX, etc.); post-install text clarified that user must **click Start** (Pinokio does not auto-launch the server). **`default: true`** on **Start** in `pinokio.js` when `env` exists.
4. **`cudnn_ops_infer64_8.dll` after ctranslate2 path** — **Pinned `ctranslate2==4.4.0`** in Windows optional extras in **`pyproject.toml`**. **Did not** fix; log still showed **Pyannote** VAD.
5. **Pyannote VAD vs faster-whisper** — assumed Pyannote was the layer still loading bad DLLs. **Implemented** optional **`vad_method`** in `load_model` (Silero), then **scoped** so:
   - default local: **no** `vad_method` (unchanged from pre-feature behaviour for unset env);
   - Pinokio: **`start.js` sets `GLITCHFRAME_WHISPERX_VAD_METHOD=silero`**.
6. **User reports:** issue **still reproduces in Pinokio** (“same issue basically”). **Afraid the app is broken** and wants a **handover to a new chat**.

---

## What is still unknown / to verify in the next session

- Whether **`vad_method` is actually supported** and applied by the **installed WhisperX** version in the Pinokio venv (TypeError would log a warning and fall back).
- Whether the **Gradio / orchestrator process** in Pinokio **inherits** `start.js`’s `env` for **long-running** work (suspect yes, but confirm).
- Whether **Silero** is used when env is set (log should **not** say “using Pyannote” for that step, if the stack respects `vad_method`).
- Whether the failure is **entirely** in **Pyannote** or also in **faster-whisper/ctranslate2** on another line — **separate** stack traces help.
- **Full traceback** and **`pip list` / `pip show whisperx ctranslate2 pyannote-audio torch`** from **`C:\pinokio\api\Glitchframe.git\env`**.

---

## Suggested next steps (for the new chat)

- Collect **one clean log** from Pinokio: from click **Align lyrics** through crash, plus **`GLITCHFRAME_WHISPERX_VAD_METHOD`** value inside the process (or add temporary logging in `lyrics_aligner` at `load_model` time).
- Confirm **`start.js` `env` is passed** to the shell that runs `python -m app` in the user’s Pinokio version (Pinokio docs / minimal repro).
- If the goal is **Pinokio-only** fixes: prefer **`install.js` / `start.js` / `pinokio.js` + documented `.env`**, and **avoid** broad pipeline defaults; revert or feature-flag pipeline edits if they do not fix Pinokio.
- Revisit **upstream** WhisperX issues (e.g. **#899**, **Silero VAD** PRs) for **Windows + cu124 + torch 2.6** combinations.
- Last resort (documented in README): **NVIDIA cuDNN 8.9** DLL copy into venv `torch\lib` or `ctranslate2` package dir — user **dislikes** manual DLL surgery; treat as **documented opt-in**.

---

## Reverting / “get back to known-good local” (owner actions)

- **Local:** do **not** set `GLITCHFRAME_WHISPERX_VAD_METHOD` in `.env` unless needed; `lyrics_aligner` then matches **pre-`vad_method` behaviour** for unset env.
- **Full revert of pipeline + extras:** use `git log` on `main`/`master` and revert commits touching `pipeline/lyrics_aligner.py`, `pyproject.toml` (ctranslate2), and optionally Pinokio files — only if the team decides the experiment is not worth keeping.

---

## Key file references

| Area | Path |
|------|------|
| Align lyrics / WhisperX | `pipeline/lyrics_aligner.py` (`_run_whisperx_forced`, `load_model`) |
| Torch load shim | `pipeline/torch_checkpoint_compat.py` |
| Pinokio | `install.js`, `start.js`, `pinokio.js` |
| Windows recovery | `docs/technical/windows-venv-recovery-guide.md` |
| Package extras | `pyproject.toml` `[project.optional-dependencies]` |
| User env sample | `.env.example` |

---

*Created for session handover; update only when the investigation changes materially.*
