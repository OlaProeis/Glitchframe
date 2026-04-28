# Session Handover

## Environment

- **Project:** Glitchframe
- **Tech Stack:** Python 3.11, CUDA/PyTorch, Gradio, librosa/demucs/whisperx, diffusers (SDXL / AnimateDiff), moderngl, skia-python, ffmpeg NVENC
- **Context file:** Always read `ai-context.md` first — it contains project rules, architecture, and model selection.
- **Python interpreter:** `.\.venv\Scripts\python.exe` (the Windows `py` launcher does NOT point at the project venv — use the full path).
- **Run tests:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` (pytest is not wired; Task Master may mention `pytest` — prefer `unittest` unless the project wires pytest).
- **Branch:** main

## Core Handover Rules

- **NO HISTORY:** Do not include past task details unless they directly impact this specific task.
- **SCOPE:** Focus ONLY on the task detailed below.

## Current Task

### Lyrics alignment: automatic CPU fallback when WhisperX / CTranslate2 GPU path fails

**Goal.** When **Align lyrics** selects **GPU** (CUDA) for WhisperX (faster-whisper / CTranslate2) but the process hits a **native load or runtime failure** typical of Windows (e.g. `cudnn_ops_infer64_8.dll`, error 1920, similar CTranslate2/cuDNN messages), **retry the same alignment pipeline once on CPU** so the user gets a result without manual env edits. Do **not** change global CUDA visibility for the rest of the app (Demucs, SDXL, AnimateDiff, compositor must keep using the GPU as today).

**Background.** See `docs/technical/pinokio-lyrics-align-windows-handover.md` for the investigation notes (PyTorch cu124 vs CTranslate2 cuDNN naming, Pinokio vs repo install). Today `_pick_device` may default or pin devices via `GLITCHFRAME_WHISPERX_DEVICE`; this task adds **recovery** when GPU was chosen or available but **fails mid-alignment**.

**Requirements (implementation outline; you implement in a follow-up session).**

1. **Scope the catch** to alignment only: wrap the WhisperX work in `pipeline/lyrics_aligner.py` (the path that calls `_run_whisperx_forced` / `load_model` / `transcribe` / `align`), not a global `torch` or `CUDA_VISIBLE_DEVICES` change for the whole process.
2. **Retry once** on CPU with the correct **compute type** for CPU (e.g. match existing `_default_compute_type("cpu")` / `int8` behavior already in the module). Log at **WARNING** with a short reason so support can see "GPU align failed, retried CPU".
3. **Detect failures** in a way that is specific enough not to mask unrelated bugs: prefer checking exception message substrings for cudnn / ctranslate / `LoadLibrary` / `dll` / `1920` (case-insensitive), or a small allowlist of exception types, rather than a bare `except Exception` that swallows everything. If uncertain, log the original exception before retry.
4. **Do not** call `torch.set_default_tensor_type`, do not clear `CUDA_VISIBLE_DEVICES` for the whole process, and do not move background generation or other stages to CPU as part of this change.
5. **Tests:** Add or extend `tests/test_lyrics_aligner.py` (or a focused test) with **mocked** `_run_whisperx_forced` raising a cudnn-like error on first call and succeeding on second call with `device="cpu"`, so the retry path is covered without downloading models.
6. **Docs:** Touch `docs/technical/lyrics-aligner.md` (short "GPU failure / CPU retry" paragraph) and `docs/index.md` only if you add a new doc file (prefer updating existing lyrics-aligner doc). Optional one-line in `.env.example` / README if a new env var is introduced (prefer **no** new env var if behavior is fully automatic).

**Non-goals.** Changing Task Master #55 (scanline_tear) or other effects work unless this task is done and you pick up scanline separately.

## Key Files

- `pipeline/lyrics_aligner.py` — `_pick_device`, `_run_whisperx_forced`, `align_lyrics` (primary integration point for try GPU → except → retry CPU)
- `pipeline/gpu_memory.py` — ensure VRAM release / `move_to_cpu` patterns stay consistent across retry so GPU memory is not leaked between attempts
- `docs/technical/lyrics-aligner.md` — user-facing behavior
- `docs/technical/pinokio-lyrics-align-windows-handover.md` — context only; update only if the implementation changes the story materially

## Context

- WhisperX uses **faster-whisper / CTranslate2** on GPU for the transcribe model; wav2vec align model uses `device=` separately. Confirm both legs behave if only the first leg retries on CPU (may need to retry the full `_run_whisperx_forced` or split carefully so you do not half-run GPU then CPU inconsistently).
- Linux/macOS users with working GPU should see **no** extra CPU path unless the GPU path actually raises.

## Verification

- `.\.venv\Scripts\python.exe -m compileall <repo-root> -q`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`
- Manual smoke (optional): on Windows with a reproducing env, Align lyrics should complete after one WARNING log when GPU Whisper fails.

## Task Master

- Add or link a Task Master item for this work if the project tracks it; there may be no numeric ID yet.

## Checklist (this handover)

- [ ] GPU alignment failure triggers **one** CPU retry with clear logging
- [ ] Background / SDXL / Demucs / render paths unchanged (no global CUDA disable)
- [ ] Tests cover retry behavior with mocks
- [ ] `lyrics-aligner.md` updated
- [ ] Run compileall + unittest

## Other open work (not in scope for this handover)

- Task Master **#55** (scanline_tear renderer) remains in the project backlog; see Task Master for details.
