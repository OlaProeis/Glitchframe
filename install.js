// Pinokio install: avoid script.start "torch.js" (can appear hung with no terminal output
// while resolving Python / downloading multi‑GB wheels).
// Windows + Python 3.11 (Pinokio venv): use the repo's pinned CUDA 12.1 lyrics stack —
// torch 2.2.2+cu121, WhisperX 3.3.0, ctranslate2 4.4.0 — matching DLL expectations for GPU Align lyrics.
//
// Pinning rationale (do not "modernise" without retesting Pinokio end-to-end on Windows):
//   * torch 2.2.2 wheels were built against NumPy 1.x → must keep numpy<2 or torch
//     prints "_ARRAY_API not found" and silently breaks tensor<->numpy interop.
//   * ctranslate2 4.4.0 expects cuDNN 8 (cudnn_ops_infer64_8.dll). torch 2.2.2+cu121
//     ALREADY ships matching cuDNN 8.9.x DLLs inside torch\lib, and
//     scripts/windows_provision_cudnn_next_to_ctranslate2.py copies them next to
//     the ctranslate2 package for LoadLibrary. We deliberately do NOT install
//     `nvidia-cudnn-cu12` — the standalone wheel introduced WinError 127 on import
//     (cuDNN 8.9.7 minor differs from torch's 8.9.x bundle and ctranslate2 4.4.0's
//     resolved exports — see docs/technical/pinokio-lyrics-align-windows-handover.md).
//     We also `pip uninstall -y nvidia-cudnn-cu12` to clean up envs that picked up
//     the wheel from a prior install run.
//   * MarkupSafe 3 + Pillow 12 break Gradio 4.x — re-pin after any torch reinstall.
//   * Speechbrain LazyModule (transitive via whisperx→pyannote-audio) crashes on
//     attribute probes when k2 is missing; app.py pre-stubs `k2` in sys.modules.
module.exports = {
  // Tell Pinokio this app needs the bundled AI stack (CUDA toolkit + cuDNN in
  // ~/pinokio/bin). Without this, Pinokio only installs the bundle if some
  // OTHER app on the system already triggered it, which made fresh-install
  // reproductions unpredictable. Our cuDNN 8.9.x DLLs still come from
  // torch\lib for ctranslate2 4.4.0 (the bundle ships cuDNN 9 with different
  // filenames, so it can't accidentally satisfy our cuDNN 8 lookup).
  requires: {
    bundle: "ai",
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        message: [
          // Some Pinokio-created venvs have no pip; bootstrap before any pip use.
          "python -m ensurepip --upgrade",
          "python -m pip install -U pip",
          // Windows lyrics GPU stack (must stay ahead of generic torch>= from requirements / -e .)
          "python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121",
          "python -m pip install -r requirements.txt",
          "python -m pip install -e .",
          'python -m pip install -e ".[all]"',
          // Re-pin cu121 trio without re-resolving deps (--no-deps). A full --force-reinstall
          // upgrades Pillow/MarkupSafe to versions Gradio 4.x rejects (UI + ingest break).
          "python -m pip install --force-reinstall --no-deps torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121",
          // Restore Gradio-compatible pins AND the NumPy 1.x ABI torch 2.2.2 needs.
          // Without numpy<2, torch logs "_ARRAY_API not found" and any tensor<->numpy
          // bridge (audio ingest, demucs, whisperx) silently misbehaves.
          "python -m pip install --force-reinstall --no-deps \"numpy>=1.26.0,<2.0\"",
          "python -m pip install \"markupsafe>=2.0,<3\" \"pillow>=10,<11\"",
          "python -m pip install \"whisperx==3.3.0\" \"faster-whisper==1.1.0\" \"ctranslate2==4.4.0\"",
          // Remove any standalone cuDNN wheel left over from a prior install (cuDNN 9
          // from unversioned `nvidia-cudnn-cu12`, or cuDNN 8.9.7 from a previous pin).
          // Both classes broke imports / GPU usage on Pinokio venvs; torch's bundled
          // cuDNN 8.9.x in torch\lib is what ctranslate2 4.4.0 resolves against.
          // `pip uninstall -y` is a no-op (exit 0) when the wheel isn't installed.
          "python -m pip uninstall -y nvidia-cudnn-cu12",
          "python scripts/windows_provision_cudnn_next_to_ctranslate2.py",
        ],
      },
    },
    {
      method: "notify",
      params: {
        html:
          "Python deps are installed (including <b>Demucs + WhisperX</b> on the <b>cu121</b> stack). Pinokio does <b>not</b> auto-launch the server &mdash; click <b>Start</b> in the sidebar. You need <code>ffmpeg</code> on your <code>PATH</code> for video encode; see README. Align lyrics defaults to <b>CPU</b> (slow but reliable); to try GPU, clear <code>GLITCHFRAME_WHISPERX_DEVICE=cpu</code> from <code>start.js</code>.",
      },
    },
  ],
};
