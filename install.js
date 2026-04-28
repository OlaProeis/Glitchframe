// Pinokio install: avoid script.start "torch.js" (can appear hung with no terminal output
// while resolving Python / downloading multi‑GB wheels).
// Windows + Python 3.11 (Pinokio venv): use the repo's pinned CUDA 12.1 lyrics stack —
// torch 2.2.2+cu121, WhisperX 3.3.0, ctranslate2 4.4.0 — matching DLL expectations for GPU Align lyrics.
module.exports = {
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
          // Restore Gradio-compatible pins after any torch-related drift (see README / requirements.txt).
          "python -m pip install \"markupsafe>=2.0,<3\" \"pillow>=10,<11\"",
          "python -m pip install \"whisperx==3.3.0\" \"faster-whisper==1.1.0\" \"ctranslate2==4.4.0\"",
          // Optional: extra cuDNN DLLs in site-packages; script copies next to ctranslate2 for LoadLibrary.
          "python -m pip install nvidia-cudnn-cu12",
          "python scripts/windows_provision_cudnn_next_to_ctranslate2.py",
        ],
      },
    },
    {
      method: "notify",
      params: {
        html:
          "Python deps are installed (including <b>Demucs + WhisperX</b> on the <b>cu121</b> stack). Pinokio does <b>not</b> auto-launch the server &mdash; click <b>Start</b> in the sidebar. You need <code>ffmpeg</code> on your <code>PATH</code> for video encode; see README. Align lyrics may use GPU if you clear <code>GLITCHFRAME_WHISPERX_DEVICE=cpu</code> from <code>start.js</code> or set <code>cuda</code> in <code>.env</code>.",
      },
    },
  ],
};
