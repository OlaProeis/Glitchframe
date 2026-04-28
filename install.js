// Pinokio install: avoid script.start "torch.js" (can appear hung with no terminal output
// while resolving Python / downloading multi‑GB wheels). Match README: venv 3.11 + PyTorch cu124, then project deps.
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
          "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
          "python -m pip install -r requirements.txt",
          "python -m pip install -e .",
          // README "full analysis + lyrics" optional extra (Demucs, WhisperX, …)
          'python -m pip install -e ".[all]"',
          // Align with PyTorch cu124: ctranslate2>=4.5 uses cuDNN9-style DLLs; older 4.4.x
          // looks for cudnn_ops_infer64_8.dll and can fail after Silero VAD (Windows).
          'python -m pip install -U "ctranslate2>=4.5.0,<5"',
          // Optional: NVIDIA cuDNN wheel into site-packages (helps some Windows ctranslate2 DLL paths).
          "python -m pip install nvidia-cudnn-cu12",
        ],
      },
    },
    {
      method: "notify",
      params: {
        html:
          "Python deps are installed (including <b>Demucs + WhisperX</b>). Pinokio does <b>not</b> auto-launch the server &mdash; click <b>Start</b> in the sidebar. You need <code>ffmpeg</code> on your <code>PATH</code> for video encode; see README if Start fails.",
      },
    },
  ],
};
