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
          "python -m pip install -U pip",
          "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
          "python -m pip install -r requirements.txt",
          "python -m pip install -e .",
          // README "full analysis + lyrics" optional extra (Demucs, WhisperX, …)
          'python -m pip install -e ".[all]"',
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
