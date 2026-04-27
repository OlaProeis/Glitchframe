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
        ],
      },
    },
    {
      method: "notify",
      params: {
        html:
          "Core install complete. For <b>Demucs + WhisperX</b> (vocal stem + lyrics), run in this app&rsquo;s terminal: <code>python -m pip install -e &quot;.[all]&quot;</code> &mdash; see README. Open the <b>Start</b> tab; you need <code>ffmpeg</code> on your PATH.",
      },
    },
  ],
};
