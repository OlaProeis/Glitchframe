module.exports = {
  run: [
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // Glitchframe targets Python 3.11+ (pyproject). Pinokio 3.3+ respects venv_python on venv creation.
          venv_python: "3.11",
        },
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        message: [
          "python -m pip install -U pip",
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
