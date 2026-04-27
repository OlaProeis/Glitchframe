module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Avoid Pyannote VAD + cuDNN8 DLL failures on Windows in Pinokio; no effect on local
          // `python -m app` unless this env is set (see pipeline/lyrics_aligner.py).
          GLITCHFRAME_WHISPERX_VAD_METHOD: "silero",
        },
        message: ["python -m app"],
        on: [
          {
            event: "/http:\\/\\/\\S+/",
            done: true,
          },
        ],
      },
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}",
      },
    },
  ],
};
