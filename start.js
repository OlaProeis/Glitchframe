module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Avoid Pyannote VAD + cuDNN DLL failures on Windows in Pinokio; no effect on local
          // `python -m app` unless this env is set (see pipeline/lyrics_aligner.py).
          GLITCHFRAME_WHISPERX_VAD_METHOD: "silero",
          // Pinokio/Windows: faster-whisper (CTranslate2) often fails to load cudnn_ops_infer64_8.dll
          // / cuDNN next to PyTorch cu124 (name mismatch, loader order). Demucs/SDXL still use GPU.
          // Align lyrics uses CPU for WhisperX only — slower but reliable. To try GPU align, delete
          // this line or set GLITCHFRAME_WHISPERX_DEVICE=cuda and fix your cuDNN stack (see README).
          GLITCHFRAME_WHISPERX_DEVICE: "cpu",
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
