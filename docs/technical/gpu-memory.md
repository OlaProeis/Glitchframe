# GPU memory lifecycle

Feature: shared helpers that release CUDA VRAM between pipeline stages so
heavyweight models (demucs, WhisperX faster-whisper, WhisperX wav2vec2
align, SDXL, AnimateDiff) never stack on top of each other inside a
single Python process.

## Why

PyTorch's caching allocator keeps freed blocks around for reuse inside
the same stage — great for throughput, bad for stages that run
sequentially and never share weights. Without an explicit cleanup,
demucs's 1 GB and WhisperX's ~3.5 GB stay resident until process exit,
so the SDXL stage tries to load 4–5 GB of FP16 weights on top of them,
forcing the driver to offload to system RAM. On the user's 24 GB card
this pushes peak VRAM from the expected 20 GB to full saturation and
makes the desktop lag while the video renders.

## Code

| Piece | Location |
|-------|----------|
| `release_cuda_memory(label=None)` | `pipeline/gpu_memory.py` |
| `move_to_cpu(obj)` | `pipeline/gpu_memory.py` |

### `release_cuda_memory`

Runs, in order, `gc.collect()` → `torch.cuda.empty_cache()` →
`torch.cuda.ipc_collect()`. Each step is wrapped in its own
`try/except` so a driver hiccup can't stop later steps from running. No
CUDA / torch installed → silent no-op. `label` is emitted in the INFO
log so the cleanup trail is readable in the Gradio run terminal.

### `move_to_cpu`

Best-effort `.cpu()` on a `torch.nn.Module` (preferred, explicit CPU
sync); falls back to `.to("cpu")`; silently no-ops on `None` or on
objects that have neither method. Use this before `del model` on
torchaudio / huggingface modules — the PyTorch allocator only frees a
module's CUDA tensors after the module is explicitly moved off the
device.

## Call sites

| Stage | Released |
|-------|----------|
| `pipeline/audio_analyzer.py::_separate_vocals_with_demucs` | `htdemucs_ft` (≈1 GB) after success *and* error |
| `pipeline/lyrics_aligner.py::_run_whisperx_forced` (transcribe) | faster-whisper large-v3 FP16 (≈3 GB) + pyannote VAD, before align model loads |
| `pipeline/lyrics_aligner.py::_run_whisperx_forced` (align) | wav2vec2 align model (≈500 MB) before returning |

The SDXL (`pipeline/background_stills.py::BackgroundStills.close`) and
AnimateDiff (`pipeline/background_animatediff.py`) stages already call
`torch.cuda.empty_cache()` directly; they can migrate to this helper
too when they're next touched, but the immediate VRAM-saturation bug was
caused by the pre-SDXL stages skipping cleanup entirely.

## Testing

`tests/test_gpu_memory.py` stubs `sys.modules["torch"]` with a fake
recording object, so every code path (CUDA available, CUDA unavailable,
torch missing, `empty_cache` failure, `is_available` failure,
`move_to_cpu` fallback, `move_to_cpu` on `None`) is exercised without
needing a real GPU.

## Convention

When you add a new stage that loads a GPU model:

1. Hold the model in a local.
2. `try: …use… finally: move_to_cpu(model); del model; release_cuda_memory("<stage name>")`.
3. If the stage loads multiple models sequentially (e.g. WhisperX's
   transcribe then align), release each one as soon as its work is done
   rather than at the end — the whole point is that they don't stack.
