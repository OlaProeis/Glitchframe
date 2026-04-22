"""
GPU memory lifecycle helper shared by every pipeline stage that loads a
model onto CUDA.

The pipeline runs multiple heavy models back-to-back within a single
Python process — demucs (``htdemucs_ft``), WhisperX (``faster-whisper``
large-v3 + wav2vec2 align model), SDXL FP16, optionally AnimateDiff — and
PyTorch's default behaviour is to hold freed allocations in its caching
allocator indefinitely. That's great for a single stage that reuses the
memory, but between stages it means the previously loaded model's
weights stay pinned in VRAM until the process exits, driving the later
stages into RAM offload and making the desktop lag.

Call :func:`release_cuda_memory` immediately after you've dropped your
last reference to a GPU-resident model (``del model`` or the caller's
``try/finally``) so the next stage starts with a clean slate. Safe to
call without CUDA installed; no-ops with a debug log if something goes
wrong so callers never need a ``try/except`` around it.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def release_cuda_memory(label: str | None = None) -> None:
    """Best-effort reclaim of CUDA VRAM + Python heap.

    Runs ``gc.collect()`` (to drop any Python wrappers still holding
    module / tensor references that prevent PyTorch from deallocating),
    then ``torch.cuda.empty_cache()`` (returns cached blocks to the
    driver), then ``torch.cuda.ipc_collect()`` (best-effort IPC handle
    sweep — important when a worker process allocated tensors shared
    with the main process).

    ``label`` is used only in debug logs so the trail shows *which*
    stage freed VRAM (e.g. ``"whisperx transcribe model"``).

    Never raises: this is called on cleanup paths where raising would
    mask the primary error.
    """
    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("gc.collect failed during release_cuda_memory(%s): %s", label, exc)

    torch: Any
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return

    try:
        if not torch.cuda.is_available():
            return
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("torch.cuda.is_available check failed (%s): %s", label, exc)
        return

    try:
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("torch.cuda.empty_cache failed (%s): %s", label, exc)
    try:
        torch.cuda.ipc_collect()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("torch.cuda.ipc_collect failed (%s): %s", label, exc)

    if label:
        LOGGER.info("Released CUDA memory after %s", label)


def move_to_cpu(obj: Any) -> None:
    """Best-effort ``.cpu()`` / ``.to('cpu')`` on a module/tensor.

    PyTorch only frees a module's CUDA weight tensors when either
    (a) the module is garbage-collected while having no ``.cuda()``
    buffers registered, or (b) the module is explicitly moved off the
    device first. For ``torch.nn.Module`` instances (wav2vec2, demucs)
    the explicit move is the reliable path — ``del`` alone frequently
    leaves the weights in the allocator cache until the next alloc.
    Safe on non-module objects and on failures (no-ops).
    """
    if obj is None:
        return
    try:
        cpu_fn = getattr(obj, "cpu", None)
        if callable(cpu_fn):
            cpu_fn()
            return
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("obj.cpu() failed: %s", exc)
    try:
        to_fn = getattr(obj, "to", None)
        if callable(to_fn):
            to_fn("cpu")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("obj.to('cpu') failed: %s", exc)


__all__ = ["release_cuda_memory", "move_to_cpu"]
