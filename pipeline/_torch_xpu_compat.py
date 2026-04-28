"""``torch.xpu`` stub for PyTorch < 2.3 (Track A: torch 2.2.2+cu121).

PyTorch added the ``torch.xpu`` submodule (Intel Discrete GPU support) in
**2.3.0**. Track A pins ``torch==2.2.2+cu121`` for the Pinokio Windows lyrics
stack (so cuDNN-era DLL names match what ctranslate2 4.4.0 expects), so
``torch.xpu`` simply isn't there.

Newer ``diffusers`` releases reference ``torch.xpu`` at import time without
guarding the access with ``hasattr(torch, "xpu")``. The result is::

    AttributeError: module 'torch' has no attribute 'xpu'

surfacing as::

    RuntimeError: AnimateDiff SDXL requires a recent diffusers install with
    AnimateDiffSDXLPipeline and DDIMScheduler ... Import failed: module
    'torch' has no attribute 'xpu'

The fix is a tiny stub: a ``ModuleType`` named ``torch.xpu`` exposing the
attributes diffusers probes (``is_available()`` returning ``False``,
``device_count()`` returning ``0``, plus a couple of bf16 / amp shims so any
``hasattr`` check + call returns the "no XPU" answer). On Track B (torch
≥ 2.4) ``torch.xpu`` already exists — we leave it untouched.

This is the exact upstream-recommended pattern (see HuggingFace transformers
issue #37838, PyTorch issue #120397: "use ``hasattr(torch, 'xpu')`` before
accessing"). We patch the consumer side because we can't fix every
``diffusers`` release the user might pull.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

LOGGER = logging.getLogger(__name__)

_PATCH_MARKER = "_glitchframe_torch_xpu_stub"


def _build_stub() -> types.ModuleType:
    """Construct a minimal ``torch.xpu`` stub mirroring the surface diffusers
    and transformers actually probe at import time."""
    stub = types.ModuleType("torch.xpu")

    def _no() -> bool:
        return False

    def _zero() -> int:
        return 0

    def _none() -> None:
        return None

    stub.is_available = _no  # type: ignore[attr-defined]
    stub.is_initialized = _no  # type: ignore[attr-defined]
    stub.device_count = _zero  # type: ignore[attr-defined]
    stub.current_device = _zero  # type: ignore[attr-defined]
    stub.is_bf16_supported = _no  # type: ignore[attr-defined]
    stub.synchronize = _none  # type: ignore[attr-defined]
    stub.empty_cache = _none  # type: ignore[attr-defined]

    # Some libs touch torch.xpu.amp.* even when XPU isn't present. Provide a
    # nested module so attribute access succeeds; calling autocast() etc.
    # would raise, but no code-path that matters runs into it when
    # is_available() returns False.
    amp = types.ModuleType("torch.xpu.amp")
    stub.amp = amp  # type: ignore[attr-defined]

    setattr(stub, _PATCH_MARKER, True)
    return stub


def patch_torch_xpu() -> bool:
    """Install a ``torch.xpu`` stub when PyTorch lacks the real submodule.

    Returns ``True`` if the stub was installed (or an existing real/stub
    ``torch.xpu`` is present and acceptable). Returns ``False`` if torch is
    not yet imported. Never raises — patching must not block startup.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False

    existing = getattr(torch, "xpu", None)
    if existing is not None:
        # Either a real torch.xpu (torch >= 2.3) or a previous stub. Don't
        # clobber a real one; mark our own as already-applied for idempotency.
        return True

    stub = _build_stub()
    try:
        torch.xpu = stub  # type: ignore[attr-defined]
        sys.modules["torch.xpu"] = stub
    except (AttributeError, TypeError) as exc:
        LOGGER.warning(
            "Could not install torch.xpu stub (%s: %s); newer diffusers "
            "imports may fail with 'module torch has no attribute xpu' on "
            "PyTorch < 2.3 (Track A).",
            type(exc).__name__,
            exc,
        )
        return False

    LOGGER.info(
        "Installed torch.xpu stub for PyTorch %s — diffusers / transformers "
        "imports that reference torch.xpu without hasattr() will now find "
        "is_available() == False and skip the XPU code path.",
        getattr(torch, "__version__", "unknown"),
    )
    return True


__all__ = ["patch_torch_xpu"]
