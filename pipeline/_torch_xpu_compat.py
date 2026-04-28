"""``torch.xpu`` stub for PyTorch < 2.3 (Track A: torch 2.2.2+cu121).

PyTorch added the ``torch.xpu`` submodule (Intel Discrete GPU support) in
**2.3.0**. Track A pins ``torch==2.2.2+cu121`` for the Pinokio Windows lyrics
stack (so cuDNN-era DLL names match what ctranslate2 4.4.0 expects), so
``torch.xpu`` simply isn't there.

Newer ``diffusers`` / ``transformers`` / ``peft`` / ``accelerate`` releases
reference ``torch.xpu.*`` at import time without guarding the access with
``hasattr(torch, "xpu")``. Concrete failures observed in the wild::

    AttributeError: module 'torch' has no attribute 'xpu'
    AttributeError: module 'torch.xpu' has no attribute 'manual_seed'

The latter comes from ``diffusers/utils/torch_utils.py`` building a
``{device -> seed_fn}`` dispatch table at module load time::

    "xpu": torch.xpu.manual_seed,

There's no point chasing every individual attribute these libraries probe —
they'll add more on every release. Instead, the stub is a ``ModuleType``
subclass with a permissive ``__getattr__``: anything not explicitly
defined returns a no-op callable that returns ``None``. Attributes whose
return value matters semantically (``is_available()`` must be ``False``,
``device_count()`` must be ``0``) are defined explicitly. Sub-namespaces
``amp`` and ``random`` are themselves stubs of the same class, so
``torch.xpu.random.manual_seed_all`` and the like also work transparently.

On Track B (torch ≥ 2.4) ``torch.xpu`` already exists — we leave it
untouched. Real Intel GPU users get the real submodule.

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
_SUBMODULE_NAMES = ("amp", "random")


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Default torch.xpu.* fallback — accepts anything, returns None.

    Matches the shape of every "void" torch.xpu function (``manual_seed``,
    ``synchronize``, ``empty_cache``, ``set_device``, ...) so consumers
    that only need the call to *succeed* don't crash. Functions that
    semantically must return a typed value (``is_available`` -> bool,
    ``device_count`` -> int) are defined explicitly on the stub instead.
    """
    return None


class _XpuStub(types.ModuleType):
    """ModuleType subclass with permissive ``__getattr__``.

    Used for both ``torch.xpu`` and its nested namespaces (``amp``,
    ``random``). Any attribute not explicitly set on the instance is
    synthesised as a no-op callable on first access (and cached on the
    module so subsequent lookups go through normal attribute resolution).

    Sub-namespace probes (e.g. ``torch.xpu.random.manual_seed_all``)
    return another ``_XpuStub``, registered in ``sys.modules`` so
    ``import torch.xpu.random`` also works.
    """

    def __getattr__(self, name: str) -> Any:
        # Let dunders fall through naturally — pretending to support
        # __init_subclass__ / __reduce__ / etc. would be hostile to debugging.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _SUBMODULE_NAMES:
            full_name = f"{self.__name__}.{name}"
            sub = _XpuStub(full_name)
            sys.modules[full_name] = sub
            setattr(self, name, sub)
            return sub
        # Synthesised no-op callable; cache on the instance so it round-trips
        # via normal attribute lookup next time (and stays identity-stable).
        fn = _noop
        # Give it a more useful name for tracebacks if it ever gets called.
        try:
            fn = type(_noop)(  # type: ignore[misc]
                _noop.__code__,
                _noop.__globals__,
                f"{self.__name__}.{name}",
                _noop.__defaults__,
                _noop.__closure__,
            )
        except Exception:  # noqa: BLE001
            fn = _noop
        setattr(self, name, fn)
        return fn


def _build_stub() -> types.ModuleType:
    """Construct a ``torch.xpu`` stub.

    Explicitly defines every attribute whose value matters to a caller.
    Unknown attributes are synthesised on demand by ``_XpuStub.__getattr__``.
    """
    stub = _XpuStub("torch.xpu")

    # Predicates — value matters (callers branch on the bool).
    stub.is_available = lambda: False  # type: ignore[attr-defined]
    stub.is_initialized = lambda: False  # type: ignore[attr-defined]
    stub.is_bf16_supported = lambda: False  # type: ignore[attr-defined]

    # Counters — value matters (range(device_count()) etc.).
    stub.device_count = lambda: 0  # type: ignore[attr-defined]
    stub.current_device = lambda: 0  # type: ignore[attr-defined]
    stub.memory_allocated = lambda *a, **k: 0  # type: ignore[attr-defined]
    stub.memory_reserved = lambda *a, **k: 0  # type: ignore[attr-defined]
    stub.max_memory_allocated = lambda *a, **k: 0  # type: ignore[attr-defined]
    stub.max_memory_reserved = lambda *a, **k: 0  # type: ignore[attr-defined]

    # Common void operations — caller doesn't read the return value, but
    # binding them eagerly avoids a __getattr__ round-trip on every call.
    stub.manual_seed = _noop  # type: ignore[attr-defined]
    stub.manual_seed_all = _noop  # type: ignore[attr-defined]
    stub.seed = _noop  # type: ignore[attr-defined]
    stub.seed_all = _noop  # type: ignore[attr-defined]
    stub.synchronize = _noop  # type: ignore[attr-defined]
    stub.empty_cache = _noop  # type: ignore[attr-defined]
    stub.set_device = _noop  # type: ignore[attr-defined]
    stub.init = _noop  # type: ignore[attr-defined]
    stub.reset_peak_memory_stats = _noop  # type: ignore[attr-defined]

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
