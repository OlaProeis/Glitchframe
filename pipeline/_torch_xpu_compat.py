"""Compat shims for missing ``torch.*`` attributes on PyTorch < 2.4.

Track A pins ``torch==2.2.2+cu121`` for the Pinokio Windows lyrics stack
(so cuDNN-era DLL names match what ctranslate2 4.4.0 expects). Newer
``diffusers`` / ``transformers`` / ``peft`` / ``accelerate`` releases
reference torch APIs that landed *after* 2.2.2 — at module import time,
without guarding the access with ``hasattr()``. Concrete failures observed
in the wild on Track A::

    AttributeError: module 'torch' has no attribute 'xpu'
    AttributeError: module 'torch.xpu' has no attribute 'manual_seed'
    AttributeError: module 'torch.distributed' has no attribute 'device_mesh'

The first two come from PyTorch 2.3 (Intel discrete GPU support).
``torch.distributed.device_mesh`` is from PyTorch 2.4 (FSDP2 / TP). Each
new diffusers/transformers release tends to add another such probe — we
hit ``torch.xpu`` first, fixed it, then immediately hit
``torch.distributed.device_mesh`` from the same import chain.

There's no winning whack-a-mole on this attribute set. The fix is a
permissive ``ModuleType`` stub: explicitly define attributes whose value
matters semantically (``is_available()`` -> ``False``, ``device_count()``
-> ``0``), and let ``__getattr__`` synthesise a no-op callable for
anything else. Sub-namespace probes (``torch.xpu.random.manual_seed_all``)
return another stub instance, registered in ``sys.modules`` so plain
``import torch.xpu.random`` also works.

On Track B (torch ≥ 2.4) the real submodules already exist — we leave them
untouched. Real Intel GPU and DTensor users get the real APIs.

This is the exact upstream-recommended pattern (see HuggingFace transformers
issue #37838, PyTorch issue #120397: "use ``hasattr(torch, 'xpu')`` before
accessing"). We patch the consumer side because we can't fix every
``diffusers`` / ``transformers`` release the user might pull.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

_PATCH_MARKER = "_glitchframe_torch_xpu_stub"
_DIST_MARKER = "_glitchframe_torch_dist_device_mesh_stub"


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Default fallback: accepts anything, returns None.

    Matches the shape of every "void" torch API (``manual_seed``,
    ``synchronize``, ``empty_cache``, ``set_device``, ...) so consumers
    that only need the call to *succeed* don't crash. Functions that
    semantically must return a typed value (``is_available`` -> bool,
    ``device_count`` -> int) are defined explicitly on the stub instead.
    """
    return None


class _PermissiveStub(types.ModuleType):
    """ModuleType subclass with permissive ``__getattr__``.

    Used as the body of every compat stub we install on torch — both
    leaves (``torch.xpu``) and nested namespaces (``torch.xpu.random``,
    ``torch.distributed.device_mesh``). Any attribute not explicitly set
    is synthesised as a no-op callable on first access (and cached on the
    module so subsequent lookups go through normal attribute resolution).

    Sub-namespace probes return another ``_PermissiveStub``, registered in
    ``sys.modules`` so ``import torch.xpu.random`` also works. The list of
    nested namespace names is configured per stub via ``_subnames``.
    """

    _subnames: tuple[str, ...] = ()

    def __getattr__(self, name: str) -> Any:
        # Let dunders fall through naturally. Pretending to support
        # __reduce__ / __init_subclass__ / etc. would confuse copy.deepcopy,
        # pickling, debuggers, and isinstance checks.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in self._subnames:
            full_name = f"{self.__name__}.{name}"
            sub = _PermissiveStub(full_name)
            sys.modules[full_name] = sub
            setattr(self, name, sub)
            return sub
        # Synthesised no-op callable; cache on the instance so it round-trips
        # via normal attribute lookup next time (and stays identity-stable).
        try:
            fn: Any = type(_noop)(  # type: ignore[misc]
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


class _XpuStub(_PermissiveStub):
    """``torch.xpu`` stub — ``amp`` and ``random`` are nested submodules."""

    _subnames = ("amp", "random")


def _build_xpu_stub() -> types.ModuleType:
    """Construct a ``torch.xpu`` stub.

    Explicitly defines every attribute whose value matters to a caller.
    Unknown attributes are synthesised on demand by ``__getattr__``.
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


# Backwards-compat alias for tests/old code that imported the previous name.
_build_stub = _build_xpu_stub


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

    stub = _build_xpu_stub()
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


def _build_device_mesh_stub() -> types.ModuleType:
    """Construct a ``torch.distributed.device_mesh`` stub.

    The real submodule (added in PyTorch 2.4) exposes ``DeviceMesh`` and
    ``init_device_mesh``. Code that probes it at import time tends to do
    ``hasattr(torch.distributed, "device_mesh")`` or just
    ``torch.distributed.device_mesh.DeviceMesh`` — both are satisfied by a
    permissive stub. Any callable use raises a clear error so anyone who
    *actually* tries to build a device mesh on Track A learns immediately.
    """
    stub = _PermissiveStub("torch.distributed.device_mesh")

    def _no_device_mesh(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "torch.distributed.device_mesh is not available in this PyTorch "
            "build (Track A: torch 2.2.x). Glitchframe runs single-GPU "
            "AnimateDiff renders, which don't need device meshes — but if "
            "you really need this API, upgrade to torch >= 2.4."
        )

    # Real symbols. Probes return a callable (so ``isinstance`` checks pass);
    # actual *use* raises with a clear explanation.
    stub.DeviceMesh = _no_device_mesh  # type: ignore[attr-defined]
    stub.init_device_mesh = _no_device_mesh  # type: ignore[attr-defined]

    setattr(stub, _DIST_MARKER, True)
    return stub


def patch_torch_distributed_device_mesh() -> bool:
    """Install a ``torch.distributed.device_mesh`` stub when missing.

    Newer ``transformers`` (4.45+ ish) imports ``torch.distributed.device_mesh``
    at module load time — fine on PyTorch ≥ 2.4 where it exists, fatal on
    Track A's torch 2.2.2 (``AttributeError: module 'torch.distributed' has
    no attribute 'device_mesh'``). The error surfaces from inside
    ``diffusers.loaders.single_file`` because it imports transformers
    transitively while loading AnimateDiff SDXL.

    Returns ``True`` if the stub was installed (or already present),
    ``False`` if torch / torch.distributed isn't loaded yet. Never raises.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False

    # torch.distributed itself exists on every recent torch; it's just the
    # device_mesh sub-attribute that's missing on 2.2.x.
    dist = getattr(torch, "distributed", None)
    if dist is None:
        return False

    existing = getattr(dist, "device_mesh", None)
    if existing is not None:
        return True

    stub = _build_device_mesh_stub()
    try:
        dist.device_mesh = stub
        sys.modules["torch.distributed.device_mesh"] = stub
    except (AttributeError, TypeError) as exc:
        LOGGER.warning(
            "Could not install torch.distributed.device_mesh stub (%s: %s); "
            "newer transformers / diffusers imports may fail on PyTorch "
            "< 2.4 (Track A).",
            type(exc).__name__,
            exc,
        )
        return False

    LOGGER.info(
        "Installed torch.distributed.device_mesh stub for PyTorch %s — "
        "transformers / diffusers / accelerate imports that reference it "
        "without hasattr() will now find an inert DeviceMesh placeholder.",
        getattr(torch, "__version__", "unknown"),
    )
    return True


def patch_all() -> dict[str, bool]:
    """Apply every torch-attribute compat shim. Idempotent.

    Returns a dict ``{patch_name: applied}`` for diagnostics. Never raises.
    """
    return {
        "torch.xpu": patch_torch_xpu(),
        "torch.distributed.device_mesh": patch_torch_distributed_device_mesh(),
    }


__all__: Iterable[str] = (
    "patch_torch_xpu",
    "patch_torch_distributed_device_mesh",
    "patch_all",
)
