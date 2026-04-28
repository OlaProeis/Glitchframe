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


def _make_stub_callable(qualname: str, *, raise_on_call: bool = False) -> type:
    """Build a class-shaped stand-in for a missing torch attribute.

    **Why a class and not a function?** Newer ``transformers`` / ``diffusers``
    use PEP 604 unions in type hints, evaluated at function-definition time::

        from torch.distributed.device_mesh import DeviceMesh
        def foo(mesh: DeviceMesh | None = None) -> None: ...
        # Python 3.10+ evaluates `DeviceMesh | None` immediately. If
        # DeviceMesh is a plain function, this raises:
        #   TypeError: unsupported operand type(s) for |:
        #              'function' and 'NoneType'

    Plain functions don't define ``__or__``; classes do (via the ``type``
    metaclass), so ``Class | None`` evaluates to ``types.UnionType``.
    Returning a class means:

      * ``StubAttr | None`` works (no TypeError at import time).
      * ``StubAttr(...)`` is callable (instantiates the class).
      * ``isinstance(x, StubAttr)`` returns ``False`` for normal objects,
        which matches the "no real support for this API" semantic.
      * Real torch APIs that take a ``Callable``-typed argument still
        accept it because classes are callable.

    ``raise_on_call=True`` makes the constructor raise so anyone *using*
    the stub (rather than just probing for its existence in a hasattr /
    annotation context) gets a clear runtime error pointing at the cause.
    """
    short_name = qualname.rsplit(".", 1)[-1]
    module_name = qualname.rsplit(".", 1)[0] if "." in qualname else "__main__"

    if raise_on_call:
        def _init(self: Any, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                f"{qualname} is a Glitchframe compatibility stub for an API "
                f"missing from this PyTorch build (Track A: torch < 2.4). "
                f"Module imports succeed via this placeholder, but actual "
                f"use requires upgrading torch to >= 2.4 (see "
                f"docs/technical/pinokio-lyrics-align-windows-handover.md)."
            )
    else:
        def _init(self: Any, *_args: Any, **_kwargs: Any) -> None:  # type: ignore[misc]
            return None

    return type(
        short_name,
        (),
        {
            "__init__": _init,
            "__qualname__": short_name,
            "__module__": module_name,
            "__repr__": lambda self: f"<glitchframe stub {qualname}>",
        },
    )


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Default fallback function for predicates whose return value matters.

    Used only for explicitly-bound predicates / counters in the xpu stub
    where a specific return value is required (e.g. ``is_available()`` must
    return ``False`` so consumer code branches *away* from the XPU path).
    For attribute *probes* that may end up in type annotations, use
    ``_make_stub_callable`` (returns a class, supports ``| None``).
    """
    return None


class _PermissiveStub(types.ModuleType):
    """ModuleType subclass with permissive ``__getattr__``.

    Used as the body of every compat stub we install on torch — both
    leaves (``torch.xpu``) and nested namespaces (``torch.xpu.random``,
    ``torch.distributed.device_mesh``). Any attribute not explicitly set
    is synthesised on first access as a class (see ``_make_stub_callable``)
    so PEP 604 type annotations like ``StubAttr | None`` evaluate cleanly.
    The synthesised class is cached on the module instance, so subsequent
    lookups go through normal attribute resolution.

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
        # Synthesise a class (not a function) so PEP 604 unions work.
        # Cache on the instance so subsequent lookups go through normal
        # attribute resolution and stay identity-stable.
        cls = _make_stub_callable(f"{self.__name__}.{name}")
        setattr(self, name, cls)
        return cls


class _XpuStub(_PermissiveStub):
    """``torch.xpu`` stub — ``amp`` and ``random`` are nested submodules."""

    _subnames = ("amp", "random")


def _build_xpu_stub() -> types.ModuleType:
    """Construct a ``torch.xpu`` stub.

    Explicitly defines every attribute whose value matters to a caller.
    Unknown attributes are synthesised on demand by ``__getattr__`` (as
    classes, see ``_make_stub_callable``).
    """
    stub = _XpuStub("torch.xpu")

    # Predicates — value matters (callers branch on the bool / int).
    # These MUST be functions returning the specific value, not classes.
    stub.is_available = lambda: False  # type: ignore[attr-defined]
    stub.is_initialized = lambda: False  # type: ignore[attr-defined]
    stub.is_bf16_supported = lambda: False  # type: ignore[attr-defined]
    stub.device_count = lambda: 0  # type: ignore[attr-defined]
    stub.current_device = lambda: 0  # type: ignore[attr-defined]
    stub.memory_allocated = lambda *a, **k: 0  # type: ignore[attr-defined]
    stub.memory_reserved = lambda *a, **k: 0  # type: ignore[attr-defined]
    stub.max_memory_allocated = lambda *a, **k: 0  # type: ignore[attr-defined]
    stub.max_memory_reserved = lambda *a, **k: 0  # type: ignore[attr-defined]

    # Void operations — caller doesn't read the return value. Use class-
    # shaped stubs so ``manual_seed | None`` style annotations don't
    # explode if some library decides to type-hint them.
    stub.manual_seed = _make_stub_callable("torch.xpu.manual_seed")  # type: ignore[attr-defined]
    stub.manual_seed_all = _make_stub_callable("torch.xpu.manual_seed_all")  # type: ignore[attr-defined]
    stub.seed = _make_stub_callable("torch.xpu.seed")  # type: ignore[attr-defined]
    stub.seed_all = _make_stub_callable("torch.xpu.seed_all")  # type: ignore[attr-defined]
    stub.synchronize = _make_stub_callable("torch.xpu.synchronize")  # type: ignore[attr-defined]
    stub.empty_cache = _make_stub_callable("torch.xpu.empty_cache")  # type: ignore[attr-defined]
    stub.set_device = _make_stub_callable("torch.xpu.set_device")  # type: ignore[attr-defined]
    stub.init = _make_stub_callable("torch.xpu.init")  # type: ignore[attr-defined]
    stub.reset_peak_memory_stats = _make_stub_callable(  # type: ignore[attr-defined]
        "torch.xpu.reset_peak_memory_stats"
    )

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
    ``init_device_mesh``. Both are surfaced as **classes** here (not
    functions) so PEP 604 type annotations evaluate cleanly::

        # transformers does this in fsdp_utils.py and similar:
        from torch.distributed.device_mesh import DeviceMesh
        def _foo(mesh: DeviceMesh | None = None) -> None: ...

    With a function-typed ``DeviceMesh``, the ``DeviceMesh | None``
    annotation raises ``TypeError: unsupported operand type(s) for |:
    'function' and 'NoneType'`` at *function-definition* time — i.e. on
    import. Class-typed ``DeviceMesh`` (an instance of ``type``) supports
    ``__or__``, so the annotation evaluates to ``types.UnionType`` and
    the import completes. See ``_make_stub_callable`` for the full
    rationale.

    Probes (``hasattr``, ``isinstance(x, DeviceMesh)``, annotation
    evaluation) all succeed. *Calling* either symbol raises ``RuntimeError``
    so anyone who actually tries to build a device mesh on Track A gets
    a clear, actionable error rather than a silent no-op.
    """
    stub = _PermissiveStub("torch.distributed.device_mesh")
    stub.DeviceMesh = _make_stub_callable(  # type: ignore[attr-defined]
        "torch.distributed.device_mesh.DeviceMesh", raise_on_call=True
    )
    stub.init_device_mesh = _make_stub_callable(  # type: ignore[attr-defined]
        "torch.distributed.device_mesh.init_device_mesh", raise_on_call=True
    )
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
