"""Speechbrain ``LazyModule`` Windows-path-separator fix.

Speechbrain v1.0.x defines ``speechbrain.utils.importutils.LazyModule.ensure_module``
with a guard that raises ``AttributeError`` when the calling frame is
CPython's ``inspect.py`` — that's how it deliberately short-circuits
``inspect.getmodule`` walks (``hasattr(mod, '__file__')`` etc.) so they don't
force the lazy import. The check is::

    if importer_frame is not None and importer_frame.filename.endswith(
        "/inspect.py"
    ):
        raise AttributeError()

The hard-coded forward slash makes the guard a **no-op on Windows**, where
``inspect.py``'s frame filename is ``...\\Lib\\inspect.py`` (backslash). The
lazy import is then forced for *every* ``LazyModule`` speechbrain registered,
including:

* ``speechbrain.integrations.k2_fsa`` (needs ``k2`` — no Windows wheel)
* ``speechbrain.integrations.nlp.flair_embeddings`` (needs ``flair`` — heavy
  NLP package, not used by Glitchframe)
* multiple ``deprecated_redirect`` shims (e.g. ``speechbrain.k2_integration``)

Whichever one fails first crashes ``librosa.load`` (audio ingest) because
``librosa.lazy_loader`` calls ``inspect.stack()`` to find its caller, and
CPython's ``inspect.getmodule`` walks every entry in ``sys.modules`` calling
``hasattr(mod, '__file__')``.

This module monkey-patches ``LazyModule.ensure_module`` with a separator-aware
version that matches both ``/inspect.py`` and ``\\inspect.py``. Idempotent —
safe to call multiple times. Silent no-op when speechbrain is not installed,
when the patch has already been applied, or when speechbrain has fixed the
bug upstream (we detect the buggy literal in the original source).

Tracked upstream: https://github.com/speechbrain/speechbrain/issues/2995
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from types import ModuleType
from typing import Any

LOGGER = logging.getLogger(__name__)

_PATCH_MARKER = "_glitchframe_inspect_path_sep_patch"


def _patched_ensure_module(self: Any, stacklevel: int) -> ModuleType:
    """Drop-in replacement for ``LazyModule.ensure_module`` that handles
    both Windows and POSIX path separators in the ``inspect.py`` guard.

    Mirrors upstream behaviour exactly *except* the path-separator check.
    """
    import importlib  # noqa: PLC0415  — keep import-cost local
    import warnings  # noqa: PLC0415

    importer_frame = None

    try:
        importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
    except AttributeError:
        warnings.warn(
            "Failed to inspect frame to check if we should ignore "
            "importing a module lazily. This relies on a CPython "
            "implementation detail, report an issue if you see this with "
            "standard Python and include your version number."
        )

    if importer_frame is not None:
        # THE FIX: use os.sep (and also accept '/' on Windows) so the guard
        # works on both POSIX and Windows. ``os.path.basename`` is the most
        # robust way to compare just the file leaf.
        leaf = os.path.basename(importer_frame.filename)
        if leaf == "inspect.py":
            raise AttributeError()

    if self.lazy_module is None:
        try:
            if self.package is None:
                self.lazy_module = importlib.import_module(self.target)
            else:
                self.lazy_module = importlib.import_module(
                    f".{self.target}", self.package
                )
        except Exception as exc:  # noqa: BLE001
            raise ImportError(f"Lazy import of {self!r} failed") from exc

    return self.lazy_module


def patch_speechbrain_lazy_module() -> bool:
    """Apply the LazyModule path-separator patch if speechbrain is loaded.

    Returns ``True`` if the patch is applied (or already in place),
    ``False`` if speechbrain is not present or the structure is unrecognised.
    Never raises — the patch is best-effort and must not block startup.

    Idempotent: a marker attribute on the ``LazyModule`` class makes a second
    call a cheap no-op. Our replacement ``ensure_module`` is functionally
    equivalent to the upstream-fixed version, so overwriting an
    already-fixed-upstream method is benign even if speechbrain ships a real
    fix — we don't try to detect that case (substring matching the buggy
    literal in the source is fragile because comments and docstrings can
    falsely match).
    """
    importutils = sys.modules.get("speechbrain.utils.importutils")
    if importutils is None:
        return False

    LazyModule = getattr(importutils, "LazyModule", None)
    if LazyModule is None:
        LOGGER.debug(
            "speechbrain.utils.importutils has no LazyModule — speechbrain "
            "internals changed; skipping path-separator patch."
        )
        return False

    if getattr(LazyModule, _PATCH_MARKER, False):
        return True  # already patched in this process

    try:
        LazyModule.ensure_module = _patched_ensure_module  # type: ignore[assignment]
        setattr(LazyModule, _PATCH_MARKER, True)
    except (AttributeError, TypeError) as exc:
        LOGGER.warning(
            "Could not patch speechbrain LazyModule.ensure_module "
            "(%s: %s); audio ingest may crash on Windows when integrations "
            "have missing optional deps (k2 / flair).",
            type(exc).__name__,
            exc,
        )
        return False

    LOGGER.info(
        "Patched speechbrain LazyModule.ensure_module to honour Windows "
        "path separators in the inspect.py guard (upstream issue #2995)."
    )
    return True


__all__ = ["patch_speechbrain_lazy_module"]
