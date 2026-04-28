"""Force ``huggingface_hub`` to copy files instead of symlinking on Windows.

Why this exists
---------------
``huggingface_hub`` lays out its cache as a content-addressed blob store
plus a per-revision ``snapshots/<commit>/`` directory whose entries are
**relative symlinks** pointing at the blobs. On Windows, creating a
symlink requires either:

1. The ``SeCreateSymbolicLinkPrivilege`` token (only granted to local
   administrators by default), OR
2. **Developer Mode** enabled in Windows Settings.

Pinokio installs run as a normal (non-admin) user with Developer Mode
**off** in the vast majority of cases. Symptom seen in the wild:

    Align lyrics failed: [WinError 1314] A required privilege is not
    held by the client: '..\\..\\blobs\\<sha>' ->
    'C:\\pinokio\\api\\Glitchframe.git\\cache\\HF_HOME\\hub\\models--Systran-
    -faster-whisper-large-v3\\snapshots\\<commit>\\preprocessor_config.json'

The blob has already been downloaded fine; only the snapshot symlink
fails. There's no recovery path inside ``faster-whisper`` / ``whisperx``
because they call ``snapshot_download(...)`` and expect it to either
return a clean snapshot dir or raise.

The fix
-------
``huggingface_hub.file_download.are_symlinks_supported(cache_dir)`` is
the single decision point: when it returns ``False``, the library
populates the snapshot directory by **copying** blobs instead of
symlinking. We monkey-patch it to always return ``False`` on Windows
**before** any HF download happens.

Trade-off: when multiple revisions of the same model end up cached at
once, blobs get duplicated rather than deduplicated (typical extra cost
~1-2 GB for our use case, which is fine). In exchange the cache works
reliably for every Pinokio user regardless of admin / Developer Mode
state, **and** is safe to copy across machines (no broken symlinks).

This matches the upstream guidance for the new
``HF_HUB_DISABLE_SYMLINKS=1`` env var introduced in
``huggingface_hub`` PR #4032 (Apr 2026). We can't rely on that env var
yet because it requires a very recent ``huggingface_hub`` version that
not every Pinokio install will have pinned, so we patch the function
directly — works on every supported version.

Idempotent. No-op when ``huggingface_hub`` isn't importable. No-op on
non-Windows hosts (where symlinks work without privilege escalation).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

LOGGER = logging.getLogger(__name__)

_PATCH_SENTINEL = "_glitchframe_symlink_patch_applied"


def patch_huggingface_disable_symlinks() -> bool:
    """Force ``huggingface_hub`` to skip symlinks on Windows.

    Returns ``True`` if the patch was applied (or had already been
    applied), ``False`` if there was nothing to patch (non-Windows host
    or ``huggingface_hub`` not importable).

    Safe to call multiple times. Safe to call before *or* after
    ``huggingface_hub`` has been imported by other code — we both
    rebind ``are_symlinks_supported`` in the module namespace **and**
    invalidate ``_are_symlinks_supported_in_dir`` so any cached ``True``
    result from before the patch is dropped.
    """
    if not sys.platform.startswith("win"):
        # On Linux / macOS symlinks are unprivileged and faster; let HF do
        # its thing. Patching here would just waste disk for no benefit.
        return False

    try:
        import huggingface_hub  # noqa: F401  -- ensure parent package loads
        from huggingface_hub import file_download as _fd  # type: ignore
    except ImportError:
        # Not in the dependency tree (CPU-only test env, smoke tests).
        return False

    if getattr(_fd, _PATCH_SENTINEL, False):
        return True

    original = getattr(_fd, "are_symlinks_supported", None)
    if not callable(original):
        # Some far-future huggingface_hub refactored the entry point we hook;
        # fail soft so the import doesn't crash the whole app -- worst case
        # the user sees the original WinError 1314 we set out to fix.
        LOGGER.warning(
            "huggingface_hub.file_download.are_symlinks_supported is missing "
            "or not callable; cannot apply Windows symlink-disable patch. "
            "If model downloads fail with WinError 1314, enable Developer "
            "Mode in Windows Settings or run Pinokio as administrator."
        )
        return False

    def _no_symlinks_on_windows(*_args: Any, **_kwargs: Any) -> bool:
        # The HF call signature is ``are_symlinks_supported(cache_dir=None)``;
        # we ignore the dir because the answer is uniformly "no, copy
        # instead" on Windows non-admin / non-developer-mode -- and the cost
        # of being wrong (extra disk usage) is far smaller than the cost of
        # the alternative (model download blows up after blobs are saved).
        return False

    # Rebind both the function and any cached "yes" answers from prior calls.
    # The cache is keyed by absolute cache_dir -- if any prior call inside
    # this process resolved True, subsequent _create_symlink calls would skip
    # straight to os.symlink() and re-hit the WinError 1314 we are trying
    # to fix.
    _fd.are_symlinks_supported = _no_symlinks_on_windows
    cache = getattr(_fd, "_are_symlinks_supported_in_dir", None)
    if isinstance(cache, dict):
        for key in list(cache.keys()):
            cache[key] = False

    setattr(_fd, _PATCH_SENTINEL, True)

    # Belt-and-braces: also export the env var if the running
    # huggingface_hub knows about it (PR #4032, recent versions). Costs
    # nothing on older versions where the env var is ignored.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    # Suppress the legacy "your machine does not support symlinks" warning
    # that older huggingface_hub versions emit on every download.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    LOGGER.info(
        "huggingface_hub: symlinks disabled on Windows; HF cache will copy "
        "blobs into snapshot dirs instead of symlinking them. This avoids "
        "WinError 1314 (SeCreateSymbolicLinkPrivilege) on non-admin / "
        "non-developer-mode installs. Trade-off: extra disk usage when "
        "multiple model revisions are cached."
    )
    return True


__all__ = ["patch_huggingface_disable_symlinks"]
