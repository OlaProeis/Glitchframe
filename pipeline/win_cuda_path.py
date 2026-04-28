"""Windows: register DLL search paths before CTranslate2 / faster-whisper import."""

from __future__ import annotations

import sys

_path_prepended = False


def ensure_windows_cuda_dll_paths() -> None:
    """Add ``torch\\lib`` and common ``site-packages`` wheel dirs to the DLL path.

    CTranslate2 loads cuDNN/CUDA by name; those DLLs often live next to
    :mod:`torch` or under ``site-packages\\nvidia\\*``. Without
    :func:`os.add_dll_directory` (Win32), dependent DLLs in ``torch\\lib`` may
    not resolve when a wheel under ``site-packages`` loads first.
    """
    if sys.platform != "win32":
        return
    import os
    from pathlib import Path

    add = getattr(os, "add_dll_directory", None)
    if not add:
        return

    def _try_add(path: Path) -> None:
        if path.is_dir():
            try:
                add(str(path))
            except OSError:
                pass

    try:
        import torch

        tlib = Path(torch.__file__).resolve().parent / "lib"
        _try_add(tlib)
        site = tlib.parent.parent
    except Exception:
        return

    # NVIDIA pip metapackages (if installed in this env, e.g. nvidia-cudnn-cu12)
    _try_add(site / "nvidia" / "cudnn" / "bin")
    _try_add(site / "nvidia" / "cublas" / "bin")
    _try_add(site / "ctranslate2")

    # Some Windows loaders resolve dependent DLLs via PATH, not only
    # add_dll_directory — prepend (once) so ctranslate2 finds torch/cuDNN.
    global _path_prepended
    if not _path_prepended and tlib.is_dir():
        extra = [
            str(tlib),
            str(site / "nvidia" / "cudnn" / "bin"),
            str(site / "nvidia" / "cublas" / "bin"),
            str(site / "ctranslate2"),
        ]
        prefix = ";".join(p for p in extra if Path(p).is_dir())
        if prefix:
            old = os.environ.get("PATH", "")
            os.environ["PATH"] = prefix + ";" + old
            _path_prepended = True


__all__ = ["ensure_windows_cuda_dll_paths"]
