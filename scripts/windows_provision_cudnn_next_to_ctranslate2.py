#!/usr/bin/env python3
"""Best-effort (Windows): copy CUDNN DLLs next to ``ctranslate2`` so LoadLibrary can resolve them.

Used after ``pip install nvidia-cudnn-cu12`` and/or when PyTorch bundles cuDNN under
``torch\\lib``. The curated Windows lyrics stack pins ``ctranslate2==4.4.0`` with
PyTorch ``2.2.2+cu121`` — matching DLL layout reduces ``cudnn_ops_infer64_8.dll`` failures.

Does **not** download NVIDIA's standalone cuDNN 8.9.7 installer; use pip wheels or
manual extraction from https://developer.nvidia.com/cudnn if pip alone is insufficient.

Safe to run multiple times. Exits 0 even when nothing is copied (e.g. Linux, missing packages).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _ctranslate2_package_dir() -> Path | None:
    try:
        import ctranslate2 as ct2  # noqa: PLC0415
    except ImportError:
        return None
    return Path(ct2.__file__).resolve().parent


def main() -> int:
    if sys.platform != "win32":
        return 0

    ct2_dir = _ctranslate2_package_dir()
    if ct2_dir is None:
        return 0

    try:
        import site  # noqa: PLC0415

        roots = [Path(p) for p in site.getsitepackages()]
    except Exception:
        roots = []

    if not roots and getattr(sys, "prefix", None):
        roots = [Path(sys.prefix) / "Lib" / "site-packages"]

    dll_names = (
        # cuDNN 8.x (Track A: torch 2.2.2+cu121 ships these in torch\lib).
        # ctranslate2 4.4.0 looks for the *_8 family at runtime; copying the full
        # set next to ctranslate2 ensures Windows LoadLibrary finds them via the
        # ctranslate2 package directory regardless of PATH ordering quirks.
        "cudnn64_8.dll",
        "cudnn_ops_infer64_8.dll",
        "cudnn_ops_train64_8.dll",
        "cudnn_cnn_infer64_8.dll",
        "cudnn_cnn_train64_8.dll",
        "cudnn_adv_infer64_8.dll",
        "cudnn_adv_train64_8.dll",
    )

    sources: list[Path] = []
    for root in roots:
        tl = root / "torch" / "lib"
        if tl.is_dir():
            sources.append(tl)
        nb = root / "nvidia" / "cudnn" / "bin"
        if nb.is_dir():
            sources.append(nb)

    copied = 0
    for src_dir in sources:
        for name in dll_names:
            src = src_dir / name
            if not src.is_file():
                continue
            dst = ct2_dir / name
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError:
                continue

    if copied:
        print(
            "windows_provision_cudnn_next_to_ctranslate2: "
            f"copied {copied} DLL(s) into {ct2_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
