"""
PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``.

WhisperX loads pyannote (and similar) checkpoints whose pickles reference
OmegaConf types such as ``ListConfig``, which the restricted unpickler
rejects. The checkpoints we load from Hugging Face for alignment are
trusted; when callers omit ``weights_only``, restore the pre-2.6 default
(``weights_only=False``) so those loads succeed.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

_applied = False
_original_torch_load: Callable[..., Any] | None = None


def _is_pytorch_2_6_or_newer() -> bool:
    try:
        import torch
    except ImportError:
        return False
    m = re.match(r"^(\d+)\.(\d+)", torch.__version__.split("+", 1)[0])
    if not m:
        return False
    major, minor = int(m.group(1)), int(m.group(2))
    return major > 2 or (major == 2 and minor >= 6)


def apply_whisperx_torch_load_compat() -> None:
    """Patch ``torch.load`` once so omitted ``weights_only`` defaults to False (PyTorch 2.6+)."""
    global _applied, _original_torch_load
    if _applied:
        return
    _applied = True
    if not _is_pytorch_2_6_or_newer():
        return
    import torch

    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]
    LOGGER.debug(
        "torch.load default set to weights_only=False for HF checkpoint compatibility "
        "(PyTorch %s)",
        torch.__version__,
    )


__all__ = ["apply_whisperx_torch_load_compat"]
