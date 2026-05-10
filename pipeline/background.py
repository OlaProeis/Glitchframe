"""
Shared background API: mode constants, :class:`BackgroundSource` protocol, and
factory for compositor-ready ``background_frame(t)`` implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from config import MODEL_CACHE_DIR

from .background_animatediff import AnimateDiffBackground
from .background_kenburns import DEFAULT_MARGIN, StaticKenBurnsBackground
from .background_stills import BackgroundStills, DEFAULT_KEYFRAME_INTERVAL

ProgressFn = Callable[[float, str], None]

MODE_SDXL_STILLS = "sdxl-stills"
MODE_STATIC_KENBURNS = "static-kenburns"
MODE_ANIMATEDIFF = "animatediff"

BACKGROUND_MODES: tuple[str, ...] = (
    MODE_SDXL_STILLS,
    MODE_STATIC_KENBURNS,
    MODE_ANIMATEDIFF,
)

# Image-generation backend for the AI stills mode. SDXL stays the default so
# existing renders behave identically; HiDream is an experimental opt-in
# requiring GLITCHFRAME_HIDREAM_* env vars (see .env.example).
IMAGE_BACKEND_SDXL = "sdxl"
IMAGE_BACKEND_HIDREAM = "hidream"
IMAGE_BACKENDS: tuple[str, ...] = (IMAGE_BACKEND_SDXL, IMAGE_BACKEND_HIDREAM)


def normalize_image_backend(backend: str | None) -> str:
    """Map UI / legacy labels to canonical image-backend strings; raises on unknown."""
    if backend is None or not str(backend).strip():
        return IMAGE_BACKEND_SDXL
    raw = str(backend).strip().lower()
    aliases = {
        "sdxl": IMAGE_BACKEND_SDXL,
        "stable diffusion xl": IMAGE_BACKEND_SDXL,
        "sdxl ai stills": IMAGE_BACKEND_SDXL,
        "hidream": IMAGE_BACKEND_HIDREAM,
        "hidream-o1": IMAGE_BACKEND_HIDREAM,
        "hidream-o1-image": IMAGE_BACKEND_HIDREAM,
    }
    if raw in aliases:
        return aliases[raw]
    if raw in IMAGE_BACKENDS:
        return raw
    raise ValueError(
        f"Unknown image backend {backend!r}; expected one of {IMAGE_BACKENDS}"
    )


@runtime_checkable
class BackgroundSource(Protocol):
    """Compositor-facing background: prepare assets, sample RGB frames, release GPU."""

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        ...

    def ensure(
        self,
        *,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> Any:
        """Generate or load cached assets; returns a mode-specific manifest dict/object."""
        ...

    def background_frame(self, t: float) -> np.ndarray:
        """``(H, W, 3)`` ``uint8`` RGB for wall-clock time ``t`` (seconds)."""
        ...

    def close(self) -> None:
        """Release GPU / large buffers."""
        ...


def normalize_background_mode(mode: str | None) -> str:
    """Map UI / legacy labels to canonical mode strings; raises on unknown."""
    if mode is None or not str(mode).strip():
        return MODE_SDXL_STILLS
    raw = str(mode).strip().lower()
    aliases = {
        "sdxl-stills": MODE_SDXL_STILLS,
        "sdxl_stills": MODE_SDXL_STILLS,
        "ai stills (fast)": MODE_SDXL_STILLS,
        "ai-stills": MODE_SDXL_STILLS,
        "static-kenburns": MODE_STATIC_KENBURNS,
        "static_kenburns": MODE_STATIC_KENBURNS,
        "static image upload": MODE_STATIC_KENBURNS,
        "kenburns": MODE_STATIC_KENBURNS,
        "animatediff": MODE_ANIMATEDIFF,
        "ai animated (animatediff, slow)": MODE_ANIMATEDIFF,
        "ai-animated": MODE_ANIMATEDIFF,
    }
    if raw in aliases:
        return aliases[raw]
    if raw in BACKGROUND_MODES:
        return raw
    raise ValueError(
        f"Unknown background mode {mode!r}; expected one of {BACKGROUND_MODES}"
    )


def create_background_source(
    mode: str,
    cache_dir: Path | str,
    *,
    preset_id: str,
    preset_prompt: str,
    static_image_path: Path | str | None = None,
    width: int = 1920,
    height: int = 1080,
    model_cache_dir: Path | None = None,
    keyframe_interval: float = DEFAULT_KEYFRAME_INTERVAL,
    sdxl_ken_burns: bool = False,
    ken_burns_margin: float | None = None,
    sdxl_rife_morph: bool = False,
    rife_exp: int = 4,
    ken_burns_rms_drive_at: Callable[[float], float] | None = None,
    image_backend: str | None = None,
) -> BackgroundSource:
    """
    Construct the background implementation for ``mode`` (use
    :func:`normalize_background_mode` first if values come from the UI).

    ``image_backend`` selects between ``sdxl`` (default; existing behavior)
    and ``hidream`` (HiDream-O1-Image via out-of-process worker — see
    :mod:`pipeline.background_stills_hidream`). Only applies to AI-stills
    modes (``MODE_SDXL_STILLS``, ``MODE_ANIMATEDIFF``).

    Raises
    ------
    ValueError
        Unknown mode or invalid arguments for the selected mode.
    """
    m = normalize_background_mode(mode)
    backend = normalize_image_backend(image_backend)
    mdir = Path(model_cache_dir) if model_cache_dir is not None else MODEL_CACHE_DIR

    kb_margin = float(DEFAULT_MARGIN if ken_burns_margin is None else ken_burns_margin)

    if m == MODE_SDXL_STILLS:
        if backend == IMAGE_BACKEND_HIDREAM:
            from .background_stills_hidream import BackgroundStillsHiDream

            return BackgroundStillsHiDream(
                cache_dir,
                preset_id=preset_id,
                preset_prompt=preset_prompt,
                width=width,
                height=height,
                keyframe_interval=keyframe_interval,
                ken_burns=bool(sdxl_ken_burns),
                ken_burns_margin=kb_margin,
                rife_morph=bool(sdxl_rife_morph),
                rife_exp=int(rife_exp),
                ken_burns_rms_drive_at=ken_burns_rms_drive_at,
            )
        return BackgroundStills(
            cache_dir,
            preset_id=preset_id,
            preset_prompt=preset_prompt,
            width=width,
            height=height,
            keyframe_interval=keyframe_interval,
            model_cache_dir=mdir,
            ken_burns=bool(sdxl_ken_burns),
            ken_burns_margin=kb_margin,
            rife_morph=bool(sdxl_rife_morph),
            rife_exp=int(rife_exp),
            ken_burns_rms_drive_at=ken_burns_rms_drive_at,
        )
    if m == MODE_STATIC_KENBURNS:
        return StaticKenBurnsBackground(
            cache_dir,
            preset_id=preset_id,
            source_image_path=static_image_path,
            width=width,
            height=height,
        )
    if m == MODE_ANIMATEDIFF:
        # AnimateDiff mode runs the user's contract:
        #   load SDXL -> generate stills -> dump SDXL -> load AnimateDiff ->
        #   for each segment, seed AnimateDiff from the closest SDXL keyframe.
        # The lifecycle is orchestrated inside ``AnimateDiffBackground.ensure``
        # (the stills source is closed there to free SDXL VRAM before the
        # AnimateDiff pipeline loads). No sample-time blending happens -- the
        # AnimateDiff frames *are* the output, not an overlay on top of SDXL.
        init_source = BackgroundStills(
            cache_dir,
            preset_id=preset_id,
            preset_prompt=preset_prompt,
            width=width,
            height=height,
            keyframe_interval=keyframe_interval,
            model_cache_dir=mdir,
            ken_burns=False,
            rife_morph=False,
        )
        return AnimateDiffBackground(
            cache_dir,
            preset_id=preset_id,
            preset_prompt=preset_prompt,
            width=width,
            height=height,
            model_cache_dir=mdir,
            init_image_source=init_source,
        )
    raise ValueError(f"Unhandled background mode: {m!r}")


__all__ = [
    "BACKGROUND_MODES",
    "BackgroundSource",
    "IMAGE_BACKEND_HIDREAM",
    "IMAGE_BACKEND_SDXL",
    "IMAGE_BACKENDS",
    "MODE_ANIMATEDIFF",
    "MODE_SDXL_STILLS",
    "MODE_STATIC_KENBURNS",
    "ProgressFn",
    "create_background_source",
    "normalize_background_mode",
    "normalize_image_backend",
]
