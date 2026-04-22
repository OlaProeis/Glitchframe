"""
Static image background with RMS-driven Ken Burns (zoom / pan / subtle rotation).

Caches the uploaded still under ``cache/<song_hash>/background/source.<ext>`` and
writes ``manifest_static_kenburns.json`` so SDXL ``manifest.json`` stays
independent. Requires ``analysis.json`` for duration and RMS envelope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.background_stills import (
    BACKGROUND_DIRNAME,
    _duration_from_analysis,
    _load_analysis,
    _smoothstep,
)


def _interp_scalar_series(
    values: Sequence[float] | None, t: float, fps: float
) -> float:
    """Linear interpolation along ``values`` sampled at ``fps`` (matches reactive_shader)."""
    if not values or fps <= 0.0:
        return 0.0
    n = len(values)
    if n == 0:
        return 0.0
    idx_f = float(t) * float(fps)
    if idx_f <= 0.0:
        return float(values[0])
    if idx_f >= n - 1:
        return float(values[-1])
    i0 = int(idx_f)
    i1 = min(i0 + 1, n - 1)
    frac = idx_f - i0
    return float((1.0 - frac) * float(values[i0]) + frac * float(values[i1]))

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest_static_kenburns.json"
SOURCE_BASENAME = "source"
MANIFEST_SCHEMA_VERSION = 1

DEFAULT_MARGIN = 1.38
ZOOM_BASE = 1.06
ZOOM_U_SCALE = 0.14
ZOOM_RMS_SCALE = 0.18
PAN_RANGE = 0.42
ROT_DEG_MAX = 1.35

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class KenBurnsManifest:
    schema_version: int
    mode: str
    preset_id: str
    source_sha256: str
    width: int
    height: int
    duration_sec: float
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "mode": str(self.mode),
            "preset_id": str(self.preset_id),
            "source_sha256": str(self.source_sha256),
            "width": int(self.width),
            "height": int(self.height),
            "duration_sec": float(self.duration_sec),
            "margin": float(self.margin),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KenBurnsManifest":
        try:
            return cls(
                schema_version=int(raw["schema_version"]),
                mode=str(raw["mode"]),
                preset_id=str(raw["preset_id"]),
                source_sha256=str(raw["source_sha256"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                duration_sec=float(raw["duration_sec"]),
                margin=float(raw.get("margin", DEFAULT_MARGIN)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Ken Burns manifest: {exc}") from exc

    def matches(
        self,
        *,
        preset_id: str,
        source_sha256: str,
        width: int,
        height: int,
        duration_sec: float,
        margin: float,
    ) -> bool:
        return (
            self.schema_version == MANIFEST_SCHEMA_VERSION
            and self.mode == "static-kenburns"
            and self.preset_id == preset_id
            and self.source_sha256 == source_sha256
            and self.width == int(width)
            and self.height == int(height)
            and abs(self.duration_sec - float(duration_sec)) < 1e-3
            and abs(self.margin - float(margin)) < 1e-6
        )


def _background_dir(cache_dir: Path) -> Path:
    return cache_dir / BACKGROUND_DIRNAME


def _manifest_path(cache_dir: Path) -> Path:
    return _background_dir(cache_dir) / MANIFEST_FILENAME


def _atomic_write_json(data: Mapping[str, Any], dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(dst)


def _atomic_copy_to(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cover_canvas(
    img: Image.Image, width: int, height: int, margin: float
) -> Image.Image:
    """RGB image covering ``width x height`` with at least ``margin`` headroom."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    sw, sh = img.size
    if sw <= 0 or sh <= 0:
        raise ValueError("Source image has invalid dimensions")
    base = max(float(width) / sw, float(height) / sh) * margin
    nw = max(1, int(sw * base + 0.5))
    nh = max(1, int(sh * base + 0.5))
    return img.resize((nw, nh), Image.LANCZOS)


def _rms_envelope_stats(analysis: Mapping[str, Any]) -> tuple[float, list[float]]:
    rms_block = analysis.get("rms") or {}
    values = rms_block.get("values")
    if not isinstance(values, list) or not values:
        return 1.0, []
    floats = [float(v) for v in values]
    mx = max(floats) if floats else 1.0
    if mx <= 1e-12:
        mx = 1.0
    return mx, floats


def _ken_burns_transform(
    work: Image.Image,
    *,
    width: int,
    height: int,
    u: float,
    rms_n: float,
) -> np.ndarray:
    """Return ``(H, W, 3) uint8`` RGB for progress ``u in [0,1]`` and RMS drive ``rms_n in [0,1]``."""
    iw, ih = work.size
    zoom = ZOOM_BASE + ZOOM_U_SCALE * u + ZOOM_RMS_SCALE * rms_n
    zoom = float(np.clip(zoom, 1.0, 2.5))
    pan_x = math.sin(u * math.pi * 1.15 + rms_n * 0.9) * PAN_RANGE
    pan_y = math.cos(u * math.pi * 0.95 - rms_n * 0.6) * PAN_RANGE * 0.72
    rot_deg = (u - 0.5) * 0.35 + (rms_n - 0.5) * ROT_DEG_MAX

    ar_out = width / height
    crop_h = ih / zoom
    crop_w = crop_h * ar_out
    if crop_w > iw / zoom:
        crop_w = iw / zoom
        crop_h = crop_w / ar_out

    cx = iw * (0.5 + pan_x * 0.08)
    cy = ih * (0.5 + pan_y * 0.08)
    left = cx - crop_w / 2.0
    top = cy - crop_h / 2.0
    left = float(np.clip(left, 0.0, max(0.0, iw - crop_w)))
    top = float(np.clip(top, 0.0, max(0.0, ih - crop_h)))

    box = (int(left), int(top), int(left + crop_w + 0.5), int(top + crop_h + 0.5))
    cropped = work.crop(box)
    if abs(rot_deg) > 1e-3:
        cropped = cropped.rotate(
            rot_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0)
        )
        cropped = _cover_canvas(cropped, width, height, 1.0)
    out = cropped.resize((width, height), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8).copy()


class StaticKenBurnsBackground:
    """
    RMS-modulated Ken Burns on a user-supplied still.

    Parameters
    ----------
    cache_dir:
        ``cache/<song_hash>/`` with ``analysis.json``.
    preset_id:
        Included in the cache manifest key (cosmetic alignment with presets).
    source_image_path:
        Path to the uploaded image; must be set before :meth:`ensure` (or pass
        a concrete path at construction time).
    width, height:
        Output resolution (default 1920×1080).
    margin:
        Cover-scale headroom for pan/zoom (larger = more room, softer crops).
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        preset_id: str,
        source_image_path: Path | str | None = None,
        width: int = 1920,
        height: int = 1080,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        cache = Path(cache_dir)
        if not cache.is_dir():
            raise FileNotFoundError(f"Cache dir does not exist: {cache}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution: {width}x{height}")
        if margin <= 1.0:
            raise ValueError(f"margin must be > 1.0, got {margin}")
        if not preset_id.strip():
            raise ValueError("preset_id must be a non-empty string")

        self._cache_dir = cache
        self._preset_id = preset_id.strip()
        self._source_path = (
            Path(source_image_path) if source_image_path is not None else None
        )
        self._width = int(width)
        self._height = int(height)
        self._margin = float(margin)

        self._analysis: dict[str, Any] | None = None
        self._duration = 0.0
        self._rms_fps = 30.0
        self._rms_norm_max = 1.0
        self._work_img: Image.Image | None = None
        self._manifest: KenBurnsManifest | None = None
        self._closed = False

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def manifest(self) -> KenBurnsManifest | None:
        return self._manifest

    def _load_manifest_disk(self) -> KenBurnsManifest | None:
        p = _manifest_path(self._cache_dir)
        if not p.is_file():
            return None
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return KenBurnsManifest.from_dict(raw)

    def ensure(
        self,
        *,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> KenBurnsManifest:
        if self._closed:
            raise RuntimeError("StaticKenBurnsBackground has been closed")

        def _report(p: float, msg: str) -> None:
            if progress is not None:
                progress(max(0.0, min(1.0, p)), msg)

        src = self._source_path
        if src is None or not str(src).strip():
            raise RuntimeError(
                "Static Ken Burns requires a source image path; "
                "upload a still when using static-kenburns mode"
            )
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"Static background image not found: {src}")

        _report(0.05, "Loading analysis…")
        analysis = _load_analysis(self._cache_dir / ANALYSIS_JSON_NAME)
        duration = _duration_from_analysis(analysis)
        sha = _file_sha256(src)
        expected = KenBurnsManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            mode="static-kenburns",
            preset_id=self._preset_id,
            source_sha256=sha,
            width=self._width,
            height=self._height,
            duration_sec=duration,
            margin=self._margin,
        )

        bg_dir = _background_dir(self._cache_dir)
        bg_dir.mkdir(parents=True, exist_ok=True)
        ext = src.suffix if src.suffix else ".png"
        cached_source = bg_dir / f"{SOURCE_BASENAME}{ext}"

        if not force:
            existing = None
            try:
                existing = self._load_manifest_disk()
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("Ignoring bad Ken Burns manifest (%s); rebuilding", exc)
            if existing is not None and existing.matches(
                preset_id=expected.preset_id,
                source_sha256=expected.source_sha256,
                width=expected.width,
                height=expected.height,
                duration_sec=expected.duration_sec,
                margin=expected.margin,
            ):
                if cached_source.is_file():
                    _report(0.5, "Loading cached static background…")
                    img = Image.open(cached_source)
                    self._work_img = _cover_canvas(img, self._width, self._height, self._margin)
                    img.close()
                    self._analysis = analysis
                    self._duration = duration
                    rms_max, _ = _rms_envelope_stats(analysis)
                    rms_block = analysis.get("rms") or {}
                    self._rms_fps = float(
                        rms_block.get("fps") or analysis.get("fps") or 30.0
                    )
                    self._rms_norm_max = rms_max
                    self._manifest = existing
                    _report(1.0, "Ken Burns background ready (cache hit)")
                    return existing

        _report(0.15, "Caching source image…")
        _atomic_copy_to(src, cached_source)

        _report(0.35, "Preparing Ken Burns canvas…")
        with Image.open(cached_source) as im:
            self._work_img = _cover_canvas(im, self._width, self._height, self._margin)

        self._analysis = analysis
        self._duration = duration
        rms_max, _ = _rms_envelope_stats(analysis)
        rms_block = analysis.get("rms") or {}
        self._rms_fps = float(rms_block.get("fps") or analysis.get("fps") or 30.0)
        self._rms_norm_max = rms_max
        self._manifest = expected
        _atomic_write_json(expected.to_dict(), _manifest_path(self._cache_dir))
        _report(1.0, "Ken Burns background ready")
        return expected

    def background_frame(self, t: float) -> np.ndarray:
        if self._closed:
            raise RuntimeError("StaticKenBurnsBackground has been closed")
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        if self._work_img is None or self._analysis is None:
            raise RuntimeError(
                "background_frame called before ensure(); prepare the background first"
            )
        dur = max(1e-6, float(self._duration))
        t_clamped = float(np.clip(t, 0.0, self._duration))
        u = _smoothstep(t_clamped / dur)
        raw_rms = _interp_scalar_series(
            (self._analysis.get("rms") or {}).get("values"),
            t_clamped,
            self._rms_fps,
        )
        rms_n = float(raw_rms) / self._rms_norm_max if self._rms_norm_max > 0 else 0.0
        rms_n = float(np.clip(rms_n, 0.0, 1.0))
        return _ken_burns_transform(
            self._work_img,
            width=self._width,
            height=self._height,
            u=u,
            rms_n=rms_n,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._work_img = None
        self._analysis = None

    def __enter__(self) -> "StaticKenBurnsBackground":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "KenBurnsManifest",
    "MANIFEST_FILENAME",
    "StaticKenBurnsBackground",
]
