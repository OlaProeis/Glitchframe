"""
SDXL AI background stills: keyframe generation + smooth interpolation.

Renders ``N = ceil(duration_sec / DEFAULT_KEYFRAME_INTERVAL)`` still images
for a song using :class:`diffusers.StableDiffusionXLPipeline` on CUDA FP16,
guided by the active preset's ``prompt`` field plus a per-section modifier
derived from ``analysis.json`` segments, then exposes a ``background_frame(t)``
callable that interpolates smoothly between consecutive keyframes via a
``smoothstep`` crossfade so the compositor can sample a background RGB
image for any wall-clock time ``t``.

Keyframes are persisted under ``cache/<song_hash>/background/keyframe_{i}.png``
alongside a ``manifest.json`` capturing the cache key
``(preset_id, prompt_hash, section_count, num_keyframes, model_id, width,
height)``. Calls reuse existing keyframes when the key matches; otherwise
generation is explicit (``force=True`` overrides a matching key).

All hard failures are raised (missing CUDA, OOM, missing ``analysis.json``,
malformed manifest). No silent regeneration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from config import MODEL_CACHE_DIR
from pipeline.audio_analyzer import ANALYSIS_JSON_NAME

LOGGER = logging.getLogger(__name__)

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
# SDXL-friendly 16:9-ish generation resolution (trained aspect ratio bucket);
# upscaled to DEFAULT_WIDTH x DEFAULT_HEIGHT with Pillow Lanczos afterwards.
DEFAULT_GEN_WIDTH = 1344
DEFAULT_GEN_HEIGHT = 768
DEFAULT_KEYFRAME_INTERVAL = 8.0  # seconds per keyframe (see N = ceil(dur/8))
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NUM_INFERENCE_STEPS = 28
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, text, watermark, logo, deformed, extra limbs, jpeg artifacts"
)

BACKGROUND_DIRNAME = "background"
MANIFEST_FILENAME = "manifest.json"
KEYFRAME_FILENAME_FMT = "keyframe_{index:04d}.png"
MANIFEST_SCHEMA_VERSION = 1

RIFE_TIMELINE_DIRNAME = "rife_timeline"
MANIFEST_RIFE_FILENAME = "manifest_rife.json"
RIFE_MANIFEST_SCHEMA_VERSION = 1

ProgressFn = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Plan / prompt helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyframePlan:
    """One keyframe's timing + prompt metadata."""

    index: int
    t_sec: float
    segment_index: int
    segment_label: int
    prompt: str


@dataclass(frozen=True)
class BackgroundManifest:
    """On-disk cache key for the generated keyframes.

    Matching on ``(preset_id, prompt_hash, section_count, num_keyframes,
    model_id, width, height)`` is enough to decide whether cached PNGs are
    reusable. ``keyframe_times`` is stored so the frame API does not have to
    recompute them from ``analysis.json`` when reading a cached set.
    """

    schema_version: int
    preset_id: str
    prompt_hash: str
    section_count: int
    num_keyframes: int
    duration_sec: float
    model_id: str
    width: int
    height: int
    keyframe_times: tuple[float, ...]
    prompts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "preset_id": self.preset_id,
            "prompt_hash": self.prompt_hash,
            "section_count": int(self.section_count),
            "num_keyframes": int(self.num_keyframes),
            "duration_sec": float(self.duration_sec),
            "model_id": self.model_id,
            "width": int(self.width),
            "height": int(self.height),
            "keyframe_times": [float(t) for t in self.keyframe_times],
            "prompts": list(self.prompts),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BackgroundManifest":
        try:
            return cls(
                schema_version=int(raw["schema_version"]),
                preset_id=str(raw["preset_id"]),
                prompt_hash=str(raw["prompt_hash"]),
                section_count=int(raw["section_count"]),
                num_keyframes=int(raw["num_keyframes"]),
                duration_sec=float(raw["duration_sec"]),
                model_id=str(raw["model_id"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                keyframe_times=tuple(float(t) for t in raw["keyframe_times"]),
                prompts=tuple(str(p) for p in raw.get("prompts", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid background manifest: {exc}") from exc

    def matches_key(
        self,
        *,
        preset_id: str,
        prompt_hash: str,
        section_count: int,
        num_keyframes: int,
        model_id: str,
        width: int,
        height: int,
    ) -> bool:
        return (
            self.schema_version == MANIFEST_SCHEMA_VERSION
            and self.preset_id == preset_id
            and self.prompt_hash == prompt_hash
            and self.section_count == int(section_count)
            and self.num_keyframes == int(num_keyframes)
            and self.model_id == model_id
            and self.width == int(width)
            and self.height == int(height)
        )


@dataclass(frozen=True)
class RifeMorphManifest:
    """Cache metadata for RIFE-densified timelines layered on SDXL keyframes."""

    schema_version: int
    base_prompt_hash: str
    preset_id: str
    section_count: int
    num_keyframes: int
    duration_sec: float
    rife_exp: int
    rife_repo_id: str
    width: int
    height: int
    frame_count: int
    times: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "base_prompt_hash": self.base_prompt_hash,
            "preset_id": self.preset_id,
            "section_count": int(self.section_count),
            "num_keyframes": int(self.num_keyframes),
            "duration_sec": float(self.duration_sec),
            "rife_exp": int(self.rife_exp),
            "rife_repo_id": self.rife_repo_id,
            "width": int(self.width),
            "height": int(self.height),
            "frame_count": int(self.frame_count),
            "times": [float(x) for x in self.times],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RifeMorphManifest":
        try:
            return cls(
                schema_version=int(raw["schema_version"]),
                base_prompt_hash=str(raw["base_prompt_hash"]),
                preset_id=str(raw["preset_id"]),
                section_count=int(raw["section_count"]),
                num_keyframes=int(raw["num_keyframes"]),
                duration_sec=float(raw["duration_sec"]),
                rife_exp=int(raw["rife_exp"]),
                rife_repo_id=str(raw["rife_repo_id"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                frame_count=int(raw["frame_count"]),
                times=tuple(float(x) for x in raw["times"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RIFE manifest: {exc}") from exc

    def matches_key(
        self,
        *,
        base_prompt_hash: str,
        preset_id: str,
        section_count: int,
        num_keyframes: int,
        duration_sec: float,
        rife_exp: int,
        rife_repo_id: str,
        width: int,
        height: int,
    ) -> bool:
        return (
            self.schema_version == RIFE_MANIFEST_SCHEMA_VERSION
            and self.base_prompt_hash == base_prompt_hash
            and self.preset_id == preset_id
            and self.section_count == int(section_count)
            and self.num_keyframes == int(num_keyframes)
            and abs(self.duration_sec - float(duration_sec)) < 1e-6
            and self.rife_exp == int(rife_exp)
            and self.rife_repo_id == rife_repo_id
            and self.width == int(width)
            and self.height == int(height)
            and len(self.times) == int(self.frame_count)
        )


def _load_analysis(analysis_path: Path) -> dict[str, Any]:
    if not analysis_path.is_file():
        raise FileNotFoundError(
            f"Missing analysis.json at {analysis_path}; run the audio analyzer first"
        )
    with analysis_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid {ANALYSIS_JSON_NAME} (expected object): {analysis_path}")
    return data


def _duration_from_analysis(analysis: Mapping[str, Any]) -> float:
    duration = analysis.get("duration_sec")
    if not isinstance(duration, (int, float)) or float(duration) <= 0.0:
        raise ValueError("analysis.duration_sec missing or non-positive")
    return float(duration)


def _segments_from_analysis(
    analysis: Mapping[str, Any], duration: float
) -> list[dict[str, Any]]:
    raw = analysis.get("segments")
    if not isinstance(raw, list) or not raw:
        # Analyzer always writes at least one segment, but stay defensive so a
        # corrupt analysis.json surfaces here rather than later in rendering.
        return [{"t_start": 0.0, "t_end": float(duration), "label": 0}]
    cleaned: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            t0 = float(entry["t_start"])
            t1 = float(entry["t_end"])
            label = int(entry.get("label", len(cleaned)))
        except (KeyError, TypeError, ValueError):
            continue
        cleaned.append({"t_start": t0, "t_end": t1, "label": label})
    if not cleaned:
        return [{"t_start": 0.0, "t_end": float(duration), "label": 0}]
    cleaned.sort(key=lambda s: s["t_start"])
    return cleaned


def _segment_index_for_time(
    segments: Sequence[Mapping[str, Any]], t: float
) -> int:
    """Return index of the segment containing ``t``; clamp outside the grid."""
    if not segments:
        return 0
    if t <= float(segments[0]["t_start"]):
        return 0
    for i, seg in enumerate(segments):
        if float(seg["t_start"]) <= t < float(seg["t_end"]):
            return i
    return len(segments) - 1


def _n_keyframes(duration: float, interval: float) -> int:
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if interval <= 0.0:
        raise ValueError(f"interval must be positive, got {interval}")
    return max(1, int(math.ceil(duration / interval)))


def _keyframe_times(duration: float, n: int) -> list[float]:
    """
    Place ``n`` keyframes evenly so the first anchors at ``t=0`` and the last
    at ``t=duration``; for ``n == 1`` place a single keyframe at the midpoint.
    This makes crossfade interpolation cover the entire song span.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return [float(duration) * 0.5]
    step = float(duration) / float(n - 1)
    return [float(i) * step for i in range(n)]


def _build_keyframe_prompt(
    preset_prompt: str,
    *,
    index: int,
    total: int,
    segment_label: int,
    segment_count: int,
    t_sec: float,
) -> str:
    """Combine preset prompt with a structural hint so scenes evolve across the song."""
    base = preset_prompt.strip()
    # Keep the modifier short and deterministic so cache hashes are stable.
    modifier = (
        f"scene {index + 1} of {total}, "
        f"song section {segment_label + 1} of {segment_count}, "
        f"t={t_sec:.1f}s"
    )
    if base:
        return f"{base}, {modifier}"
    return modifier


def plan_keyframes(
    analysis: Mapping[str, Any],
    preset_prompt: str,
    *,
    interval: float = DEFAULT_KEYFRAME_INTERVAL,
) -> list[KeyframePlan]:
    """Deterministic keyframe plan for an ``analysis.json`` dict + preset prompt."""
    duration = _duration_from_analysis(analysis)
    segments = _segments_from_analysis(analysis, duration)
    n = _n_keyframes(duration, interval)
    times = _keyframe_times(duration, n)

    plans: list[KeyframePlan] = []
    seg_count = len(segments)
    for i, t in enumerate(times):
        seg_idx = _segment_index_for_time(segments, t)
        seg_label = int(segments[seg_idx].get("label", seg_idx))
        plans.append(
            KeyframePlan(
                index=i,
                t_sec=float(t),
                segment_index=int(seg_idx),
                segment_label=seg_label,
                prompt=_build_keyframe_prompt(
                    preset_prompt,
                    index=i,
                    total=n,
                    segment_label=seg_label,
                    segment_count=seg_count,
                    t_sec=float(t),
                ),
            )
        )
    return plans


def prompt_hash(
    *,
    preset_prompt: str,
    prompts: Sequence[str],
    model_id: str,
    width: int,
    height: int,
) -> str:
    """SHA-256 over everything that should invalidate the cached keyframes."""
    h = hashlib.sha256()
    h.update(b"v1")  # bumped if the hashing layout changes
    h.update(b"\x00")
    h.update(preset_prompt.strip().encode("utf-8"))
    h.update(b"\x00")
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(f"{int(width)}x{int(height)}".encode("utf-8"))
    for p in prompts:
        h.update(b"\x00")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _background_dir(cache_dir: Path) -> Path:
    return cache_dir / BACKGROUND_DIRNAME


def _manifest_path(cache_dir: Path) -> Path:
    return _background_dir(cache_dir) / MANIFEST_FILENAME


def _keyframe_path(cache_dir: Path, index: int) -> Path:
    return _background_dir(cache_dir) / KEYFRAME_FILENAME_FMT.format(index=index)


def _atomic_write_png(img: Image.Image, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    img.save(str(tmp), format="PNG", optimize=False)
    tmp.replace(dst)


def _atomic_write_json(data: Mapping[str, Any], dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(dst)


def _load_manifest(cache_dir: Path) -> BackgroundManifest | None:
    p = _manifest_path(cache_dir)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return BackgroundManifest.from_dict(raw)


def _load_keyframes(
    cache_dir: Path, n: int, width: int, height: int
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(n):
        path = _keyframe_path(cache_dir, i)
        if not path.is_file():
            raise FileNotFoundError(f"Cached keyframe missing: {path}")
        with Image.open(path) as im:
            img = im.convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            frames.append(np.asarray(img, dtype=np.uint8).copy())
    return frames


def _rife_timeline_dir(cache_dir: Path) -> Path:
    return _background_dir(cache_dir) / RIFE_TIMELINE_DIRNAME


def _rife_manifest_path(cache_dir: Path) -> Path:
    return _background_dir(cache_dir) / MANIFEST_RIFE_FILENAME


def _rife_frame_path(cache_dir: Path, index: int) -> Path:
    return _rife_timeline_dir(cache_dir) / f"rife_{index:06d}.png"


def _load_rife_manifest(cache_dir: Path) -> RifeMorphManifest | None:
    p = _rife_manifest_path(cache_dir)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return RifeMorphManifest.from_dict(raw)


def _rife_timeline_pngs_complete(cache_dir: Path, count: int) -> bool:
    if count < 1:
        return False
    for i in range(count):
        if not _rife_frame_path(cache_dir, i).is_file():
            return False
    return True


def _load_rife_timeline_frames(
    cache_dir: Path, count: int, width: int, height: int
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(count):
        path = _rife_frame_path(cache_dir, i)
        if not path.is_file():
            raise FileNotFoundError(f"Cached RIFE frame missing: {path}")
        with Image.open(path) as im:
            img = im.convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            frames.append(np.asarray(img, dtype=np.uint8).copy())
    return frames


def _persist_rife_timeline(
    cache_dir: Path,
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    manifest: RifeMorphManifest,
) -> None:
    td = _rife_timeline_dir(cache_dir)
    td.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(frames):
        img = Image.fromarray(arr, mode="RGB")
        _atomic_write_png(img, _rife_frame_path(cache_dir, i))
    _atomic_write_json(manifest.to_dict(), _rife_manifest_path(cache_dir))


# ---------------------------------------------------------------------------
# SDXL pipeline wrapper
# ---------------------------------------------------------------------------


def _require_cuda() -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is required for SDXL background generation"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available; SDXL background generation requires a CUDA GPU"
        )
    return "cuda:0"


def _load_sdxl_pipeline(
    model_id: str,
    *,
    cache_dir: Path,
    device: str,
) -> Any:
    """Load ``StableDiffusionXLPipeline`` at FP16 and move it to ``device``."""
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionXLPipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "diffusers / torch must be installed for SDXL background generation"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=str(cache_dir),
            add_watermarker=False,
        )
    except Exception as exc:  # noqa: BLE001
        # fp16 variant is not always published (e.g. some SDXL forks); retry
        # without the variant so the caller still gets a working pipeline.
        LOGGER.info(
            "Loading SDXL %s without fp16 variant (fallback after: %s)", model_id, exc
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=str(cache_dir),
            add_watermarker=False,
        )

    try:
        pipe = pipe.to(device)
    except Exception as exc:  # noqa: BLE001 - device moves can OOM or fail hard
        raise RuntimeError(
            f"Failed to move SDXL pipeline to device {device!r}: {exc}"
        ) from exc

    # Memory footprint reducers. All safe no-ops on older diffusers versions.
    # VAE slicing/tiling decode one latent slice at a time (saves ~2 GB peak).
    for method_name in ("enable_vae_slicing", "enable_vae_tiling"):
        method = getattr(pipe, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("SDXL %s failed: %s", method_name, exc)

    # xformers if present, else attention slicing — big VRAM reducer during
    # denoising so we never spill into shared system memory.
    xformers_enabled = False
    enable_xformers = getattr(pipe, "enable_xformers_memory_efficient_attention", None)
    if callable(enable_xformers):
        try:
            enable_xformers()
            xformers_enabled = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("xformers attention unavailable: %s", exc)
    if not xformers_enabled:
        enable_slicing = getattr(pipe, "enable_attention_slicing", None)
        if callable(enable_slicing):
            try:
                enable_slicing("auto")
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("attention slicing failed: %s", exc)

    if hasattr(pipe, "set_progress_bar_config"):
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:  # noqa: BLE001
            pass
    return pipe


def _resize_rgb(img: Image.Image, width: int, height: int) -> np.ndarray:
    """Return an ``(H, W, 3) uint8`` RGB array resized via Lanczos."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


KeyframeCallback = Callable[[int, int, int, int], None]
"""``(prompt_index, prompts_total, step_index, steps_total) -> None``."""


def _generate_images(
    pipe: Any,
    prompts: Sequence[str],
    *,
    negative_prompt: str,
    gen_width: int,
    gen_height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seeds: Sequence[int | None] | None = None,
    on_image: Callable[[int, Image.Image], None] | None = None,
    step_callback: KeyframeCallback | None = None,
) -> list[Image.Image]:
    """
    Generate SDXL images one-by-one (batch size = 1) to keep VRAM bounded.

    Batched generation (list of prompts in a single call) scales activation
    memory with len(prompts) and on ~24 GB GPUs silently spills into shared
    system RAM, slowing generation by 10–100×. A serial loop avoids that and
    also enables meaningful per-keyframe progress reporting.

    ``seeds`` (optional) is a parallel list of per-prompt seeds so partial
    resume runs reproduce the same image as a fresh run would.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch not available") from exc

    total_prompts = len(prompts)
    if seeds is not None and len(seeds) != total_prompts:
        raise ValueError(
            f"seeds length {len(seeds)} must match prompts length {total_prompts}"
        )
    images: list[Image.Image] = []
    for i, prompt in enumerate(prompts):
        generator: Any = None
        seed_i = seeds[i] if seeds is not None else None
        if seed_i is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed_i))

        if step_callback is not None:
            step_callback(i, total_prompts, 0, int(num_inference_steps))

        def _pipe_step_cb(
            _pipe: Any,
            step_idx: int,
            _timestep: Any,
            callback_kwargs: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            if step_callback is not None:
                # diffusers reports step_idx as 0-based; report 1-based so the
                # UI never shows "step 0 of N" after kickoff.
                step_callback(i, total_prompts, step_idx + 1, int(num_inference_steps))
            return callback_kwargs

        try:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(gen_width),
                height=int(gen_height),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                callback_on_step_end=_pipe_step_cb,
            )
        except TypeError:
            # Older diffusers without callback_on_step_end — fall back silently.
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(gen_width),
                height=int(gen_height),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            )
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            raise RuntimeError(
                "CUDA out of memory during SDXL keyframe generation; "
                "try a lower gen resolution or free VRAM (close other GPU apps)"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"SDXL generation failed on keyframe {i}: {exc}") from exc

        batch = getattr(out, "images", None)
        if not isinstance(batch, list) or not batch:
            raise RuntimeError("SDXL pipeline returned no images")
        img = batch[0]
        images.append(img)
        if on_image is not None:
            on_image(i, img)

        # Release activations between keyframes so peak VRAM stays flat.
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    return images


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _crossfade(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear ``uint8`` blend of two RGB frames at ``alpha in [0, 1]``."""
    if a.shape != b.shape:
        raise ValueError(f"Frame shape mismatch: {a.shape} vs {b.shape}")
    if alpha <= 0.0:
        return a.copy()
    if alpha >= 1.0:
        return b.copy()
    # int32 intermediate so (bf - af) * 256 cannot overflow int16 bounds.
    af = a.astype(np.int32)
    bf = b.astype(np.int32)
    alpha_q8 = int(round(alpha * 256.0))
    blend = af + ((bf - af) * alpha_q8 >> 8)
    return np.clip(blend, 0, 255).astype(np.uint8)


def _interpolate_frame(
    frames: Sequence[np.ndarray], times: Sequence[float], t: float
) -> np.ndarray:
    n = len(frames)
    if n != len(times) or n == 0:
        raise ValueError("frames and times must be the same non-zero length")
    if n == 1:
        return frames[0].copy()
    if t <= times[0]:
        return frames[0].copy()
    if t >= times[-1]:
        return frames[-1].copy()
    # Find the bracketing pair (times is monotonically increasing).
    for i in range(n - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        if t0 <= t <= t1:
            span = max(1e-6, t1 - t0)
            raw_alpha = (t - t0) / span
            return _crossfade(frames[i], frames[i + 1], _smoothstep(raw_alpha))
    # Unreachable given the bounds checks above; fall back defensively.
    return frames[-1].copy()


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


class BackgroundStills:
    """
    AI background stills renderer for the compositor.

    Parameters
    ----------
    cache_dir:
        Per-song cache directory (``cache/<song_hash>/``) containing
        ``analysis.json``. Keyframes + manifest land under
        ``cache_dir/background/``.
    preset_id:
        Preset stem (e.g. ``neon-synthwave``) used as the cache key component.
    preset_prompt:
        Raw preset ``prompt`` field forwarded to SDXL; combined with a
        per-section modifier per keyframe.
    width, height:
        Output resolution (defaults to 1920×1080). The upstream SDXL output is
        generated at ``gen_width x gen_height`` and then Lanczos-resized.
    gen_width, gen_height:
        Native SDXL generation resolution (defaults to 1344×768, an SDXL
        training aspect-ratio bucket). Must match the bucket constraints.
    keyframe_interval:
        Seconds per keyframe (``N = ceil(duration / interval)``).
    model_id:
        HF model id passed to :meth:`StableDiffusionXLPipeline.from_pretrained`.
    num_inference_steps, guidance_scale, negative_prompt, seed:
        SDXL sampling controls.
    model_cache_dir:
        Optional override for the HF cache (defaults to ``config.MODEL_CACHE_DIR``).

    Use as a context manager (or call :meth:`close`) to release the SDXL
    pipeline / VRAM deterministically.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        preset_id: str,
        preset_prompt: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        gen_width: int = DEFAULT_GEN_WIDTH,
        gen_height: int = DEFAULT_GEN_HEIGHT,
        keyframe_interval: float = DEFAULT_KEYFRAME_INTERVAL,
        model_id: str = DEFAULT_MODEL_ID,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: int | None = 0,
        model_cache_dir: Path | None = None,
        ken_burns: bool = False,
        ken_burns_margin: float | None = None,
        rife_morph: bool = False,
        rife_exp: int = 4,
        rife_repo_id: str = "MonsterMMORPG/RIFE_4_26",
    ) -> None:
        cache = Path(cache_dir)
        if not cache.is_dir():
            raise FileNotFoundError(f"Cache dir does not exist: {cache}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution: {width}x{height}")
        if gen_width <= 0 or gen_height <= 0:
            raise ValueError(f"Invalid gen resolution: {gen_width}x{gen_height}")
        if gen_width % 8 != 0 or gen_height % 8 != 0:
            raise ValueError(
                f"SDXL requires gen resolution divisible by 8, got {gen_width}x{gen_height}"
            )
        if keyframe_interval <= 0.0:
            raise ValueError(
                f"keyframe_interval must be positive, got {keyframe_interval}"
            )
        if not preset_id.strip():
            raise ValueError("preset_id must be a non-empty string")

        self._cache_dir = cache
        self._preset_id = preset_id.strip()
        self._preset_prompt = preset_prompt or ""
        self._width = int(width)
        self._height = int(height)
        self._gen_width = int(gen_width)
        self._gen_height = int(gen_height)
        self._keyframe_interval = float(keyframe_interval)
        self._model_id = str(model_id)
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._negative_prompt = str(negative_prompt)
        self._seed = None if seed is None else int(seed)
        self._model_cache_dir = (
            Path(model_cache_dir) if model_cache_dir is not None else MODEL_CACHE_DIR
        )
        self._ken_burns = bool(ken_burns)
        if ken_burns_margin is not None:
            km = float(ken_burns_margin)
            if km <= 1.0:
                raise ValueError(f"ken_burns_margin must be > 1.0, got {km}")
            self._ken_burns_margin = km
        else:
            self._ken_burns_margin = 1.38  # keep in sync with background_kenburns.DEFAULT_MARGIN

        re = int(rife_exp)
        if re < 2:
            re = 2
        elif re > 6:
            re = 6
        self._rife_morph = bool(rife_morph)
        self._rife_exp = re
        self._rife_repo = str(rife_repo_id).strip() or "MonsterMMORPG/RIFE_4_26"

        self._plans: list[KeyframePlan] | None = None
        self._frames: list[np.ndarray] | None = None
        self._manifest: BackgroundManifest | None = None
        self._timeline_times: tuple[float, ...] | None = None
        self._pipe: Any = None
        self._closed = False
        self._kb_duration: float = 0.0
        self._kb_analysis: dict[str, Any] | None = None

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def manifest(self) -> BackgroundManifest | None:
        return self._manifest

    @property
    def keyframes(self) -> tuple[np.ndarray, ...]:
        return tuple(self._frames) if self._frames else ()

    # -- planning -----------------------------------------------------------

    def _load_analysis(self) -> dict[str, Any]:
        return _load_analysis(self._cache_dir / ANALYSIS_JSON_NAME)

    def _plan(self, analysis: Mapping[str, Any]) -> list[KeyframePlan]:
        if self._plans is None:
            self._plans = plan_keyframes(
                analysis,
                self._preset_prompt,
                interval=self._keyframe_interval,
            )
        return self._plans

    def _expected_manifest(
        self,
        analysis: Mapping[str, Any],
        plans: Sequence[KeyframePlan],
    ) -> BackgroundManifest:
        prompts = tuple(p.prompt for p in plans)
        ph = prompt_hash(
            preset_prompt=self._preset_prompt,
            prompts=prompts,
            model_id=self._model_id,
            width=self._gen_width,
            height=self._gen_height,
        )
        segments = _segments_from_analysis(
            analysis, _duration_from_analysis(analysis)
        )
        return BackgroundManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            preset_id=self._preset_id,
            prompt_hash=ph,
            section_count=len(segments),
            num_keyframes=len(plans),
            duration_sec=_duration_from_analysis(analysis),
            model_id=self._model_id,
            width=self._gen_width,
            height=self._gen_height,
            keyframe_times=tuple(p.t_sec for p in plans),
            prompts=prompts,
        )

    def _prepare_ken_burns_state(self, analysis: Mapping[str, Any]) -> None:
        if not self._ken_burns:
            self._kb_analysis = None
            self._kb_duration = 0.0
            return
        self._kb_analysis = dict(analysis)
        self._kb_duration = _duration_from_analysis(analysis)

    def _apply_rife_morph_if_needed(
        self,
        mf: BackgroundManifest,
        *,
        force: bool,
        report: ProgressFn,
    ) -> None:
        """Replace ``self._frames`` with a RIFE-dense timeline when enabled."""
        self._timeline_times = None
        if not self._rife_morph:
            return
        kf = self._frames
        if kf is None or mf.num_keyframes < 2:
            LOGGER.info("RIFE morph skipped (need at least two SDXL keyframes)")
            return

        if not force:
            try:
                rm = _load_rife_manifest(self._cache_dir)
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("Ignoring unreadable RIFE manifest (%s)", exc)
                rm = None
            if rm is not None and rm.matches_key(
                base_prompt_hash=mf.prompt_hash,
                preset_id=mf.preset_id,
                section_count=mf.section_count,
                num_keyframes=mf.num_keyframes,
                duration_sec=mf.duration_sec,
                rife_exp=self._rife_exp,
                rife_repo_id=self._rife_repo,
                width=self._width,
                height=self._height,
            ) and _rife_timeline_pngs_complete(self._cache_dir, rm.frame_count):
                try:
                    self._frames = _load_rife_timeline_frames(
                        self._cache_dir,
                        rm.frame_count,
                        self._width,
                        self._height,
                    )
                except FileNotFoundError as exc:
                    LOGGER.info(
                        "RIFE manifest matches but a frame is missing (%s); will regenerate",
                        exc,
                    )
                else:
                    self._timeline_times = tuple(rm.times)
                    report(1.0, "Reused cached RIFE morph timeline")
                    return

        from pipeline.rife_runtime import rife_build_morph_timeline

        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for RIFE morph") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RIFE morph requires a CUDA GPU; disable “Morph keyframes (RIFE)” "
                "or run on hardware with CUDA"
            )
        device = torch.device("cuda:0")
        dense_frames, dense_times = rife_build_morph_timeline(
            tuple(kf),
            mf.keyframe_times,
            exp=self._rife_exp,
            device=device,
            repo_id=self._rife_repo,
            progress=report,
        )
        final_m = RifeMorphManifest(
            schema_version=RIFE_MANIFEST_SCHEMA_VERSION,
            base_prompt_hash=mf.prompt_hash,
            preset_id=mf.preset_id,
            section_count=mf.section_count,
            num_keyframes=mf.num_keyframes,
            duration_sec=mf.duration_sec,
            rife_exp=self._rife_exp,
            rife_repo_id=self._rife_repo,
            width=self._width,
            height=self._height,
            frame_count=len(dense_frames),
            times=tuple(float(t) for t in dense_times),
        )
        _persist_rife_timeline(self._cache_dir, dense_frames, dense_times, final_m)
        self._frames = dense_frames
        self._timeline_times = final_m.times

    # -- generation / cache -------------------------------------------------

    def ensure(
        self,
        *,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> BackgroundManifest:
        """Alias for :meth:`ensure_keyframes` (common :class:`BackgroundSource` API)."""
        return self.ensure_keyframes(force=force, progress=progress)

    def ensure_keyframes(
        self,
        *,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> BackgroundManifest:
        """
        Ensure all keyframe PNGs + ``manifest.json`` are present and valid.

        Reuses cached outputs when the computed manifest key matches; otherwise
        regenerates via SDXL. ``force=True`` regenerates even on a key match.
        Returns the active manifest.
        """
        if self._closed:
            raise RuntimeError("BackgroundStills has been closed")

        def _report(p: float, msg: str) -> None:
            if progress is not None:
                progress(max(0.0, min(1.0, p)), msg)

        _report(0.0, "Loading analysis…")
        analysis = self._load_analysis()
        plans = self._plan(analysis)
        expected = self._expected_manifest(analysis, plans)

        if not force:
            try:
                existing = _load_manifest(self._cache_dir)
            except Exception as exc:  # noqa: BLE001
                LOGGER.info(
                    "Ignoring unreadable background manifest (%s); will regenerate",
                    exc,
                )
                existing = None

            if existing is not None and existing.matches_key(
                preset_id=expected.preset_id,
                prompt_hash=expected.prompt_hash,
                section_count=expected.section_count,
                num_keyframes=expected.num_keyframes,
                model_id=expected.model_id,
                width=expected.width,
                height=expected.height,
            ):
                try:
                    frames = _load_keyframes(
                        self._cache_dir,
                        existing.num_keyframes,
                        self._width,
                        self._height,
                    )
                except FileNotFoundError as exc:
                    LOGGER.info(
                        "Background manifest matches but a keyframe is missing "
                        "(%s); will regenerate",
                        exc,
                    )
                else:
                    self._frames = frames
                    self._manifest = existing
                    self._prepare_ken_burns_state(analysis)
                    _report(1.0, "Reused cached background keyframes")
                    self._apply_rife_morph_if_needed(existing, force=force, report=_report)
                    return existing

        frames = self._generate_and_persist(expected, _report)
        self._frames = frames
        self._manifest = expected
        self._prepare_ken_burns_state(analysis)
        self._apply_rife_morph_if_needed(expected, force=force, report=_report)
        return expected

    def _generate_and_persist(
        self,
        manifest: BackgroundManifest,
        report: ProgressFn,
    ) -> list[np.ndarray]:
        bg_dir = _background_dir(self._cache_dir)
        bg_dir.mkdir(parents=True, exist_ok=True)

        total = max(1, manifest.num_keyframes)
        frames: list[np.ndarray | None] = [None] * total
        prompts = list(manifest.prompts)

        # Resume partial runs: reuse any keyframe PNGs already on disk so an
        # earlier crash (or a cancelled hang) doesn't force regenerating the
        # ones that already succeeded.
        missing_indices: list[int] = []
        missing_prompts: list[str] = []
        for i in range(total):
            path = _keyframe_path(self._cache_dir, i)
            if path.is_file():
                try:
                    with Image.open(path) as im:
                        frames[i] = _resize_rgb(im.convert("RGB"), self._width, self._height)
                    continue
                except Exception as exc:  # noqa: BLE001
                    LOGGER.info(
                        "Ignoring unreadable cached keyframe %s (%s); will regenerate",
                        path, exc,
                    )
            missing_indices.append(i)
            missing_prompts.append(prompts[i])

        if not missing_prompts:
            _atomic_write_json(manifest.to_dict(), _manifest_path(self._cache_dir))
            report(1.0, f"Reused {total} cached keyframes")
            return [f for f in frames if f is not None]

        resumed = total - len(missing_prompts)
        if resumed > 0:
            report(
                0.1,
                f"Resuming: {resumed}/{total} keyframes cached, generating {len(missing_prompts)}",
            )

        device = _require_cuda()
        report(0.05, f"Loading SDXL {self._model_id}…")
        if self._pipe is None:
            self._pipe = _load_sdxl_pipeline(
                self._model_id,
                cache_dir=self._model_cache_dir,
                device=device,
            )

        missing_total = len(missing_prompts)

        def _on_step(
            prompt_idx: int, prompts_total: int, step_idx: int, steps_total: int
        ) -> None:
            # Map (prompt_idx, step_idx) -> [0.1, 0.95] so every denoising step
            # ticks the bar and users see why the render feels slow.
            per_prompt = 1.0 / max(1, prompts_total)
            frac_in_prompt = min(1.0, step_idx / max(1, steps_total))
            overall = prompt_idx * per_prompt + frac_in_prompt * per_prompt
            target_idx = missing_indices[prompt_idx]
            report(
                0.1 + 0.85 * overall,
                f"SDXL keyframe {target_idx + 1}/{total}"
                f" (step {step_idx}/{steps_total})",
            )

        def _on_image(i: int, img: Image.Image) -> None:
            target_idx = missing_indices[i]
            _atomic_write_png(img, _keyframe_path(self._cache_dir, target_idx))
            frames[target_idx] = _resize_rgb(img, self._width, self._height)
            report(
                0.1 + 0.85 * (i + 1) / max(1, missing_total),
                f"Keyframe {target_idx + 1}/{total} saved",
            )

        # Use target-index-derived seeds so resumed keyframes match a fresh
        # full run byte-for-byte (important when verifying cache behaviour).
        seeds: list[int | None] = (
            [self._seed + idx for idx in missing_indices]
            if self._seed is not None
            else [None] * missing_total
        )

        report(
            0.1,
            f"Generating {missing_total} SDXL keyframes (batch=1, {self._num_inference_steps} steps each)",
        )
        _generate_images(
            self._pipe,
            missing_prompts,
            negative_prompt=self._negative_prompt,
            gen_width=self._gen_width,
            gen_height=self._gen_height,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            seeds=seeds,
            on_image=_on_image,
            step_callback=_on_step,
        )

        if any(f is None for f in frames):
            missing_after = [i for i, f in enumerate(frames) if f is None]
            raise RuntimeError(
                f"SDXL generation left {len(missing_after)} keyframes missing: {missing_after}"
            )

        _atomic_write_json(manifest.to_dict(), _manifest_path(self._cache_dir))
        report(1.0, "Background keyframes ready")
        return [f for f in frames if f is not None]  # type: ignore[misc]

    # -- frame API ----------------------------------------------------------

    def background_frame(self, t: float) -> np.ndarray:
        """
        Return the interpolated RGB background frame at time ``t`` seconds.

        ``t`` is clamped to ``[0, duration_sec]`` so callers can feed the song
        clock straight through without bounds checks. Output is ``(H, W, 3)``
        ``uint8`` in top-left origin.
        """
        if self._closed:
            raise RuntimeError("BackgroundStills has been closed")
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        if self._frames is None or self._manifest is None:
            raise RuntimeError(
                "background_frame called before ensure_keyframes(); "
                "generate or load keyframes first"
            )
        times_src = (
            self._timeline_times
            if self._timeline_times is not None
            else self._manifest.keyframe_times
        )
        base = _interpolate_frame(
            self._frames,
            times_src,
            float(t),
        )
        if not self._ken_burns:
            return base
        if self._kb_analysis is None:
            raise RuntimeError(
                "SDXL Ken Burns enabled but analysis state is missing; "
                "call ensure_keyframes() first"
            )
        from pipeline.background_kenburns import apply_ken_burns_to_rgb_array

        return apply_ken_burns_to_rgb_array(
            base,
            width=self._width,
            height=self._height,
            margin=self._ken_burns_margin,
            t=float(t),
            duration_sec=self._kb_duration,
            analysis=self._kb_analysis,
        )

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the SDXL pipeline and frame memory. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._frames = None
        self._timeline_times = None
        self._kb_analysis = None
        pipe = self._pipe
        self._pipe = None
        if pipe is None:
            return
        try:
            import torch  # type: ignore
        except Exception:  # noqa: BLE001
            return
        try:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Ignoring SDXL release error: %s", exc)

    def __enter__(self) -> "BackgroundStills":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer
        try:
            self.close()
        except Exception:
            pass


__all__: Sequence[str] = [
    "BACKGROUND_DIRNAME",
    "BackgroundManifest",
    "BackgroundStills",
    "DEFAULT_GEN_HEIGHT",
    "DEFAULT_GEN_WIDTH",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_HEIGHT",
    "DEFAULT_KEYFRAME_INTERVAL",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_WIDTH",
    "KEYFRAME_FILENAME_FMT",
    "KeyframePlan",
    "MANIFEST_FILENAME",
    "RIFE_MANIFEST_SCHEMA_VERSION",
    "RIFE_TIMELINE_DIRNAME",
    "RifeMorphManifest",
    "plan_keyframes",
    "prompt_hash",
]
