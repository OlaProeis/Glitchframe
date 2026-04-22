"""
Per-segment AnimateDiff loops (SDXL) with cached PNG frames and segment crossfades.

Uses :class:`diffusers.AnimateDiffSDXLPipeline` with a motion adapter when
available. If diffusers does not provide SDXL AnimateDiff classes, a clear
:class:`RuntimeError` is raised (no silent fallback to other modes).

Frames are written as ``anim_{seg:03d}_{f:04d}.png`` under
``cache/<song_hash>/background/`` with ``manifest_animatediff.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from config import MODEL_CACHE_DIR
from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.background_stills import (
    BACKGROUND_DIRNAME,
    DEFAULT_NEGATIVE_PROMPT,
    _atomic_write_png,
    _duration_from_analysis,
    _load_analysis,
    _resize_rgb,
    _segment_index_for_time,
    _segments_from_analysis,
    _smoothstep,
    _crossfade,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest_animatediff.json"
ANIM_FILENAME_FMT = "anim_{seg:03d}_{f:04d}.png"
# v2: motion-language prompts (no more "scene N of M / t=X.Xs" structural
# hints) + per-preset ``MOTION_FLAVORS`` + raised inference steps +
# motion-specific negative prompt addendum. Existing v1 caches are ignored
# and regenerated on first run after upgrade.
MANIFEST_SCHEMA_VERSION = 2

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-sdxl-beta"
DEFAULT_GEN_WIDTH = 1024
DEFAULT_GEN_HEIGHT = 1024
DEFAULT_NUM_FRAMES = 16
DEFAULT_GUIDANCE_SCALE = 7.5
# AnimateDiff benefits from more denoising steps than plain SDXL stills;
# at 28 steps temporal attention often produces soft/ghosty frames, 35 is
# the sweet spot for the SDXL-beta motion adapter.
DEFAULT_NUM_INFERENCE_STEPS = 35
SEGMENT_CROSSFADE_HALF = 0.35
COMPOSITION_FPS = 30.0

# Motion-specific additions to the negative prompt. The stills default
# already covers "low quality / text / watermark / deformed"; here we add
# failure modes that only matter for video loops.
ANIMATEDIFF_NEGATIVE_PROMPT = (
    DEFAULT_NEGATIVE_PROMPT
    + ", static frame, frozen motion, stutter, duplicate frames,"
    + " jerky camera, hard cut, scene cut, flickering, morphing shapes,"
    + " distorted proportions, rolling shutter"
)

# Per-preset motion language. AnimateDiff responds to "camera moves /
# elements drift" phrasing far better than structural scene hints, so we
# describe the *kind* of motion we want rather than the song position.
MOTION_FLAVORS: Mapping[str, str] = {
    "cosmic": (
        "slow cosmic drift, subtle parallax between dust layers,"
        " shimmering distant stars, ethereal floating motion"
    ),
    "glitch-vhs": (
        "subtle handheld shake, analog tape tracking drift,"
        " occasional jittery bursts, camera mostly steady"
    ),
    "lofi-warm": (
        "almost still, gentle breathing motion, drifting dust motes,"
        " soft warm ambient sway"
    ),
    "minimal-mono": (
        "very slow push-in, calm deliberate camera, subtle light shifts,"
        " meditative pacing"
    ),
    "neon-synthwave": (
        "slow forward drive, gentle horizon sway,"
        " distant city parallax, cinematic retrowave motion"
    ),
    "organic-liquid": (
        "smooth fluid flow, slow swirling motion,"
        " liquid ink drifting, macro close-up wobble"
    ),
}

DEFAULT_MOTION_FLAVOR = (
    "slow cinematic motion, gentle parallax, smooth continuous movement,"
    " stable framing"
)


def _pacing_cue(index: int, total: int) -> str:
    """Return a short pacing modifier that varies by song position.

    AnimateDiff likes a single concise intent per loop, but completely
    identical prompts across every segment make long songs feel static, so
    we pick one of three pacing words based on whether the segment sits in
    the opening quartile, middle half, or closing quartile.
    """
    if total <= 1:
        return "steady motion"
    rel = (float(index) + 0.5) / float(total)
    if rel < 0.25:
        return "establishing shot, slow motion"
    if rel > 0.75:
        return "slower fade-out motion"
    return "steady motion"


def _build_motion_prompt(
    preset_prompt: str,
    *,
    preset_id: str,
    index: int,
    total: int,
) -> str:
    """Build an AnimateDiff prompt focused on *motion*, not scene structure.

    The keyframe-style builder used for SDXL stills appends "scene N of M /
    t=X.Xs" hints which help diversify still keyframes but hurt AnimateDiff
    — the motion adapter treats them as content and often produces soft,
    off-topic frames. Here we skip the structural metadata and instead
    attach a preset-specific motion flavor plus a single pacing cue.
    """
    base = (preset_prompt or "").strip()
    flavor = MOTION_FLAVORS.get(preset_id, DEFAULT_MOTION_FLAVOR).strip()
    pacing = _pacing_cue(index, total)
    tail = "cinematic, high detail, coherent frames"
    parts = [p for p in (base, flavor, pacing, tail) if p]
    return ", ".join(parts)

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class AnimateDiffManifest:
    schema_version: int
    mode: str
    preset_id: str
    prompt_hash: str
    section_count: int
    num_frames: int
    duration_sec: float
    model_id: str
    motion_adapter_id: str
    gen_width: int
    gen_height: int
    out_width: int
    out_height: int
    segment_starts: tuple[float, ...]
    segment_ends: tuple[float, ...]
    segment_labels: tuple[int, ...]
    prompts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "mode": str(self.mode),
            "preset_id": str(self.preset_id),
            "prompt_hash": str(self.prompt_hash),
            "section_count": int(self.section_count),
            "num_frames": int(self.num_frames),
            "duration_sec": float(self.duration_sec),
            "model_id": str(self.model_id),
            "motion_adapter_id": str(self.motion_adapter_id),
            "gen_width": int(self.gen_width),
            "gen_height": int(self.gen_height),
            "out_width": int(self.out_width),
            "out_height": int(self.out_height),
            "segment_starts": [float(x) for x in self.segment_starts],
            "segment_ends": [float(x) for x in self.segment_ends],
            "segment_labels": [int(x) for x in self.segment_labels],
            "prompts": list(self.prompts),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AnimateDiffManifest":
        try:
            return cls(
                schema_version=int(raw["schema_version"]),
                mode=str(raw["mode"]),
                preset_id=str(raw["preset_id"]),
                prompt_hash=str(raw["prompt_hash"]),
                section_count=int(raw["section_count"]),
                num_frames=int(raw["num_frames"]),
                duration_sec=float(raw["duration_sec"]),
                model_id=str(raw["model_id"]),
                motion_adapter_id=str(raw["motion_adapter_id"]),
                gen_width=int(raw["gen_width"]),
                gen_height=int(raw["gen_height"]),
                out_width=int(raw["out_width"]),
                out_height=int(raw["out_height"]),
                segment_starts=tuple(float(x) for x in raw["segment_starts"]),
                segment_ends=tuple(float(x) for x in raw["segment_ends"]),
                segment_labels=tuple(int(x) for x in raw["segment_labels"]),
                prompts=tuple(str(p) for p in raw.get("prompts", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid AnimateDiff manifest: {exc}") from exc

    def matches_key(
        self,
        *,
        preset_id: str,
        prompt_hash: str,
        section_count: int,
        num_frames: int,
        model_id: str,
        motion_adapter_id: str,
        gen_width: int,
        gen_height: int,
        out_width: int,
        out_height: int,
        duration_sec: float,
    ) -> bool:
        return (
            self.schema_version == MANIFEST_SCHEMA_VERSION
            and self.mode == "animatediff"
            and self.preset_id == preset_id
            and self.prompt_hash == prompt_hash
            and self.section_count == int(section_count)
            and self.num_frames == int(num_frames)
            and self.model_id == model_id
            and self.motion_adapter_id == motion_adapter_id
            and self.gen_width == int(gen_width)
            and self.gen_height == int(gen_height)
            and self.out_width == int(out_width)
            and self.out_height == int(out_height)
            and abs(self.duration_sec - float(duration_sec)) < 1e-3
        )


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / BACKGROUND_DIRNAME / MANIFEST_FILENAME


def _anim_path(cache_dir: Path, seg: int, frame: int) -> Path:
    return cache_dir / BACKGROUND_DIRNAME / ANIM_FILENAME_FMT.format(
        seg=seg, f=frame
    )


def _atomic_write_json(data: Mapping[str, Any], dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(dst)


def _prompt_hash_segments(prompts: Sequence[str], **extras: str) -> str:
    h = hashlib.sha256()
    h.update(b"animatediff-v1")
    for k in sorted(extras.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(extras[k]).encode("utf-8"))
    for p in prompts:
        h.update(b"\x00")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _require_cuda() -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for AnimateDiff backgrounds") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available; AnimateDiff background generation requires a CUDA GPU. "
            "If this used to work, pip may have replaced your CUDA PyTorch with a CPU "
            "wheel (e.g. after installing whisperx). Reinstall from the PyTorch CUDA "
            "index, then restore Gradio pins — see comments in requirements.txt."
        )
    return "cuda:0"


def _import_animatediff_sdxl() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "AnimateDiff SDXL requires a recent diffusers install with "
            "AnimateDiffSDXLPipeline and DDIMScheduler "
            "(e.g. pip install -U diffusers). "
            f"Import failed: {exc}"
        ) from exc
    try:
        from diffusers.models import MotionAdapter  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            from diffusers import MotionAdapter  # type: ignore
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                "MotionAdapter could not be imported from diffusers; "
                f"upgrade diffusers. Import failed: {exc2}"
            ) from exc2
    return torch, MotionAdapter, (AnimateDiffSDXLPipeline, DDIMScheduler)


def _load_pipe(
    model_id: str,
    motion_adapter_id: str,
    *,
    cache_dir: Path,
    device: str,
) -> Any:
    torch, MotionAdapter, pair = _import_animatediff_sdxl()
    AnimateDiffSDXLPipeline, DDIMScheduler = pair
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        adapter = MotionAdapter.from_pretrained(
            motion_adapter_id,
            torch_dtype=torch.float16,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load motion adapter {motion_adapter_id!r}: {exc}"
        ) from exc

    # Force-cast the adapter right after load. In some ``diffusers`` releases
    # ``MotionAdapter.from_pretrained`` silently ignores ``torch_dtype`` for
    # a subset of its parameters / buffers (particularly the temporal
    # positional embedding), so the attached adapter ends up mixed-precision
    # when the pipeline is later cast to fp16. A direct ``.half()`` visits
    # every registered parameter *and* buffer.
    try:
        adapter = adapter.half()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("MotionAdapter.half() after load failed: %s", exc)

    try:
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load DDIM scheduler for {model_id!r}: {exc}"
        ) from exc

    try:
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            model_id,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.info(
            "AnimateDiff SDXL load without fp16 variant (retry after: %s)", exc
        )
        try:
            pipe = AnimateDiffSDXLPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                cache_dir=str(cache_dir),
            )
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load AnimateDiffSDXLPipeline for {model_id!r}: {exc2}"
            ) from exc2

    try:
        pipe = pipe.to(device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to move AnimateDiff pipeline to {device!r}: {exc}"
        ) from exc

    # Force-unify dtype across every submodule.
    #
    # When the fp16 variant isn't published on the hub and we fall back to
    # loading fp32 weights with ``torch_dtype=torch.float16``, diffusers
    # occasionally leaves a submodule in fp32 — most commonly the VAE (which
    # is numerically unstable in fp16 and is sometimes kept in fp32 by
    # default) or the MotionAdapter cross-attention when ``low_cpu_mem_usage``
    # is on. At inference time the UNet expects Half but receives Float and
    # raises "expected scalar type Half but found Float".
    #
    # ``pipe.to(dtype=...)`` traverses all registered submodules; the per-
    # module fallback covers older diffusers releases whose ``.to()`` signature
    # doesn't accept dtype as a kwarg.
    try:
        pipe.to(dtype=torch.float16)
    except TypeError:
        for name in (
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "motion_adapter",
            "image_encoder",
            "controlnet",
        ):
            mod = getattr(pipe, name, None)
            if mod is None:
                continue
            half = getattr(mod, "half", None)
            if callable(half):
                try:
                    half()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("AnimateDiff %s.half() failed: %s", name, exc)

    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(pipe, "enable_vae_tiling"):
        try:
            pipe.enable_vae_tiling()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(pipe, "set_progress_bar_config"):
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:  # noqa: BLE001
            pass
    return pipe


def _generate_segment_frames(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    gen_width: int,
    gen_height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
) -> list[Image.Image]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch not available") from exc

    generator: Any = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Autocast shields us from the remaining fp32 pockets inside the
    # pipeline — diffusers computes some time/positional embeddings and
    # MotionAdapter buffers in fp32 regardless of ``torch_dtype`` or
    # ``pipe.to(dtype=...)``, so feeding them into the fp16 UNet raises
    # "expected scalar type Half but found Float". ``torch.autocast`` casts
    # mismatched inputs to fp16 at op boundaries, which is the canonical
    # workaround and the pattern used by Hugging Face's own AnimateDiff
    # SDXL examples.
    autocast_ctx: Any
    if hasattr(torch, "autocast"):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:  # pragma: no cover - torch <1.10 doesn't exist in our pin range
        from contextlib import nullcontext

        autocast_ctx = nullcontext()

    try:
        with torch.inference_mode(), autocast_ctx:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(gen_width),
                height=int(gen_height),
                num_frames=int(num_frames),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            "CUDA out of memory during AnimateDiff generation; "
            "reduce gen_width/gen_height, num_frames, or free VRAM"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"AnimateDiff generation failed: {exc}") from exc

    frames = getattr(out, "frames", None)
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("AnimateDiff pipeline returned no frames")
    first = frames[0]
    if not isinstance(first, list) or len(first) != int(num_frames):
        raise RuntimeError(
            f"Expected {num_frames} AnimateDiff frames, got "
            f"{len(first) if isinstance(first, list) else type(first)}"
        )
    return list(first)


def _loop_sample(
    frames: Sequence[np.ndarray], t_local: float, *, loop_duration: float
) -> np.ndarray:
    n = len(frames)
    if n == 0:
        raise ValueError("no frames")
    if loop_duration <= 1e-9:
        return frames[0].copy()
    phase = (t_local % loop_duration) / loop_duration
    idx_f = phase * n
    i0 = int(idx_f) % n
    i1 = (i0 + 1) % n
    frac = idx_f - int(idx_f)
    if i0 == n - 1 and i1 == 0:
        pass
    return _crossfade(frames[i0], frames[i1], frac)


class AnimateDiffBackground:
    """
    SDXL + AnimateDiff motion segments; see module docstring for cache layout.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        preset_id: str,
        preset_prompt: str,
        width: int = 1920,
        height: int = 1080,
        gen_width: int = DEFAULT_GEN_WIDTH,
        gen_height: int = DEFAULT_GEN_HEIGHT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        model_id: str = DEFAULT_MODEL_ID,
        motion_adapter_id: str = DEFAULT_MOTION_ADAPTER_ID,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        negative_prompt: str = ANIMATEDIFF_NEGATIVE_PROMPT,
        seed: int | None = 0,
        model_cache_dir: Path | None = None,
        composition_fps: float = COMPOSITION_FPS,
        crossfade_half: float = SEGMENT_CROSSFADE_HALF,
    ) -> None:
        cache = Path(cache_dir)
        if not cache.is_dir():
            raise FileNotFoundError(f"Cache dir does not exist: {cache}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution: {width}x{height}")
        if gen_width % 8 != 0 or gen_height % 8 != 0:
            raise ValueError(
                f"AnimateDiff gen size must be divisible by 8, got {gen_width}x{gen_height}"
            )
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        if not preset_id.strip():
            raise ValueError("preset_id must be a non-empty string")

        self._cache_dir = cache
        self._preset_id = preset_id.strip()
        self._preset_prompt = preset_prompt or ""
        self._width = int(width)
        self._height = int(height)
        self._gen_width = int(gen_width)
        self._gen_height = int(gen_height)
        self._num_frames = int(num_frames)
        self._model_id = str(model_id)
        self._motion_adapter_id = str(motion_adapter_id)
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._negative_prompt = str(negative_prompt)
        self._seed = None if seed is None else int(seed)
        self._model_cache_dir = (
            Path(model_cache_dir) if model_cache_dir is not None else MODEL_CACHE_DIR
        )
        self._composition_fps = float(composition_fps)
        self._crossfade_half = float(crossfade_half)

        self._segments: list[dict[str, Any]] = []
        self._segment_frames: list[list[np.ndarray]] = []
        self._manifest: AnimateDiffManifest | None = None
        self._duration = 0.0
        self._loop_duration = float(num_frames) / float(composition_fps)
        self._pipe: Any = None
        self._closed = False

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def manifest(self) -> AnimateDiffManifest | None:
        return self._manifest

    def _build_expected_manifest(
        self, analysis: Mapping[str, Any], prompts: Sequence[str]
    ) -> AnimateDiffManifest:
        duration = _duration_from_analysis(analysis)
        segments = _segments_from_analysis(analysis, duration)
        ph = _prompt_hash_segments(
            prompts,
            model_id=self._model_id,
            motion_adapter_id=self._motion_adapter_id,
        )
        return AnimateDiffManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            mode="animatediff",
            preset_id=self._preset_id,
            prompt_hash=ph,
            section_count=len(segments),
            num_frames=self._num_frames,
            duration_sec=duration,
            model_id=self._model_id,
            motion_adapter_id=self._motion_adapter_id,
            gen_width=self._gen_width,
            gen_height=self._gen_height,
            out_width=self._width,
            out_height=self._height,
            segment_starts=tuple(float(s["t_start"]) for s in segments),
            segment_ends=tuple(float(s["t_end"]) for s in segments),
            segment_labels=tuple(int(s.get("label", i)) for i, s in enumerate(segments)),
            prompts=tuple(prompts),
        )

    def _segment_prompts(
        self, segments: Sequence[Mapping[str, Any]], duration: float
    ) -> list[str]:
        del duration  # kept for API stability; pacing now derives from index
        n = len(segments)
        out: list[str] = []
        for i in range(n):
            out.append(
                _build_motion_prompt(
                    self._preset_prompt,
                    preset_id=self._preset_id,
                    index=i,
                    total=n,
                )
            )
        return out

    def _try_load_disk(self, manifest: AnimateDiffManifest) -> bool:
        buf: list[list[np.ndarray]] = []
        for s in range(manifest.section_count):
            row: list[np.ndarray] = []
            for f in range(manifest.num_frames):
                p = _anim_path(self._cache_dir, s, f)
                if not p.is_file():
                    return False
                with Image.open(p) as im:
                    row.append(
                        _resize_rgb(im, self._width, self._height)
                    )
            buf.append(row)
        self._segment_frames = buf
        self._manifest = manifest
        return True

    def ensure(
        self,
        *,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> AnimateDiffManifest:
        if self._closed:
            raise RuntimeError("AnimateDiffBackground has been closed")

        def _report(p: float, msg: str) -> None:
            if progress is not None:
                progress(max(0.0, min(1.0, p)), msg)

        _report(0.0, "Loading analysis…")
        analysis = _load_analysis(self._cache_dir / ANALYSIS_JSON_NAME)
        duration = _duration_from_analysis(analysis)
        segments = _segments_from_analysis(analysis, duration)
        self._segments = list(segments)
        self._duration = duration
        prompts = self._segment_prompts(segments, duration)
        expected = self._build_expected_manifest(analysis, prompts)

        bg_dir = self._cache_dir / BACKGROUND_DIRNAME
        bg_dir.mkdir(parents=True, exist_ok=True)

        if not force:
            mp = _manifest_path(self._cache_dir)
            if mp.is_file():
                try:
                    with mp.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                    existing = AnimateDiffManifest.from_dict(raw)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.info("Ignoring bad AnimateDiff manifest: %s", exc)
                    existing = None
                if existing is not None and existing.matches_key(
                    preset_id=expected.preset_id,
                    prompt_hash=expected.prompt_hash,
                    section_count=expected.section_count,
                    num_frames=expected.num_frames,
                    model_id=expected.model_id,
                    motion_adapter_id=expected.motion_adapter_id,
                    gen_width=expected.gen_width,
                    gen_height=expected.gen_height,
                    out_width=expected.out_width,
                    out_height=expected.out_height,
                    duration_sec=expected.duration_sec,
                ):
                    if self._try_load_disk(existing):
                        _report(1.0, "Reused cached AnimateDiff segments")
                        return existing

        device = _require_cuda()
        _report(0.05, "Loading AnimateDiff SDXL…")
        if self._pipe is None:
            self._pipe = _load_pipe(
                self._model_id,
                self._motion_adapter_id,
                cache_dir=self._model_cache_dir,
                device=device,
            )

        self._segment_frames = []
        total = max(1, expected.section_count)
        for s in range(expected.section_count):
            _report(
                0.1 + 0.85 * (s / total),
                f"AnimateDiff segment {s + 1}/{total}…",
            )
            images = _generate_segment_frames(
                self._pipe,
                prompt=prompts[s],
                negative_prompt=self._negative_prompt,
                gen_width=self._gen_width,
                gen_height=self._gen_height,
                num_frames=self._num_frames,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                seed=self._seed,
            )
            row: list[np.ndarray] = []
            for f, img in enumerate(images):
                arr = _resize_rgb(img, self._width, self._height)
                _atomic_write_png(img, _anim_path(self._cache_dir, s, f))
                row.append(arr)
            self._segment_frames.append(row)

        _atomic_write_json(expected.to_dict(), _manifest_path(self._cache_dir))
        self._manifest = expected
        self._release_generation_pipeline()
        _report(1.0, "AnimateDiff backgrounds ready")
        return expected

    def _frame_for_segment(self, seg_idx: int, t: float) -> np.ndarray:
        if seg_idx < 0 or seg_idx >= len(self._segment_frames):
            raise IndexError(f"segment index out of range: {seg_idx}")
        segs = self._segments
        t0 = float(segs[seg_idx]["t_start"])
        t1 = float(segs[seg_idx]["t_end"])
        local = max(0.0, t - t0)
        span = max(1e-6, t1 - t0)
        local = float(np.clip(local, 0.0, span))
        return _loop_sample(
            self._segment_frames[seg_idx],
            local,
            loop_duration=self._loop_duration,
        )

    def _release_generation_pipeline(self) -> None:
        """Unload SDXL+AnimateDiff from VRAM; compositing only needs ``_segment_frames``."""
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
            LOGGER.debug("AnimateDiff pipeline release: %s", exc)

    def background_frame(self, t: float) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AnimateDiffBackground has been closed")
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        if not self._segment_frames or not self._segments:
            raise RuntimeError(
                "background_frame called before ensure(); generate segments first"
            )
        t_eff = float(np.clip(t, 0.0, self._duration))
        idx = _segment_index_for_time(self._segments, t_eff)
        fh = self._crossfade_half
        base = self._frame_for_segment(idx, t_eff)

        if idx < len(self._segments) - 1:
            boundary = float(self._segments[idx]["t_end"])
            if t_eff >= boundary - fh and t_eff <= boundary + fh:
                other = self._frame_for_segment(idx + 1, t_eff)
                denom = max(1e-6, 2.0 * fh)
                alpha = _smoothstep((t_eff - (boundary - fh)) / denom)
                return _crossfade(base, other, alpha)

        if idx > 0:
            boundary_prev = float(self._segments[idx]["t_start"])
            if t_eff <= boundary_prev + fh and t_eff >= boundary_prev - fh:
                other = self._frame_for_segment(idx - 1, t_eff)
                denom = max(1e-6, 2.0 * fh)
                alpha = _smoothstep((t_eff - (boundary_prev - fh)) / denom)
                return _crossfade(other, base, alpha)

        return base

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._segment_frames = []
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
            LOGGER.debug("AnimateDiff release error: %s", exc)

    def __enter__(self) -> "AnimateDiffBackground":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "ANIMATEDIFF_NEGATIVE_PROMPT",
    "AnimateDiffBackground",
    "AnimateDiffManifest",
    "DEFAULT_MOTION_FLAVOR",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "MOTION_FLAVORS",
]
