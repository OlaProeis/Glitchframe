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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    # Imported lazily to avoid the circular dependency
    # ``pipeline.background`` -> ``pipeline.background_animatediff`` ->
    # ``pipeline.background``. Only used for type hints below.
    from pipeline.background import BackgroundSource

from config import MODEL_CACHE_DIR
from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.background_stills import (
    BACKGROUND_DIRNAME,
    DEFAULT_NEGATIVE_PROMPT,
    KEYFRAME_FILENAME_FMT as STILLS_KEYFRAME_FILENAME_FMT,
    MANIFEST_FILENAME as STILLS_MANIFEST_FILENAME,
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
# SDXL's stock VAE is numerically unstable in fp16: the (16-frame x latent)
# tensor produced by AnimateDiff overflows fp16 range during decode and
# silently returns NaN -> all-black PNGs. madebyollin's drop-in is the
# community-standard fix (Apache-2.0, ~330 MB) and is fully fp16-safe.
DEFAULT_FP16_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
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
# When conditioning AnimateDiff on an SDXL keyframe we follow Stable Diffusion
# img2img semantics (matches ``StableDiffusionXLPipeline.get_timesteps``):
# ``strength`` maps to how deep into the noise schedule we start — higher =
# more denoising steps / freer motion, lower = closer to the still. Clamped to
# (0.06, 1] because strength≈0 yields an empty timestep slice.
DEFAULT_INIT_IMAGE_STRENGTH = 0.38

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
    "cosmic-flow": (
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


# ---------------------------------------------------------------------------
# SDXL-stills bridge: read keyframe times + PNGs from disk so AnimateDiff can
# use them as init latents without holding a reference to the (now closed)
# BackgroundStills source.
# ---------------------------------------------------------------------------


def _stills_manifest_path(cache_dir: Path) -> Path:
    """Path to the SDXL-stills manifest under ``cache_dir/background/``."""
    return cache_dir / BACKGROUND_DIRNAME / STILLS_MANIFEST_FILENAME


def _stills_keyframe_path(cache_dir: Path, index: int) -> Path:
    return cache_dir / BACKGROUND_DIRNAME / STILLS_KEYFRAME_FILENAME_FMT.format(
        index=index
    )


def _read_stills_keyframe_times(cache_dir: Path) -> tuple[float, ...]:
    """Read ``keyframe_times`` from the SDXL-stills manifest if it exists.

    Returns an empty tuple when the manifest is missing or malformed -- the
    caller treats that as "no init images available" and AnimateDiff runs
    in pure text-to-video mode.
    """
    p = _stills_manifest_path(cache_dir)
    if not p.is_file():
        return ()
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read SDXL stills manifest at %s: %s", p, exc)
        return ()
    times = raw.get("keyframe_times")
    if not isinstance(times, list) or not times:
        return ()
    try:
        return tuple(float(t) for t in times)
    except (TypeError, ValueError):
        return ()


def _pick_stills_keyframe_index(
    times: Sequence[float], target_t: float
) -> int | None:
    """Return the index of the keyframe whose time is closest to ``target_t``.

    Returns ``None`` when ``times`` is empty.
    """
    if not times:
        return None
    return int(min(range(len(times)), key=lambda i: abs(times[i] - target_t)))


def _load_stills_keyframe_pil(
    cache_dir: Path,
    index: int,
    *,
    target_size: tuple[int, int],
) -> "Image.Image | None":
    """Load the SDXL keyframe PNG at ``index`` and resize to ``target_size``.

    The returned PIL image is in RGB and matches the AnimateDiff generation
    resolution so its VAE encoding produces the expected latent shape. The
    function never raises on a missing file (returns ``None``) so the caller
    can fall back to text-to-video gracefully.
    """
    path = _stills_keyframe_path(cache_dir, index)
    if not path.is_file():
        return None
    try:
        from PIL import Image as _PILImage  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        img = _PILImage.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load SDXL keyframe %s: %s", path, exc)
        return None
    if img.size != target_size:
        img = img.resize(target_size, _PILImage.LANCZOS)
    return img


def _prompt_2_for_index(prompts: Sequence[str], index: int) -> str:
    """Return the prompt to feed SDXL's *second* text encoder for segment ``index``.

    For all segments except the last, this is the *next* segment's prompt --
    SDXL's dual-encoder design lets the joint conditioning blend the two,
    which gently pulls each AnimateDiff loop toward what comes next and
    softens cross-segment transitions. The last segment has no "next", so
    its ``prompt_2`` mirrors its own prompt (no morph beyond the song end).
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if index < 0 or index >= len(prompts):
        raise IndexError(f"prompt index {index} out of range 0..{len(prompts) - 1}")
    if index + 1 < len(prompts):
        return prompts[index + 1]
    return prompts[index]


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


def _load_fp16_safe_vae(
    vae_id: str,
    *,
    cache_dir: Path,
    torch_module: Any,
) -> Any:
    """Load the community fp16-safe SDXL VAE; raises with a clear message on failure.

    The default SDXL VAE produces NaN latents in fp16 once the (16-frame x
    latent) tensor passes through decode, which surfaces as all-black output
    PNGs -- the exact failure mode the AnimateDiff mode was tagged broken for.
    Replacing ``pipe.vae`` with ``madebyollin/sdxl-vae-fp16-fix`` resolves it
    without touching dtype anywhere else.
    """
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "diffusers must be installed to load the fp16-safe SDXL VAE"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=torch_module.float16,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load fp16-safe SDXL VAE {vae_id!r}: {exc}. "
            "If you're offline, pre-cache it once with internet access; "
            "the AnimateDiff mode requires it to avoid black-frame output."
        ) from exc


def _import_animatediff_sdxl() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore

        # Track A pins torch 2.2.2 — predates ``torch.xpu`` (2.3) and
        # ``torch.distributed.device_mesh`` (2.4). Newer diffusers /
        # transformers reference both at import time without ``hasattr()``
        # guards. Apply our compat stubs before the diffusers import so
        # AnimateDiff SDXL can load on Track A. No-op on Track B (torch >=
        # 2.4) and idempotent if app startup already applied them.
        try:
            from pipeline._torch_xpu_compat import patch_all as _patch_all_torch_compat

            _patch_all_torch_compat()
        except Exception:  # noqa: BLE001
            pass

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
    vae_id: str = DEFAULT_FP16_VAE_ID,
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

    # Swap in the fp16-safe SDXL VAE before moving to GPU so the device /
    # dtype unification below applies to it uniformly. The stock SDXL VAE
    # silently NaNs in fp16 during AnimateDiff decode (all-black output);
    # see ``_load_fp16_safe_vae`` docstring for details.
    safe_vae = _load_fp16_safe_vae(
        vae_id, cache_dir=cache_dir, torch_module=torch
    )
    pipe.vae = safe_vae

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
    # occasionally leaves a submodule in fp32 (most commonly MotionAdapter
    # cross-attention when ``low_cpu_mem_usage`` is on). At inference time
    # the UNet expects Half but receives Float and raises "expected scalar
    # type Half but found Float".
    #
    # The VAE was historically the worst offender here, but we now load
    # ``madebyollin/sdxl-vae-fp16-fix`` explicitly above so its dtype is
    # already fp16 -- this pass is purely defensive for the rest of the
    # pipeline.
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


def _vae_encode_tiled_keyframe_latents(
    pipe: Any,
    init_image: "Image.Image",
    *,
    num_frames: int,
    gen_width: int,
    gen_height: int,
    torch_module: Any,
) -> Any:
    """Deterministic VAE encode of ``init_image`` → scaled latent tiled over ``num_frames``.

    Shape ``(1, 4, num_frames, H/8, W/8)``. Uses ``latent_dist.mode()`` like SDXL
    img2img (stable encode vs ``sample()`` RNG jitter).
    """
    target_size = (int(gen_width), int(gen_height))
    if init_image.size != target_size:
        init_image = init_image.resize(target_size, Image.LANCZOS)

    arr = np.asarray(init_image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) * 2.0
    tensor = torch_module.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    device = getattr(pipe, "_execution_device", None) or getattr(
        pipe, "device", "cuda"
    )
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    tensor = tensor.to(device=device, dtype=vae_dtype)

    with torch_module.no_grad():
        dist = vae.encode(tensor).latent_dist
        enc_mode = getattr(dist, "mode", None)
        encoded = enc_mode() if callable(enc_mode) else dist.sample()
        scaling = float(getattr(vae.config, "scaling_factor", 0.13025))
        encoded = encoded * scaling

    latents = encoded.unsqueeze(2).repeat(1, 1, int(num_frames), 1, 1)
    return latents.to(dtype=torch_module.float16)


def _animatediff_img2img_generate(
    pipe: Any,
    *,
    prompt: str,
    prompt_2: str | None,
    negative_prompt: str,
    negative_prompt_2: str | None,
    gen_width: int,
    gen_height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    init_image: "Image.Image",
    init_strength: float,
    torch_module: Any,
) -> list["Image.Image"]:
    """Run AnimateDiff SDXL with true img2img init (SDXL-still → temporally denoise).

    Passing ``latents=`` into ``AnimateDiffSDXLPipeline.__call__`` is **not**
    img2img: ``prepare_latents`` assumes Gaussian noise (× ``init_noise_sigma``,
    which is ``1.0`` for DDIM) and denoises from timestep *zero* of the full
    schedule. Feeding clean VAE latents there makes the UNet see wrong statistics,
    which produces abstract garbage unrelated to the SDXL still.

    This path mirrors ``StableDiffusionXLImg2ImgPipeline``: slice the timestep
    tensor by ``strength``, ``scheduler.add_noise`` the encoded still at the
    first kept timestep, divide by ``init_noise_sigma`` so ``prepare_latents``
    restores ``add_noise`` output, then run the denoise loop on the sliced
    schedule (copied from ``AnimateDiffSDXLPipeline.__call__``, minus IP-Adapter /
    FreeInit / callbacks).
    """
    from diffusers.pipelines.animatediff.pipeline_animatediff_sdxl import (
        rescale_noise_cfg,
    )
    from diffusers.utils.torch_utils import randn_tensor

    torch = torch_module
    device = pipe._execution_device
    height = int(gen_height)
    width = int(gen_width)
    num_inference_steps = int(num_inference_steps)
    eta = 0.0
    guidance_rescale = 0.0
    output_type = "pil"

    pipe.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    )

    pipe._guidance_scale = float(guidance_scale)
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = None
    pipe._cross_attention_kwargs = None
    pipe._denoising_end = None
    pipe._interrupt = False

    batch_size = 1
    num_videos_per_prompt = 1

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_videos_per_prompt=num_videos_per_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    full_schedule = scheduler.timesteps.clone()
    order = int(getattr(scheduler, "order", 1))

    strength = float(init_strength)
    strength = float(np.clip(strength, 0.05, 1.0))
    init_timestep = min(
        int(num_inference_steps * strength),
        num_inference_steps,
    )
    init_timestep = max(init_timestep, 1)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = full_schedule[t_start * order :].clone()
    scheduler.timesteps = timesteps
    scheduler.num_inference_steps = len(timesteps)

    z_clean = _vae_encode_tiled_keyframe_latents(
        pipe,
        init_image,
        num_frames=num_frames,
        gen_width=gen_width,
        gen_height=gen_height,
        torch_module=torch,
    )
    noise = randn_tensor(
        z_clean.shape,
        generator=generator,
        device=device,
        dtype=z_clean.dtype,
    )
    first_ts = timesteps[0:1].long().to(device=device)
    z_noisy = scheduler.add_noise(z_clean, noise, first_ts)
    sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))
    latents = z_noisy / sigma

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = int(pipe.text_encoder_2.config.projection_dim)

    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    negative_add_time_ids = add_time_ids

    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.repeat_interleave(repeats=num_frames, dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_videos_per_prompt, 1)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(
            batch_size * num_videos_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    ip_adapter_image = None
    ip_adapter_image_embeds = None

    with pipe.progress_bar(total=len(timesteps)) as progress_bar:
        for _i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue
            latent_model_input = (
                torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                raise RuntimeError("IP-Adapter path not implemented for img2img wrapper")

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if pipe.do_classifier_free_guidance and pipe.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=pipe.guidance_rescale,
                )

            latents = scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]
            progress_bar.update()

    needs_upcasting = pipe.vae.dtype == torch.float16 and getattr(
        pipe.vae.config, "force_upcast", False
    )
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    video_tensor = pipe.decode_latents(latents)
    video = pipe.video_processor.postprocess_video(
        video=video_tensor, output_type=output_type
    )
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    pipe.maybe_free_model_hooks()

    if not isinstance(video, list) or not video:
        raise RuntimeError("AnimateDiff img2img returned no video batch")
    first = video[0]
    if not isinstance(first, list) or len(first) != int(num_frames):
        raise RuntimeError(
            f"Expected {num_frames} frames from img2img path, got "
            f"{len(first) if isinstance(first, list) else type(first)}"
        )
    return list(first)


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
    prompt_2: str | None = None,
    negative_prompt_2: str | None = None,
    init_image: "Image.Image | None" = None,
    init_image_strength: float = DEFAULT_INIT_IMAGE_STRENGTH,
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

    # SDXL still → proper img2img noise injection + sliced schedule (see
    # ``_animatediff_img2img_generate``). Passing raw encoded latents into
    # ``pipe(latents=…)`` does **not** preserve the still (DDIM
    # ``init_noise_sigma`` is 1.0 — the UNet denoises from the wrong state).
    if init_image is not None:
        try:
            with torch.inference_mode(), autocast_ctx:
                return _animatediff_img2img_generate(
                    pipe,
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    gen_width=int(gen_width),
                    gen_height=int(gen_height),
                    num_frames=int(num_frames),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    init_image=init_image,
                    init_strength=float(init_image_strength),
                    torch_module=torch,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "AnimateDiff img2img from SDXL still failed (%s); "
                "falling back to text-to-video for this segment.",
                exc,
            )

    pipe_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": int(gen_width),
        "height": int(gen_height),
        "num_frames": int(num_frames),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
    }
    if prompt_2 is not None and prompt_2 != prompt:
        pipe_kwargs["prompt_2"] = prompt_2
    if negative_prompt_2 is not None and negative_prompt_2 != negative_prompt:
        pipe_kwargs["negative_prompt_2"] = negative_prompt_2

    try:
        with torch.inference_mode(), autocast_ctx:
            out = pipe(**pipe_kwargs)
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
        vae_id: str = DEFAULT_FP16_VAE_ID,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        negative_prompt: str = ANIMATEDIFF_NEGATIVE_PROMPT,
        seed: int | None = 0,
        model_cache_dir: Path | None = None,
        composition_fps: float = COMPOSITION_FPS,
        crossfade_half: float = SEGMENT_CROSSFADE_HALF,
        init_image_source: "BackgroundSource | None" = None,
        init_image_strength: float = DEFAULT_INIT_IMAGE_STRENGTH,
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
        if not 0.05 <= float(init_image_strength) <= 1.0:
            raise ValueError(
                "init_image_strength must be in [0.05, 1] "
                f"(img2img noise fraction), got {init_image_strength}"
            )

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
        self._vae_id = str(vae_id)
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._negative_prompt = str(negative_prompt)
        self._seed = None if seed is None else int(seed)
        self._model_cache_dir = (
            Path(model_cache_dir) if model_cache_dir is not None else MODEL_CACHE_DIR
        )
        self._composition_fps = float(composition_fps)
        self._crossfade_half = float(crossfade_half)
        # SDXL stills source whose keyframe PNGs feed AnimateDiff as init
        # latents (not blended on top). Generation order in ``ensure()``:
        # run stills -> close stills (free SDXL VRAM) -> load AnimateDiff ->
        # for each segment, VAE-encode the closest SDXL keyframe and use its
        # latent as the AnimateDiff init noise. The actual SDXL keyframe is
        # never composited on top of the AnimateDiff output -- AnimateDiff
        # IS the SDXL frame, just animated.
        self._init_image_source: "BackgroundSource | None" = init_image_source
        self._init_image_strength = float(init_image_strength)
        # Cached SDXL keyframe times once the init source is ensured; lets us
        # pick the right keyframe per segment without re-reading disk.
        self._init_keyframe_times: tuple[float, ...] = ()

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
        # Hash each segment's effective conditioning pair so that enabling /
        # disabling cross-segment prompt morph invalidates the cache. The
        # ``|||`` separator is unlikely to appear inside a prompt but is
        # only ever read by the hasher, never by the pipeline.
        hashed_prompts = [
            f"{p}|||{_prompt_2_for_index(prompts, i)}"
            for i, p in enumerate(prompts)
        ]
        # The presence of SDXL init images materially changes every segment's
        # output (each loop is now seeded from a real image instead of pure
        # noise), so it must participate in the cache key. We hash the tuple
        # of keyframe times that will actually be consumed -- swapping a
        # keyframe interval or disabling stills both invalidate cleanly.
        # SDXL-stills img2img: changing strength invalidates AnimateDiff PNGs.
        init_key = (
            "img2img-v1|"
            f"s={self._init_image_strength:.4f}|"
            "t="
            + ",".join(f"{t:.3f}" for t in self._init_keyframe_times)
            if self._init_keyframe_times
            else "init=none"
        )
        ph = _prompt_hash_segments(
            hashed_prompts,
            model_id=self._model_id,
            motion_adapter_id=self._motion_adapter_id,
            vae_id=self._vae_id,
            init_key=init_key,
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

    def _init_image_for_segment(self, seg_idx: int) -> "Image.Image | None":
        """Pick the SDXL keyframe whose time is closest to the segment start.

        Returns ``None`` when no SDXL stills cache is available -- AnimateDiff
        then falls back to plain text-to-video for that segment. The PIL
        image is resized to ``(gen_width, gen_height)`` so its VAE encoding
        matches the AnimateDiff latent shape.
        """
        if not self._init_keyframe_times or not self._segments:
            return None
        if seg_idx < 0 or seg_idx >= len(self._segments):
            return None
        segment = self._segments[seg_idx]
        target_t = float(segment.get("t_start", 0.0))
        idx = _pick_stills_keyframe_index(self._init_keyframe_times, target_t)
        if idx is None:
            return None
        return _load_stills_keyframe_pil(
            self._cache_dir,
            idx,
            target_size=(self._gen_width, self._gen_height),
        )

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

        # When an SDXL init-image source is wired, we generate it first so
        # its keyframe PNGs are on disk by the time AnimateDiff segment
        # generation needs them. We then close the SDXL pipeline so its
        # ~5 GB VRAM is released before the AnimateDiff pipeline loads --
        # avoids OOM on smaller GPUs and is the explicit user contract:
        # "load SDXL, make stills, dump SDXL, load AnimateDiff."
        init_share = 0.4 if self._init_image_source is not None else 0.0
        anim_lo = init_share
        anim_span = 1.0 - init_share

        def _report_anim(p: float, msg: str) -> None:
            if progress is not None:
                clamped = max(0.0, min(1.0, p))
                progress(anim_lo + clamped * anim_span, msg)

        if self._init_image_source is not None:
            def _report_init(p: float, msg: str) -> None:
                if progress is not None:
                    clamped = max(0.0, min(1.0, p))
                    progress(clamped * init_share, f"[stills] {msg}")

            self._init_image_source.ensure(force=force, progress=_report_init)
            # Snapshot keyframe times so we can pick the right one per
            # segment, then close the source to free SDXL VRAM. The PNGs
            # remain on disk under cache_dir/background/keyframe_*.png.
            self._init_keyframe_times = _read_stills_keyframe_times(
                self._cache_dir
            )
            try:
                self._init_image_source.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Init-image source close failed: %s", exc)
            self._init_image_source = None

        _report = _report_anim

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
                vae_id=self._vae_id,
            )

        self._segment_frames = []
        total = max(1, expected.section_count)
        for s in range(expected.section_count):
            _report(
                0.1 + 0.85 * (s / total),
                f"AnimateDiff segment {s + 1}/{total}…",
            )
            prompt_2 = _prompt_2_for_index(prompts, s)
            init_image = self._init_image_for_segment(s)
            images = _generate_segment_frames(
                self._pipe,
                prompt=prompts[s],
                prompt_2=prompt_2,
                negative_prompt=self._negative_prompt,
                gen_width=self._gen_width,
                gen_height=self._gen_height,
                num_frames=self._num_frames,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                seed=self._seed,
                init_image=init_image,
                init_image_strength=self._init_image_strength,
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
        # The init-image source (SDXL stills) is normally closed inside
        # ``ensure()`` once its keyframes are on disk, but if ``ensure`` was
        # never called (or raised early) we still need to release any pipeline
        # it may have loaded.
        init_source = self._init_image_source
        self._init_image_source = None
        if init_source is not None:
            try:
                init_source.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("AnimateDiff init-image source release error: %s", exc)
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
    "DEFAULT_FP16_VAE_ID",
    "DEFAULT_INIT_IMAGE_STRENGTH",
    "DEFAULT_MOTION_FLAVOR",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "MOTION_FLAVORS",
]
