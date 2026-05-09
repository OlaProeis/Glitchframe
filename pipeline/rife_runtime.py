"""Download Practical-RIFE weights and run GPU interpolation between SDXL keyframes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from config import MODEL_CACHE_DIR
from pipeline.rife_vendor.ifnet_hdv3 import IFNet

LOGGER = logging.getLogger(__name__)

DEFAULT_RIFE_REPO = "MonsterMMORPG/RIFE_4_26"
RIFE_WEIGHTS_SUBPATH = "train_log/flownet.pkl"
RIFE_CACHE_SUBDIR = "rife_practical"


ProgressFn = Callable[[float, str], None]


def _rife_use_fp16(device: torch.device) -> bool:
    """Whether to run IFNet in half precision.

    Defaults to ``True`` on CUDA (the 3090-class hardware our SDXL stack
    targets has fast tensor-core FP16 and IFNet's 5-block architecture is
    numerically stable in half precision in inference). Set
    ``GLITCHFRAME_RIFE_FP16=0`` to force FP32 if you ever hit a model with
    saturating activations or want byte-exact reproducibility.
    """
    if device.type != "cuda":
        return False
    flag = os.environ.get("GLITCHFRAME_RIFE_FP16", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return True


def rife_weights_dir(repo_id: str = DEFAULT_RIFE_REPO) -> Path:
    safe = repo_id.replace("/", "--")
    return Path(MODEL_CACHE_DIR) / RIFE_CACHE_SUBDIR / safe


def ensure_rife_flownet_path(
    *,
    repo_id: str = DEFAULT_RIFE_REPO,
    local_files_only: bool = False,
) -> Path:
    """Return path to ``flownet.pkl``, downloading via ``huggingface_hub`` when needed."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=repo_id,
        filename=RIFE_WEIGHTS_SUBPATH,
        local_dir=str(rife_weights_dir(repo_id)),
        local_dir_use_symlinks=False,
        local_files_only=local_files_only,
    )
    return Path(p)


class RIFEInferenceWrapper:
    """Thin wrapper matching Practical-RIFE ``Model.inference`` for IFNet v4.x."""

    def __init__(self, flownet: IFNet, *, fp16: bool = False) -> None:
        self.flownet = flownet
        self.version = 4.25
        self.fp16 = bool(fp16)

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
    ) -> torch.Tensor:
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        _flow, _mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]


def load_rife_model(
    *,
    device: torch.device,
    repo_id: str = DEFAULT_RIFE_REPO,
    local_files_only: bool = False,
    fp16: bool | None = None,
) -> RIFEInferenceWrapper:
    """Load the Practical-RIFE flownet onto ``device``.

    ``fp16`` defaults to :func:`_rife_use_fp16` (auto-on for CUDA). FP16
    cuts inference time roughly in half on tensor-core GPUs and halves the
    model's VRAM footprint with no observable quality loss for IFNet v4.x in
    inference mode (outputs are quantised to uint8 immediately afterwards).
    """
    weight_path = ensure_rife_flownet_path(
        repo_id=repo_id, local_files_only=local_files_only
    )
    use_fp16 = _rife_use_fp16(device) if fp16 is None else bool(fp16)
    net = IFNet()
    try:
        state = torch.load(
            weight_path, map_location=device, weights_only=False
        )
    except TypeError:
        state = torch.load(weight_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected RIFE checkpoint at {weight_path}")
    fixed: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k.replace("module.", "", 1) if k.startswith("module.") else k
        fixed[nk] = v
    net.load_state_dict(fixed, strict=False)
    net.eval()
    net.to(device)
    if use_fp16:
        # ``.half()`` after ``.to(device)`` so any FP32 buffers are converted
        # in place; warplayer's grid cache is keyed on dtype and will create
        # an FP16 grid on first use.
        net.half()
    return RIFEInferenceWrapper(net, fp16=use_fp16)


def _prepare_pair_tensors(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if rgb0.shape != rgb1.shape or rgb0.ndim != 3 or rgb0.shape[2] != 3:
        raise ValueError("Expected matching HxWx3 uint8 RGB arrays")
    t0 = torch.from_numpy(rgb0).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=dtype
    ) / 255.0
    t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=dtype
    ) / 255.0
    _n, _c, h, w = t0.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    pad = (0, pw - w, 0, ph - h)
    t0 = F.pad(t0, pad)
    t1 = F.pad(t1, pad)
    return t0, t1, h, w


def _tensor_to_rgb(out: torch.Tensor, h: int, w: int) -> np.ndarray:
    x = (out[0, :3].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
    return np.ascontiguousarray(x)


def rife_exp_interpolate_pair(
    model: RIFEInferenceWrapper,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    *,
    exp: int,
    device: torch.device,
    include_endpoints: bool = True,
) -> list[np.ndarray]:
    """RIFE subdivision (``2 ** exp``) between ``rgb0`` and ``rgb1``.

    With ``include_endpoints=True`` (default) returns the legacy uniform grid
    ``[rgb0, IFNet(1/n), …, IFNet((n-1)/n), rgb1]`` — ``n + 1`` frames where
    the first and last are the exact source stills.

    With ``include_endpoints=False`` returns ``n`` IFNet predictions sampled
    at the **centered** timesteps ``[(i + 0.5)/n for i in 0..n-1]``. The
    morph timeline builder uses this mode so adjacent segments don't snap to
    the exact SDXL still at every internal keyframe boundary: the flanking
    dense samples are both equally-soft IFNet predictions (one decelerating
    into the keyframe, one accelerating away), so a linear blend at the
    boundary produces a continuous-velocity midpoint instead of a sharp
    pixel-content discontinuity.

    Inference runs inside :func:`torch.inference_mode`, which skips autograd
    bookkeeping (Practical-RIFE upstream does the equivalent via
    ``torch.set_grad_enabled(False)``). Without this flag PyTorch builds a
    no-op autograd graph for every forward pass, costing ~15–20 % extra
    wall-clock and roughly doubling transient VRAM.
    """
    if exp < 1:
        raise ValueError(f"rife exp must be >= 1, got {exp}")
    n = 2**exp
    dtype = torch.float16 if model.fp16 else torch.float32
    t0, t1, h, w = _prepare_pair_tensors(rgb0, rgb1, device, dtype=dtype)
    out_list: list[np.ndarray] = []
    if include_endpoints:
        out_list.append(_tensor_to_rgb(t0, h, w))
        with torch.inference_mode():
            for i in range(n - 1):
                mid = model.inference(t0, t1, (i + 1) * 1.0 / n)
                out_list.append(_tensor_to_rgb(mid, h, w))
        out_list.append(_tensor_to_rgb(t1, h, w))
    else:
        with torch.inference_mode():
            for i in range(n):
                # Centered timestep: i + 0.5 instead of i+1, so samples sit
                # symmetrically inside (0, 1) and never coincide with the
                # exact endpoints.
                ts = (i + 0.5) / n
                mid = model.inference(t0, t1, ts)
                out_list.append(_tensor_to_rgb(mid, h, w))
    return out_list


def rife_build_morph_timeline(
    keyframes: Sequence[np.ndarray],
    keyframe_times: Sequence[float],
    *,
    exp: int,
    device: torch.device,
    repo_id: str = DEFAULT_RIFE_REPO,
    progress: ProgressFn | None = None,
    on_frame: Callable[[int, np.ndarray, float], None] | None = None,
    keep_frames: bool = True,
) -> tuple[list[np.ndarray], list[float]]:
    """Build one continuous-velocity dense RIFE timeline across all keyframes.

    Each segment ``[kf_i, kf_{i+1}]`` is sampled at **centered** timesteps
    ``[(j + 0.5)/n for j in 0..n-1]`` (where ``n = 2**exp``), so the dense
    timeline contains only IFNet predictions internally — never the exact
    SDXL still at an internal keyframe boundary. The exact start (``kf_0``)
    and end (``kf_{n-1}``) stills are kept as the very first and very last
    frames so the song opens/closes on the sharp generated image.

    Why this matters for perceived smoothness: with the legacy "include
    endpoints + dedupe" scheme the timeline read ``soft_ifnet → SHARP_still
    → soft_ifnet`` at every internal keyframe time, producing a brief
    snap-to-still pause that read as a framerate dip even with a uniform
    sample spacing. Centered sampling makes the two flanking dense samples
    around every internal keyframe equally soft IFNet predictions (one
    decelerating into the keyframe, one accelerating away), so a linear
    blend at ``t_kf`` produces a continuous-velocity midpoint instead of a
    sharp pixel-content discontinuity.

    The dense temporal spacing is uniform ``T_seg/n`` both *within* a
    segment and *across* every internal keyframe boundary (the last centered
    sample of segment ``N`` sits at ``t_kf - T_seg/(2n)``, the first of
    segment ``N+1`` at ``t_kf + T_seg/(2n)``). The only spacing irregularity
    is the short ``T_seg/(2n)`` gap between the prepended exact start still
    and the first IFNet sample (and the symmetric one at the end), where
    motion is naturally just the start/end still slightly evolved — visually
    smooth.

    ``on_frame``, if given, is invoked for every output frame as soon as it
    is available with ``(global_index, rgb_uint8, t_sec)``. Callers can use
    this to stream frames straight to disk (or a thread pool of writers)
    without waiting for the whole timeline to finish — important at high
    ``rife_exp`` where the full timeline easily exceeds 30 GB of RAM at
    1080p.

    ``keep_frames`` controls whether the returned ``frames`` list is built.
    Set ``False`` together with ``on_frame`` for zero-copy streaming when
    the caller persists frames externally and reads them back lazily; the
    returned ``times`` list is still complete.
    """
    if len(keyframes) != len(keyframe_times):
        raise ValueError("keyframes and times length mismatch")
    n = len(keyframes)
    if n < 2:
        raise ValueError("RIFE morph needs at least two SDXL keyframes")

    def _report(p: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, p)), msg)

    # IFNet runs the same input shape thousands of times back-to-back, so
    # cudnn's algorithm-search cache pays for itself almost immediately;
    # leaving this off forces cudnn to redo a generic heuristic per call.
    # We only flip the flag while running RIFE and restore the previous
    # value to avoid affecting other CUDA users (SDXL etc.).
    prev_cudnn_benchmark: bool | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            prev_cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
            torch.backends.cudnn.benchmark = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Could not enable cudnn.benchmark: %s", exc)

    _report(0.0, "Loading RIFE model…")
    model = load_rife_model(device=device, repo_id=repo_id)
    frames_out: list[np.ndarray] = []
    times_out: list[float] = []
    global_idx = 0

    n_steps = 1 << int(exp)  # = 2 ** exp; centered-sample count per segment
    total_segs = n - 1
    try:
        # Frame 0: exact start still — sharp opening for the song.
        start_t = float(keyframe_times[0])
        if on_frame is not None:
            on_frame(global_idx, np.ascontiguousarray(keyframes[0]), start_t)
        if keep_frames:
            frames_out.append(np.ascontiguousarray(keyframes[0]))
        times_out.append(start_t)
        global_idx += 1

        for seg_i in range(total_segs):
            t0 = float(keyframe_times[seg_i])
            t1 = float(keyframe_times[seg_i + 1])
            seg_span = t1 - t0
            seg_frames = rife_exp_interpolate_pair(
                model,
                keyframes[seg_i],
                keyframes[seg_i + 1],
                exp=exp,
                device=device,
                include_endpoints=False,
            )
            # ``seg_frames`` is exactly ``n_steps`` IFNet predictions at
            # centered timesteps; map each to wall-clock time so the gap to
            # the previous/next segment matches the within-segment spacing.
            for j, fr in enumerate(seg_frames):
                tau = t0 + seg_span * (j + 0.5) / n_steps
                if on_frame is not None:
                    on_frame(global_idx, fr, tau)
                if keep_frames:
                    frames_out.append(fr)
                times_out.append(tau)
                global_idx += 1
            frac = (seg_i + 1) / max(1, total_segs)
            _report(0.05 + 0.9 * frac, f"RIFE segment {seg_i + 1}/{total_segs}")

        # Final frame: exact end still — sharp closing for the song.
        end_t = float(keyframe_times[-1])
        if on_frame is not None:
            on_frame(global_idx, np.ascontiguousarray(keyframes[-1]), end_t)
        if keep_frames:
            frames_out.append(np.ascontiguousarray(keyframes[-1]))
        times_out.append(end_t)
        global_idx += 1
    finally:
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("RIFE cuda empty_cache: %s", exc)
        if prev_cudnn_benchmark is not None:
            try:
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Could not restore cudnn.benchmark: %s", exc)

    _report(
        0.97,
        "RIFE interpolation done — saving morph frames to cache "
        "(may take several minutes; UI may look idle briefly between updates)…",
    )
    return frames_out, times_out
