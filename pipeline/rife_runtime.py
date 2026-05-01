"""Download Practical-RIFE weights and run GPU interpolation between SDXL keyframes."""

from __future__ import annotations

import logging
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

    def __init__(self, flownet: IFNet) -> None:
        self.flownet = flownet
        self.version = 4.25

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
) -> RIFEInferenceWrapper:
    weight_path = ensure_rife_flownet_path(
        repo_id=repo_id, local_files_only=local_files_only
    )
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
    return RIFEInferenceWrapper(net)


def _prepare_pair_tensors(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if rgb0.shape != rgb1.shape or rgb0.ndim != 3 or rgb0.shape[2] != 3:
        raise ValueError("Expected matching HxWx3 uint8 RGB arrays")
    t0 = torch.from_numpy(rgb0).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t1 = torch.from_numpy(rgb1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t0 = t0.to(device)
    t1 = t1.to(device)
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
) -> list[np.ndarray]:
    """Uniform RIFE subdivision (``2 ** exp``) between ``rgb0`` and ``rgb1`` (inclusive)."""
    if exp < 1:
        raise ValueError(f"rife exp must be >= 1, got {exp}")
    n = 2**exp
    t0, t1, h, w = _prepare_pair_tensors(rgb0, rgb1, device)
    out_list: list[np.ndarray] = []
    out_list.append(_tensor_to_rgb(t0, h, w))
    for i in range(n - 1):
        mid = model.inference(t0, t1, (i + 1) * 1.0 / n)
        out_list.append(_tensor_to_rgb(mid, h, w))
    out_list.append(_tensor_to_rgb(t1, h, w))
    return out_list


def rife_build_morph_timeline(
    keyframes: Sequence[np.ndarray],
    keyframe_times: Sequence[float],
    *,
    exp: int,
    device: torch.device,
    repo_id: str = DEFAULT_RIFE_REPO,
    progress: ProgressFn | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Concatenate per-segment RIFE tracks into one dense timeline (dedupe boundaries)."""
    if len(keyframes) != len(keyframe_times):
        raise ValueError("keyframes and times length mismatch")
    n = len(keyframes)
    if n < 2:
        raise ValueError("RIFE morph needs at least two SDXL keyframes")

    def _report(p: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, p)), msg)

    _report(0.0, "Loading RIFE model…")
    model = load_rife_model(device=device, repo_id=repo_id)
    frames_out: list[np.ndarray] = []
    times_out: list[float] = []

    total_segs = n - 1
    try:
        for seg_i in range(total_segs):
            t0 = float(keyframe_times[seg_i])
            t1 = float(keyframe_times[seg_i + 1])
            seg_frames = rife_exp_interpolate_pair(
                model,
                keyframes[seg_i],
                keyframes[seg_i + 1],
                exp=exp,
                device=device,
            )
            for j, fr in enumerate(seg_frames):
                tau = t0 + (t1 - t0) * (j / max(1, len(seg_frames) - 1))
                if frames_out and j == 0:
                    continue
                frames_out.append(fr)
                times_out.append(tau)
            frac = (seg_i + 1) / max(1, total_segs)
            _report(0.05 + 0.9 * frac, f"RIFE segment {seg_i + 1}/{total_segs}")
    finally:
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("RIFE cuda empty_cache: %s", exc)

    _report(1.0, "RIFE morph timeline ready")
    return frames_out, times_out
