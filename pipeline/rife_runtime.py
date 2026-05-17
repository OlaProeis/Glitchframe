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

# IFNet timestep inset for centered morph sampling.
#
# Without an inset, centered sampling at ``(j+0.5)/n`` places the boundary
# samples of every segment at ``s ≈ 0.5/n`` and ``s ≈ (n-0.5)/n`` — for the
# default ``rife_exp=4`` (n=16) that's ``s ≈ 0.031`` and ``s ≈ 0.969``. At
# those near-endpoint timesteps IFNet's flow displacement collapses toward
# zero, so the predicted frames are visually near-identical to the SDXL
# keyframe pixel-wise. The compositor then renders an extended window of
# "near-keyframe" content around every internal keyframe time, which the
# viewer perceives as the morph "pausing on the original still".
#
# Pushing the IFNet timestep distribution inward by ``DEFAULT_IFNET_TIMESTEP_INSET``
# guarantees every dense sample carries meaningful flow (the boundary
# samples sit at ``s ≈ inset`` and ``s ≈ 1 - inset``), so the perceived
# motion stays continuous through every keyframe. Wall-clock spacing remains
# uniform centered ``T_seg/n`` (decoupled from IFNet timestep), so apparent
# motion velocity is constant — only the visual *content* of each frame is
# shifted away from the keyframe.
#
# v7.1 default ``0.12`` (unchanged from v7 / v4 / v5). With ``rife_exp=4``
# (n=16) the boundary body samples land at ``s ≈ 0.144`` / ``s ≈ 0.856`` —
# visibly distinct from the keyframes (so no in-body plateau) yet close
# enough to the keyframe that the linear interpolation from the last body
# sample to the v7.1 IFNet-rendered anchor at ``t_kf`` (an IFNet inference
# at ``s = 1.0`` on the *prior* pair) travels only ``inset = 0.12`` IFNet
# timesteps over the ``T_seg/(2n)`` boundary cell. At ``T_seg = 10 s``,
# n = 16, fps = 30 that's a ~5× per-frame visual change vs. the body —
# perceived as a brief speed-up *into* every keyframe, not a discrete jump.
#
# History: v4 introduced the inset at 0.12; v6 raised it to 0.25 together
# with a cross-pair bridge at every internal keyframe (amplified an
# orthogonal "spatial-jump" artifact by adding a *third* pair-mismatched
# warp into the boundary blend); v7 replaced the cross-pair bridge with
# the *exact SDXL still* at every internal ``t_kf`` to eliminate cross-pair
# blending entirely. v7's design rested on the assumption that the SDXL
# still was byte-identical to ``IFNet(kf_i, kf_{i+1}, s=1.0)`` ("modulo
# IFNet's identity behaviour at the endpoints"). That assumption is wrong:
# Practical-RIFE's IFNet has *no* identity branch at the endpoints — it
# unconditionally runs all five flow-refinement blocks with ``timestep``
# baked into the feature concatenation, so its render at ``s = 1.0`` is
# visually close to ``kf_{i+1}`` but carries the network's characteristic
# warping / merge-mask texture, which differs from the SDXL VAE's output.
# Inserting the raw VAE still as the anchor therefore introduced a 1-frame
# *texture* discontinuity (raw VAE crispness amid IFNet smoothing) at every
# internal boundary — perceived as a "blip" or "skip" by the viewer, on
# every keyframe. v7.1 replaces the exact-still anchor with an IFNet
# inference at ``s = 1.0`` on the prior pair so the anchor shares texture
# with its neighbours; cross-pair spatial alignment at the boundary is
# preserved because both the pre-anchor (last body of seg_i, pair A) and
# the anchor itself live on the same pair A, and the residual pair switch
# on the post-anchor side (anchor on A → first body of seg_{i+1} on B) is
# between two IFNet renders both dominated by ``kf_{i+1}`` content, so its
# flow residual is minimal. The inset value stays at 0.12 (the
# anchor-velocity argument is unchanged).
#
# Override at runtime via ``GLITCHFRAME_RIFE_IFNET_INSET`` (clamped to the
# safe range below); when set, this value is honoured byte-exactly and the
# v7.2 boundary-velocity cap (see :data:`DEFAULT_BOUNDARY_VELOCITY_RATIO`
# below) is bypassed entirely. Set the env var to ``0`` to recover the
# legacy centered-only behaviour for byte-exact reproducibility against
# older bakes; tune towards ``0.0`` if you want the morph to settle
# visibly onto every SDXL keyframe (at the cost of brief perceived
# pauses there).
DEFAULT_IFNET_TIMESTEP_INSET = 0.12
_IFNET_TIMESTEP_INSET_MIN = 0.0
# Capping at 0.45 keeps both halves of the (warped) [inset, 1-inset]
# interval non-degenerate; in practice values much above ~0.30 over-compress
# the visible motion range and start to feel "muted" rather than "fluid".
_IFNET_TIMESTEP_INSET_MAX = 0.45

# v7.2 boundary-velocity cap.
#
# The v7 / v7.1 internal-keyframe anchor is an IFNet sample at ``s = 1.0``
# (or the exact SDXL still, pre-v7.1) placed at every internal ``t_kf``.
# The IFNet-timestep gap from the last body sample (at ``s = 1 - inset``,
# approximately, after the centered-inset warp) to the anchor at ``s = 1.0``
# is ``inset`` per boundary cell, traversed in ``T_seg / (2 * n_steps)``
# wall-clock seconds. The ratio of that boundary IFNet-timestep velocity to
# the body's own IFNet-timestep velocity ``(1 - 2 * inset) / T_seg`` is
#
#   ratio  =  2 * n_steps * inset / (1 - 2 * inset)
#
# (T_seg cancels). At ``inset = 0.12`` this gives ~5× the body pace at
# ``rife_exp = 4`` (n_steps = 16) — the v7 design target, which reads as a
# brief speed-up *into* every keyframe rather than a discrete jump. But
# the ratio scales linearly in ``n_steps``: at ``rife_exp = 8``
# (n_steps = 256) the same constant inset produces ~80× body velocity at
# every anchor — about 25 % IFNet-timestep change per video frame at
# 30 fps, perceived as a clear "skip" before *and* a symmetric cross-pair
# residual after every keyframe boundary. (The pre-v7 cross-pair bridge
# masked this with a different artifact; v7.1 fixed the texture blip and
# made the velocity spike the dominant residual at high ``rife_exp``.)
#
# v7.2 caps the inset by inverting the ratio formula:
#
#   inset_cap         =  ratio / (2 * n_steps + 2 * ratio)
#   inset_effective   =  min(DEFAULT_IFNET_TIMESTEP_INSET, inset_cap)
#
# so the boundary IFNet-timestep velocity stays at ``ratio`` × body
# velocity *or less* for every ``n_steps``. At the v7.1 reference
# ``rife_exp = 4`` (n_steps = 16) the cap evaluates to 0.119 — just below
# the legacy default 0.12 — and the ``min`` returns 0.119; the difference
# is below user-perceptible motion thresholds so the common path is
# preserved. At ``rife_exp ≤ 2`` (n_steps ≤ 4) the cap exceeds 0.12 and
# the ``min`` returns the legacy default, leaving low-``rife_exp``
# behaviour byte-identical. At higher ``rife_exp`` the cap kicks in
# (e.g. 0.0362 at n_steps = 64, 0.0096 at n_steps = 256), which also
# tightens the boundary body samples toward ``s = 0`` / ``s = 1``; the
# post-anchor cross-pair residual (anchor on pair A at ``s = 1.0`` →
# first body of next segment on pair B at ``s = inset``) is correspond-
# ingly minimised because both flanking samples now sit at IFNet
# timesteps where the network's render is essentially ``kf_{i+1}``.
#
# The cap is applied *only* when no explicit ``ifnet_timestep_inset``
# kwarg and no ``GLITCHFRAME_RIFE_IFNET_INSET`` env var are set; an
# explicit override is honoured byte-for-byte so reproducibility and
# manual experimentation continue to work. Override the velocity-ratio
# target via ``GLITCHFRAME_RIFE_BOUNDARY_VELOCITY_RATIO`` (clamped to
# ``[0.0, _BOUNDARY_VELOCITY_RATIO_MAX]``).
DEFAULT_BOUNDARY_VELOCITY_RATIO = 5.0
_BOUNDARY_VELOCITY_RATIO_MIN = 0.0
# Cap the target ratio at 100 because beyond that the derived inset
# approaches the safe-max 0.45 (e.g. at n_steps = 256 the inset hits
# ~0.164) and we'd be re-introducing the v7.0 perceived-pause regime via
# the back door.
_BOUNDARY_VELOCITY_RATIO_MAX = 100.0


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


def _resolve_ifnet_timestep_inset(override: float | None = None) -> float:
    """Resolve the IFNet timestep inset, honouring an env var when set.

    Order of precedence: explicit ``override`` > ``GLITCHFRAME_RIFE_IFNET_INSET``
    > :data:`DEFAULT_IFNET_TIMESTEP_INSET`. Values are clamped to
    ``[_IFNET_TIMESTEP_INSET_MIN, _IFNET_TIMESTEP_INSET_MAX]``; unparseable
    strings fall back to the default with a debug log entry.

    This resolver does NOT apply the v7.2 boundary-velocity cap — it always
    returns either the explicit override or the constant
    :data:`DEFAULT_IFNET_TIMESTEP_INSET`. The morph timeline builder uses
    :func:`_resolve_ifnet_timestep_inset_explicit` + the cap instead so it
    can scale inset with ``n_steps``; callers wanting the raw legacy default
    keep using this helper.
    """
    if override is not None:
        v = float(override)
    else:
        raw = os.environ.get("GLITCHFRAME_RIFE_IFNET_INSET", "").strip()
        if raw == "":
            v = DEFAULT_IFNET_TIMESTEP_INSET
        else:
            try:
                v = float(raw)
            except ValueError:
                LOGGER.debug(
                    "Ignoring non-numeric GLITCHFRAME_RIFE_IFNET_INSET=%r; "
                    "using default %.3f",
                    raw,
                    DEFAULT_IFNET_TIMESTEP_INSET,
                )
                v = DEFAULT_IFNET_TIMESTEP_INSET
    return max(_IFNET_TIMESTEP_INSET_MIN, min(_IFNET_TIMESTEP_INSET_MAX, v))


def _resolve_ifnet_timestep_inset_explicit(
    override: float | None = None,
) -> float | None:
    """Return the explicit IFNet timestep inset override, or ``None`` when none is set.

    Order of precedence: explicit ``override`` kwarg >
    ``GLITCHFRAME_RIFE_IFNET_INSET`` env var. Returns ``None`` when both are
    absent (or the env var is unparseable) so the caller can apply the
    v7.2 boundary-velocity cap instead of the raw legacy default. Returned
    values are clamped to ``[_IFNET_TIMESTEP_INSET_MIN,
    _IFNET_TIMESTEP_INSET_MAX]``.
    """
    if override is not None:
        v = float(override)
        return max(_IFNET_TIMESTEP_INSET_MIN, min(_IFNET_TIMESTEP_INSET_MAX, v))
    raw = os.environ.get("GLITCHFRAME_RIFE_IFNET_INSET", "").strip()
    if raw == "":
        return None
    try:
        v = float(raw)
    except ValueError:
        LOGGER.debug(
            "Ignoring non-numeric GLITCHFRAME_RIFE_IFNET_INSET=%r; "
            "falling back to v7.2 boundary-velocity cap",
            raw,
        )
        return None
    return max(_IFNET_TIMESTEP_INSET_MIN, min(_IFNET_TIMESTEP_INSET_MAX, v))


def _resolve_boundary_velocity_ratio() -> float:
    """Resolve the v7.2 target boundary IFNet-timestep velocity / body velocity ratio.

    Reads ``GLITCHFRAME_RIFE_BOUNDARY_VELOCITY_RATIO`` (defaults to
    :data:`DEFAULT_BOUNDARY_VELOCITY_RATIO`). Values are clamped to
    ``[_BOUNDARY_VELOCITY_RATIO_MIN, _BOUNDARY_VELOCITY_RATIO_MAX]``;
    unparseable strings fall back to the default with a debug log entry.
    """
    raw = os.environ.get("GLITCHFRAME_RIFE_BOUNDARY_VELOCITY_RATIO", "").strip()
    if raw == "":
        return DEFAULT_BOUNDARY_VELOCITY_RATIO
    try:
        v = float(raw)
    except ValueError:
        LOGGER.debug(
            "Ignoring non-numeric GLITCHFRAME_RIFE_BOUNDARY_VELOCITY_RATIO=%r; "
            "using default %.2f",
            raw,
            DEFAULT_BOUNDARY_VELOCITY_RATIO,
        )
        return DEFAULT_BOUNDARY_VELOCITY_RATIO
    return max(_BOUNDARY_VELOCITY_RATIO_MIN, min(_BOUNDARY_VELOCITY_RATIO_MAX, v))


def _inset_from_boundary_velocity_ratio(n_steps: int, ratio: float) -> float:
    """Compute the inset that produces the given boundary-velocity ratio at ``n_steps``.

    Inverts ``ratio = 2 * n_steps * inset / (1 - 2 * inset)``:

        inset = ratio / (2 * n_steps + 2 * ratio)

    Clamped to ``[_IFNET_TIMESTEP_INSET_MIN, _IFNET_TIMESTEP_INSET_MAX]``.
    """
    if n_steps <= 0 or ratio <= 0.0:
        return 0.0
    inset = ratio / (2.0 * n_steps + 2.0 * ratio)
    return max(_IFNET_TIMESTEP_INSET_MIN, min(_IFNET_TIMESTEP_INSET_MAX, inset))


def _resolve_effective_inset(
    n_steps: int,
    override: float | None = None,
) -> float:
    """Resolve the effective IFNet timestep inset for ``n_steps`` body samples.

    When an explicit ``override`` kwarg or ``GLITCHFRAME_RIFE_IFNET_INSET``
    env var is set, that value is honoured byte-exactly (clamped only to
    the safe ``[0, 0.45]`` range) — the v7.2 cap is bypassed so byte-exact
    reproduction against older bakes / manual experimentation continues to
    work.

    Otherwise the effective inset is ``min(DEFAULT_IFNET_TIMESTEP_INSET,
    inset_cap)`` where ``inset_cap`` is derived from the target
    boundary-velocity ratio (default :data:`DEFAULT_BOUNDARY_VELOCITY_RATIO`,
    override via ``GLITCHFRAME_RIFE_BOUNDARY_VELOCITY_RATIO``). The
    ``min`` makes the cap a **no-op** below the velocity-knee — at
    ``rife_exp ≤ 4`` (n_steps ≤ 16) the constant ``0.12`` already satisfies
    the 5 × ratio target so the v7.1 default is preserved byte-for-byte.
    At higher ``rife_exp`` the cap shrinks the inset to keep the
    IFNet-timestep velocity spike at every internal-keyframe anchor at
    ``ratio`` × body velocity regardless of ``n_steps``.

    See the :data:`DEFAULT_BOUNDARY_VELOCITY_RATIO` block comment for the
    full derivation and the trade-offs at high ``rife_exp``.
    """
    explicit = _resolve_ifnet_timestep_inset_explicit(override)
    if explicit is not None:
        return explicit
    ratio = _resolve_boundary_velocity_ratio()
    inset_cap = _inset_from_boundary_velocity_ratio(n_steps, ratio)
    return min(DEFAULT_IFNET_TIMESTEP_INSET, inset_cap)


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


def rife_ifnet_at_timestep(
    model: RIFEInferenceWrapper,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    *,
    timestep: float,
    device: torch.device,
) -> np.ndarray:
    """Single IFNet inference between ``rgb0`` and ``rgb1`` at ``timestep``.

    ``timestep`` is the IFNet temporal parameter ``s ∈ [0, 1]`` (``s = 0``
    ≈ ``rgb0``, ``s = 1`` ≈ ``rgb1``). The morph timeline builder uses this
    helper to render the **bookend** frames at song start and end at IFNet
    timesteps that match the body's wall-clock-to-IFNet-timestep velocity,
    eliminating the perceived "skip" right before the song's closing still
    (and the symmetric one at the song open). See
    :func:`rife_build_morph_timeline` for the velocity-matching derivation.

    Inference runs inside :func:`torch.inference_mode` for the same reason
    as :func:`rife_exp_interpolate_pair` — no autograd graph for inference,
    cuts ~15–20 % wall-clock and roughly halves transient VRAM versus a
    plain ``model.eval()`` call.
    """
    dtype = torch.float16 if model.fp16 else torch.float32
    t0, t1, h, w = _prepare_pair_tensors(rgb0, rgb1, device, dtype=dtype)
    with torch.inference_mode():
        mid = model.inference(t0, t1, float(timestep))
    return _tensor_to_rgb(mid, h, w)


def rife_exp_interpolate_pair(
    model: RIFEInferenceWrapper,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    *,
    exp: int,
    device: torch.device,
    include_endpoints: bool = True,
    ifnet_timestep_inset: float = 0.0,
) -> list[np.ndarray]:
    """RIFE subdivision (``2 ** exp``) between ``rgb0`` and ``rgb1``.

    With ``include_endpoints=True`` (default) returns the legacy uniform grid
    ``[rgb0, IFNet(1/n), …, IFNet((n-1)/n), rgb1]`` — ``n + 1`` frames where
    the first and last are the exact source stills. ``ifnet_timestep_inset``
    is ignored in this mode (legacy bake / tests rely on the byte-exact
    uniform grid).

    With ``include_endpoints=False`` returns ``n`` IFNet predictions sampled
    at **inset-warped centered** timesteps ``[inset + (1 - 2*inset)*(i + 0.5)/n
    for i in 0..n-1]``. The morph timeline builder uses this mode so adjacent
    segments don't snap to the exact SDXL still at every internal keyframe
    boundary: the flanking dense samples are equally-soft IFNet predictions
    on both sides of the keyframe, and the inset guarantees they carry
    *visible* flow displacement instead of collapsing onto near-keyframe
    pixels. With ``ifnet_timestep_inset=0`` this reduces to the legacy plain
    centered sampling (boundary samples at ``s ≈ 0.5/n`` and ``s ≈ (n-0.5)/n``,
    visually near-identical to the keyframes — the perceptual "pause" the
    inset is meant to eliminate).

    Inference runs inside :func:`torch.inference_mode`, which skips autograd
    bookkeeping (Practical-RIFE upstream does the equivalent via
    ``torch.set_grad_enabled(False)``). Without this flag PyTorch builds a
    no-op autograd graph for every forward pass, costing ~15–20 % extra
    wall-clock and roughly doubling transient VRAM.
    """
    if exp < 1:
        raise ValueError(f"rife exp must be >= 1, got {exp}")
    inset = max(
        _IFNET_TIMESTEP_INSET_MIN,
        min(_IFNET_TIMESTEP_INSET_MAX, float(ifnet_timestep_inset)),
    )
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
        span = 1.0 - 2.0 * inset
        with torch.inference_mode():
            for i in range(n):
                # Inset-warped centered timestep: legacy ``(i + 0.5)/n`` is
                # remapped from (0, 1) into ``(inset, 1 - inset)`` so even
                # the boundary samples sit a ``span/2 = (1-2*inset)/(2n)``
                # beyond the inset — that is, comfortably away from the
                # IFNet "near-zero motion" zones at ``s -> 0`` and ``s -> 1``.
                ts = inset + span * (i + 0.5) / n
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
    ifnet_timestep_inset: float | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Build one continuous-velocity dense RIFE timeline across all keyframes.

    Each segment ``[kf_i, kf_{i+1}]`` body is sampled at **inset-warped
    centered** IFNet timesteps ``[inset + (1 - 2*inset) * (j + 0.5)/n
    for j in 0..n-1]`` (where ``n = 2**exp`` and ``inset`` defaults to
    :data:`DEFAULT_IFNET_TIMESTEP_INSET`). The body samples carry visible
    motion instead of collapsing onto near-keyframe pixels.

    **v7.1 internal-keyframe anchors.** At every *internal* keyframe time
    ``t_{i+1}`` (between ``seg_i`` and ``seg_{i+1}``) we insert one extra
    IFNet inference — ``IFNet(kf_i, kf_{i+1}, s=1.0)`` on the *prior* pair
    — and place it at exactly ``t = t_{i+1}``. This is *IFNet's own render
    of* ``kf_{i+1}`` (visually very close to the SDXL still but processed
    through the network's flow + merge-mask pipeline), not the exact SDXL
    keyframe itself.

    Why not the exact SDXL still? Practical-RIFE's :class:`IFNet` has *no*
    identity branch at the endpoints — at ``s = 0`` or ``s = 1`` it still
    runs all five flow-refinement blocks with ``timestep`` baked into the
    feature concatenation, so its output is *not* byte-identical to either
    input keyframe. The render at ``s = 1.0`` carries the network's
    characteristic warping / merge-mask texture, which differs from the
    SDXL VAE's output (the VAE is sharper / more saturated; IFNet smooths
    via its convolutional warping). v7 placed the raw VAE still as the
    anchor on the (incorrect) assumption that it was byte-equivalent to
    ``IFNet(kf_i, kf_{i+1}, s=1.0)``; the resulting frame-by-frame texture
    hand-off VAE → IFNet at every internal boundary read as a brief 1–2
    video-frame "blip" / "skip" in the rendered video (visible regardless
    of how clean the underlying SDXL keyframes are). v7.1 fixes this by
    having the anchor share IFNet's texture signature with its neighbours.

    The compositor's pixel-space linear blend around the anchor:

    * just before ``t_{i+1}``: blend within pair ``(kf_i, kf_{i+1})`` —
      the last body of ``seg_i`` at ``s ≈ 1 - inset`` blending into the
      anchor at ``s = 1.0``. Same flow field, same texture signature, no
      spatial mismatch and no texture jump.
    * exactly at ``t_{i+1}``: ``IFNet(kf_i, kf_{i+1}, s = 1.0)`` — IFNet's
      own rendering of ``kf_{i+1}``, visually dominated by ``kf_{i+1}``
      content.
    * just after ``t_{i+1}``: blend across pairs — anchor (pair A, IFNet
      at ``s = 1.0``) → first body of ``seg_{i+1}`` (pair B, IFNet at
      ``s ≈ inset``). Both samples are IFNet renders (same texture
      signature, so no v7-style VAE↔IFNet blip), and both are dominated
      by ``kf_{i+1}`` content (anchor is full warp toward ``kf_{i+1}``;
      first body is only ``inset`` along the next flow), so the cross-pair
      flow residual at the boundary is at its minimum visible scale.

    Trade-off vs. v7: +1 IFNet inference per internal boundary
    (``total_segs − 1`` extra forward passes — sub-second on a 3090 for
    typical 20–40 keyframe songs). Trade-off vs. v6: the
    cross-pair-mismatched bridge in the middle of the boundary window is
    gone, which is what made v6 read as a hard *spatial* jump rather than
    a texture blip; v7 fixed the spatial jump but introduced the texture
    blip; v7.1 fixes both. Pre-anchor blend stays entirely within pair A
    (best case: same pair, same texture). Post-anchor blend stays within
    "two IFNet renders both close to ``kf_{i+1}``" — perceptually smooth
    even when the underlying optical-flow pairs differ. The residual
    IFNet-timestep velocity spike from the last body (``s = 1 - inset``)
    to the anchor (``s = 1.0``) over ``T_seg/(2n)`` wall-clock seconds is
    ``2 * n * inset / span`` body-velocities, perceived as a brief
    speed-up *into* every keyframe rather than a discrete jump.

    The song's first and last frames are **velocity-matched IFNet bookends**
    at ``t = keyframe_times[0]`` and ``t = keyframe_times[-1]`` — namely
    IFNet at ``s = inset`` (using the first keyframe pair ``kf_0, kf_1``)
    and ``s = 1 - inset`` (using the last keyframe pair ``kf_{N-1}, kf_N``).
    A naive bookend at the *exact* SDXL still (``s = 0`` / ``s = 1``) would
    leave an IFNet jump of ``inset + span/(2n)`` between the bookend and
    the first body sample over only ``T_seg/(2n)`` wall-clock seconds: at
    the legacy ``inset = 0.12`` and ``exp = 4`` that's ``0.144`` IFNet
    timesteps in ``0.25 s``, ~6× the body's pace ``span/T_seg``. The viewer
    perceives that as a brief "skip" right at the song open and close,
    even though the morph in between is smooth. Sampling the bookends at
    ``s = inset`` instead reduces the IFNet jump to ``span/(2n)`` (half a
    body step) over the same ``T_seg/(2n)`` wall-clock — exactly the body
    velocity ``span/T_seg``. The visual cost is tiny: IFNet at ``s = inset``
    is within ~``inset`` flow displacement of ``kf_0`` (and symmetrically
    for ``kf_N``), so the song still visibly opens/closes on the generated
    image, just with a hint of motion instead of a freeze that snaps.

    Why the body inset matters in the first place: under plain centered
    sampling ``(j + 0.5)/n``, the boundary samples of every segment land at
    ``s ≈ 0.5/n`` and ``s ≈ (n - 0.5)/n`` (≈ 0.031 / 0.969 at ``exp=4``).
    IFNet's flow displacement near those endpoints is essentially zero, so
    those frames are visually indistinguishable from the SDXL keyframe —
    the compositor renders a ~``T_seg/n`` window of "stuck on the original
    still" content around every internal keyframe time, perceived by the
    viewer as a pause that breaks the otherwise-smooth morph. Insetting
    the IFNet timestep distribution forces every dense sample to carry
    visible flow (the closest-to-boundary samples now sit at
    ``s ≈ inset`` and ``s ≈ 1 - inset``), and the v7 default
    ``inset = 0.12`` keeps each body sample visibly in motion while
    keeping the IFNet-timestep gap between the last body and the exact
    keyframe anchor small (only ``inset`` IFNet timesteps over a
    ``T_seg/(2n)`` boundary cell). Wall-clock spacing remains uniform
    centered ``T_seg/n``.

    Wall-clock placement remains uniform centered ``T_seg/n`` for body
    samples (decoupled from the IFNet timestep): each body sample lands at
    ``t_kf_i + T_seg * (j + 0.5)/n``, both *within* a segment and *across*
    every internal keyframe boundary. Internal-keyframe anchors land
    exactly at each internal ``t_kf`` (so the boundary blend windows are
    halved to ``T_seg/(2n)`` per side, but both halves stay within a
    single optical-flow pair because the anchor is the canonical SDXL
    still — see the v7 derivation above). Bookend wall-clock placement
    at ``t = keyframe_times[0]`` / ``t = keyframe_times[-1]`` keeps the
    song's very first and last rendered frame pinned to ``t = 0`` /
    ``t = duration``; combined with the velocity-matched IFNet timestep
    there is no longer a jump in IFNet velocity between the bookends and
    the body, so the compositor's linear sampling produces a
    uniform-velocity rendering across the entire song aside from the
    intentional speed-up *into* each internal keyframe anchor.

    ``ifnet_timestep_inset`` overrides :data:`DEFAULT_IFNET_TIMESTEP_INSET`
    and the ``GLITCHFRAME_RIFE_IFNET_INSET`` env var. Pass ``None`` (default)
    to let the **v7.2 boundary-velocity cap** kick in: the effective inset
    becomes ``min(0.12, ratio / (2 * n_steps + 2 * ratio))`` where
    ``ratio`` defaults to :data:`DEFAULT_BOUNDARY_VELOCITY_RATIO` (5 ×;
    override via ``GLITCHFRAME_RIFE_BOUNDARY_VELOCITY_RATIO``). At
    ``rife_exp ≤ 4`` (n_steps ≤ 16) the cap is a no-op — the v7.1
    default of 0.12 already satisfies the 5 × ratio target — so the
    common path is preserved byte-for-byte. At higher ``rife_exp`` the
    cap shrinks the inset so the IFNet-timestep velocity spike at every
    internal-keyframe anchor stays at ``ratio`` × body velocity (≈ 5 %
    per video frame at 30 fps for ratio = 5, a "brief speed-up *into*
    the anchor" rather than a discrete jump). Without the cap,
    ``inset = 0.12`` at ``rife_exp = 8`` (n_steps = 256) produces a
    ~80 × body-velocity spike — ~25 % IFNet timestep change per video
    frame — which reads as a clear "skip" right before every keyframe
    (and a symmetric cross-pair residual after).

    Set ``ifnet_timestep_inset = 0`` (or
    ``GLITCHFRAME_RIFE_IFNET_INSET = 0``) for the legacy plain centered
    sampling — at ``inset = 0`` the bookends collapse to ``s = 0`` /
    ``s = 1`` (≈ exact stills under IFNet) and the body samples reach
    close to the keyframes on each side, so the IFNet-timestep velocity
    into every internal-keyframe anchor collapses to ~body velocity (no
    boundary speed-up). The trade-off at ``inset = 0`` is that the body
    samples near every keyframe become visually near-identical to the
    keyframe itself (IFNet's near-endpoint motion collapse), bringing
    back the "stops on the still" perception inside the body — milder
    at higher ``n_steps`` because the near-identical window shrinks to
    ``T_seg / n`` wall-clock per side. An explicit override bypasses the
    v7.2 cap entirely so byte-exact reproduction against older bakes
    continues to work. The cache count formula stays
    ``frame_count = total_segs * n_steps + (total_segs - 1) + 2`` (body
    samples + internal-keyframe anchors + the two velocity-matched song
    bookends).

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

    # n_steps body samples per segment — also the parameter the v7.2
    # boundary-velocity cap uses to shrink the inset at higher ``rife_exp``.
    # Computed up front so the inset resolver can apply the cap before any
    # other RIFE work happens.
    n_steps = 1 << int(exp)  # = 2 ** exp; centered-sample count per segment
    inset = _resolve_effective_inset(n_steps=n_steps, override=ifnet_timestep_inset)

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

    total_segs = n - 1
    try:
        # Bookend at song start: IFNet at s = inset (instead of the legacy
        # exact ``kf_0`` still). Sampling here at ``s = inset`` rather than
        # ``s = 0`` makes the wall-clock-to-IFNet-timestep velocity from the
        # bookend to the first body sample equal to the body's own velocity
        # ``span / T_seg_0`` (rather than ~6× higher, which the viewer reads
        # as a "skip" at the song open). See the function docstring for the
        # full velocity derivation. ``inset = 0`` recovers the legacy
        # ``s = 0`` bookend (≈ ``kf_0``) and a uniform velocity already
        # holds, so this code path is correct for both the default and the
        # legacy reproducibility setting.
        start_t = float(keyframe_times[0])
        start_bookend = rife_ifnet_at_timestep(
            model,
            keyframes[0],
            keyframes[1],
            timestep=inset,
            device=device,
        )
        if on_frame is not None:
            on_frame(global_idx, start_bookend, start_t)
        if keep_frames:
            frames_out.append(start_bookend)
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
                ifnet_timestep_inset=inset,
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

            # v7.1 internal-keyframe anchor. After every segment except
            # the last, render ``IFNet(kf_i, kf_{i+1}, s=1.0)`` on the
            # *prior* pair and place it at ``t = keyframe_times[seg_i+1]``.
            # This is IFNet's own rendering of ``kf_{i+1}`` — visually
            # dominated by ``kf_{i+1}`` content but processed through the
            # network's warp + merge-mask pipeline, so it shares the
            # *texture* signature of every IFNet body sample on this
            # segment (smooth, mildly warped, same VAE-free look).
            #
            # v7 placed the raw SDXL still here instead, on the assumption
            # that it was byte-identical to ``IFNet(s=1.0)`` "modulo IFNet
            # identity at the endpoints". That assumption is wrong:
            # :class:`IFNet` has *no* identity branch at the endpoints
            # (see ``pipeline/rife_vendor/ifnet_hdv3.py`` — it always runs
            # all five blocks with ``timestep`` baked into the feature
            # concatenation), so the VAE still and ``IFNet(s=1.0)`` differ
            # by a per-network signature: the VAE output is *sharper / more
            # saturated*; the IFNet render is slightly smoother. v7 had
            # the compositor blend ``IFNet(seg_i body, s≈0.86)`` → ``VAE
            # still`` → ``IFNet(seg_{i+1} body, s≈0.14)`` — a 1–2-video-
            # frame texture pop on every boundary, visible regardless of
            # how clean the underlying SDXL still was (the artifact the
            # user reported in v7).
            #
            # v6 went the other way and placed an IFNet sample on the
            # *cross pair* ``(kf_i, kf_{i+2})`` at ``s = 0.5`` here. That
            # sample lives in a flow space neither flanking body sample
            # lives in (its dominant content is at slightly different
            # pixel coordinates than the body samples' kf_{i+1}-dominant
            # warps), so the compositor's pixel-space linear blend
            # produced a visible per-frame translation of the dominant
            # object across the boundary window — the "clear move of
            # the object" / "skipping entire frames" symptom v7 fixed.
            #
            # v7.1 keeps both wins. The pre-anchor blend (last body of
            # ``seg_i`` at ``s ≈ 1 - inset`` → anchor at ``s = 1.0``)
            # stays within pair A — same flow field *and* same texture
            # signature, no spatial or stylistic mismatch. The post-
            # anchor blend (anchor on pair A at ``s = 1.0`` → first body
            # of ``seg_{i+1}`` on pair B at ``s ≈ inset``) still crosses
            # pairs, but: (a) both samples are IFNet renders (no VAE↔IFNet
            # texture hand-off — the v7 regression); (b) both are
            # dominated by ``kf_{i+1}`` content (anchor is full warp
            # toward ``kf_{i+1}``; first body of pair B is only ``inset``
            # along the next flow), so the cross-pair flow residual at
            # the boundary is at its minimum visible scale (far smaller
            # than v6's bridge, which sat at ``s = 0.5`` of a different
            # pair entirely).
            #
            # The trade-off is identical to v7 on the velocity dimension:
            # the IFNet-timestep velocity from the last body (``s = 1 -
            # inset``) to the anchor (``s = 1.0``) over the ``T_seg/(2n)``
            # boundary cell is ``inset / (T_seg/(2n)) = 2 * n * inset /
            # T_seg`` IFNet timesteps per second, vs. the body's ``span /
            # T_seg``. At ``inset = 0.12``, ``n = 16`` that's ~5× the body
            # pace — a brief perceived speed-up *into* every keyframe
            # rather than a discrete jump. ``inset = 0`` collapses the
            # spike to zero (perfect velocity continuity) at the cost of
            # letting the per-pair body samples settle visibly onto the
            # keyframe (the original "stops on the still" symptom).
            #
            # Cost vs. v7: +1 IFNet inference per internal boundary
            # (``total_segs - 1`` extra forward passes per bake —
            # sub-second on a 3090 for typical 20–40-keyframe songs).
            if seg_i < total_segs - 1:
                anchor = rife_ifnet_at_timestep(
                    model,
                    keyframes[seg_i],
                    keyframes[seg_i + 1],
                    timestep=1.0,
                    device=device,
                )
                if on_frame is not None:
                    on_frame(global_idx, anchor, t1)
                if keep_frames:
                    frames_out.append(anchor)
                times_out.append(t1)
                global_idx += 1

            frac = (seg_i + 1) / max(1, total_segs)
            _report(0.05 + 0.9 * frac, f"RIFE segment {seg_i + 1}/{total_segs}")

        # Bookend at song end: IFNet at ``s = 1 - inset`` on the last
        # keyframe pair (instead of the legacy exact ``kf_{N-1}`` still).
        # This is the symmetric fix to the start bookend above — see the
        # function docstring for the full derivation. The visible "skip"
        # the user reported between the last RIFE frame and the closing
        # original still was exactly this velocity discontinuity.
        end_t = float(keyframe_times[-1])
        end_bookend = rife_ifnet_at_timestep(
            model,
            keyframes[-2],
            keyframes[-1],
            timestep=1.0 - inset,
            device=device,
        )
        if on_frame is not None:
            on_frame(global_idx, end_bookend, end_t)
        if keep_frames:
            frames_out.append(end_bookend)
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
