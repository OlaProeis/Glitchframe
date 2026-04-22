"""Optional logo overlay: Pillow load/resize, NumPy alpha blend onto video frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_LOGO_POSITION_ALIASES: dict[str, tuple[str, ...]] = {
    "top-left": ("top-left", "topleft", "tl"),
    "top-right": ("top-right", "topright", "tr"),
    "bottom-left": ("bottom-left", "bottomleft", "bl"),
    "bottom-right": ("bottom-right", "bottomright", "br"),
    "center": ("center", "centre", "middle"),
}


def normalize_logo_position(label: str) -> str:
    """Map UI labels like ``Top-left`` to canonical ``top-left``."""
    key = label.strip().lower().replace(" ", "-")
    for canonical, aliases in _LOGO_POSITION_ALIASES.items():
        if key == canonical or key in aliases:
            return canonical
    raise ValueError(f"Unknown logo position: {label!r}")


def load_logo_rgba(path: str | Path) -> np.ndarray:
    """Load a PNG (or other image) as ``uint8`` RGBA ``(H, W, 4)``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Logo file not found: {p}")
    with Image.open(p) as im:
        rgba = im.convert("RGBA")
        return np.asarray(rgba, dtype=np.uint8)


def _fit_logo_hw(lh: int, lw: int, fh: int, fw: int) -> tuple[int, int]:
    """Scale logo down (preserving aspect ratio) so it fits inside the frame."""
    if lh <= 0 or lw <= 0:
        raise ValueError("Logo has invalid dimensions")
    if lh <= fh and lw <= fw:
        return lh, lw
    scale = min(fw / lw, fh / lh)
    nh = max(1, int(round(lh * scale)))
    nw = max(1, int(round(lw * scale)))
    return nh, nw


def prepare_logo_rgba(logo_rgba: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """Resize logo if it is larger than the frame (LANCZOS). Same array if no resize."""
    if logo_rgba.ndim != 3 or logo_rgba.shape[2] != 4:
        raise ValueError("logo_rgba must have shape (H, W, 4)")
    lh, lw = int(logo_rgba.shape[0]), int(logo_rgba.shape[1])
    nh, nw = _fit_logo_hw(lh, lw, frame_h, frame_w)
    if nh == lh and nw == lw:
        return logo_rgba
    pil = Image.fromarray(logo_rgba, mode="RGBA")
    resized = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def _origin_for_position(
    position: str,
    frame_h: int,
    frame_w: int,
    logo_h: int,
    logo_w: int,
) -> tuple[int, int]:
    fh, fw, lh, lw = frame_h, frame_w, logo_h, logo_w
    pos = normalize_logo_position(position)
    if pos == "top-left":
        return 0, 0
    if pos == "top-right":
        return fw - lw, 0
    if pos == "bottom-left":
        return 0, fh - lh
    if pos == "bottom-right":
        return fw - lw, fh - lh
    # center
    return (fw - lw) // 2, (fh - lh) // 2


def _scale_logo_rgba(logo_rgba: np.ndarray, scale: float) -> np.ndarray:
    """Return ``logo_rgba`` scaled uniformly around its centre.

    Used by the beat-pulse path: the prepared (fit-to-frame) logo is resized
    each frame by a small factor (typically 1.00–1.08). ``BILINEAR`` is plenty
    for these deltas and is ~5–10× faster than ``LANCZOS``, which matters
    when scaling happens at 30+ fps on modestly sized logos. Returns the
    input unchanged when the resized dimensions would not differ.
    """
    if scale is None or abs(scale - 1.0) < 1e-3:
        return logo_rgba
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    lh, lw = int(logo_rgba.shape[0]), int(logo_rgba.shape[1])
    nh = max(1, int(round(lh * scale)))
    nw = max(1, int(round(lw * scale)))
    if nh == lh and nw == lw:
        return logo_rgba
    pil = Image.fromarray(logo_rgba, mode="RGBA")
    resized = pil.resize((nw, nh), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _hex_to_rgb_u8(h: str | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not h or not str(h).strip():
        return fallback
    s = str(h).strip().lstrip("#")
    if len(s) != 6:
        return fallback
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return fallback


def resolve_logo_glow_rgb(
    shadow_hex: str | None, base_hex: str
) -> tuple[int, int, int]:
    """Neon tint: prefer preset shadow (outline) colour, else a magenta from base."""
    base_fb = _hex_to_rgb_u8(base_hex, (186, 104, 255))
    return _hex_to_rgb_u8(shadow_hex, base_fb)


def _blend_premult_rgba_patch(
    dst: np.ndarray,
    premult_rgba: np.ndarray,
    x0: int,
    y0: int,
) -> None:
    """Alpha-composite premultiplied RGBA patch onto ``dst`` (H,W,3) uint8."""
    if premult_rgba.ndim != 3 or premult_rgba.shape[2] != 4:
        raise ValueError("premult_rgba must be (H, W, 4)")
    fh, fw = int(dst.shape[0]), int(dst.shape[1])
    ph, pw = int(premult_rgba.shape[0]), int(premult_rgba.shape[1])
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(fw, x0 + pw)
    dst_y1 = min(fh, y0 + ph)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return
    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    patch = premult_rgba[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
    sa = patch[:, :, 3:4]
    base = dst[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32) / 255.0
    out = patch[:, :, :3] + base * (1.0 - sa)
    dst[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(out * 255.0, 0.0, 255.0).astype(
        np.uint8
    )


def _neon_glow_patch(
    logo_rgba: np.ndarray,
    *,
    glow_rgb: tuple[int, int, int],
    amount: float,
    blur_radius: float,
    pad: int,
    opacity_pct: float,
) -> tuple[np.ndarray, int] | None:
    """Build a padded premultiplied RGBA halo from the logo alpha."""
    if amount <= 1e-5:
        return None
    lh, lw = int(logo_rgba.shape[0]), int(logo_rgba.shape[1])
    p = max(4, int(pad))
    H, W = lh + 2 * p, lw + 2 * p
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[p : p + lh, p : p + lw] = logo_rgba[:, :, 3]
    pil_a = Image.fromarray(alpha, mode="L")
    br = max(0.5, float(blur_radius))
    blurred = np.asarray(pil_a.filter(ImageFilter.GaussianBlur(radius=br)), dtype=np.float32)
    blurred *= float(np.clip(amount, 0.0, 1.5)) * (float(opacity_pct) / 100.0)
    blurred = np.clip(blurred, 0.0, 255.0)
    a_u8 = blurred.astype(np.uint8)
    gr, gg, gb = glow_rgb
    a_f = blurred / 255.0
    r = (gr * a_f).astype(np.uint8)
    g = (gg * a_f).astype(np.uint8)
    b = (gb * a_f).astype(np.uint8)
    return np.stack([r, g, b, a_u8], axis=-1), p


def composite_logo_onto_frame(
    frame: np.ndarray,
    logo_rgba: np.ndarray,
    position: str,
    opacity_pct: float,
    *,
    inplace: bool = False,
    scale: float = 1.0,
    glow_amount: float = 0.0,
    glow_rgb: tuple[int, int, int] | None = None,
    glow_blur_radius: float = 5.0,
    glow_pad_px: int = 32,
) -> np.ndarray:
    """
    Alpha-blend ``logo_rgba`` onto ``frame``.

    ``frame`` is ``uint8`` with shape ``(H, W, 3)`` or ``(H, W, 4)``.
    User opacity is ``opacity_pct`` in ``[0, 100]`` applied uniformly with the logo alpha.
    ``scale`` (default ``1.0``) lets callers pulse the logo's size per frame
    (see :mod:`pipeline.beat_pulse`); the scaled logo is re-anchored so the
    origin chosen by ``position`` still lines up with the correct frame edge.

    ``glow_amount`` in ``[0, 1]`` (can slightly exceed 1 for peaks) drives a
    blurred neon halo **behind** the logo using ``glow_rgb``.
    """
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("frame must have shape (H, W, 3) or (H, W, 4)")
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    logo = prepare_logo_rgba(logo_rgba, fh, fw)
    logo = _scale_logo_rgba(logo, float(scale))
    lh, lw = int(logo.shape[0]), int(logo.shape[1])

    op = float(np.clip(opacity_pct, 0.0, 100.0)) / 100.0
    if op <= 0.0:
        return frame if inplace else frame.copy()

    x0, y0 = _origin_for_position(position, fh, fw, lh, lw)
    dst = frame if inplace else frame.copy()

    if (
        glow_amount > 1e-5
        and glow_rgb is not None
        and dst.shape[2] == 3
    ):
        gp = _neon_glow_patch(
            logo,
            glow_rgb=glow_rgb,
            amount=float(glow_amount),
            blur_radius=glow_blur_radius,
            pad=glow_pad_px,
            opacity_pct=float(np.clip(opacity_pct, 0.0, 100.0)),
        )
        if gp is not None:
            glow_patch, pad = gp
            _blend_premult_rgba_patch(dst, glow_patch, x0 - pad, y0 - pad)

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(fw, x0 + lw)
    dst_y1 = min(fh, y0 + lh)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return dst

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    patch_logo = logo[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    la = (patch_logo[:, :, 3:4] / 255.0) * op
    lrgb = patch_logo[:, :, :3] / 255.0

    if dst.shape[2] == 3:
        patch = dst[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32) / 255.0
        out = patch * (1.0 - la) + lrgb * la
        dst[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(out * 255.0, 0.0, 255.0).astype(
            np.uint8
        )
        return dst

    patch = dst[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    drgb = patch[:, :, :3] / 255.0
    da = patch[:, :, 3:4] / 255.0
    out_a = la + da * (1.0 - la)
    out_rgb = np.divide(
        lrgb * la + drgb * da * (1.0 - la),
        out_a,
        out=np.zeros_like(lrgb),
        where=out_a > 1e-6,
    )
    dst[dst_y0:dst_y1, dst_x0:dst_x1, :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    dst[dst_y0:dst_y1, dst_x0:dst_x1, 3:4] = np.clip(out_a * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    return dst


def composite_logo_from_path(
    frame: np.ndarray,
    logo_path: str | Path | None,
    position: str,
    opacity_pct: float,
    *,
    inplace: bool = False,
    scale: float = 1.0,
    glow_amount: float = 0.0,
    glow_rgb: tuple[int, int, int] | None = None,
    glow_blur_radius: float = 5.0,
    glow_pad_px: int = 32,
) -> np.ndarray:
    """Load logo from disk; no-op copy if path is missing."""
    if logo_path is None or not str(logo_path).strip():
        return frame if inplace else frame.copy()
    rgba = load_logo_rgba(logo_path)
    return composite_logo_onto_frame(
        frame,
        rgba,
        position,
        opacity_pct,
        inplace=inplace,
        scale=scale,
        glow_amount=glow_amount,
        glow_rgb=glow_rgb,
        glow_blur_radius=glow_blur_radius,
        glow_pad_px=glow_pad_px,
    )
