"""Full-screen ASCII “wallpaper” + voidcat, CPU-rasterised (numpy).

* **Grid:** full-screen cell matrix; each cell is a character with a colour.
* **Beat ring:** a travelling highlight ``beat_phase=0`` (centre) → ``1`` (edge),
  with palette hot-shift and “denser” symbols on the ring, plus ``bass_hit`` bloom.
* **Side cat:** a small multiline cat that **roams the flanks** for the full song
  (Lissajous-style drift, slow left/right, beat/bar nudge) with **higher** contrast
  on drops. **Column** band optional alpha for centre logo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# 6 columns × 7 rows, row-major, top line is ``head`` (small screen coords).
# Scale with nearest-neighbor to each cell.
_GLYPH_BITS: dict[str, list[int]] = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    ".": [0, 0, 0, 0, 0, 0, 0b000001],
    "-": [0, 0, 0, 0b111111, 0, 0, 0],
    ":": [0, 0b001001, 0, 0, 0, 0b001001, 0],
    "=": [0, 0, 0b111111, 0, 0b111111, 0, 0],
    "*": [0, 0b001001, 0b010110, 0b111111, 0b010110, 0b001001, 0],
    "+": [0, 0, 0b000100, 0b000100, 0b011111, 0b000100, 0b000100],
    "#": [0, 0b001001, 0b111111, 0b001001, 0b001001, 0b111111, 0b001001],
    "%": [0, 0b100001, 0b000010, 0b000100, 0b001000, 0b010000, 0b100001],
    "@": [0, 0b001110, 0b010001, 0b010110, 0b010100, 0b010100, 0b001000],
    "o": [0, 0b000100, 0b001010, 0b010001, 0b010001, 0b001010, 0b000100],
    "^": [0, 0b000100, 0b001010, 0b010001, 0, 0, 0],
    "v": [0, 0, 0, 0, 0b100001, 0b010010, 0b001100],
    "|": [0, 0b000100, 0b000100, 0b000100, 0b000100, 0b000100, 0b000100],
    "/": [0, 0b000001, 0b000010, 0b000100, 0b001000, 0b010000, 0b100000],
    "\\": [0, 0b100000, 0b010000, 0b001000, 0b000100, 0b000010, 0b000001],
    "(": [0, 0, 0b000100, 0b001000, 0b010000, 0b001000, 0b000100],
    ")": [0, 0, 0b000100, 0b000010, 0b000001, 0b000010, 0b000100],
    ">": [0, 0, 0, 0b000100, 0b001010, 0b000001, 0],
}

_CHARSET = " .-:=*+#%@"
# Denser / “lit” end of the same charset (used on beat ring + highlights)
_CHARSET_HOT = "#%@*+="

_GLYPH_CACHE: dict[str, np.ndarray] = {}


def _bits_to_mask6x7(bits: list[int]) -> np.ndarray:
    m = np.zeros((7, 6), dtype=np.float32)
    for y, row in enumerate(bits[1:]):
        for x in range(6):
            if (row >> (5 - x)) & 1:
                m[y, x] = 1.0
    return m


for _ch, _bits in _GLYPH_BITS.items():
    _GLYPH_CACHE[_ch] = _bits_to_mask6x7(_bits)


# Cute multiline cat (fixed width); roams the side columns (see
# :func:`_wandering_cat_placement`).
_CAT_LINES = [
    "  /\\  /\\  ",
    " ( o  o ) ",
    "  > ^ <   ",
    " /|   |\\ ",
    "  =   =   ",
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.strip().lstrip("#")
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return (40, 40, 55)


@dataclass(frozen=True, slots=True)
class _CatPlacement:
    start_ci: int
    start_cj: int
    side: str  # "left" | "right"


@dataclass(frozen=True, slots=True)
class VoidcatAsciiContext:
    """Precomputed per-render metadata for the ASCII overlay layer."""

    hero_drop_t: float | None
    hero_confidence: float
    # All drop events (t, confidence); used to **pump** the cat, not to pin it.
    drop_events: tuple[tuple[float, float], ...]
    # Drift + palette seed; prefer ``analysis["song_hash"]``.
    song_hash: str
    side: str  # default flank hint for legacy (wander picks L/R from time+hash)
    palette_rgb: list[tuple[int, int, int]]
    # When True, skip the centre column band entirely (legacy void-only centre).
    sides_only: bool = False
    # Centre band [0.32, 0.68] of columns: multiply glyph alpha (0=transparent, 1=full).
    # Default ~0.5 keeps full ASCII in the middle but slightly softer for a centre logo.
    center_alpha_mul: float = 0.52


_MIN_DROP_CONF = 0.12
# Base visibility 0..1: cat is always on-screen; drop envelopes add on top.
_CAT_AMBIENT = 0.32
_CAT_DROP_BOOST = 0.52
# Pre / post / decay seconds for the side-cat visibility envelope (per drop).
_PRE_DROP_SEC = 0.24
_POST_ATTACK_SEC = 0.2
_DECAY_SEC = 1.25


def build_voidcat_ascii_context(
    analysis: Mapping[str, Any],
    palette_hex: Sequence[str] | None,
) -> VoidcatAsciiContext:
    """Pick hero drop metadata, all drop events for pulse, and palette."""
    sh = str(analysis.get("song_hash") or "default")
    h = int(hashlib.sha256(sh.encode("utf-8")).hexdigest()[:8], 16)
    side: str = "left" if (h % 2 == 0) else "right"

    drops: list[dict[str, Any]] = list(
        (analysis.get("events") or {}).get("drops") or []
    )
    hero_t: float | None = None
    conf = 0.0
    events: list[tuple[float, float]] = []
    for raw in drops:
        if not isinstance(raw, dict) or "t" not in raw:
            continue
        tc = float(raw["t"])
        cf = float(max(0.0, min(1.0, float(raw.get("confidence", 0.0)))))
        events.append((tc, cf))
    events.sort(key=lambda p: p[0])
    if events:
        best = max(events, key=lambda p: p[1])
        hero_t, conf = best[0], best[1]

    pal = [_hex_to_rgb(x) for x in (list(palette_hex) if palette_hex else [])]
    if len(pal) < 5:
        pal = [
            (20, 20, 32),
            (100, 90, 140),
            (230, 230, 245),
            (255, 50, 90),
            (120, 250, 255),
        ][:5]

    return VoidcatAsciiContext(
        hero_drop_t=hero_t,
        hero_confidence=conf,
        drop_events=tuple(events),
        song_hash=sh,
        side=side,
        palette_rgb=pal,
    )


_GRAY = (58, 56, 72)


def _cell_norm_radius(ci: int, cj: int, ncols: int, nrows: int) -> float:
    """Unit-ish radius from the frame centre (0 in middle, ~1 at corners)."""
    u = (ci + 0.5) / max(1, ncols)
    v = (cj + 0.5) / max(1, nrows)
    du = (u - 0.5) * 2.0
    dv = (v - 0.5) * 2.0
    d = float(np.hypot(du, dv))
    return min(1.0, d / 0.9)


def _beat_ring_strength(
    ci: int, cj: int, ncols: int, nrows: int, umap: Mapping[str, Any]
) -> float:
    """
    0..1: cell lies on the travelling beat-synchronised **ring** (centre → out).
    ``beat_phase`` 0 = front near the middle, 1 = near the frame edge.
    """
    d_norm = _cell_norm_radius(ci, cj, ncols, nrows)
    bp = max(0.0, min(1.0, float(umap.get("beat_phase", 0.0))))
    bh = max(0.0, min(1.0, float(umap.get("bass_hit", 0.0))))
    width = 0.07 + 0.1 * (1.0 - min(1.0, bh * 1.1))
    dist = abs(d_norm - bp)
    ring = max(0.0, 1.0 - dist / max(width, 0.04))
    ring = float(np.clip(ring, 0.0, 1.0)) ** 0.75
    ring = max(
        ring,
        0.32 * min(1.0, bh) * max(0.0, 1.0 - d_norm * 0.55),
    )
    return min(1.0, float(ring))


def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _hero_cat_weight(t: float, hero_t: float | None, conf: float) -> float:
    """Legacy single-hero envelope (kept for tests that compare old behaviour)."""
    if hero_t is None or conf < 0.1:
        return 0.0
    return _drop_cat_weight(t, float(hero_t), conf)


def _drop_cat_weight(t: float, drop_t: float, conf: float) -> float:
    """
    One drop: small ramp *before* the marker (anticipation), peak through the
    attack window, then exponential decay. ``conf`` scales overall strength.
    """
    if conf < _MIN_DROP_CONF:
        return 0.0
    dt = t - float(drop_t)
    if dt < -_PRE_DROP_SEC or dt > _POST_ATTACK_SEC + 5.0:
        return 0.0
    pre = 0.0
    if dt < 0.0:
        u = _smoothstep01((dt + _PRE_DROP_SEC) / max(_PRE_DROP_SEC, 1e-6))
        pre = 0.45 * u * u
    post = 0.0
    if dt >= 0.0:
        if dt < _POST_ATTACK_SEC:
            post = 0.35 + 0.65 * _smoothstep01(dt / max(_POST_ATTACK_SEC, 1e-6))
        else:
            post = float(
                np.exp(-(dt - _POST_ATTACK_SEC) / max(_DECAY_SEC, 1e-6))
            )
    a = max(pre, post)
    return float(max(0.0, min(1.0, a * (0.35 + 0.65 * conf))))


def _cat_micro_sway(umap: Mapping[str, Any]) -> float:
    """Small 0..~0.09 wobble from beat phase + bass so the cat feels alive."""
    bp = float(umap.get("beat_phase", 0.0))
    bh = max(0.0, min(1.0, float(umap.get("bass_hit", 0.0))))
    return 0.05 * bh + 0.04 * float(np.sin(12.566370614359172 * bp))


def _wandering_cat_placement(
    t: float,
    umap: Mapping[str, Any],
    song_hash: str,
    ncols: int,
    nrows: int,
    c_lo: int,
    c_hi: int,
    row_len: int,
    n_lines: int,
) -> _CatPlacement:
    """Flank-only cell position that **moves** continuously for the whole song."""
    h0 = int(
        hashlib.sha256(f"{song_hash}:w0".encode("utf-8")).hexdigest()[:8],
        16,
    )
    h1 = int(
        hashlib.sha256(f"{song_hash}:w1".encode("utf-8")).hexdigest()[:8],
        16,
    )
    # Slow left ↔ right (tens of seconds per half-period at 0.055 rad/s).
    side: str = (
        "right" if float(np.sin(0.055 * t + (h0 & 255) * 0.01)) > 0.0 else "left"
    )
    bp = float(umap.get("beat_phase", 0.0))
    barp = float(umap.get("bar_phase", 0.0))
    # Incommensurate phases so the path does not repeat on a short loop.
    phx = (
        0.11 * t
        + 1.15 * 6.283185307179586 * bp
        + 0.48 * 6.283185307179586 * barp
        + (h0 % 7) * 0.31
    )
    phy = (
        0.087 * t
        + 0.95 * 6.283185307179586 * barp
        + 0.19 * t * (0.5 + (h1 & 31) / 62.0)
    )
    hx = 0.5 + 0.5 * float(np.sin(phx))
    hy = 0.5 + 0.5 * float(np.sin(phy))
    margin = 1
    if side == "left":
        last_ok = c_lo - row_len
        if last_ok < margin:
            start_ci = 0
        else:
            wspan = last_ok - margin
            start_ci = margin + int(hx * max(0.0, float(wspan) + 1e-6))
    else:
        first_ok = c_hi
        last_ok = ncols - row_len
        if first_ok > last_ok:
            start_ci = max(0, ncols - row_len - 1)
        else:
            wspan = last_ok - first_ok
            start_ci = first_ok + int(hx * max(0.0, float(wspan) + 1e-6))
    v_span = max(1, nrows - n_lines - 2)
    start_cj = 1 + int(hy * float(max(0, v_span - 1)))
    start_ci = int(max(0, min(int(start_ci), ncols - row_len - 1)))
    start_cj = int(max(0, min(int(start_cj), nrows - n_lines - 1)))
    return _CatPlacement(start_ci=start_ci, start_cj=start_cj, side=side)


def _best_cat_state(
    t: float,
    ctx: VoidcatAsciiContext,
    ncols: int,
    nrows: int,
    c_lo: int,
    c_hi: int,
    umap: Mapping[str, Any],
) -> tuple[float, _CatPlacement | None, float]:
    """
    Returns ``(w_cat, placement | None, w_pulse)`` where ``w_pulse`` is the
    strongest 0..1 **drop** envelope at ``t`` (no ambient). The cat is always
    visible: ``w_cat`` combines ambient, drop boost, and micro-sway.
    """
    lines = _CAT_LINES
    row_len = max(len(s) for s in lines) if lines else 0
    n_lines = len(lines)
    if row_len <= 0:
        return 0.0, None, 0.0

    w_pulse = 0.0
    if ctx.drop_events:
        for dt, cf in ctx.drop_events:
            w_pulse = max(
                w_pulse, _drop_cat_weight(t, float(dt), float(cf))
            )
    else:
        w_pulse = _hero_cat_weight(t, ctx.hero_drop_t, ctx.hero_confidence)

    micro = _cat_micro_sway(umap)
    w_cat = min(
        1.0, float(_CAT_AMBIENT) + float(_CAT_DROP_BOOST) * w_pulse + micro
    )
    pl = _wandering_cat_placement(
        t,
        umap,
        ctx.song_hash,
        ncols,
        nrows,
        c_lo,
        c_hi,
        row_len,
        n_lines,
    )
    return w_cat, pl, w_pulse


def _lerp3(
    a: tuple[int, int, int], b: tuple[int, int, int], u: float
) -> tuple[int, int, int]:
    u = max(0.0, min(1.0, u))
    return (
        int(a[0] * (1 - u) + b[0] * u),
        int(a[1] * (1 - u) + b[1] * u),
        int(a[2] * (1 - u) + b[2] * u),
    )


def _pick_char(
    ci: int,
    cj: int,
    t: float,
    umap: Mapping[str, Any],
    *,
    ring: float = 0.0,
) -> str:
    bp = int(float(umap.get("beat_phase", 0.0)) * 7) % 4
    t_hi = float(umap.get("transient_hi", 0.0))
    on_ring = ring > 0.3
    src = _CHARSET_HOT if on_ring else _CHARSET
    n = len(src)
    k = (ci * 37 + cj * 19 + int(t * 1.9) + bp * 2) % n
    if t_hi > 0.32 and (not on_ring) and ((ci + cj + int(t * 12)) & 1) == 0:
        k = (k + 4) % n
    return src[k]


def _color_animated(
    ci: int,
    cj: int,
    ncols: int,
    nrows: int,
    t: float,
    pal: list[tuple[int, int, int]],
    umap: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Per-cell colour: travelling waves + palette, driven by time + analysis uniforms."""
    rms = float(umap.get("rms", 0.0))
    bp = float(umap.get("beat_phase", 0.0))
    barp = float(umap.get("bar_phase", 0.0))
    o_env = float(umap.get("onset_env", 0.0))
    o_p = float(umap.get("onset_pulse", 0.0))
    bh = float(umap.get("bass_hit", 0.0))
    tension = float(umap.get("build_tension", 0.0))
    d_hold = float(umap.get("drop_hold", 0.0))
    t_lo = float(umap.get("transient_lo", 0.0))
    t_mid = float(umap.get("transient_mid", 0.0))
    t_hi = float(umap.get("transient_hi", 0.0))

    u = (ci + 0.5) / max(1, ncols)
    v = (cj + 0.5) / max(1, nrows)
    ta = 6.283185307179586 * bp
    tb = 6.283185307179586 * barp

    # Diagonal and radial waves: shift with beat + wall-clock time
    w1 = 0.5 + 0.5 * float(
        np.sin(2.0 * t + 5.5 * u + 4.0 * v + 0.9 * ta + 0.25 * t_hi * 3.0)
    )
    w2 = 0.5 + 0.5 * float(
        np.sin(1.35 * t - 3.2 * u + 5.1 * v + 1.1 * tb - 0.4 * t_mid * 2.0)
    )
    w3 = 0.5 + 0.5 * float(
        np.sin(0.7 * t + 9.0 * (u * u + v * v) + 0.5 * t_lo * 2.0)
    )
    hit = 0.22 * o_env + 0.28 * o_p + 0.2 * t_hi + 0.12 * t_mid
    wmix = w1 * 0.38 + w2 * 0.32 + w3 * 0.2 + 0.1 * hit
    wmix = max(0.0, min(1.0, wmix + 0.12 * rms + 0.1 * d_hold + 0.12 * bh))

    n = len(pal)
    if n < 1:
        return (128, 128, 128)
    pos = wmix * float(n) + 0.4 * t + 0.2 * o_p
    i0 = int(np.floor(pos)) % n
    i1 = (i0 + 1) % n
    lf = float(pos - np.floor(pos))
    rgb = _lerp3(pal[i0], pal[i1], max(0.0, min(1.0, lf)))
    # Drop afterglow + bass nudge hot colours
    hot = float(max(0.0, min(1.0, 0.4 * d_hold + 0.25 * bh + 0.15 * o_p)))
    rgb = _lerp3(rgb, pal[min(3, n - 1)], hot)
    # Pre-drop: cool and pull toward flat grey
    if tension > 0.01:
        rgb = _lerp3(rgb, _GRAY, 0.45 * max(0.0, min(1.0, tension)))
    return rgb


def _blend_premult_patch(
    dst: np.ndarray,
    patch: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    """Blend premultiplied RGBA ``patch`` onto ``dst`` (same format) in-place."""
    ph, pw = patch.shape[:2]
    y1 = max(0, y0)
    x1 = max(0, x0)
    y2 = min(dst.shape[0], y0 + ph)
    x2 = min(dst.shape[1], x0 + pw)
    if y1 >= y2 or x1 >= x2:
        return
    py0, px0 = y1 - y0, x1 - x0
    sub_d = dst[y1:y2, x1:x2, :]
    sub_s = patch[py0 : py0 + (y2 - y1), px0 : px0 + (x2 - x1), :].astype(
        np.float32
    ) / 255.0
    d = sub_d.astype(np.float32) / 255.0
    sa = sub_s[..., 3:4]
    d[..., :3] = sub_s[..., :3] + d[..., :3] * (1.0 - sa)
    d[..., 3:4] = sub_s[..., 3:4] + d[..., 3:4] * (1.0 - sa)
    sub_d[:] = np.clip(np.round(d * 255.0), 0, 255).astype(np.uint8)


def _glyph_patch(
    ch: str,
    rgb: tuple[int, int, int],
    alpha: float,
    scale: int,
) -> np.ndarray:
    """Return premultiplied RGBA patch (uint8) for one scaled glyph."""
    if ch not in _GLYPH_CACHE:
        ch = "."
    m = _GLYPH_CACHE[ch]
    k = np.kron(m, np.ones((scale, scale), dtype=np.float32))
    # k * alpha: per-pixel straight alpha 0-1, then 8-bit premul R = a * R_straight
    a = float(max(0.0, min(1.0, alpha))) * k
    r0, g0, b0 = float(rgb[0]), float(rgb[1]), float(rgb[2])
    pr = np.clip(a * r0, 0.0, 255.0)
    pg = np.clip(a * g0, 0.0, 255.0)
    pb = np.clip(a * b0, 0.0, 255.0)
    pa = np.clip(a * 255.0, 0.0, 255.0)
    pm = np.stack([pr, pg, pb, pa], axis=-1)
    return np.clip(np.round(pm), 0, 255).astype(np.uint8)


def _blit_voidcat_silhouette(
    out: np.ndarray,
    *,
    w_cat: float,
    pl: _CatPlacement,
    c_lo: int,
    c_hi: int,
    ncols: int,
    nrows: int,
    cell_w: int,
    cell_h: int,
    scale: int,
    pal: list[tuple[int, int, int]],
) -> None:
    """Draw the multiline cat and optional * laser column in ``out``."""
    lines = _CAT_LINES
    row_len = max(len(s) for s in lines) if lines else 0
    # w_cat includes ambient (~0.32); map to stable draw strength. Bumped
    # alongside the grid base alpha (2026-04) so the side cat reads against
    # SDXL backgrounds without leaning on the now-near-transparent
    # ``void_ascii_bg`` shader pass.
    a_cat = 0.32 + 0.58 * min(1.0, w_cat)
    c_rgb = _lerp3(pal[1], pal[3], 0.42 + 0.25 * min(1.0, w_cat))
    for li, line in enumerate(lines):
        for k, ch0 in enumerate(line):
            ch = ch0
            if ch not in _GLYPH_CACHE:
                if ch0 in "·":
                    ch = "."
                else:
                    ch = "."
            cj = pl.start_cj + li
            if cj >= nrows:
                break
            ci = pl.start_ci + k
            if c_lo <= ci < c_hi or ci < 0 or ci >= ncols:
                continue
            y0b = cj * cell_h
            x0b = ci * cell_w
            p = _glyph_patch(ch, c_rgb, a_cat, scale)
            _blend_premult_patch(out, p, y0b, x0b)
    laser_ci = c_lo - 1 if pl.side == "left" else c_hi
    if 0 <= laser_ci < ncols and not (c_lo <= laser_ci < c_hi):
        for li in (1, 2, 3):
            cj = pl.start_cj + li
            if cj >= nrows:
                break
            y0b = cj * cell_h
            x0b = laser_ci * cell_w
            lrgb = _lerp3(pal[3], pal[4], 0.4)
            p2 = _glyph_patch("*", lrgb, (0.35 + 0.45 * min(1.0, w_cat)), scale)
            _blend_premult_patch(out, p2, y0b, x0b)


def render_voidcat_cat_overlay_rgba(
    width: int,
    height: int,
    t: float,
    *,
    uniforms: Mapping[str, Any],
    ctx: VoidcatAsciiContext,
    center_x0: float = 0.32,
    center_x1: float = 0.68,
) -> np.ndarray:
    """
    Full-frame **premul** RGBA with only the side cat + trace (background
    clear). Compositing this **after** chromatic/scanline keeps the critter
    legible when those effects are strong.
    """
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width/height must be positive")
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ncols = 64
    nrows = 36
    cell_w = w // ncols
    cell_h = h // nrows
    if cell_w < 2 or cell_h < 2:
        return out
    scale = min(5, max(1, cell_w // 6, cell_h // 7))
    c_lo = int(ncols * center_x0)
    c_hi = int(ncols * center_x1)
    c_lo = max(0, min(c_lo, ncols - 1))
    c_hi = max(c_lo + 1, min(c_hi, ncols))
    w_cat, pl, _cconf = _best_cat_state(
        float(t), ctx, ncols, nrows, c_lo, c_hi, uniforms
    )
    if pl is None:
        return out
    _blit_voidcat_silhouette(
        out,
        w_cat=w_cat,
        pl=pl,
        c_lo=c_lo,
        c_hi=c_hi,
        ncols=ncols,
        nrows=nrows,
        cell_w=cell_w,
        cell_h=cell_h,
        scale=scale,
        pal=ctx.palette_rgb,
    )
    return out


def render_voidcat_ascii_rgba(
    width: int,
    height: int,
    t: float,
    *,
    uniforms: Mapping[str, Any],
    ctx: VoidcatAsciiContext,
    center_x0: float = 0.32,
    center_x1: float = 0.68,
    omit_cat: bool = False,
) -> np.ndarray:
    """
    Return ``(H, W, 4)`` **premultiplied** RGBA ``uint8`` (same convention as
    kinetic type).

    **Default (``sides_only=False``):** the whole frame is a coloured ASCII
    field; colours animate with time and ``uniforms`` (beat/bar phase, onset,
    transients, etc.). The centre **column** band (``[center_x0, center_x1]``)
    uses ``center_alpha_mul`` (typically &lt; 1) so a centre logo stays legible
    on top. Set ``sides_only=True`` to leave the centre **fully** transparent
    (legacy void-only middle). The drop cat is drawn in the left or right
    flanks only, not in the centre band. Set ``omit_cat=True`` to raster the
    grid only (for a second sharp cat pass composited after frame FX).
    """
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width/height must be positive")

    out = np.zeros((h, w, 4), dtype=np.uint8)
    ncols = 64
    nrows = 36
    cell_w = w // ncols
    cell_h = h // nrows
    if cell_w < 2 or cell_h < 2:
        return out
    scale = min(5, max(1, cell_w // 6, cell_h // 7))

    c_lo = int(ncols * center_x0)
    c_hi = int(ncols * center_x1)
    c_lo = max(0, min(c_lo, ncols - 1))
    c_hi = max(c_lo + 1, min(c_hi, ncols))

    rms = float(uniforms.get("rms", 0.0))
    t_hi = float(uniforms.get("transient_hi", 0.0))
    bass = float(uniforms.get("bass_hit", 0.0))
    hold = float(uniforms.get("drop_hold", 0.0))
    pal = ctx.palette_rgb

    # Stay readable on the SDXL/AnimateDiff still that the reactive shader
    # composites over (the void_ascii_bg shader is intentionally near-zero
    # alpha — see ``assets/shaders/void_ascii_bg.frag``). Higher base alpha
    # plus a fuller audio lift so the grid is the visible feature instead
    # of getting swallowed by the SDXL still and the centre logo.
    base_alpha = 0.46 + 0.38 * rms + 0.20 * t_hi + 0.18 * bass + 0.18 * hold
    cam = max(0.0, min(1.0, float(ctx.center_alpha_mul)))
    w_cat, pl_cat, _wc = _best_cat_state(
        float(t), ctx, ncols, nrows, c_lo, c_hi, uniforms
    )
    if bool(omit_cat):
        w_cat, pl_cat = 0.0, None

    for cj in range(nrows):
        y0 = cj * cell_h
        for ci in range(ncols):
            in_center = c_lo <= ci < c_hi
            if bool(ctx.sides_only) and in_center:
                continue
            r_ring = _beat_ring_strength(ci, cj, ncols, nrows, uniforms)
            ch = _pick_char(ci, cj, t, uniforms, ring=r_ring)
            rgb = _color_animated(
                ci, cj, ncols, nrows, t, pal, uniforms
            )
            if r_ring > 0.01 and len(pal) >= 2:
                rgb = _lerp3(rgb, pal[min(3, len(pal) - 1)], 0.38 * r_ring)
                if len(pal) > 4:
                    rgb = _lerp3(rgb, pal[4], 0.2 * r_ring * r_ring)
            a = base_alpha * (0.72 + 0.28 * float((ci * 3 + cj * 5) % 5) / 4.0)
            a *= 1.0 + 0.55 * r_ring
            a = min(0.95, a)
            if in_center and not bool(ctx.sides_only):
                a *= cam
            p = _glyph_patch(ch, rgb, a, scale)
            _blend_premult_patch(out, p, y0, ci * cell_w)

    if pl_cat is not None and w_cat > 0.01 and not bool(omit_cat):
        _blit_voidcat_silhouette(
            out,
            w_cat=w_cat,
            pl=pl_cat,
            c_lo=c_lo,
            c_hi=c_hi,
            ncols=ncols,
            nrows=nrows,
            cell_w=cell_w,
            cell_h=cell_h,
            scale=scale,
            pal=pal,
        )

    return out


def sanity_check_ascii_layer(
    width: int = 320,
    height: int = 180,
) -> dict[str, float | int | bool]:
    """Return simple stats to verify the layer is **visible** and **beat-reactive**.

    Call before a long run: ``if not sanity_check_ascii_layer()[\"ok\"]: ...``

    No GPU; uses a minimal fake ``analysis`` and two ``beat_phase`` values so
    the average RGB should change (ring in different place).
    """
    fake: dict[str, Any] = {
        "beats": [0.0, 0.5, 1.0],
        "downbeats": [0.0, 2.0],
        "tempo": {"bpm": 120.0},
        "rms": {"fps": 20.0, "values": [0.3, 0.35, 0.4, 0.32]},
        "spectrum": {"fps": 20.0, "values": [[0.2] * 8] * 8},
        "onsets": {
            "peaks": [0.2, 0.5],
            "strength": [0.0, 0.0, 0.5, 0.0],
            "frame_rate_hz": 10.0,
        },
        "events": {"drops": []},
        "song_hash": "sanity",
    }
    from pipeline.reactive_shader import uniforms_at_time

    ctx = build_voidcat_ascii_context(fake, None)
    u = uniforms_at_time(fake, 0.5, num_bands=8)
    for k in (
        "bass_hit",
        "transient_lo",
        "transient_mid",
        "transient_hi",
        "drop_hold",
    ):
        u[k] = 0.0

    def _one(bp: float) -> np.ndarray:
        u2 = {**u, "beat_phase": bp}
        return render_voidcat_ascii_rgba(
            width, height, 0.0, uniforms=u2, ctx=ctx
        )

    r0 = _one(0.15)
    r1 = _one(0.85)
    m0 = float(r0[:, :, :3].mean())
    m1 = float(r1[:, :, :3].mean())
    a_max = int(max(r0[:, :, 3].max(), r1[:, :, 3].max()))
    dif = float(abs(m0 - m1))
    ok = m0 > 0.4 and m1 > 0.4 and a_max > 2 and dif > 0.05
    return {
        "mean_rgb_bp015": m0,
        "mean_rgb_bp085": m1,
        "mean_rgb_delta": dif,
        "max_channel_alpha_pair": a_max,
        "ok": ok,
    }


if __name__ == "__main__":
    r = sanity_check_ascii_layer()
    print(r)
    raise SystemExit(0 if r["ok"] else 1)
