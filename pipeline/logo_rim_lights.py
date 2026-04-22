"""
Preprocessing and per-frame field for advanced logo rim lighting (tasks 24, 25+).

``compute_logo_rim_prep`` (task 24) derives per-pixel line/stroke energy, alpha
weights, image-space centroid, and a confidence score from a single RGBA logo.

``compute_logo_rim_light_patch`` (tasks 25--27) consumes that prep plus a time
``t`` and a :class:`RimLightConfig` to produce a padded **premultiplied RGBA**
patch
ready for :func:`pipeline.logo_composite._blend_premult_rgba_patch`. It combines
an **outer halo** (radial falloff from the alpha edge), a configurable
**inward bleed** (distance transform into the shape), optional **line-local**
energy (when ``prep.use_line_features``), a **traveling angular wave**
``cos(waves * theta + 2*pi*phase_hz*t)`` centred on the logo centroid, and
(optional) **2--3** phase-offset / HSV-spread emissive **layers** with
``hue_drift_per_sec`` and ``song_hash``-seeded tints; halo-only logos are
limited to **two** tints.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import zlib
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

# Below this mean line energy (relative to effective alpha), ignore stroke features.
_MIN_LINE_CONFIDENCE_DEFAULT: Final[float] = 0.06


@dataclass(frozen=True, slots=True)
class LogoRimPrep:
    """Cached masks and flags for one prepared logo RGBA frame."""

    line_mask: np.ndarray
    """Float32 (H, W) in [0, 1]. Higher = more stroke/edge-like energy."""
    alpha_f: np.ndarray
    """Float32 (H, W) in [0, 1] from the original alpha channel."""
    centroid_xy: tuple[float, float]
    """(cx, cy) pixel coordinates in the same space as the logo array (x right, y down)."""
    line_confidence: float
    """Scalar in [0, 1]. Higher = trust stroke-local lighting."""
    use_line_features: bool
    """If False, later stages should use silhouette / halo only."""


def _scharr_magnitude(gray: np.ndarray) -> np.ndarray:
    """Edge magnitude in [0, +inf); gray is 2D float."""
    g = np.asarray(gray, dtype=np.float64)
    kx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float64)
    ky = kx.T
    ex = ndimage.convolve(g, kx, mode="nearest")
    ey = ndimage.convolve(g, ky, mode="nearest")
    return np.hypot(ex, ey).astype(np.float32, copy=False)


def compute_logo_rim_prep(
    logo_rgba: np.ndarray,
    *,
    min_line_confidence: float = _MIN_LINE_CONFIDENCE_DEFAULT,
    luma_bright_q: float = 0.72,
    edge_vs_luma: float = 0.45,
    morph_dilate: int = 0,
) -> LogoRimPrep:
    """
    Build line mask, alpha, centroid, and line-feature confidence.

    * **Luma** uses max(R, G, B) so white-on-dark line art is emphasized.
    * **Edges** use a Scharr magnitude on luma * alpha to respect transparency.
    * **Bright fill ratio** penalizes solid glyphs (mostly bright interior): sparse
      strokes get higher ``line_confidence`` than a flat white disc.

    Parameters
    ----------
    logo_rgba
        uint8 (H, W, 4) RGBA.
    min_line_confidence
        If ``line_confidence`` falls below this, ``use_line_features`` is False.
    luma_bright_q
        Quantile in (0, 1) for adaptive high-luma threshold on ``w = luma * alpha_f``.
    edge_vs_luma
        In [0, 1], weight of normalized edge map vs luma*alpha in the blend.
    morph_dilate
        Optional 1--2 px binary dilation on the *high* luma support (0 = off).
    """
    if logo_rgba.ndim != 3 or logo_rgba.shape[2] != 4:
        raise ValueError("logo_rgba must have shape (H, W, 4)")
    if not np.issubdtype(logo_rgba.dtype, np.unsignedinteger):
        raise TypeError("logo_rgba must be uint8 RGBA")
    h, w = int(logo_rgba.shape[0]), int(logo_rgba.shape[1])
    if h < 1 or w < 1:
        raise ValueError("logo_rgba has invalid dimensions")

    a_u8 = logo_rgba[:, :, 3].astype(np.float32)
    alpha_f = np.clip(a_u8 / 255.0, 0.0, 1.0)
    a_sum = float(np.sum(alpha_f))
    r = logo_rgba[:, :, 0].astype(np.float32) / 255.0
    gg = logo_rgba[:, :, 1].astype(np.float32) / 255.0
    b = logo_rgba[:, :, 2].astype(np.float32) / 255.0
    luma = np.maximum(np.maximum(r, gg), b)

    ys, xs = np.indices((h, w), dtype=np.float32)
    if a_sum < 1e-6:
        cxy = (float(w) * 0.5, float(h) * 0.5)
        z = np.zeros((h, w), dtype=np.float32)
        return LogoRimPrep(
            line_mask=z,
            alpha_f=alpha_f.astype(np.float32, copy=False),
            centroid_xy=cxy,
            line_confidence=0.0,
            use_line_features=False,
        )

    cx = float(np.sum(xs * alpha_f) / a_sum)
    cy = float(np.sum(ys * alpha_f) / a_sum)

    w_luma = luma * alpha_f
    m = w_luma > 1e-4
    if not np.any(m):
        z = np.zeros((h, w), dtype=np.float32)
        return LogoRimPrep(
            line_mask=z,
            alpha_f=alpha_f.astype(np.float32, copy=False),
            centroid_xy=(cx, cy),
            line_confidence=0.0,
            use_line_features=False,
        )

    # Adaptive: bright strokes (white lines) rise above the dark surround.
    q = float(
        np.clip(
            luma_bright_q,
            0.5,
            0.99,
        )
    )
    t_hi = float(np.quantile(w_luma[m], q))
    t_lo = float(np.quantile(w_luma[m], min(0.35, q * 0.5)))
    high = np.clip((w_luma - t_lo) / (t_hi - t_lo + 1e-6), 0.0, 1.0)
    if morph_dilate > 0:
        support = (high > 0.35).astype(np.uint8)
        for _ in range(int(np.clip(morph_dilate, 0, 3))):
            support = ndimage.binary_dilation(support, structure=np.ones((3, 3), dtype=bool))
        high = np.maximum(high, support.astype(np.float32) * 0.4)

    edge_mag = _scharr_magnitude(w_luma)
    e_max = float(np.max(edge_mag)) + 1e-9
    edge_n = (edge_mag / e_max).astype(np.float32, copy=False)

    ew = float(np.clip(edge_vs_luma, 0.0, 1.0))
    energy = (1.0 - ew) * w_luma + ew * edge_n
    e95 = float(np.quantile(energy[alpha_f > 0.02], 0.95)) + 1e-6
    line_mask = np.clip(energy / e95, 0.0, 1.0).astype(np.float32, copy=False)

    # Penalize solid bright fills: stroke art leaves much of the interior dark / low w_luma.
    mean_wm = float(np.sum(w_luma) / a_sum)
    mean_hi = float(np.sum((w_luma > t_hi * 0.85).astype(np.float32) * alpha_f) / a_sum)
    fill_penalty = np.clip((mean_hi - 0.15) / 0.75, 0.0, 1.0)
    spatial = float(np.sum(line_mask * alpha_f) / a_sum)
    line_confidence = float(
        np.clip(
            spatial * (1.0 - 0.55 * fill_penalty) + 0.15 * (mean_wm * (1.0 - 0.4 * mean_hi)),
            0.0,
            1.0,
        )
    )

    # Halo-only: do not feed stroke line_mask; compositor can use alpha_f for silhouette.
    use_line = line_confidence >= min_line_confidence and mean_hi <= 0.92
    if not use_line:
        line_mask = np.zeros((h, w), dtype=np.float32)

    return LogoRimPrep(
        line_mask=line_mask,
        alpha_f=alpha_f.astype(np.float32, copy=False),
        centroid_xy=(cx, cy),
        line_confidence=line_confidence,
        use_line_features=use_line,
    )


# ---------------------------------------------------------------------------
# Task 25: traveling-wave rim light field.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RimLightConfig:
    """Parameters for :func:`compute_logo_rim_light_patch`.

    Geometry is expressed in **patch pixels** (same scale as the prepared
    ``logo_rgba``). Time is in **seconds**. A single frame's patch is assembled
    as::

        emissive = ( halo + inward_mix * inward_bleed + line_boost * line_mask )
                   * angular_wave(theta, t)
                   * intensity

    The result is then premultiplied with ``rim_rgb`` and clamped.
    """

    pad_px: int = 24
    """Border added around the logo so halo / blur reach outside safely."""

    rim_rgb: tuple[int, int, int] = (255, 180, 220)
    """Emissive colour (linear-ish RGB, 0--255). Final alpha drives intensity."""

    intensity: float = 1.0
    """Overall multiplicative gain on the combined emissive field (>=0)."""

    opacity_pct: float = 100.0
    """Final opacity in [0, 100]; multiplies the premultiplied alpha."""

    halo_spread_px: float = 14.0
    """Outward falloff distance from the alpha edge (radial halo)."""

    halo_boost: float = 1.0
    """Weight of the outer halo term."""

    inward_mix: float = 0.5
    """In ``[0, 1]``. ``0`` = halo only; ``1`` = full inward bleed into the shape."""

    inward_depth_px: float = 18.0
    """How far (px) the inward light reads from the alpha edge toward the core."""

    line_boost: float = 1.0
    """Weight of ``prep.line_mask`` contribution (zero automatically when
    ``prep.use_line_features`` is False)."""

    blur_px: float = 3.0
    """Gaussian blur on the combined radial field (before angular modulation)."""

    waves: int = 3
    """Number of wave lobes around the centroid."""

    wave_sharpness: float = 1.5
    """Exponent applied to the cosine lobes; >=1 sharpens bright lights."""

    phase_hz: float = 0.25
    """Global phase speed; lobes complete ``phase_hz`` revolutions per second."""

    phase_offset: float = 0.0
    """Additional constant phase (radians)."""

    wave_floor: float = 0.0
    """Minimum of the angular modulation in ``[0, 1)``. ``0`` = full lobes,
    higher values keep a baseline rim glow between peaks."""

    # --- Task 26: multi-colour layers (set ``rim_color_layers`` to 2--3). ---
    rim_color_layers: int = 1
    """``1`` = single ``rim_rgb`` (task-25 path). ``2``--``3`` = weighted coloured
    layers sharing the same blurred base field, each with its own angular phase
    offset and HSV-spread tint."""

    color_spread_rad: float = 2.0 * math.pi / 3.0
    """HSV **hue** delta between **adjacent** layers, in **radians** (not the
    same unit as the 0--1 :mod:`colorsys` hue; divide by :math:`2\\pi` to
    convert). Default ``2\\pi/3`` (~120°) gives a readable **pink / cyan / violet**
    triad from a single magenta-leaning ``rim_rgb``."""

    layer_phase_offsets: tuple[float, ...] = ()
    """Radians **added to** the travelling-phase ``phi`` per layer, after
    ``phase_offset``. If empty, defaults are: two layers → ``(0, \\pi)``; three
    → ``(0, 2\\pi/3, 4\\pi/3)`` so the lobes stay interleaved."""

    hue_drift_per_sec: float = 0.0
    """Number of full **0--1** hue sweeps per second (i.e. Hz on the HSV
    **hue** ring) applied identically to every layer after ``color_spread`` and
    ``song_hash`` — smooth colour motion without affecting wave ``phase_hz``."""

    song_hash: str | int | bytes | None = None
    """Optional value mixed into a small stable per-run hue offset; same song
    and config yield the same **bytes**; change this to vary the palette without
    touching ``rim_rgb`` (Adler-32 of UTF-8 when ``str``; integer truncated)."""

    flicker_amount: float = 0.0
    """In ``[0, 1]`` — scales a ~11 Hz sinusoid on the total emissive **gain**;
    zero disables (default). Slight **neon** shimmer, no extra RNG."""


# ---------------------------------------------------------------------------
# Task 27: audio-reactive multipliers (snare + bass) in absolute time.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RimAudioModulation:
    """Per-frame scalers for :func:`compute_logo_rim_light_patch`.

    ``glow_strength_mul`` emulates a snare-driven intensity lift (analog to the
    classic neon ``glow_amount``). ``phase_offset_rad`` nudges the travelling
    wave (brief snare “kick” after slew). ``inward_strength_mul`` modulates
    how far light reads into the glyph from the **bass** envelope (breathing).
    Identity values keep the unmodulated look.
    """

    glow_strength_mul: float = 1.0
    """Multiplies the combined emissive **gain** (``intensity * opacity * …``)."""

    phase_offset_rad: float = 0.0
    """Added to the angular wave phase (radians) after ``RimLightConfig`` offsets."""

    inward_strength_mul: float = 1.0
    """Scales the effective **inward_mix** (``inward_mix * this``, clamped to ``[0,1]``)."""


@dataclass
class RimModulationState:
    """Slew memory for :func:`advance_rim_audio_modulation` (per render / thread)."""

    snare_slew: float = 0.0
    bass_slew: float = 0.0


@dataclass(frozen=True, slots=True)
class RimAudioTuning:
    """Declared ranges and smoothing times; safe defaults match compositor use."""

    # Snare → glow gain: ``1 + global_strength * glow_snare_max_delta * snare``.
    global_strength: float = 1.0
    glow_snare_max_delta: float = 0.36
    # Snare (post-slew) → extra angular phase, radians at ``snare_slew=1``.
    phase_snare_max_rad: float = 0.55
    # Bass (post-slew) → inward spread: ``1 + d * (2 * bass_slew - 1)``, ``d`` below.
    inward_bass_max_delta: float = 0.12
    snare_phase_attack_sec: float = 0.012
    snare_phase_release_sec: float = 0.095
    bass_slew_tau_sec: float = 0.14


def _slew_toward(current: float, target: float, dt_sec: float, tau_sec: float) -> float:
    t = max(1e-5, float(tau_sec))
    a = 1.0 - math.exp(-float(dt_sec) / t)
    return current + (float(target) - current) * a


def rim_modulation_instant(
    snare_env: float,
    bass_env: float,
    *,
    tuning: RimAudioTuning = RimAudioTuning(),
) -> RimAudioModulation:
    """Map raw ``[0,1]`` envelopes to modulation (no state / no smoothing)."""
    s = max(0.0, min(1.0, float(snare_env)))
    b = max(0.0, min(1.0, float(bass_env)))
    g = float(tuning.global_strength)
    d = float(tuning.glow_snare_max_delta) * g
    glow = max(0.6, min(1.6, 1.0 + d * s))
    ph = g * float(tuning.phase_snare_max_rad) * s
    ib = float(tuning.inward_bass_max_delta) * g
    inward = 1.0 + ib * (2.0 * b - 1.0)
    inward = max(1.0 - ib, min(1.0 + ib, inward))
    return RimAudioModulation(
        glow_strength_mul=glow, phase_offset_rad=ph, inward_strength_mul=inward
    )


def advance_rim_audio_modulation(
    state: RimModulationState,
    *,
    snare_env: float,
    bass_env: float,
    dt_sec: float,
    tuning: RimAudioTuning = RimAudioTuning(),
) -> RimAudioModulation:
    """
    One audio clock step. Snare drives **glow** and (after asymmetric slew)
    **phase**; bass is lightly slewed and drives **inward** spread.

    ``dt_sec`` should match the real frame step (e.g. ``1/fps``) so attack /
    release match the ear on the snare/mid track.
    """
    s = max(0.0, min(1.0, float(snare_env)))
    b = max(0.0, min(1.0, float(bass_env)))
    dt = max(1e-6, float(dt_sec))
    g = float(tuning.global_strength)
    d_g = float(tuning.glow_snare_max_delta) * g
    d_ph = float(tuning.phase_snare_max_rad) * g
    ib = float(tuning.inward_bass_max_delta) * g

    # Asymmetric slew on snare for phase only (kills 1-frame stutter on kicks).
    if s >= state.snare_slew:
        tau = float(tuning.snare_phase_attack_sec)
    else:
        tau = float(tuning.snare_phase_release_sec)
    state.snare_slew = _slew_toward(state.snare_slew, s, dt, tau)

    tb = max(1e-4, float(tuning.bass_slew_tau_sec))
    ab = 1.0 - math.exp(-dt / tb)
    state.bass_slew = state.bass_slew + (b - state.bass_slew) * ab

    glow = max(0.6, min(1.6, 1.0 + d_g * s))
    phase = d_ph * state.snare_slew
    inward = 1.0 + ib * (2.0 * state.bass_slew - 1.0)
    inward = max(1.0 - ib, min(1.0 + ib, inward))
    return RimAudioModulation(
        glow_strength_mul=glow,
        phase_offset_rad=phase,
        inward_strength_mul=inward,
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep clamped to ``[0, 1]``."""
    x = np.clip(x, 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32, copy=False)


def _gaussian_blur_f32(arr: np.ndarray, radius_px: float) -> np.ndarray:
    """Gaussian blur a float32 2D array using PIL (scaled to uint8 internally).

    The field is rescaled to ``[0, 255]`` for PIL then restored; this keeps
    the single-channel blur fast without pulling SciPy's FFT path in.
    """
    if radius_px <= 0.25:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-6:
        return arr
    norm = (arr - lo) / (hi - lo)
    u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(radius_px)))
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return (out * (hi - lo) + lo).astype(np.float32, copy=False)


def _song_hash_bytes(h: str | int | bytes | None) -> bytes | None:
    if h is None:
        return None
    if isinstance(h, bytes):
        return h
    if isinstance(h, int):
        return str(int(h)).encode("utf-8", errors="replace")
    return str(h).encode("utf-8", errors="replace")


def _stable_hue_bump01(song_hash: str | int | bytes | None) -> float:
    """Deterministic small offset in ``[0, 1)`` for HSV hue repeatability."""
    b = _song_hash_bytes(song_hash)
    if not b:
        return 0.0
    # Adler-32: fast, same song → same nudge; pair with sha256 for a wider range.
    a = zlib.adler32(b) & 0xFFFFFFFF
    s = hashlib.sha256(b).digest()
    w = int.from_bytes(s[:4], "little", signed=False) ^ a
    return (w % 1_000_000) / 1_000_000.0 * 0.12


def _flicker_gain(t: float, flicker_amount: float) -> float:
    u = float(np.clip(flicker_amount, 0.0, 1.0))
    if u <= 1e-9:
        return 1.0
    wobble = 0.5 + 0.5 * math.sin(2.0 * math.pi * 11.0 * float(t))
    return 1.0 - u + u * wobble


def _effective_rim_color_layers(
    use_line_features: bool, requested: int, multicolor: bool
) -> int:
    """Halo / dual-tone: without stroke features, at most two tints (spec)."""
    r = int(np.clip(int(requested), 1, 3))
    if not multicolor or r <= 1:
        return 1
    if not use_line_features and r > 1:
        return 2
    return r


def _default_layer_phase_offsets(n: int) -> tuple[float, ...]:
    if n <= 1:
        return (0.0,)
    if n == 2:
        return (0.0, float(math.pi))
    return (0.0, float(2.0 * math.pi / 3.0), float(4.0 * math.pi / 3.0))


def _resolve_layer_phase_offsets(
    n: int, custom: tuple[float, ...]
) -> tuple[float, ...]:
    d = _default_layer_phase_offsets(n)
    if custom and len(custom) >= n:
        return tuple(float(custom[i]) for i in range(n))
    return d


def _srgb_to_linear_f32(x: np.ndarray) -> np.ndarray:
    s = np.clip(x.astype(np.float32), 0.0, 1.0)
    lo = s <= 0.04045
    return np.where(lo, s / 12.92, ((s + 0.055) / 1.055) ** 2.4).astype(np.float32, copy=False)


def _linear_to_srgb_f32(x: np.ndarray) -> np.ndarray:
    s = np.clip(x.astype(np.float32), 0.0, 1.0)
    lo = s <= 0.0031308
    hi = 1.055 * (np.power(np.maximum(s, 0.0), 1.0 / 2.4) - 0.055)
    return np.where(lo, 12.92 * s, hi).astype(np.float32, copy=False)


def _layer_srgb_tints(
    n: int,
    rim_rgb: tuple[int, int, int],
    *,
    color_spread_rad: float,
    t: float,
    song_hash: str | int | bytes | None,
    hue_drift_per_sec: float,
) -> list[tuple[int, int, int]]:
    # Neon-friendly baseline: spread hues from a single rim_rgb, slightly boost
    # saturation (pink / cyan / violet triad is the reference look).
    r0, g0, b0 = (c / 255.0 for c in rim_rgb)
    h, l, s = colorsys.rgb_to_hls(r0, g0, b0)
    s = min(1.0, s * 1.12)
    h = (h + _stable_hue_bump01(song_hash) + (hue_drift_per_sec * float(t))) % 1.0
    spread = float(color_spread_rad) / (2.0 * math.pi)
    out: list[tuple[int, int, int]] = []
    for k in range(n):
        hk = (h + k * spread) % 1.0
        r, g, b = colorsys.hls_to_rgb(hk, l, s)
        out.append(
            (
                int(np.clip(round(r * 255.0), 0, 255)),
                int(np.clip(round(g * 255.0), 0, 255)),
                int(np.clip(round(b * 255.0), 0, 255)),
            )
        )
    return out


def rim_base_rgb_from_preset(shadow_hex: str | None, base_hex: str) -> tuple[int, int, int]:
    """Primary rim glow / shadow tint from a preset, matching the neon path."""
    from pipeline.logo_composite import resolve_logo_glow_rgb

    return resolve_logo_glow_rgb(shadow_hex, base_hex)


def _angular_modulation(
    theta: np.ndarray, t: float, config: RimLightConfig, extra_phase: float
) -> np.ndarray:
    phi = (
        2.0
        * float(np.pi)
        * float(config.phase_hz)
        * float(t)
        + float(config.phase_offset)
        + float(extra_phase)
    )
    waves = max(1, int(config.waves))
    sharpness = max(1.0, float(config.wave_sharpness))
    raw = 0.5 + 0.5 * np.cos(waves * theta + phi, dtype=np.float32)
    raw = np.power(raw, sharpness, dtype=np.float32)
    floor = float(np.clip(config.wave_floor, 0.0, 0.95))
    return (floor + (1.0 - floor) * raw).astype(np.float32, copy=False)


def _build_zero_patch(h: int, w: int, pad: int) -> tuple[np.ndarray, int]:
    """Empty premultiplied RGBA patch matching the (H+2p, W+2p) contract."""
    return np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8), pad


def compute_logo_rim_light_patch(
    prep: LogoRimPrep,
    *,
    t: float,
    config: RimLightConfig = RimLightConfig(),
    audio_mod: RimAudioModulation | None = None,
) -> tuple[np.ndarray, int]:
    """Build a padded premultiplied-RGBA rim-light patch for one frame.

    Returns ``(patch, pad)`` where ``patch`` has shape ``(H + 2*pad, W + 2*pad, 4)``
    and is intended to be blended at ``(x0 - pad, y0 - pad)`` via
    :func:`pipeline.logo_composite._blend_premult_rgba_patch` -- matching the
    :func:`pipeline.logo_composite.build_classic_neon_glow_patch` contract.

    Optional ``audio_mod`` (task 27) applies time-varying **snare** / **bass**
    scalers in absolute time ``t``; ``None`` keeps the unmodulated look.

    The field combines:

    * an **outer halo** that falls off radially from the alpha edge with
      :func:`_smoothstep` over ``halo_spread_px``;
    * an **inward bleed** weighted by the distance transform inside the alpha,
      so thin strokes light up all the way through while wide fills only glow
      near their rim (gated by ``inward_mix``);
    * the prep's **line mask** (stroke-local energy) when
      ``prep.use_line_features`` is True -- otherwise this term is zero and
      the result degrades cleanly to a halo-only rim;
    * a **traveling angular wave** ``cos(waves * theta + phi(t))`` around
      ``prep.centroid_xy`` that rotates the bright lobes over time at
      ``config.phase_hz`` cycles per second.

    The returned patch is fully premultiplied (``rgb = rim_rgb * a`` in the
    single-colour case; multiple layers add linear sRGB, then re-associate for
    premult, see implementation) and never produces NaNs.
    """
    alpha_f = prep.alpha_f
    if alpha_f.ndim != 2:
        raise ValueError("prep.alpha_f must be 2D")
    h, w = int(alpha_f.shape[0]), int(alpha_f.shape[1])
    pad = max(0, int(config.pad_px))
    op = float(np.clip(config.opacity_pct, 0.0, 100.0)) / 100.0
    gain = float(max(0.0, config.intensity)) * op
    if audio_mod is not None:
        gain *= max(0.0, min(2.0, float(audio_mod.glow_strength_mul)))
    if h < 1 or w < 1 or gain <= 1e-6:
        return _build_zero_patch(max(h, 1), max(w, 1), pad)
    am_phase = float(audio_mod.phase_offset_rad) if audio_mod is not None else 0.0

    n_layers = _effective_rim_color_layers(
        prep.use_line_features, int(config.rim_color_layers), int(config.rim_color_layers) > 1
    )
    fl = _flicker_gain(float(t), float(config.flicker_amount))

    H, W = h + 2 * pad, w + 2 * pad
    alpha_pad = np.zeros((H, W), dtype=np.float32)
    alpha_pad[pad : pad + h, pad : pad + w] = alpha_f
    line_pad = np.zeros((H, W), dtype=np.float32)
    if prep.use_line_features and float(np.max(prep.line_mask)) > 0.0:
        line_pad[pad : pad + h, pad : pad + w] = prep.line_mask

    alpha_bin = alpha_pad > 0.5
    if not np.any(alpha_bin):
        return _build_zero_patch(h, w, pad)

    # Distance transforms: outside gives how far a pixel sits from the logo
    # edge (radial halo), inside gives how deep it sits within the shape
    # (inward bleed falls off with depth, so thin strokes fully light up).
    dt_out = ndimage.distance_transform_edt(~alpha_bin).astype(np.float32, copy=False)
    dt_in = ndimage.distance_transform_edt(alpha_bin).astype(np.float32, copy=False)

    halo_spread = max(1e-3, float(config.halo_spread_px))
    halo = _smoothstep(1.0 - dt_out / halo_spread) * max(0.0, float(config.halo_boost))

    inward_depth = max(1e-3, float(config.inward_depth_px))
    inward_w = _smoothstep(1.0 - dt_in / inward_depth) * alpha_pad
    inward_mix = float(np.clip(config.inward_mix, 0.0, 1.0))
    if audio_mod is not None:
        im = min(
            1.0,
            inward_mix
            * max(0.0, min(2.0, float(audio_mod.inward_strength_mul))),
        )
    else:
        im = inward_mix
    inward = inward_w * im

    line_term = line_pad * max(0.0, float(config.line_boost))

    base = halo + inward + line_term
    base = _gaussian_blur_f32(base, float(config.blur_px))

    # Travelling angular wave. Centroid is in patch coords; shift to padded.
    cx, cy = prep.centroid_xy
    ys, xs = np.indices((H, W), dtype=np.float32)
    theta = np.arctan2(ys - (float(cy) + pad), xs - (float(cx) + pad))

    if n_layers <= 1:
        ang = _angular_modulation(theta, float(t), config, 0.0 + am_phase)
        emissive = np.clip(base * ang * gain * fl, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.any(emissive):
            return _build_zero_patch(h, w, pad)
        a_u8 = (emissive * 255.0).astype(np.uint8)
        r_u8 = (emissive * float(config.rim_rgb[0])).astype(np.uint8)
        g_u8 = (emissive * float(config.rim_rgb[1])).astype(np.uint8)
        b_u8 = (emissive * float(config.rim_rgb[2])).astype(np.uint8)
        return np.stack([r_u8, g_u8, b_u8, a_u8], axis=-1), pad

    # Multi-colour: re-use the same scalar ``base``; one angular modulation per
    # layer, colours from HSV spread + ``hue_drift_per_sec`` + ``song_hash``.
    phases = _resolve_layer_phase_offsets(n_layers, config.layer_phase_offsets)
    w_layer = 1.0 / float(n_layers)
    rim3 = (int(config.rim_rgb[0]), int(config.rim_rgb[1]), int(config.rim_rgb[2]))
    tints = _layer_srgb_tints(
        n_layers,
        rim3,
        color_spread_rad=float(config.color_spread_rad),
        t=float(t),
        song_hash=config.song_hash,
        hue_drift_per_sec=float(config.hue_drift_per_sec),
    )
    e_acc = np.zeros((H, W), dtype=np.float32)
    r_acc = np.zeros((H, W), dtype=np.float32)
    g_acc = np.zeros((H, W), dtype=np.float32)
    b_acc = np.zeros((H, W), dtype=np.float32)
    for k in range(n_layers):
        ang = _angular_modulation(theta, float(t), config, phases[k] + am_phase)
        e = np.clip(base * ang * gain * w_layer * fl, 0.0, 1.0).astype(np.float32, copy=False)
        r_k, g_k, b_k = tints[k]
        lrgb = _srgb_to_linear_f32(
            np.array([r_k, g_k, b_k], dtype=np.float32) / 255.0
        )
        e_acc += e
        r_acc += e * lrgb[0]
        g_acc += e * lrgb[1]
        b_acc += e * lrgb[2]

    if not np.any(e_acc):
        return _build_zero_patch(h, w, pad)

    s = np.clip(e_acc, 0.0, 1.0)
    s_safe = np.maximum(s, 1e-6)
    c_r = r_acc / s_safe
    c_g = g_acc / s_safe
    c_b = b_acc / s_safe
    c_r = np.clip(c_r, 0.0, 1.0)
    c_g = np.clip(c_g, 0.0, 1.0)
    c_b = np.clip(c_b, 0.0, 1.0)
    r_s = _linear_to_srgb_f32(c_r)
    g_s = _linear_to_srgb_f32(c_g)
    b_s = _linear_to_srgb_f32(c_b)
    a_u8 = (s * 255.0).astype(np.uint8)
    r_u8 = (r_s * s * 255.0).clip(0.0, 255.0).astype(np.uint8)
    g_u8 = (g_s * s * 255.0).clip(0.0, 255.0).astype(np.uint8)
    b_u8 = (b_s * s * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return np.stack([r_u8, g_u8, b_u8, a_u8], axis=-1), pad
