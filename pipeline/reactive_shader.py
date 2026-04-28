"""
Offscreen :mod:`moderngl` reactive shader pass.

Loads a fragment shader from :data:`config.SHADERS_DIR` (paired with the shared
:file:`passthrough.vert`), creates a standalone GL context with an RGBA FBO at
the target resolution, and renders single frames driven by analysis-derived
uniforms (``beat_phase``, ``bar_phase``, ``band_energies[8]``, ``rms``,
``onset_pulse``, ``onset_env``, ``build_tension``, ``time``, ``intensity``).

The main entry points are :class:`ReactiveShader` and :func:`uniforms_at_time`,
which together let the compositor map an ``analysis.json`` bundle onto the
shader uniforms for any wall-clock time ``t``. All GL failures are raised as
:class:`RuntimeError` with enough context to debug headless/container hosts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import moderngl
import numpy as np

from config import SHADERS_DIR

from .builtin_shaders import BUILTIN_SHADERS

LOGGER = logging.getLogger(__name__)

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_NUM_BANDS = 8
DEFAULT_SHADER = "spectrum_bars"
DEFAULT_VERTEX_SHADER = "passthrough.vert"

# exp(-ONSET_DECAY_PER_SEC * dt) collapses to ~0 after roughly 0.5 s.
ONSET_DECAY_PER_SEC = 6.0

# Normalisation percentile for the continuous onset-strength envelope.
# Dividing by the 95th percentile (instead of the raw max) keeps a single
# outlier spike from crushing the rest of the track into near-zero.
ONSET_ENV_NORM_PERCENTILE = 95.0

# Key under which the normalised onset envelope is memoised on the analysis
# dict. Caching is keyed by ``id(strength_list)`` so a swapped-in analysis
# bundle with a new ``strength`` array naturally misses and recomputes.
ONSET_ENV_CACHE_KEY = "_onset_env_cache"

# Memoise synthetic downbeats derived from the beat list (``beats_per_bar`` stride)
# on the in-memory analysis dict for the duration of a render — avoids O(n)
# Python list builds thousands of times per song.
_BAR_PHASE_SYNTH_CACHE_KEY = "_gf_bar_phase_synth_v1"

# Palette uniform layout. Must match ``uniform vec3 u_palette[PALETTE_SLOTS]``
# in every fragment shader that reads the preset palette. PALETTE_SLOTS is a
# fixed ceiling; presets shorter than this are padded by repeating the last
# color, and the shader is told the "real" length via ``u_palette_size`` so it
# can use ``band % u_palette_size`` style indexing without reading padding.
PALETTE_SLOTS = 5
DEFAULT_PALETTE: tuple[str, ...] = (
    "#1A0A2E",  # deep indigo
    "#4361EE",  # cobalt
    "#4CC9F0",  # cyan
    "#F72585",  # magenta
    "#FFD166",  # amber
)

_BACKGROUND_TEXTURE_UNIT = 0

# Second sampler unit reserved for the optional feedback / warp framebuffer
# (``u_prev_frame``). Kept distinct from the background unit so the compositor
# path (``u_background`` on unit 0) and the Milkdrop-style feedback loop on
# unit 1 can coexist without a manual rebind step per frame.
_PREV_FRAME_TEXTURE_UNIT = 1


def _parse_hex_color(hex_str: str) -> tuple[float, float, float]:
    """Strict ``#RRGGBB`` (case-insensitive) to ``(r, g, b)`` floats in ``[0, 1]``."""
    s = str(hex_str).strip()
    if len(s) != 7 or s[0] != "#":
        raise ValueError(
            f"palette color must be '#RRGGBB' (7 chars, leading '#'), got {hex_str!r}"
        )
    try:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
    except ValueError as exc:
        raise ValueError(f"invalid hex in palette color {hex_str!r}: {exc}") from exc
    return (r / 255.0, g / 255.0, b / 255.0)


def _build_palette_uniform(
    palette: Sequence[str] | None,
) -> tuple[list[float], int]:
    """
    Parse a hex palette into ``(flat_rgb_floats, effective_size)``.

    The returned flat list always has ``PALETTE_SLOTS * 3`` floats so it lines
    up with the fixed-size GLSL uniform array. If ``palette`` is shorter than
    ``PALETTE_SLOTS`` the last color is repeated for padding, and
    ``effective_size`` reports the original (non-padded) length so the shader
    can index modulo that value instead of walking into repeated padding.
    ``None`` or empty input falls back to :data:`DEFAULT_PALETTE`.
    """
    colors: list[str] = [str(c) for c in (palette or ())]
    if not colors:
        colors = list(DEFAULT_PALETTE)
    parsed: list[tuple[float, float, float]] = [_parse_hex_color(c) for c in colors]
    effective = min(len(parsed), PALETTE_SLOTS)
    if effective <= 0:
        raise ValueError("palette must contain at least one color")
    while len(parsed) < PALETTE_SLOTS:
        parsed.append(parsed[-1])
    parsed = parsed[:PALETTE_SLOTS]
    flat: list[float] = []
    for r, g, b in parsed:
        flat.extend([float(r), float(g), float(b)])
    return flat, effective


# ---------------------------------------------------------------------------
# Shader resolution & CPU composite (for tests / non-GL callers)
# ---------------------------------------------------------------------------


def resolve_builtin_shader_stem(
    shader_stem: str,
    *,
    shaders_dir: Path | None = None,
) -> str:
    """
    Return ``shader_stem`` if it is allowlisted and ``{stem}.frag`` exists.

    Raises ``ValueError`` for unknown stems and ``FileNotFoundError`` when the
    fragment file is missing — no fallback shader.
    """
    stem = str(shader_stem).strip()
    if not stem:
        raise ValueError("shader stem must be non-empty")
    if stem not in BUILTIN_SHADERS:
        raise ValueError(
            f"unknown reactive shader {stem!r}; expected one of {list(BUILTIN_SHADERS)}"
        )
    root = Path(shaders_dir) if shaders_dir is not None else SHADERS_DIR
    frag = root / f"{stem}.frag"
    if not frag.is_file():
        raise FileNotFoundError(f"Fragment shader not found: {frag}")
    return stem


def composite_premultiplied_rgba_over_rgb(
    rgba: np.ndarray,
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Alpha-blend premultiplied ``(H, W, 4)`` ``uint8`` RGBA over ``(H, W, 3)``
    ``uint8`` RGB. Returns ``(H, W, 3)`` ``uint8``.
    """
    a = np.asarray(rgba, dtype=np.uint8)
    b = np.asarray(rgb, dtype=np.uint8)
    if a.shape[-1] != 4:
        raise ValueError(f"rgba must have shape (H, W, 4), got {a.shape}")
    if b.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {b.shape}")
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"shape mismatch: {a.shape[:2]} vs {b.shape[:2]}")

    src = a.astype(np.float32) / 255.0
    dst = b.astype(np.float32) / 255.0
    sa = src[..., 3:4]
    out = src[..., :3] + dst * (1.0 - sa)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Uniform helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShaderUniforms:
    """Typed bag of reactive uniforms; use :meth:`as_dict` for rendering."""

    time: float = 0.0
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    rms: float = 0.0
    onset_pulse: float = 0.0
    onset_env: float = 0.0
    build_tension: float = 0.0
    intensity: float = 1.0
    band_energies: tuple[float, ...] = tuple([0.0] * DEFAULT_NUM_BANDS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "beat_phase": float(self.beat_phase),
            "bar_phase": float(self.bar_phase),
            "rms": float(self.rms),
            "onset_pulse": float(self.onset_pulse),
            "onset_env": float(self.onset_env),
            "build_tension": float(self.build_tension),
            "intensity": float(self.intensity),
            "band_energies": [float(x) for x in self.band_energies],
        }


def _bisect_right_floats(seq: Sequence[float], t: float) -> int:
    """Pure-Python :func:`bisect.bisect_right` that tolerates non-list inputs."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if t < seq[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _beat_phase_at(beats: Sequence[float], t: float, *, bpm: float) -> float:
    """
    Phase within the current beat interval in ``[0, 1)``.

    Uses neighbouring beat times when available; extrapolates with a fallback
    period derived from ``bpm`` when ``t`` is outside the beat grid.
    """
    if not beats:
        return 0.0

    idx = _bisect_right_floats(beats, t)
    if 0 < idx < len(beats):
        prev_t = float(beats[idx - 1])
        next_t = float(beats[idx])
        span = next_t - prev_t
        if span <= 1e-6:
            return 0.0
        return float(np.clip((t - prev_t) / span, 0.0, 1.0))

    fallback = 60.0 / bpm if bpm and bpm > 1e-3 else 0.5
    if fallback <= 1e-6:
        return 0.0

    if idx == 0:
        anchor = float(beats[0])
        delta = (anchor - t) % fallback
        phase = 1.0 - (delta / fallback)
    else:
        anchor = float(beats[-1])
        delta = (t - anchor) % fallback
        phase = delta / fallback
    return float(np.clip(phase, 0.0, 1.0))


def _interp_scalar_series(
    values: Sequence[float] | None, t: float, fps: float
) -> float:
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


def _interp_scalar_series_1d_np(values: np.ndarray, t: float, fps: float) -> float:
    """Like :func:`_interp_scalar_series` but samples a 1D float array without ``tolist()``."""
    if values.size == 0 or fps <= 0.0:
        return 0.0
    idx_f = float(t) * float(fps)
    if idx_f <= 0.0:
        return float(values[0])
    n = int(values.size)
    if idx_f >= n - 1:
        return float(values[-1])
    i0 = int(idx_f)
    i1 = min(i0 + 1, n - 1)
    frac = idx_f - i0
    return float((1.0 - frac) * float(values[i0]) + frac * float(values[i1]))


def _interp_bands(
    values: Any,
    t: float,
    fps: float,
    num_bands: int,
) -> list[float]:
    """Linear time-axis interpolation of ``(frames, bands)`` spectrum data."""
    empty = [0.0] * num_bands
    if values is None or fps <= 0.0:
        return empty
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return empty

    n_frames = arr.shape[0]
    idx_f = float(t) * float(fps)
    if idx_f <= 0.0:
        row = arr[0]
    elif idx_f >= n_frames - 1:
        row = arr[-1]
    else:
        i0 = int(idx_f)
        i1 = min(i0 + 1, n_frames - 1)
        frac = idx_f - i0
        row = (1.0 - frac) * arr[i0] + frac * arr[i1]

    row = np.clip(row, 0.0, 1.0)
    out = row.tolist()
    if len(out) >= num_bands:
        return [float(x) for x in out[:num_bands]]
    out = [float(x) for x in out]
    out.extend([0.0] * (num_bands - len(out)))
    return out


def _onset_pulse_at(
    peaks: Sequence[float], t: float, *, decay: float = ONSET_DECAY_PER_SEC
) -> float:
    if not peaks:
        return 0.0
    idx = _bisect_right_floats(peaks, t)
    if idx == 0:
        return 0.0
    last = float(peaks[idx - 1])
    dt = max(0.0, float(t) - last)
    return float(np.exp(-decay * dt))


def _normalise_onset_strength(
    strength: Sequence[float],
    *,
    percentile: float = ONSET_ENV_NORM_PERCENTILE,
) -> np.ndarray:
    """
    Scale a raw onset-strength series into a roughly ``[0, 1]`` envelope.

    Divides by the ``percentile``-th percentile (default 95th) so a single
    outlier spike can't crush the rest of the track into near-zero, then
    clips to ``[0, 1]``. Empty / non-positive inputs yield an empty array.
    """
    arr = np.asarray(strength, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    pivot = float(np.percentile(arr, float(percentile)))
    if pivot <= 1e-6:
        # Completely silent / flat analysis: nothing meaningful to normalise.
        return np.zeros_like(arr)
    return np.clip(arr / pivot, 0.0, 1.0).astype(np.float32, copy=False)


def _interp_onset_strength(analysis: Mapping[str, Any], t: float) -> float:
    """
    Sample the continuous onset-strength envelope at time ``t`` (seconds).

    Reads ``analysis['onsets']`` (schema: ``{strength, frame_rate_hz, ...}``)
    and linearly interpolates between frames. The sampler uses
    ``frame_rate_hz`` from the onset block itself — **not** ``analysis.fps``
    — because onset analysis runs at a different hop than the mel spectrum.

    Normalisation (divide by 95th percentile, clip to ``[0, 1]``) is done
    once per ``strength`` array and memoised on the analysis dict under
    :data:`ONSET_ENV_CACHE_KEY`, keyed by ``id(strength)``. Caching on the
    dict (rather than a module-global ``WeakValueDictionary``) keeps the
    helper stateless between songs and lets garbage collection drop the
    cache when the analysis dict does.

    Returns ``0.0`` for ``t < 0``, when the block is missing, when
    ``strength`` is empty, or when ``frame_rate_hz`` is non-positive.
    """
    if float(t) < 0.0:
        return 0.0
    onsets = analysis.get("onsets") if isinstance(analysis, Mapping) else None
    if not isinstance(onsets, Mapping):
        return 0.0

    strength = onsets.get("strength")
    if not strength:
        return 0.0

    # Onset analysis uses its own hop; accept ``fps`` as a legacy alias but
    # prefer ``frame_rate_hz`` since that's what ``_onset_features`` emits.
    rate_raw = onsets.get("frame_rate_hz")
    if rate_raw is None:
        rate_raw = onsets.get("fps")
    try:
        rate = float(rate_raw) if rate_raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    if rate <= 0.0:
        return 0.0

    cache_key = id(strength)
    cache = analysis.get(ONSET_ENV_CACHE_KEY) if isinstance(analysis, Mapping) else None
    normalised: np.ndarray | None = None
    if isinstance(cache, dict):
        cached = cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            normalised = cached
    if normalised is None:
        normalised = _normalise_onset_strength(strength)
        # Best-effort write-through cache. Mapping may be read-only (e.g.
        # ``MappingProxyType`` in tests) — swallow the failure and just
        # recompute next time rather than raising on a performance path.
        try:
            if not isinstance(cache, dict):
                cache = {}
                analysis[ONSET_ENV_CACHE_KEY] = cache  # type: ignore[index]
            cache[cache_key] = normalised
        except (TypeError, KeyError):
            pass

    if normalised.size == 0:
        return 0.0
    return _interp_scalar_series_1d_np(normalised, float(t), rate)


def _bar_phase_at(
    t: float,
    downbeats: Sequence[float],
    *,
    beats_per_bar: int = 4,
    bpm: float | None = None,
    beats: Sequence[float] | None = None,
    cache_parent: MutableMapping[str, Any] | None = None,
) -> float:
    """
    Phase within the current ``beats_per_bar``-beat bar in ``[0, 1)``.

    Resolution order:

    1. **Downbeats given.** Interpolate between the nearest downbeat
       boundaries, so ``t`` halfway between two consecutive downbeats
       returns ``0.5``. Outside the downbeat grid, extrapolate using the
       median downbeat span as the bar period.
    2. **Only beats given.** Group beats into runs of ``beats_per_bar``
       and treat every ``beats_per_bar``-th beat as a synthetic downbeat,
       then interpolate as above.
    3. **Fallback.** With neither grid, use ``60 / bpm * beats_per_bar``
       as the bar period and phase-lock to ``t = 0``.
    """
    if beats_per_bar <= 0:
        raise ValueError(f"beats_per_bar must be positive, got {beats_per_bar}")

    t_f = float(t)

    def _phase_in_span(prev: float, nxt: float) -> float:
        span = nxt - prev
        if span <= 1e-6:
            return 0.0
        return float(np.clip((t_f - prev) / span, 0.0, 1.0)) % 1.0

    def _extrapolate(anchors: Sequence[float], period: float) -> float:
        if period <= 1e-6:
            return 0.0
        if t_f < float(anchors[0]):
            delta = (float(anchors[0]) - t_f) % period
            phase = 1.0 - (delta / period)
        else:
            delta = (t_f - float(anchors[-1])) % period
            phase = delta / period
        return float(np.clip(phase, 0.0, 1.0)) % 1.0

    # 1. Downbeat grid (use the live sequence — no per-frame list copy).
    if downbeats:
        db = downbeats
        idx = _bisect_right_floats(db, t_f)
        if 0 < idx < len(db):
            return _phase_in_span(float(db[idx - 1]), float(db[idx]))
        if len(db) >= 2:
            spans = np.diff(np.asarray(db, dtype=np.float64))
            period = float(np.median(spans))
        elif bpm and bpm > 1e-3:
            period = 60.0 / float(bpm) * float(beats_per_bar)
        else:
            period = 0.0
        return _extrapolate(db, period)

    # 2. Beat grid — treat every beats_per_bar-th beat as a synthetic downbeat.
    if beats:
        n_beats = len(beats)
        synth: Sequence[float]
        cached: Any = None
        if cache_parent is not None:
            try:
                cached = cache_parent.get(_BAR_PHASE_SYNTH_CACHE_KEY)
            except Exception:  # noqa: BLE001
                cached = None
        if (
            isinstance(cached, dict)
            and cached.get("beats_id") == id(beats)
            and cached.get("step") == beats_per_bar
            and cached.get("n_beats") == n_beats
        ):
            synth = cached["synth"]
        else:
            synth = tuple(float(beats[i]) for i in range(0, n_beats, beats_per_bar))
            if cache_parent is not None:
                try:
                    cache_parent[_BAR_PHASE_SYNTH_CACHE_KEY] = {
                        "beats_id": id(beats),
                        "step": beats_per_bar,
                        "n_beats": n_beats,
                        "synth": synth,
                    }
                except (TypeError, KeyError):
                    pass
        if synth:
            idx = _bisect_right_floats(synth, t_f)
            if 0 < idx < len(synth):
                return _phase_in_span(float(synth[idx - 1]), float(synth[idx]))
            if len(synth) >= 2:
                spans = np.diff(np.asarray(synth, dtype=np.float64))
                period = float(np.median(spans))
            elif bpm and bpm > 1e-3:
                period = 60.0 / float(bpm) * float(beats_per_bar)
            else:
                period = 0.0
            return _extrapolate(synth, period)

    # 3. Pure bpm fallback: anchor bar 0 at t = 0.
    if bpm and bpm > 1e-3:
        period = 60.0 / float(bpm) * float(beats_per_bar)
        if period <= 1e-6:
            return 0.0
        return float((t_f % period) / period)
    return 0.0


def uniforms_at_time(
    analysis: Mapping[str, Any],
    t: float,
    *,
    num_bands: int = DEFAULT_NUM_BANDS,
    intensity: float = 1.0,
    onset_decay: float = ONSET_DECAY_PER_SEC,
) -> dict[str, Any]:
    """
    Map an ``analysis.json`` dict onto reactive uniforms at time ``t`` seconds.

    Missing sections (no beats, no onsets, malformed spectrum) degrade to
    zero-valued uniforms rather than raising.
    """
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}")

    beats_raw = analysis.get("beats")
    beats: Sequence[float] = beats_raw if beats_raw else ()
    down_raw = analysis.get("downbeats")
    downbeats: Sequence[float] = down_raw if down_raw else ()
    tempo = analysis.get("tempo") or {}
    bpm = float(tempo.get("bpm") or 0.0)
    beat_phase = _beat_phase_at(beats, float(t), bpm=bpm)
    cache_parent = analysis if isinstance(analysis, dict) else None
    bar_phase = _bar_phase_at(
        float(t),
        downbeats,
        beats_per_bar=4,
        bpm=bpm if bpm > 0.0 else None,
        beats=beats if beats else None,
        cache_parent=cache_parent,
    )

    spec = analysis.get("spectrum") or {}
    spec_fps = float(spec.get("fps") or analysis.get("fps") or 0.0)
    bands = _interp_bands(spec.get("values"), float(t), spec_fps, num_bands)

    rms_block = analysis.get("rms") or {}
    rms_fps = float(rms_block.get("fps") or analysis.get("fps") or 0.0)
    rms_val = _interp_scalar_series(rms_block.get("values"), float(t), rms_fps)

    onset_peaks = (analysis.get("onsets") or {}).get("peaks") or []
    onset_val = _onset_pulse_at(onset_peaks, float(t), decay=onset_decay)
    onset_env_val = _interp_onset_strength(analysis, float(t))

    # Schema v2 ``events`` block. Defensive: pre-v2 caches re-analyze on load,
    # but a hand-rolled test dict may omit it entirely — fall through to zero.
    events = analysis.get("events") or {}
    build_block = events.get("build_tension") or {}
    build_fps = float(build_block.get("fps") or analysis.get("fps") or 0.0)
    build_val = _interp_scalar_series(
        build_block.get("values"), float(t), build_fps
    )

    return {
        "time": float(t),
        "beat_phase": float(beat_phase),
        "bar_phase": float(bar_phase),
        "rms": float(rms_val),
        "onset_pulse": float(onset_val),
        "onset_env": float(onset_env_val),
        "build_tension": float(build_val),
        "intensity": float(np.clip(intensity, 0.0, 1.0)),
        "band_energies": bands,
    }


# ---------------------------------------------------------------------------
# GL renderer
# ---------------------------------------------------------------------------


class ReactiveShader:
    """
    Offscreen moderngl fragment-shader renderer.

    Parameters
    ----------
    shader_name:
        Fragment-shader stem under :data:`SHADERS_DIR` (e.g. ``spectrum_bars``).
    width, height:
        Output resolution in pixels (defaults to 1920×1080).
    num_bands:
        Number of spectrum bands consumed by the shader (must match the
        ``band_energies[N]`` declaration).
    shaders_dir:
        Optional override for the shader root (defaults to
        :data:`config.SHADERS_DIR`).
    vertex_shader:
        Vertex-shader filename under ``shaders_dir`` (default
        ``passthrough.vert``).
    feedback_enabled:
        Opt into the ping-pong feedback framebuffer (``u_prev_frame`` +
        ``u_has_prev``) for Milkdrop-style trails / tunnels. ``None`` (the
        default) auto-detects by checking whether the fragment shader
        declares the ``u_prev_frame`` sampler — existing shaders pay
        zero cost. ``True`` / ``False`` forces the behaviour either way.
        When enabled the renderer allocates a second RGBA8 FBO at the
        output resolution (~8 MB at 1920×1080; ~16 MB including the
        primary colour attachment).

    Use as a context manager or call :meth:`close` explicitly to release GL
    resources. The instance is **not** thread-safe.
    """

    def __init__(
        self,
        shader_name: str = DEFAULT_SHADER,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_bands: int = DEFAULT_NUM_BANDS,
        shaders_dir: Path | None = None,
        vertex_shader: str = DEFAULT_VERTEX_SHADER,
        palette: Sequence[str] | None = None,
        feedback_enabled: bool | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution: {width}x{height}")
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")

        self._width = int(width)
        self._height = int(height)
        self._num_bands = int(num_bands)
        self._shader_name = str(shader_name)
        # Parse palette up front so bad hex fails construction, not rendering.
        self._palette_flat, self._palette_size = _build_palette_uniform(palette)
        # Reused every frame in :meth:`_apply_uniforms` (avoid ``list(...)`` alloc).
        self._palette_uniform_list: list[float] = list(self._palette_flat)

        root = Path(shaders_dir) if shaders_dir is not None else SHADERS_DIR
        frag_path = root / f"{shader_name}.frag"
        vert_path = root / vertex_shader
        if not frag_path.is_file():
            raise FileNotFoundError(f"Fragment shader not found: {frag_path}")
        if not vert_path.is_file():
            raise FileNotFoundError(f"Vertex shader not found: {vert_path}")

        self._ctx: moderngl.Context | None = None
        self._program: moderngl.Program | None = None
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._color_tex: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._bg_tex: moderngl.Texture | None = None
        # Feedback / ping-pong state. ``_fbo_alt`` wraps ``_prev_tex`` so the
        # two framebuffers can swap roles after every render; ``_color_tex``
        # always tracks "the texture we just rendered into" and ``_prev_tex``
        # "the texture the next frame reads as u_prev_frame".
        self._feedback_requested: bool | None = (
            feedback_enabled if feedback_enabled is None else bool(feedback_enabled)
        )
        self._feedback_enabled: bool = False
        self._prev_tex: moderngl.Texture | None = None
        self._fbo_alt: moderngl.Framebuffer | None = None
        self._has_prev_frame: bool = False

        try:
            self._ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:  # noqa: BLE001 - host-specific GL failure
            raise RuntimeError(
                "Failed to create standalone moderngl context "
                "(OpenGL 3.3+ required)"
            ) from exc

        try:
            frag_src = frag_path.read_text(encoding="utf-8")
            vert_src = vert_path.read_text(encoding="utf-8")
            try:
                self._program = self._ctx.program(
                    vertex_shader=vert_src, fragment_shader=frag_src
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Shader compile/link failure ({shader_name}): {exc}"
                ) from exc

            quad = np.asarray(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
            self._vbo = self._ctx.buffer(quad.tobytes())
            self._vao = self._ctx.vertex_array(
                self._program, [(self._vbo, "2f", "in_pos")]
            )
            self._color_tex = self._ctx.texture(
                (self._width, self._height), 4, dtype="f1"
            )
            self._fbo = self._ctx.framebuffer(
                color_attachments=[self._color_tex]
            )
            self._bg_tex = self._ctx.texture(
                (self._width, self._height), 3, dtype="u1"
            )
            try:
                self._program["u_background"] = _BACKGROUND_TEXTURE_UNIT
            except KeyError:
                pass

            # Feedback FBO: auto-detect by default so existing shaders that
            # don't declare ``u_prev_frame`` pay zero cost. Explicit ``True``
            # / ``False`` from the caller wins over the auto-detect result.
            shader_wants_feedback = True
            try:
                self._program["u_prev_frame"]
            except KeyError:
                shader_wants_feedback = False

            if self._feedback_requested is None:
                self._feedback_enabled = shader_wants_feedback
            else:
                self._feedback_enabled = bool(self._feedback_requested)

            if self._feedback_enabled:
                self._prev_tex = self._ctx.texture(
                    (self._width, self._height), 4, dtype="f1"
                )
                # Nearest filtering is safe for a 1:1 feedback pass; switch to
                # linear only if/when a shader intentionally samples off-grid.
                self._prev_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._fbo_alt = self._ctx.framebuffer(
                    color_attachments=[self._prev_tex]
                )
                try:
                    self._program["u_prev_frame"] = _PREV_FRAME_TEXTURE_UNIT
                except KeyError:
                    # Caller forced ``feedback_enabled=True`` on a shader that
                    # doesn't read the sampler — still allocate the FBO so the
                    # ping-pong lifecycle is observable, but the binding here
                    # is a no-op.
                    pass
                # Prime the "previous" texture to opaque-zero so frame 0's
                # ``texture(u_prev_frame, ...)`` reads well-defined pixels even
                # though ``u_has_prev`` will report zero. Using the alt FBO's
                # clear avoids a separate ``texture.write`` upload path.
                self._fbo_alt.use()
                self._ctx.clear(0.0, 0.0, 0.0, 0.0)

            self._fbo.use()
            self._ctx.viewport = (0, 0, self._width, self._height)
            self._ctx.enable(moderngl.BLEND)
        except Exception:
            self.close()
            raise

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def num_bands(self) -> int:
        return self._num_bands

    @property
    def shader_name(self) -> str:
        return self._shader_name

    @property
    def feedback_enabled(self) -> bool:
        """Whether the ping-pong feedback framebuffer is live for this instance."""
        return self._feedback_enabled

    @property
    def has_prev_frame(self) -> bool:
        """``True`` once at least one frame has been rendered into the feedback loop."""
        return self._has_prev_frame

    def reset_feedback(self) -> None:
        """
        Clear the feedback history so the next frame sees ``u_has_prev = 0``.

        Zeros the previous-frame texture as well so a shader that samples
        ``u_prev_frame`` regardless of ``u_has_prev`` reads deterministic
        (0, 0, 0, 0) pixels rather than stale trail content. No-op when
        feedback was not enabled for this instance.
        """
        if not self._feedback_enabled:
            return
        if self._fbo_alt is None or self._ctx is None or self._fbo is None:
            return
        self._fbo_alt.use()
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.use()
        self._has_prev_frame = False

    def write_background_rgb(self, background_rgb: np.ndarray) -> None:
        """
        Upload ``(H, W, 3)`` ``uint8`` RGB (top-left origin) to the ``u_background``
        texture (OpenGL bottom-up row order).
        """
        if self._bg_tex is None:
            raise RuntimeError("ReactiveShader has been closed")
        arr = np.asarray(background_rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"background_rgb must be (H, W, 3) uint8, got shape {arr.shape}"
            )
        if arr.shape[0] != self._height or arr.shape[1] != self._width:
            raise ValueError(
                f"background_rgb size {arr.shape[1]}×{arr.shape[0]} does not match "
                f"FBO {self._width}×{self._height}"
            )
        flipped = np.ascontiguousarray(np.flipud(arr))
        self._bg_tex.write(flipped.tobytes())

    def _bind_background_sampler(self) -> None:
        if self._program is None or self._bg_tex is None:
            return
        try:
            self._program["u_background"]
        except KeyError:
            return
        self._bg_tex.use(location=_BACKGROUND_TEXTURE_UNIT)

    def _bind_prev_frame_sampler(self) -> None:
        if not self._feedback_enabled:
            return
        if self._program is None or self._prev_tex is None:
            return
        try:
            self._program["u_prev_frame"]
        except KeyError:
            return
        self._prev_tex.use(location=_PREV_FRAME_TEXTURE_UNIT)

    # -- uniform handling ---------------------------------------------------

    def _pad_bands(self, value: Iterable[float]) -> list[float]:
        bands = [float(x) for x in value]
        if len(bands) >= self._num_bands:
            return bands[: self._num_bands]
        bands.extend([0.0] * (self._num_bands - len(bands)))
        return bands

    def _set_uniform(self, name: str, value: Any) -> None:
        if self._program is None:
            raise RuntimeError("ReactiveShader has been closed")
        try:
            uniform = self._program[name]
        except KeyError:
            # Shader does not declare this uniform; silently ignore so callers
            # can share a single uniforms dict across multiple shaders.
            return

        # moderngl exposes ``array_length`` (number of array elements) and
        # ``dimension`` (components per element: 1 for float/int, 3 for vec3,
        # 16 for mat4). Total float count is the product — the library's own
        # format helper uses the same formula. Using ``array_length`` alone
        # truncates ``vec3[5]`` to 5 floats instead of 15.
        array_len = int(getattr(uniform, "array_length", 1) or 1)
        dim = int(getattr(uniform, "dimension", 1) or 1)
        total_floats = max(1, array_len * dim)
        try:
            if array_len > 1:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size < total_floats:
                    pad = np.zeros(total_floats - arr.size, dtype=np.float32)
                    arr = np.concatenate([arr, pad])
                else:
                    arr = arr[:total_floats]
                uniform.write(arr.tobytes())
            elif isinstance(value, (list, tuple)):
                uniform.value = tuple(float(v) for v in value)
            else:
                uniform.value = value
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to set uniform {name!r} to {value!r}: {exc}"
            ) from exc

    def _apply_uniforms(self, uniforms: Mapping[str, Any]) -> None:
        defaults: dict[str, Any] = {
            "time": 0.0,
            "beat_phase": 0.0,
            "bar_phase": 0.0,
            "rms": 0.0,
            "onset_pulse": 0.0,
            "onset_env": 0.0,
            "build_tension": 0.0,
            "drop_hold": 0.0,
            "transient_lo": 0.0,
            "transient_mid": 0.0,
            "transient_hi": 0.0,
            "bass_hit": 0.0,
            "intensity": 1.0,
            "u_comp_background": 0.0,
            "u_has_prev": 0.0,
            "resolution": (float(self._width), float(self._height)),
            "band_energies": [0.0] * self._num_bands,
            # Palette is sticky per instance (set once at construction), but we
            # re-upload it every frame so a future reset path can't desync the
            # GPU state.
            "u_palette": self._palette_uniform_list,
            "u_palette_size": int(self._palette_size),
        }

        merged: dict[str, Any] = dict(defaults)
        if uniforms:
            merged.update(uniforms)

        merged["band_energies"] = self._pad_bands(merged["band_energies"])
        # Resolution is fixed to the FBO size regardless of caller input.
        merged["resolution"] = (float(self._width), float(self._height))

        for name, value in merged.items():
            if name == "u_background":
                continue
            self._set_uniform(name, value)

    # -- rendering ----------------------------------------------------------

    def render_frame(
        self, uniforms: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        """Render one frame and return an ``(H, W, 4)`` ``uint8`` RGBA array."""
        effective = dict(uniforms) if uniforms else {}
        return self._render_frame_with_effective(effective)

    def _render_frame_with_effective(self, effective: dict[str, Any]) -> np.ndarray:
        """Internal: mutate ``effective`` for feedback uniforms then draw."""
        if self._ctx is None or self._fbo is None or self._vao is None:
            raise RuntimeError("ReactiveShader has been closed")

        # Feedback path: inject ``u_has_prev`` based on whether we've already
        # produced at least one frame, and make the previous-frame texture
        # sampleable before the draw call. ``setdefault`` preserves any
        # explicit override the caller passes in (handy for manual tests).
        if self._feedback_enabled:
            effective.setdefault(
                "u_has_prev", 1.0 if self._has_prev_frame else 0.0
            )

        self._fbo.use()
        self._ctx.viewport = (0, 0, self._width, self._height)
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._bind_background_sampler()
        self._bind_prev_frame_sampler()
        self._apply_uniforms(effective)
        self._vao.render(moderngl.TRIANGLE_STRIP)

        err = self._ctx.error
        if err and err != "GL_NO_ERROR":
            raise RuntimeError(f"GL error during render: {err}")

        raw = self._fbo.read(components=4, dtype="f1")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            self._height, self._width, 4
        )

        if self._feedback_enabled and self._fbo_alt is not None:
            # Ping-pong swap: the texture we just wrote becomes next frame's
            # ``u_prev_frame``, and the previous ``_prev_tex`` (now stale)
            # becomes the write target. Framebuffers carry their colour
            # attachments with them, so swapping both pairs keeps pairing
            # consistent without rebuilding either FBO.
            self._fbo, self._fbo_alt = self._fbo_alt, self._fbo
            self._color_tex, self._prev_tex = self._prev_tex, self._color_tex
            self._has_prev_frame = True

        # moderngl returns framebuffer pixels bottom-up; flip to top-left origin
        # so downstream code can treat the array like any other image.
        return np.flipud(arr).copy()

    def render_frame_composited_rgb(
        self,
        uniforms: Mapping[str, Any] | None,
        background_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Render one frame with ``u_comp_background = 1`` so the fragment shader
        alpha-blends the reactive layer over ``background_rgb``, and return
        ``(H, W, 3)`` ``uint8`` RGB (opaque) for downstream typography / compositor.
        """
        self.write_background_rgb(background_rgb)
        effective: dict[str, Any] = dict(uniforms or {})
        effective["u_comp_background"] = 1.0
        rgba = self._render_frame_with_effective(effective)
        return rgba[:, :, :3].copy()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release GL resources. Safe to call multiple times."""
        for attr in (
            "_fbo",
            "_fbo_alt",
            "_color_tex",
            "_prev_tex",
            "_bg_tex",
            "_vao",
            "_vbo",
            "_program",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Ignoring release error on %s: %s", attr, exc)
                setattr(self, attr, None)
        if self._ctx is not None:
            try:
                self._ctx.release()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Ignoring context release error: %s", exc)
            self._ctx = None

    def __enter__(self) -> "ReactiveShader":
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
    "BUILTIN_SHADERS",
    "DEFAULT_HEIGHT",
    "DEFAULT_NUM_BANDS",
    "DEFAULT_PALETTE",
    "DEFAULT_SHADER",
    "DEFAULT_VERTEX_SHADER",
    "DEFAULT_WIDTH",
    "ONSET_DECAY_PER_SEC",
    "ONSET_ENV_CACHE_KEY",
    "ONSET_ENV_NORM_PERCENTILE",
    "PALETTE_SLOTS",
    "ReactiveShader",
    "ShaderUniforms",
    "composite_premultiplied_rgba_over_rgb",
    "resolve_builtin_shader_stem",
    "uniforms_at_time",
]
