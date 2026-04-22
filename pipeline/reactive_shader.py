"""
Offscreen :mod:`moderngl` reactive shader pass.

Loads a fragment shader from :data:`config.SHADERS_DIR` (paired with the shared
:file:`passthrough.vert`), creates a standalone GL context with an RGBA FBO at
the target resolution, and renders single frames driven by analysis-derived
uniforms (``beat_phase``, ``band_energies[8]``, ``rms``, ``onset_pulse``,
``time``, ``intensity``).

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
from typing import Any, Iterable, Mapping, Sequence

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
    rms: float = 0.0
    onset_pulse: float = 0.0
    intensity: float = 1.0
    band_energies: tuple[float, ...] = tuple([0.0] * DEFAULT_NUM_BANDS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "beat_phase": float(self.beat_phase),
            "rms": float(self.rms),
            "onset_pulse": float(self.onset_pulse),
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

    beats = list(analysis.get("beats") or [])
    tempo = analysis.get("tempo") or {}
    bpm = float(tempo.get("bpm") or 0.0)
    beat_phase = _beat_phase_at(beats, float(t), bpm=bpm)

    spec = analysis.get("spectrum") or {}
    spec_fps = float(spec.get("fps") or analysis.get("fps") or 0.0)
    bands = _interp_bands(spec.get("values"), float(t), spec_fps, num_bands)

    rms_block = analysis.get("rms") or {}
    rms_fps = float(rms_block.get("fps") or analysis.get("fps") or 0.0)
    rms_val = _interp_scalar_series(rms_block.get("values"), float(t), rms_fps)

    onset_peaks = (analysis.get("onsets") or {}).get("peaks") or []
    onset_val = _onset_pulse_at(onset_peaks, float(t), decay=onset_decay)

    return {
        "time": float(t),
        "beat_phase": float(beat_phase),
        "rms": float(rms_val),
        "onset_pulse": float(onset_val),
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
            "rms": 0.0,
            "onset_pulse": 0.0,
            "bass_hit": 0.0,
            "intensity": 1.0,
            "u_comp_background": 0.0,
            "resolution": (float(self._width), float(self._height)),
            "band_energies": [0.0] * self._num_bands,
            # Palette is sticky per instance (set once at construction), but we
            # re-upload it every frame so a future reset path can't desync the
            # GPU state.
            "u_palette": list(self._palette_flat),
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
        if self._ctx is None or self._fbo is None or self._vao is None:
            raise RuntimeError("ReactiveShader has been closed")

        self._fbo.use()
        self._ctx.viewport = (0, 0, self._width, self._height)
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._bind_background_sampler()
        self._apply_uniforms(uniforms or {})
        self._vao.render(moderngl.TRIANGLE_STRIP)

        err = self._ctx.error
        if err and err != "GL_NO_ERROR":
            raise RuntimeError(f"GL error during render: {err}")

        raw = self._fbo.read(components=4, dtype="f1")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            self._height, self._width, 4
        )
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
        u: dict[str, Any] = dict(uniforms or {})
        u["u_comp_background"] = 1.0
        rgba = self.render_frame(u)
        return rgba[:, :, :3].copy()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release GL resources. Safe to call multiple times."""
        for attr in ("_fbo", "_color_tex", "_bg_tex", "_vao", "_vbo", "_program"):
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
    "PALETTE_SLOTS",
    "ReactiveShader",
    "ShaderUniforms",
    "composite_premultiplied_rgba_over_rgb",
    "resolve_builtin_shader_stem",
    "uniforms_at_time",
]
