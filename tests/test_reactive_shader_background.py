"""GL-backed tests for ``ReactiveShader.render_frame_composited_rgb``.

Locks in two contracts that previously regressed silently:

1. The ``u_background`` texture is created with ``dtype="f1"`` (normalised
   GL_RGB8, sampleable through ``sampler2D``). With ``dtype="u1"`` the
   texture is GL_RGB8UI and ``texture(u_background, ...)`` silently
   returns ``vec4(0)`` on at least NVIDIA, which made the entire
   compositor branch (``u_comp_background == 1``) paint the SDXL /
   AnimateDiff still as black for every preset. See the related
   ``alpha contract`` section in
   ``docs/technical/reactive-shader-layer.md``.

2. Every bundled fragment shader writes a *content-driven* alpha. With a
   bright background and silent uniforms, the rendered output must read
   meaningfully closer to the background than to the shader's own dark
   content, i.e. the overlay does not unconditionally hide the SDXL
   still.

The module probes for an OpenGL 3.3+ standalone context at import time
and gates every test on it so CI hosts without a GL stack skip cleanly.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

try:  # pragma: no cover - probe result depends on host GL stack
    import moderngl

    _probe_ctx = moderngl.create_standalone_context(require=330)
    _probe_ctx.release()
    _GL_AVAILABLE = True
except Exception:  # noqa: BLE001 - any driver/runtime failure means skip
    _GL_AVAILABLE = False

if _GL_AVAILABLE:  # Import lazily so the skip works on GL-less hosts.
    from pipeline.reactive_shader import BUILTIN_SHADERS, ReactiveShader


_SILENT_UNIFORMS: dict[str, Any] = {
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
    "band_energies": [0.0] * 8,
}


@unittest.skipUnless(_GL_AVAILABLE, "OpenGL 3.3+ standalone context not available")
class TestBackgroundTextureFormat(unittest.TestCase):
    """Pin the ``u_background`` texture format so it stays sampler2D-compatible."""

    def test_bg_texture_is_normalised_uint8(self) -> None:
        # ``"f1"`` -> GL_RGB8 (normalised; sampler2D reads as floats in [0, 1]).
        # ``"u1"`` -> GL_RGB8UI (integer; only usampler2D can read it; sampler2D
        # silently returns vec4(0)).
        with ReactiveShader("particles", width=32, height=32) as shader:
            self.assertEqual(shader._bg_tex.dtype, "f1")
            self.assertEqual(shader._bg_tex.components, 3)


@unittest.skipUnless(_GL_AVAILABLE, "OpenGL 3.3+ standalone context not available")
class TestBackgroundComposite(unittest.TestCase):
    """``render_frame_composited_rgb`` must let the background show through."""

    def _bright_bg(self) -> np.ndarray:
        return np.full((32, 32, 3), (200, 60, 200), dtype=np.uint8)

    def _black_bg(self) -> np.ndarray:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def test_changing_bg_changes_composited_output(self) -> None:
        # Hard regression test for the GL_RGB8UI bug. If
        # ``texture(u_background, ...)`` returned vec4(0) regardless of
        # uploaded pixels, the diff between black-bg and bright-bg renders
        # would be zero. We require a non-trivial difference for every
        # bundled shader.
        for shader_name in BUILTIN_SHADERS:
            with self.subTest(shader=shader_name):
                with ReactiveShader(shader_name, width=32, height=32) as r:
                    out_black = r.render_frame_composited_rgb(
                        _SILENT_UNIFORMS, self._black_bg()
                    )
                with ReactiveShader(shader_name, width=32, height=32) as r:
                    out_bright = r.render_frame_composited_rgb(
                        _SILENT_UNIFORMS, self._bright_bg()
                    )
                diff = np.abs(
                    out_bright.astype(np.int32) - out_black.astype(np.int32)
                ).mean()
                # A differently coloured background MUST visibly change the
                # composited output; otherwise we've regressed back to
                # painting the bg as black.
                self.assertGreater(
                    diff,
                    1.0,
                    msg=(
                        f"shader {shader_name!r} ignored u_background "
                        f"(bright vs black mean diff = {diff:.2f})"
                    ),
                )

    def test_silent_uniforms_let_bg_dominate(self) -> None:
        # Shaders whose alpha is properly content-driven should let a bright
        # background dominate during silent passages. ``synth_grid`` is the
        # documented exception (vista with intentionally opaque ground), so
        # we hold it to a looser bar.
        bg = self._bright_bg()
        loose_shaders = {"synth_grid"}
        for shader_name in BUILTIN_SHADERS:
            if shader_name in loose_shaders:
                continue
            with self.subTest(shader=shader_name):
                with ReactiveShader(shader_name, width=32, height=32) as r:
                    out = r.render_frame_composited_rgb(_SILENT_UNIFORMS, bg)
                # Background mean is (200+60+200)/3 ≈ 153. Output mean must be
                # within ~80 of that, otherwise the shader is opaque even when
                # silent (the spectral_milkdrop / paper_grain / vhs / liquid
                # / nebula bug class).
                bg_mean = bg.mean()
                out_mean = out.mean()
                self.assertGreater(
                    out_mean,
                    bg_mean - 80,
                    msg=(
                        f"shader {shader_name!r} masked the background even "
                        f"with silent uniforms (out_mean={out_mean:.1f} vs "
                        f"bg_mean={bg_mean:.1f})"
                    ),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
