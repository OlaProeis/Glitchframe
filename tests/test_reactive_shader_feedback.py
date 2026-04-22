"""GL-backed tests for the ping-pong feedback framebuffer on ``ReactiveShader``.

These exercise the Milkdrop-style ``u_prev_frame`` / ``u_has_prev`` contract
end-to-end and therefore need a live OpenGL 3.3+ standalone context. The
module probes for one at import time and gates every test on success so CI
hosts without a GL stack skip cleanly instead of erroring out.

What we verify (task 41 test strategy):

* Auto-detect leaves feedback **off** for shaders that don't declare the
  sampler — existing shaders keep paying zero cost.
* Auto-detect turns feedback **on** for shaders that do declare it, and the
  pre-swap ``_has_prev_frame`` flag reports the lifecycle correctly.
* A pass-through shader that samples ``u_prev_frame`` produces the same
  pixels on frame ``N+1`` as the raw frame ``N`` output — proving the
  ping-pong swap + sampler binding are wired the right way round.
* ``u_has_prev`` is ``0`` on the very first frame (shader sees the primed
  opaque-zero texture, not stale state).
* ``reset_feedback()`` restores the "this is the first frame" behaviour so
  reused instances (e.g. the Gradio reactive preview) never leak trails
  across render sessions.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:  # pragma: no cover - probe result depends on host GL stack
    import moderngl

    _probe_ctx = moderngl.create_standalone_context(require=330)
    _probe_ctx.release()
    _GL_AVAILABLE = True
except Exception:  # noqa: BLE001 - any driver/runtime failure means skip
    _GL_AVAILABLE = False

if _GL_AVAILABLE:  # Import lazily so the skip works on GL-less hosts.
    from pipeline.reactive_shader import ReactiveShader


# Minimal fullscreen-quad vertex shader, duplicated here so the tests don't
# depend on the repo-level ``assets/shaders/`` layout.
_VERT_SRC = """#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Pass-through trail shader: on frame 0 emits a deterministic UV-gradient
# pattern; on subsequent frames reads ``u_prev_frame`` at the same UV and
# outputs it verbatim. Frame N+1 must therefore match frame N exactly.
_FEEDBACK_FRAG_SRC = """#version 330
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_prev_frame;
uniform float u_has_prev;
void main() {
    if (u_has_prev > 0.5) {
        out_color = texture(u_prev_frame, v_uv);
    } else {
        out_color = vec4(v_uv.x, v_uv.y, 0.5, 1.0);
    }
}
"""

# Plain shader that doesn't declare the feedback sampler — used to verify
# the auto-detect keeps feedback off (and the feedback FBO unallocated).
_PLAIN_FRAG_SRC = """#version 330
in vec2 v_uv;
out vec4 out_color;
void main() {
    out_color = vec4(v_uv, 0.25, 1.0);
}
"""


def _write_shader_pack(tmpdir: Path, stem: str, frag_src: str) -> Path:
    """Materialise a (vert, frag) pair under ``tmpdir`` for ``ReactiveShader``."""
    (tmpdir / "passthrough.vert").write_text(_VERT_SRC, encoding="utf-8")
    frag = tmpdir / f"{stem}.frag"
    frag.write_text(frag_src, encoding="utf-8")
    return frag


def _make_shader(
    tmpdir: Path,
    stem: str,
    frag_src: str,
    *,
    size: int = 32,
    feedback_enabled: bool | None = None,
) -> "ReactiveShader":
    _write_shader_pack(tmpdir, stem, frag_src)
    # Bypass the ``BUILTIN_SHADERS`` allowlist by constructing against the
    # temp dir directly — ``ReactiveShader.__init__`` only file-checks the
    # resolved paths, which is exactly what we want for isolated tests.
    return ReactiveShader(
        shader_name=stem,
        width=size,
        height=size,
        shaders_dir=tmpdir,
        feedback_enabled=feedback_enabled,
    )


@unittest.skipUnless(_GL_AVAILABLE, "OpenGL 3.3+ standalone context not available")
class TestFeedbackAutoDetect(unittest.TestCase):
    def test_shader_without_u_prev_frame_leaves_feedback_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(Path(td), "plain_test", _PLAIN_FRAG_SRC)
            try:
                self.assertFalse(shader.feedback_enabled)
                self.assertFalse(shader.has_prev_frame)
                # A render must not flip any feedback state on a disabled shader.
                shader.render_frame()
                self.assertFalse(shader.has_prev_frame)
            finally:
                shader.close()

    def test_shader_with_u_prev_frame_enables_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(Path(td), "trail_test", _FEEDBACK_FRAG_SRC)
            try:
                self.assertTrue(shader.feedback_enabled)
                self.assertFalse(shader.has_prev_frame)
            finally:
                shader.close()

    def test_explicit_feedback_false_overrides_autodetect(self) -> None:
        # Shader declares the sampler, but caller forces it off — the FBO
        # must not be allocated and the has_prev flag must stay false.
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(
                Path(td),
                "trail_test",
                _FEEDBACK_FRAG_SRC,
                feedback_enabled=False,
            )
            try:
                self.assertFalse(shader.feedback_enabled)
            finally:
                shader.close()


@unittest.skipUnless(_GL_AVAILABLE, "OpenGL 3.3+ standalone context not available")
class TestFeedbackPingPong(unittest.TestCase):
    def test_frame_zero_sees_has_prev_zero(self) -> None:
        # On frame 0 the shader takes the ``u_has_prev == 0`` branch and
        # emits the UV-gradient pattern — not a zero-filled texture read
        # from the primed ``u_prev_frame``. The centre pixel's blue
        # channel is 0.5 in the "fresh" branch and 0 in the "trail" branch,
        # so that single byte is enough to distinguish them.
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(Path(td), "trail_test", _FEEDBACK_FRAG_SRC)
            try:
                self.assertFalse(shader.has_prev_frame)
                frame0 = shader.render_frame()
                centre = frame0[frame0.shape[0] // 2, frame0.shape[1] // 2]
                # Blue should be ~0.5 * 255 == 128; allow a small rounding band.
                self.assertGreater(int(centre[2]), 100)
                self.assertLess(int(centre[2]), 160)
                self.assertEqual(int(centre[3]), 255)
                # Lifecycle flipped exactly once after the first render.
                self.assertTrue(shader.has_prev_frame)
            finally:
                shader.close()

    def test_frame_n_plus_one_echoes_frame_n(self) -> None:
        # Core of task 41's test strategy: a shader that reads u_prev_frame
        # verbatim must output the previous frame's pixels on frame N+1.
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(Path(td), "trail_test", _FEEDBACK_FRAG_SRC)
            try:
                frame0 = shader.render_frame()
                frame1 = shader.render_frame()
            finally:
                shader.close()
        # Identical pixels (the shader is a pure pass-through of the sampled
        # previous frame; nearest filtering keeps the texels 1:1 with UVs).
        self.assertEqual(frame0.shape, frame1.shape)
        self.assertTrue(
            np.array_equal(frame0, frame1),
            msg="frame N+1 did not echo frame N through u_prev_frame",
        )

    def test_reset_feedback_restores_first_frame_behaviour(self) -> None:
        # After two renders the shader is on its "echo the trail" path. A
        # reset_feedback() call must zero the trail *and* flip has_prev back
        # to False so the next render takes the frame-0 branch again.
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(Path(td), "trail_test", _FEEDBACK_FRAG_SRC)
            try:
                frame0 = shader.render_frame()
                shader.render_frame()
                self.assertTrue(shader.has_prev_frame)
                shader.reset_feedback()
                self.assertFalse(shader.has_prev_frame)
                frame_after_reset = shader.render_frame()
            finally:
                shader.close()
        self.assertTrue(
            np.array_equal(frame0, frame_after_reset),
            msg="reset_feedback did not restore the frame-0 output",
        )

    def test_reset_feedback_is_noop_when_disabled(self) -> None:
        # Non-feedback shader: reset_feedback must be a safe no-op so
        # callers can call it unconditionally between render sessions.
        with tempfile.TemporaryDirectory() as td:
            shader = _make_shader(Path(td), "plain_test", _PLAIN_FRAG_SRC)
            try:
                shader.render_frame()
                shader.reset_feedback()
                # Rendering after a no-op reset still works.
                shader.render_frame()
            finally:
                shader.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
