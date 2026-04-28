"""Pinokio / Windows resilience guards: k2 stub, NumPy ABI pin, CPU retry.

These tests pin the user-facing behaviours that fix the cluster of failures
fork users hit on Pinokio (Windows + Python 3.11 venv):

1. ``app.py`` pre-stubs the ``k2`` module in ``sys.modules`` so that
   ``speechbrain.integrations.k2_fsa`` (a ``LazyModule`` whose ``__getattr__``
   force-imports ``k2``) does not crash ``inspect.getmodule`` walks triggered by
   ``librosa``'s lazy_loader during audio ingest.

2. ``requirements.txt`` / ``pyproject.toml`` keep NumPy on the 1.x ABI: torch
   2.2.2 (Track A) was compiled against NumPy 1.x and silently breaks under
   NumPy 2.x with ``Failed to initialize NumPy: _ARRAY_API not found``.

3. ``align_lyrics`` retries WhisperX on CPU+int8 ONCE when the GPU run fails
   with a CTranslate2 / cuDNN / load-library error, so a broken Windows GPU
   stack doesn't make Align lyrics unusable for fork users.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestK2StubRegisteredEarly(unittest.TestCase):
    """``app.py`` must register a ``k2`` stub before any speechbrain import.

    Speechbrain 1.0.x defines ``speechbrain.integrations.k2_fsa`` as a
    ``LazyModule``. Its ``__getattr__`` force-imports the target on ANY
    attribute access — including ``hasattr(mod, "__file__")`` from
    ``inspect.getmodule``. Pythons stdlib ``inspect.stack`` walks every entry
    in ``sys.modules`` and probes ``__file__``, so the very first call from
    ``librosa.lazy_loader.load`` (audio ingest) detonates with::

        ImportError: Lazy import of LazyModule(... k2_fsa ...) failed

    Pre-registering ``sys.modules['k2'] = ModuleType('k2')`` makes the
    ``import k2`` guard inside speechbrain pass, so the lazy module loads
    cleanly and ``__file__`` becomes a normal attribute.
    """

    def test_app_py_pre_registers_k2_stub_before_other_imports(self) -> None:
        """The k2 stub block must precede ``import gradio`` and any pipeline
        imports — those are the lines that pull in whisperx → speechbrain."""
        app_text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")

        k2_stub_idx = app_text.find('sys.modules.setdefault("k2"')
        self.assertGreater(
            k2_stub_idx,
            -1,
            "app.py must register a k2 stub via sys.modules.setdefault",
        )

        gradio_idx = app_text.find("import gradio")
        self.assertGreater(gradio_idx, 0, "app.py must import gradio")
        self.assertLess(
            k2_stub_idx,
            gradio_idx,
            "k2 stub must run BEFORE 'import gradio' (which can transitively "
            "pull speechbrain via downstream init order in some configs)",
        )

        pipeline_idx = app_text.find("from pipeline.")
        self.assertGreater(pipeline_idx, 0, "app.py must import pipeline modules")
        self.assertLess(
            k2_stub_idx,
            pipeline_idx,
            "k2 stub must run BEFORE any 'from pipeline.' import — those load "
            "librosa / whisperx which transitively imports speechbrain",
        )


class TestNumpyAbiPinned(unittest.TestCase):
    """Track A torch 2.2.2 needs NumPy 1.x — pin must survive a refactor.

    Symptom on Pinokio when this drifts: torch logs
    ``UserWarning: Failed to initialize NumPy: _ARRAY_API not found`` and any
    tensor<->numpy bridge (audio ingest, demucs, whisperx) silently produces
    garbage. Newer torch (>=2.4) handles NumPy 2 fine, but our pinned wheel
    does not.
    """

    def test_pyproject_pins_numpy_below_2(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'numpy>=1\.26\.0,\s*<2\.0',
            "pyproject.toml must pin numpy>=1.26.0,<2.0 in [project] dependencies",
        )

    def test_requirements_txt_pins_numpy_below_2(self) -> None:
        text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'numpy>=1\.26\.0,\s*<2\.0',
            "requirements.txt must pin numpy>=1.26.0,<2.0",
        )

    def test_install_js_force_reinstalls_numpy_below_2(self) -> None:
        """Pinokio install.js extras step pulls transitive numpy>=2; the script
        must --force-reinstall numpy<2 afterwards (same pattern as
        markupsafe/pillow) so torch 2.2.2's NumPy 1.x ABI is honoured at runtime."""
        text = (REPO_ROOT / "install.js").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'numpy>=1\.26\.0,\s*<2\.0',
            "install.js must explicitly install numpy>=1.26.0,<2.0",
        )
        self.assertIn(
            "--force-reinstall",
            text,
            "install.js must force-reinstall the post-extras pins (numpy / torch trio)",
        )


class TestInstallJsCudnn8Pin(unittest.TestCase):
    """``ctranslate2==4.4.0`` looks for cuDNN 8 (cudnn_ops_infer64_8.dll). pip's
    unversioned ``nvidia-cudnn-cu12`` resolves to cuDNN 9 (different DLL names)
    which does NOT satisfy ct2 4.4. Pin the cuDNN 8.9.7 wheel explicitly."""

    def test_install_js_pins_cudnn_8_9_7(self) -> None:
        text = (REPO_ROOT / "install.js").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'nvidia-cudnn-cu12==8\.9\.7\.\d+',
            "install.js must pin nvidia-cudnn-cu12==8.9.7.x to match ctranslate2 4.4.0",
        )


class TestIsCudnnClassError(unittest.TestCase):
    """``_is_cudnn_class_error`` must catch the actual Pinokio failure modes
    (cuDNN DLL load, ctranslate2 native crash, error 1920) WITHOUT swallowing
    legitimate alignment errors so the CPU retry stays narrowly scoped."""

    def test_cudnn_dll_load_failure_matches(self) -> None:
        from pipeline.lyrics_aligner import _is_cudnn_class_error

        exc = RuntimeError(
            "Could not load library cudnn_ops_infer64_8.dll. Error code 1920"
        )
        self.assertTrue(_is_cudnn_class_error(exc))

    def test_cudnn9_dll_load_failure_matches(self) -> None:
        from pipeline.lyrics_aligner import _is_cudnn_class_error

        exc = RuntimeError("Could not load library cudnn_ops64_9.dll")
        self.assertTrue(_is_cudnn_class_error(exc))

    def test_loadlibrary_generic_matches(self) -> None:
        from pipeline.lyrics_aligner import _is_cudnn_class_error

        exc = OSError(
            "[WinError 126] LoadLibraryExW failed to load ctranslate2.dll"
        )
        self.assertTrue(_is_cudnn_class_error(exc))

    def test_chained_cause_is_inspected(self) -> None:
        """faster-whisper sometimes wraps the real cuDNN OSError in a generic
        RuntimeError; ``_is_cudnn_class_error`` must walk ``__cause__``."""
        from pipeline.lyrics_aligner import _is_cudnn_class_error

        inner = OSError("Could not load library cudnn_cnn_infer64_8.dll")
        outer = RuntimeError("Failed to initialize Whisper model")
        outer.__cause__ = inner
        self.assertTrue(_is_cudnn_class_error(outer))

    def test_chained_context_is_inspected(self) -> None:
        from pipeline.lyrics_aligner import _is_cudnn_class_error

        inner = OSError("cublas64_12.dll not found")
        outer = RuntimeError("oops")
        outer.__context__ = inner
        self.assertTrue(_is_cudnn_class_error(outer))

    def test_unrelated_alignment_error_does_not_match(self) -> None:
        """Lyrics-vs-audio mismatch / empty user tokens / etc. must NOT trigger
        the GPU→CPU fallback (CPU would also fail and just hide the real bug).
        """
        from pipeline.lyrics_aligner import _is_cudnn_class_error

        for exc in (
            ValueError("lyrics_text has no word tokens after tokenisation"),
            FileNotFoundError("vocals.wav"),
            RuntimeError("CTC backtrack failed: no path through alignment graph"),
            KeyError("segments"),
        ):
            self.assertFalse(
                _is_cudnn_class_error(exc),
                f"{type(exc).__name__}('{exc}') should NOT be classified as a "
                "cuDNN-class error — CPU retry would mask the real failure",
            )


class TestAlignLyricsCpuRetryOnCudnnFailure(unittest.TestCase):
    """``align_lyrics`` must catch a single cuDNN-class crash on the GPU path
    and retry on CPU+int8 ONCE. SDXL / Demucs / render must NOT be touched —
    the fallback is scoped to the WhisperX call."""

    def _patch_minimal_align_dependencies(
        self, stack: Any, tmp_dir: Path, lyrics: str
    ) -> dict[str, Any]:
        """Stub everything outside ``_run_whisperx_forced`` so the test
        exercises the retry decision in isolation. Keeps the test fast and
        hermetic — no torch / whisperx import required."""
        from pipeline import lyrics_aligner as la

        cache_dir = tmp_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / la.VOCALS_WAV_NAME).write_bytes(b"RIFFstub")

        stack.enter_context(mock.patch.object(la, "_pick_device", return_value="cuda"))
        stack.enter_context(
            mock.patch.object(la, "_default_compute_type", return_value="float16")
        )

        stack.enter_context(
            mock.patch.object(
                la,
                "_timings_and_scores_for_user_tokens",
                lambda user_tokens, _w: (
                    [(float(i), float(i) + 0.5) for i in range(len(user_tokens))],
                    [None] * len(user_tokens),
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(la, "_polish_timings", lambda _u, t: t)
        )
        stack.enter_context(
            mock.patch.object(la, "_enforce_monotonic_per_line", lambda _u, t: t)
        )
        return {"cache_dir": cache_dir, "lyrics": lyrics}

    def test_gpu_cudnn_failure_falls_back_to_cpu_int8_once(self) -> None:
        from pipeline import lyrics_aligner as la

        with mock.patch.object(la, "_run_whisperx_forced") as mock_forced:
            cudnn_err = RuntimeError(
                "Could not load library cudnn_ops_infer64_8.dll. Error code 1920"
            )
            mock_forced.side_effect = [cudnn_err, ([], "en", [])]

            with self.subTest(stage="setup"):
                import contextlib
                import tempfile

                stack = contextlib.ExitStack()
                with stack:
                    tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    self._patch_minimal_align_dependencies(
                        stack, tmp / "deadbeef", "hello world\n"
                    )
                    la.align_lyrics(tmp / "deadbeef", "hello world\n")

            self.assertEqual(
                mock_forced.call_count,
                2,
                "Expected exactly one GPU attempt + one CPU retry",
            )

            first_kwargs = mock_forced.call_args_list[0].kwargs
            self.assertEqual(first_kwargs["device"], "cuda")
            self.assertEqual(first_kwargs["compute_type"], "float16")

            second_kwargs = mock_forced.call_args_list[1].kwargs
            self.assertEqual(
                second_kwargs["device"],
                "cpu",
                "Retry must run on CPU (GPU stack is broken on this machine)",
            )
            self.assertEqual(
                second_kwargs["compute_type"],
                "int8",
                "CPU retry must use int8 (float16 is GPU-only in faster-whisper)",
            )

    def test_unrelated_error_is_not_retried(self) -> None:
        """A lyrics-tokenisation / file error must NOT trigger CPU retry —
        CPU would fail the same way and mask the real bug."""
        from pipeline import lyrics_aligner as la
        import contextlib
        import tempfile

        with contextlib.ExitStack() as stack, mock.patch.object(
            la, "_run_whisperx_forced"
        ) as mock_forced:
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            self._patch_minimal_align_dependencies(
                stack, tmp / "deadbeef", "hello world\n"
            )
            mock_forced.side_effect = ValueError(
                "CTC backtrack failed: no path through alignment graph"
            )

            with self.assertRaises(ValueError):
                la.align_lyrics(tmp / "deadbeef", "hello world\n")

            self.assertEqual(
                mock_forced.call_count,
                1,
                "Unrelated errors must NOT trigger the CPU retry",
            )

    def test_cpu_path_does_not_retry_on_failure(self) -> None:
        """If we already started on CPU and still failed, retrying on CPU again
        is pointless — must surface the original exception."""
        from pipeline import lyrics_aligner as la
        import contextlib
        import tempfile

        with contextlib.ExitStack() as stack, mock.patch.object(
            la, "_run_whisperx_forced"
        ) as mock_forced, mock.patch.object(
            la, "_pick_device", return_value="cpu"
        ), mock.patch.object(
            la, "_default_compute_type", return_value="int8"
        ):
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            cache_dir = tmp / "deadbeef"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / la.VOCALS_WAV_NAME).write_bytes(b"RIFFstub")

            mock_forced.side_effect = RuntimeError(
                "Could not load library cudnn_ops_infer64_8.dll"
            )

            with self.assertRaises(RuntimeError):
                la.align_lyrics(cache_dir, "hello world\n")

            self.assertEqual(
                mock_forced.call_count,
                1,
                "Already on CPU — no second attempt to make",
            )


if __name__ == "__main__":
    unittest.main()
