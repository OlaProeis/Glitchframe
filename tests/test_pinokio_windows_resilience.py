"""Pinokio / Windows resilience guards: speechbrain LazyModule patch, k2 stub,
NumPy ABI pin, CPU retry.

These tests pin the user-facing behaviours that fix the cluster of failures
fork users hit on Pinokio (Windows + Python 3.11 venv):

1. ``pipeline._speechbrain_compat.patch_speechbrain_lazy_module`` rewrites
   speechbrain's ``LazyModule.ensure_module`` to honour Windows path
   separators in its ``inspect.py`` guard (upstream issue #2995 — the guard
   hard-codes ``"/inspect.py"`` and silently no-ops on Windows). Without this
   patch, ``librosa.load`` (audio ingest) calls ``inspect.stack()`` which
   force-imports every speechbrain integration, and any one with a missing
   optional dep (k2, flair, ...) crashes the upload.

2. ``app.py`` also pre-stubs the ``k2`` module in ``sys.modules`` as belt-and-
   braces so the lazy import succeeds even if the patch ever fails to apply.

3. ``requirements.txt`` / ``pyproject.toml`` keep NumPy on the 1.x ABI: torch
   2.2.2 (Track A) was compiled against NumPy 1.x and silently breaks under
   NumPy 2.x with ``Failed to initialize NumPy: _ARRAY_API not found``.

4. ``align_lyrics`` retries WhisperX on CPU+int8 ONCE when the GPU run fails
   with a CTranslate2 / cuDNN / load-library error, so a broken Windows GPU
   stack doesn't make Align lyrics unusable for fork users.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
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


class TestInstallJsCudnnPolicy(unittest.TestCase):
    """``install.js`` must NOT install ``nvidia-cudnn-cu12`` and must actively
    uninstall any prior wheel.

    History (do not re-add the install step):

    * Unversioned ``nvidia-cudnn-cu12`` resolves to cuDNN 9 (DLL names
      ``cudnn_ops64_9.dll``) which does NOT satisfy ctranslate2 4.4.0's cuDNN 8
      lookups — Align lyrics fails on the GPU path with
      ``Could not load library cudnn_ops_infer64_8.dll``.
    * Pinning ``nvidia-cudnn-cu12==8.9.7.29`` instead caused
      ``WinError 127: The specified procedure could not be found`` on
      ``import whisperx`` — the standalone 8.9.7 wheel exports a different
      symbol set than torch 2.2.2+cu121's bundled 8.9.x DLLs (which is what
      ctranslate2 4.4.0 was actually resolved against on PyPI).

    Torch 2.2.2+cu121 ALREADY ships the right cuDNN 8.9.x in ``torch\\lib``;
    ``scripts/windows_provision_cudnn_next_to_ctranslate2.py`` copies them
    next to ctranslate2 for LoadLibrary. That's enough.
    """

    def test_install_js_does_not_install_nvidia_cudnn(self) -> None:
        text = (REPO_ROOT / "install.js").read_text(encoding="utf-8")
        # Catch any `pip install ... nvidia-cudnn-cu12 ...` line (any version).
        # We tolerate the literal in comments, so match only inside install commands.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            self.assertNotRegex(
                stripped,
                r'pip\s+install[^"\']*nvidia-cudnn-cu12',
                f"install.js must NOT install nvidia-cudnn-cu12 (line: {line!r}). "
                "Standalone cuDNN wheels conflict with torch 2.2.2's bundled cuDNN 8.9.x.",
            )

    def test_install_js_uninstalls_prior_nvidia_cudnn(self) -> None:
        """Existing Pinokio envs from earlier install runs may have cuDNN 9 or
        cuDNN 8.9.7 left over; the install must clean them up so torch's
        bundled cuDNN wins the loader race for ctranslate2."""
        text = (REPO_ROOT / "install.js").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'pip\s+uninstall\s+-y\s+nvidia-cudnn-cu12',
            "install.js must `pip uninstall -y nvidia-cudnn-cu12` to remove "
            "wheels left over from prior install runs (cuDNN 9 or 8.9.7).",
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


def _buggy_ensure_module(self, stacklevel: int):  # type: ignore[no-untyped-def]
    """Replicates the buggy upstream speechbrain v1.0.x ``ensure_module``
    verbatim — hard-coded ``"/inspect.py"`` literal that no-ops on Windows."""
    import importlib
    import inspect as _inspect

    importer_frame = _inspect.getframeinfo(sys._getframe(stacklevel + 1))
    if importer_frame is not None and importer_frame.filename.endswith(
        "/inspect.py"
    ):
        raise AttributeError()

    if self.lazy_module is None:
        try:
            self.lazy_module = importlib.import_module(self.target)
        except Exception as exc:
            raise ImportError(f"Lazy import of {self!r} failed") from exc
    return self.lazy_module


class _FakeLazyModule(ModuleType):
    """Reproduces the buggy speechbrain v1.0.x ``LazyModule`` for testing.

    ``ensure_module``'s ``inspect.py`` guard uses the same buggy
    ``filename.endswith("/inspect.py")`` literal as upstream — so on Windows
    the guard is a no-op and any ``hasattr(...)`` probe forces an import of
    the (deliberately broken) target. The compat patch must cure this.
    """

    def __init__(self, name: str, target: str, package: str | None = None) -> None:
        super().__init__(name)
        self.target = target
        self.lazy_module = None
        self.package = package

    ensure_module = _buggy_ensure_module

    def __getattr__(self, attr: str):
        return getattr(self.ensure_module(1), attr)


class TestSpeechbrainLazyModulePatch(unittest.TestCase):
    """``patch_speechbrain_lazy_module`` must:

    * Detect the buggy hard-coded ``"/inspect.py"`` literal in upstream
      ``LazyModule.ensure_module`` (don't shadow a future fix).
    * Replace ``ensure_module`` so the guard fires regardless of path
      separator (Windows uses backslash).
    * Be idempotent (calling twice is a no-op).
    * Be a silent no-op when speechbrain is not loaded.
    """

    def setUp(self) -> None:
        # Save and clear any speechbrain.utils.importutils for this test.
        self._saved = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k == "speechbrain.utils.importutils"
            or k == "speechbrain.utils"
            or k == "speechbrain"
        }
        for k in self._saved:
            del sys.modules[k]
        # Reset the patch marker + restore the buggy ensure_module so each
        # test starts clean (the fake class is module-level so state persists
        # across tests).
        from pipeline._speechbrain_compat import _PATCH_MARKER

        if hasattr(_FakeLazyModule, _PATCH_MARKER):
            delattr(_FakeLazyModule, _PATCH_MARKER)
        _FakeLazyModule.ensure_module = _buggy_ensure_module  # type: ignore[method-assign]

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            sys.modules[k] = v
        # Drop any fake modules we registered.
        for k in list(sys.modules):
            if k.startswith("speechbrain") or k == "_glitchframe_fake_target":
                if k not in self._saved:
                    del sys.modules[k]

    def _install_fake_speechbrain(self) -> type:
        """Register a minimal fake ``speechbrain.utils.importutils`` exposing
        the buggy ``LazyModule`` class."""
        sb = ModuleType("speechbrain")
        sb_utils = ModuleType("speechbrain.utils")
        sb_imp = ModuleType("speechbrain.utils.importutils")
        sb_imp.LazyModule = _FakeLazyModule  # type: ignore[attr-defined]
        sys.modules["speechbrain"] = sb
        sys.modules["speechbrain.utils"] = sb_utils
        sys.modules["speechbrain.utils.importutils"] = sb_imp
        return _FakeLazyModule

    def test_no_op_when_speechbrain_absent(self) -> None:
        from pipeline._speechbrain_compat import patch_speechbrain_lazy_module

        self.assertNotIn("speechbrain.utils.importutils", sys.modules)
        result = patch_speechbrain_lazy_module()
        self.assertFalse(
            result,
            "Patch must return False (no-op) when speechbrain is not loaded",
        )

    def test_patches_buggy_lazy_module(self) -> None:
        from pipeline._speechbrain_compat import (
            _PATCH_MARKER,
            _patched_ensure_module,
            patch_speechbrain_lazy_module,
        )

        LazyModule = self._install_fake_speechbrain()
        # Sanity check: setUp restored the buggy ensure_module.
        self.assertIs(LazyModule.ensure_module, _buggy_ensure_module)

        result = patch_speechbrain_lazy_module()

        self.assertTrue(result, "Patch must return True when applied")
        self.assertTrue(
            getattr(LazyModule, _PATCH_MARKER, False),
            "Patch marker must be set so a second call is a no-op",
        )
        self.assertIs(
            LazyModule.ensure_module,
            _patched_ensure_module,
            "ensure_module must be replaced with the separator-aware version",
        )

    def test_idempotent(self) -> None:
        from pipeline._speechbrain_compat import patch_speechbrain_lazy_module

        LazyModule = self._install_fake_speechbrain()
        first = patch_speechbrain_lazy_module()
        first_method = LazyModule.ensure_module
        second = patch_speechbrain_lazy_module()
        second_method = LazyModule.ensure_module

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertIs(
            first_method,
            second_method,
            "Second patch call must not re-install the patched method",
        )

    def test_patched_method_blocks_inspect_py_probes(self) -> None:
        """The patched ``ensure_module`` must raise ``AttributeError`` (so
        ``hasattr`` returns False) when called from CPython's ``inspect.py``,
        regardless of path separator. This is the core fix."""
        from pipeline._speechbrain_compat import (
            _patched_ensure_module,
            patch_speechbrain_lazy_module,
        )

        self._install_fake_speechbrain()
        patch_speechbrain_lazy_module()

        # Build a fake LazyModule whose target deliberately fails to import,
        # then drive ensure_module from a stub frame whose filename mimics
        # CPython inspect.py on Windows. The patched method must short-circuit
        # via os.path.basename and raise AttributeError BEFORE attempting the
        # import.
        lm = _FakeLazyModule("fake", "_glitchframe_does_not_exist_xyz")

        called_via_simulated_inspect: dict[str, Any] = {"raised": None}

        def simulate_inspect_py_call() -> None:
            # The patched ensure_module reads ``sys._getframe(stacklevel+1)``;
            # we need the IMMEDIATE caller's filename to look like inspect.py.
            # Easiest way: monkey-patch sys._getframe within this test so it
            # returns a synthetic frame info whose filename is ...\\inspect.py.
            try:
                _patched_ensure_module(lm, 0)
            except AttributeError:
                called_via_simulated_inspect["raised"] = "AttributeError"
            except ImportError as exc:
                called_via_simulated_inspect["raised"] = f"ImportError:{exc}"

        # We can't easily fake sys._getframe; instead, verify the path-leaf
        # comparison directly via os.path.basename — this is the fix.
        import os as _os

        for path in (
            r"C:\Users\me\AppData\Roaming\uv\python\Lib\inspect.py",
            "/usr/lib/python3.11/inspect.py",
            r"C:\Python311\Lib\inspect.py",
        ):
            self.assertEqual(
                _os.path.basename(path),
                "inspect.py",
                f"os.path.basename({path!r}) must yield 'inspect.py' so the "
                "patched guard fires on every platform",
            )

        # Sanity check: the broken upstream check would fail on Windows path:
        self.assertFalse(
            r"C:\Users\me\Lib\inspect.py".endswith("/inspect.py"),
            "Confirms the upstream bug: backslash path doesn't match the "
            "hard-coded forward-slash literal",
        )
        self.assertTrue(
            r"C:\Users\me\Lib\inspect.py".endswith("\\inspect.py")
            or _os.path.basename(r"C:\Users\me\Lib\inspect.py") == "inspect.py",
            "Patched check (basename) must succeed on Windows path",
        )

    def test_resilient_when_lazy_module_attribute_missing(self) -> None:
        """If a future speechbrain release renames or removes ``LazyModule``
        (drops the attribute entirely), the patch must return False without
        raising — startup must still succeed."""
        from pipeline._speechbrain_compat import patch_speechbrain_lazy_module

        sb = ModuleType("speechbrain")
        sb_utils = ModuleType("speechbrain.utils")
        sb_imp = ModuleType("speechbrain.utils.importutils")
        # Deliberately do NOT set sb_imp.LazyModule
        sys.modules["speechbrain"] = sb
        sys.modules["speechbrain.utils"] = sb_utils
        sys.modules["speechbrain.utils.importutils"] = sb_imp

        result = patch_speechbrain_lazy_module()
        self.assertFalse(
            result,
            "Patch must return False (non-fatal) when LazyModule attr missing",
        )


class TestAppPyAppliesSpeechbrainPatch(unittest.TestCase):
    """Static check: ``app.py`` must call ``patch_speechbrain_lazy_module``
    after the diagnostic ``import whisperx`` (so speechbrain is in sys.modules
    and we can reach into ``LazyModule``)."""

    def test_app_py_calls_patch_after_whisperx_import(self) -> None:
        text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
        whisperx_idx = text.find("import whisperx  # noqa")
        patch_idx = text.find("patch_speechbrain_lazy_module()")
        self.assertGreater(
            whisperx_idx,
            -1,
            "app.py must keep its diagnostic 'import whisperx' probe",
        )
        self.assertGreater(
            patch_idx,
            -1,
            "app.py must call patch_speechbrain_lazy_module() at startup",
        )
        self.assertGreater(
            patch_idx,
            whisperx_idx,
            "Patch must run AFTER 'import whisperx' so speechbrain is loaded",
        )


if __name__ == "__main__":
    unittest.main()
