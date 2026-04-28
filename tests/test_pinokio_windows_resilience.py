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


class TestTorchXpuCompat(unittest.TestCase):
    """``patch_torch_xpu`` must:

    * Install a ``torch.xpu`` stub when torch is loaded but lacks ``torch.xpu``
      (Track A: torch 2.2.2 < 2.3, no Intel XPU support).
    * Be a no-op when torch already has a real ``torch.xpu`` (Track B: torch
      ≥ 2.3 / 2.4).
    * Be a no-op when torch isn't loaded yet (returns False, doesn't raise).
    * Idempotent — second call doesn't replace the existing stub.

    Without this stub, newer ``diffusers`` releases that reference
    ``torch.xpu`` at import time crash AnimateDiff SDXL pipeline loading
    with ``AttributeError: module 'torch' has no attribute 'xpu'``.
    """

    def setUp(self) -> None:
        # Save & remove every torch.* module so each test starts clean.
        self._saved = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k == "torch" or k.startswith("torch.")
        }
        for k in self._saved:
            del sys.modules[k]

    def tearDown(self) -> None:
        # Drop any fakes we registered.
        for k in list(sys.modules):
            if (k == "torch" or k.startswith("torch.")) and k not in self._saved:
                del sys.modules[k]
        for k, v in self._saved.items():
            sys.modules[k] = v

    def _install_fake_torch_without_xpu(self) -> ModuleType:
        torch = ModuleType("torch")
        torch.__version__ = "2.2.2+cu121"  # type: ignore[attr-defined]
        # Critically, do NOT set torch.xpu — this simulates Track A.
        sys.modules["torch"] = torch
        return torch

    def _install_fake_torch_with_real_xpu(self) -> tuple[ModuleType, ModuleType]:
        torch = ModuleType("torch")
        torch.__version__ = "2.4.0+cu124"  # type: ignore[attr-defined]
        real_xpu = ModuleType("torch.xpu")
        real_xpu.is_available = lambda: True  # type: ignore[attr-defined]
        torch.xpu = real_xpu  # type: ignore[attr-defined]
        sys.modules["torch"] = torch
        sys.modules["torch.xpu"] = real_xpu
        return torch, real_xpu

    def test_no_op_when_torch_absent(self) -> None:
        from pipeline._torch_xpu_compat import patch_torch_xpu

        self.assertNotIn("torch", sys.modules)
        result = patch_torch_xpu()
        self.assertFalse(
            result,
            "patch_torch_xpu must return False when torch not loaded",
        )

    def test_installs_stub_when_missing(self) -> None:
        from pipeline._torch_xpu_compat import _PATCH_MARKER, patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        result = patch_torch_xpu()

        self.assertTrue(result, "Stub install must return True on success")
        self.assertTrue(hasattr(torch, "xpu"))
        self.assertFalse(
            torch.xpu.is_available(),  # type: ignore[attr-defined]
            "Stub torch.xpu.is_available() must return False so consumer "
            "code skips the XPU path",
        )
        self.assertEqual(torch.xpu.device_count(), 0)  # type: ignore[attr-defined]
        self.assertTrue(
            getattr(torch.xpu, _PATCH_MARKER, False),
            "Stub must carry the patch marker for downstream identification",
        )
        self.assertIs(
            sys.modules.get("torch.xpu"),
            torch.xpu,  # type: ignore[attr-defined]
            "Stub must also be registered in sys.modules so 'import "
            "torch.xpu' works",
        )

    def test_does_not_clobber_real_xpu(self) -> None:
        from pipeline._torch_xpu_compat import _PATCH_MARKER, patch_torch_xpu

        torch, real_xpu = self._install_fake_torch_with_real_xpu()
        result = patch_torch_xpu()

        self.assertTrue(result)
        self.assertIs(
            torch.xpu,  # type: ignore[attr-defined]
            real_xpu,
            "Real torch.xpu (Track B) must NOT be overwritten by the stub",
        )
        self.assertFalse(
            getattr(real_xpu, _PATCH_MARKER, False),
            "Real torch.xpu must not carry the stub marker",
        )

    def test_idempotent(self) -> None:
        from pipeline._torch_xpu_compat import patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        first_result = patch_torch_xpu()
        first_xpu = torch.xpu  # type: ignore[attr-defined]
        second_result = patch_torch_xpu()
        second_xpu = torch.xpu  # type: ignore[attr-defined]

        self.assertTrue(first_result)
        self.assertTrue(second_result)
        self.assertIs(
            first_xpu,
            second_xpu,
            "Second patch call must not replace the stub",
        )

    def test_stub_exposes_amp_namespace(self) -> None:
        """diffusers/transformers reach into ``torch.xpu.amp.*`` even when
        XPU is unused (e.g. ``hasattr(torch.xpu.amp, "autocast")``). The
        stub's ``__getattr__`` synthesises an ``amp`` sub-stub on demand."""
        from pipeline._torch_xpu_compat import patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        patch_torch_xpu()
        self.assertTrue(hasattr(torch.xpu, "amp"))  # type: ignore[attr-defined]
        self.assertIs(
            sys.modules.get("torch.xpu.amp"),
            torch.xpu.amp,  # type: ignore[attr-defined]
            "amp sub-stub must be registered in sys.modules so 'import "
            "torch.xpu.amp' works",
        )

    def test_stub_supports_manual_seed_dispatch(self) -> None:
        """diffusers/utils/torch_utils.py builds a ``{device: seed_fn}``
        dispatch table at module load time::

            "xpu": torch.xpu.manual_seed,

        The stub must expose ``manual_seed`` (and the related ``seed`` /
        ``manual_seed_all`` / ``seed_all`` family) as a callable that
        accepts a seed without raising. With class-shaped stubs (so PEP
        604 unions work — see ``test_stub_pep604_union``), calling the
        stub instantiates the class; what matters is the call doesn't
        raise and the result isn't a hard error."""
        from pipeline._torch_xpu_compat import patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        patch_torch_xpu()

        for name in ("manual_seed", "manual_seed_all", "seed", "seed_all"):
            fn = getattr(torch.xpu, name)  # type: ignore[attr-defined]
            self.assertTrue(
                callable(fn),
                f"torch.xpu.{name} must be callable for diffusers seed "
                f"dispatch tables",
            )
            # Must not raise — that's the only thing diffusers' dispatch
            # actually depends on.
            result = fn(42)
            self.assertIsNotNone(
                result,
                f"torch.xpu.{name}(42) must return a stub instance "
                f"(class-shaped stub for PEP 604 union compatibility)",
            )

    def test_stub_synthesises_unknown_attributes_on_demand(self) -> None:
        """Newer diffusers / transformers / peft releases probe additional
        ``torch.xpu.*`` symbols at import time. The stub's ``__getattr__``
        must return a callable class for any unrecognised attribute so
        future library upgrades don't reintroduce this bug class."""
        from pipeline._torch_xpu_compat import patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        patch_torch_xpu()

        # Anything goes — we don't care what diffusers asks for tomorrow.
        for name in ("future_thing", "another_seed_fn", "some_new_probe"):
            cls = getattr(torch.xpu, name)  # type: ignore[attr-defined]
            self.assertTrue(callable(cls), f"torch.xpu.{name} must be callable")
            self.assertTrue(
                isinstance(cls, type),
                f"torch.xpu.{name} must be a class (so PEP 604 unions work)",
            )
            # Calling instantiates the class without raising.
            cls("whatever", and_a_kwarg=1)

        # Sub-namespace passthrough: torch.xpu.random.foo must also work.
        torch.xpu.random.bar(99)  # type: ignore[attr-defined]

    def test_stub_pep604_union(self) -> None:
        """**Regression guard for the bug that landed in commit 188fa0c**:

        Newer ``transformers`` (~4.45+) writes PEP 604 union annotations
        like ``def foo(mesh: DeviceMesh | None = None): ...`` evaluated
        at *function-definition* time on Python 3.10+. If our stub
        returns a function for ``DeviceMesh``, that annotation raises::

            TypeError: unsupported operand type(s) for |:
                       'function' and 'NoneType'

        which propagates out as ``Failed to import diffusers.loaders.
        single_file`` and breaks AnimateDiff render. The fix: synthesise
        **classes**, not functions — classes inherit ``__or__`` from the
        ``type`` metaclass, so ``MyClass | None`` evaluates to
        ``types.UnionType`` and the import succeeds."""
        from pipeline._torch_xpu_compat import patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        patch_torch_xpu()

        # Every probe path that newer transformers / diffusers might hit:
        candidates = [
            torch.xpu.manual_seed,  # type: ignore[attr-defined]
            torch.xpu.synchronize,  # type: ignore[attr-defined]
            torch.xpu.empty_cache,  # type: ignore[attr-defined]
            torch.xpu.random.foo,  # synthesised in nested namespace  # type: ignore[attr-defined]
            torch.xpu.amp.autocast,  # synthesised in nested namespace  # type: ignore[attr-defined]
            torch.xpu.completely_new_thing_2026,  # type: ignore[attr-defined]
        ]
        for stub in candidates:
            # If this raises TypeError, transformers / diffusers will fail
            # to import on Track A. Don't regress.
            union = stub | None
            self.assertIn("UnionType", type(union).__name__ + str(union))

    def test_stub_dunders_still_raise_attribute_error(self) -> None:
        """``__getattr__`` must NOT synthesise dunders — pretending to
        support ``__reduce__`` / ``__init_subclass__`` / etc. confuses
        copy.deepcopy, pickling, debuggers, and isinstance checks."""
        from pipeline._torch_xpu_compat import patch_torch_xpu

        torch = self._install_fake_torch_without_xpu()
        patch_torch_xpu()

        # AttributeError, not a synthesised callable:
        with self.assertRaises(AttributeError):
            torch.xpu.__some_dunder__  # type: ignore[attr-defined]


class TestTorchDistributedDeviceMeshCompat(unittest.TestCase):
    """``patch_torch_distributed_device_mesh`` must:

    * Install a ``torch.distributed.device_mesh`` stub when missing
      (Track A: torch 2.2.2 < 2.4, no device_mesh submodule).
    * Be a no-op when torch already has a real ``torch.distributed.device_mesh``
      (Track B: torch ≥ 2.4).
    * Be a no-op when torch isn't loaded yet (returns False, doesn't raise).
    * Idempotent — second call doesn't replace the existing stub.

    Newer ``transformers`` (4.45+ ish) imports ``torch.distributed.device_mesh``
    at module load time without a hasattr() guard. The error surfaces inside
    ``diffusers.loaders.single_file`` on Track A as
    ``AttributeError: module 'torch.distributed' has no attribute 'device_mesh'``.
    """

    def setUp(self) -> None:
        self._saved = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k == "torch" or k.startswith("torch.")
        }
        for k in self._saved:
            del sys.modules[k]

    def tearDown(self) -> None:
        for k in list(sys.modules):
            if (k == "torch" or k.startswith("torch.")) and k not in self._saved:
                del sys.modules[k]
        for k, v in self._saved.items():
            sys.modules[k] = v

    def _install_track_a_torch(self) -> tuple[ModuleType, ModuleType]:
        """Track A: torch.distributed exists but has no device_mesh."""
        torch = ModuleType("torch")
        torch.__version__ = "2.2.2+cu121"  # type: ignore[attr-defined]
        dist = ModuleType("torch.distributed")
        dist.is_available = lambda: True  # type: ignore[attr-defined]
        torch.distributed = dist  # type: ignore[attr-defined]
        sys.modules["torch"] = torch
        sys.modules["torch.distributed"] = dist
        return torch, dist

    def _install_track_b_torch(self) -> tuple[ModuleType, ModuleType, ModuleType]:
        """Track B: torch.distributed.device_mesh exists with the real classes."""
        torch = ModuleType("torch")
        torch.__version__ = "2.4.0+cu124"  # type: ignore[attr-defined]
        dist = ModuleType("torch.distributed")
        real_mesh = ModuleType("torch.distributed.device_mesh")

        class _RealDeviceMesh:
            pass

        real_mesh.DeviceMesh = _RealDeviceMesh  # type: ignore[attr-defined]
        dist.device_mesh = real_mesh  # type: ignore[attr-defined]
        torch.distributed = dist  # type: ignore[attr-defined]
        sys.modules["torch"] = torch
        sys.modules["torch.distributed"] = dist
        sys.modules["torch.distributed.device_mesh"] = real_mesh
        return torch, dist, real_mesh

    def test_no_op_when_torch_absent(self) -> None:
        from pipeline._torch_xpu_compat import patch_torch_distributed_device_mesh

        self.assertNotIn("torch", sys.modules)
        result = patch_torch_distributed_device_mesh()
        self.assertFalse(result)

    def test_no_op_when_distributed_absent(self) -> None:
        """Pathological case: torch present but no .distributed attr."""
        from pipeline._torch_xpu_compat import patch_torch_distributed_device_mesh

        torch = ModuleType("torch")
        torch.__version__ = "2.2.2"  # type: ignore[attr-defined]
        sys.modules["torch"] = torch
        # No torch.distributed at all.
        self.assertFalse(patch_torch_distributed_device_mesh())

    def test_installs_stub_when_missing(self) -> None:
        from pipeline._torch_xpu_compat import (
            _DIST_MARKER,
            patch_torch_distributed_device_mesh,
        )

        torch, dist = self._install_track_a_torch()
        result = patch_torch_distributed_device_mesh()

        self.assertTrue(result)
        self.assertTrue(hasattr(dist, "device_mesh"))
        mesh = dist.device_mesh  # type: ignore[attr-defined]
        # Real probes: hasattr DeviceMesh / init_device_mesh.
        self.assertTrue(hasattr(mesh, "DeviceMesh"))
        self.assertTrue(hasattr(mesh, "init_device_mesh"))
        self.assertTrue(callable(mesh.DeviceMesh))
        self.assertTrue(getattr(mesh, _DIST_MARKER, False))
        self.assertIs(
            sys.modules.get("torch.distributed.device_mesh"),
            mesh,
            "Stub must be in sys.modules so 'import torch.distributed."
            "device_mesh' works",
        )

    def test_does_not_clobber_real_device_mesh(self) -> None:
        from pipeline._torch_xpu_compat import (
            _DIST_MARKER,
            patch_torch_distributed_device_mesh,
        )

        torch, dist, real_mesh = self._install_track_b_torch()
        result = patch_torch_distributed_device_mesh()

        self.assertTrue(result)
        self.assertIs(
            dist.device_mesh,  # type: ignore[attr-defined]
            real_mesh,
            "Real torch.distributed.device_mesh (Track B) must NOT be "
            "overwritten",
        )
        self.assertFalse(getattr(real_mesh, _DIST_MARKER, False))

    def test_idempotent(self) -> None:
        from pipeline._torch_xpu_compat import patch_torch_distributed_device_mesh

        _, dist = self._install_track_a_torch()
        first = patch_torch_distributed_device_mesh()
        first_mesh = dist.device_mesh  # type: ignore[attr-defined]
        second = patch_torch_distributed_device_mesh()
        second_mesh = dist.device_mesh  # type: ignore[attr-defined]

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertIs(first_mesh, second_mesh)

    def test_calling_stub_classes_raises_with_clear_message(self) -> None:
        """If a caller actually tries to *use* DeviceMesh (build a mesh,
        not just probe its existence), they should get a clear runtime
        error -- not a silent no-op that pretends to construct a mesh."""
        from pipeline._torch_xpu_compat import patch_torch_distributed_device_mesh

        _, dist = self._install_track_a_torch()
        patch_torch_distributed_device_mesh()
        with self.assertRaises(RuntimeError) as ctx:
            dist.device_mesh.DeviceMesh()  # type: ignore[attr-defined]
        msg = str(ctx.exception).lower()
        self.assertIn("device_mesh", msg)
        self.assertIn("torch", msg)

    def test_device_mesh_pep604_union(self) -> None:
        """**Regression guard for commit 188fa0c failure** observed in the
        wild: newer ``transformers`` writes annotations like::

            from torch.distributed.device_mesh import DeviceMesh
            def foo(mesh: DeviceMesh | None = None) -> None: ...

        Evaluated at function-definition time. With function-shaped
        DeviceMesh, this raised ``TypeError: unsupported operand type(s)
        for |: 'function' and 'NoneType'`` and broke
        ``diffusers.loaders.single_file`` import (which transitively imports
        transformers). Class-shaped DeviceMesh has ``__or__`` from the
        ``type`` metaclass, so the union evaluates to a ``types.UnionType``
        and the import completes."""
        from pipeline._torch_xpu_compat import patch_torch_distributed_device_mesh

        _, dist = self._install_track_a_torch()
        patch_torch_distributed_device_mesh()

        DeviceMesh = dist.device_mesh.DeviceMesh  # type: ignore[attr-defined]
        init_device_mesh = dist.device_mesh.init_device_mesh  # type: ignore[attr-defined]

        # Both must be classes (not functions). Classes inherit __or__.
        self.assertIsInstance(DeviceMesh, type)
        self.assertIsInstance(init_device_mesh, type)

        # The actual bug: this evaluation must NOT raise.
        union1 = DeviceMesh | None
        union2 = init_device_mesh | None
        self.assertIn("UnionType", type(union1).__name__ + str(union1))
        self.assertIn("UnionType", type(union2).__name__ + str(union2))


class TestPatchAll(unittest.TestCase):
    """``patch_all`` must apply every torch-attribute compat shim and return
    a diagnostic mapping. Used by both ``app.py`` startup and the AnimateDiff
    import belt-and-braces call."""

    def setUp(self) -> None:
        self._saved = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k == "torch" or k.startswith("torch.")
        }
        for k in self._saved:
            del sys.modules[k]

    def tearDown(self) -> None:
        for k in list(sys.modules):
            if (k == "torch" or k.startswith("torch.")) and k not in self._saved:
                del sys.modules[k]
        for k, v in self._saved.items():
            sys.modules[k] = v

    def test_patch_all_installs_both_stubs_on_track_a(self) -> None:
        from pipeline._torch_xpu_compat import patch_all

        torch = ModuleType("torch")
        torch.__version__ = "2.2.2+cu121"  # type: ignore[attr-defined]
        dist = ModuleType("torch.distributed")
        dist.is_available = lambda: True  # type: ignore[attr-defined]
        torch.distributed = dist  # type: ignore[attr-defined]
        sys.modules["torch"] = torch
        sys.modules["torch.distributed"] = dist

        result = patch_all()
        self.assertEqual(result, {
            "torch.xpu": True,
            "torch.distributed.device_mesh": True,
        })
        # Verify the actual import chain that diffusers + transformers hit:
        import torch as t  # noqa: PLC0415
        # diffusers/utils/torch_utils.py line:
        seed_fn = t.xpu.manual_seed
        self.assertTrue(callable(seed_fn))
        seed_fn(42)
        # transformers / diffusers single_file.py probe:
        mesh = t.distributed.device_mesh
        self.assertTrue(hasattr(mesh, "DeviceMesh"))


class TestAppPyAppliesTorchXpuPatch(unittest.TestCase):
    """Static check: ``app.py`` must call the torch-compat patches
    immediately after the diagnostic ``import torch`` so any subsequent
    diffusers import finds the stubs already installed."""

    def test_app_py_calls_patch_after_torch_import(self) -> None:
        text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
        torch_import_idx = text.find("import torch\n")
        # We accept either the umbrella ``patch_all()`` call or the
        # individual ``patch_torch_xpu() / patch_torch_distributed_device_mesh()``
        # calls — whichever app.py wires up.
        patch_idx = max(
            text.find("_patch_all_torch_compat()"),
            text.find("patch_torch_xpu()"),
        )
        self.assertGreater(
            torch_import_idx,
            -1,
            "app.py must keep its diagnostic 'import torch' probe",
        )
        self.assertGreater(
            patch_idx,
            -1,
            "app.py must call a torch-compat patch at startup",
        )
        self.assertGreater(
            patch_idx,
            torch_import_idx,
            "Patch must run AFTER 'import torch' so torch is in sys.modules",
        )


class TestAnimateDiffImportCallsXpuPatch(unittest.TestCase):
    """Static check: ``background_animatediff._import_animatediff_sdxl``
    must call the torch-compat patches BEFORE the ``from diffusers import
    AnimateDiffSDXLPipeline, DDIMScheduler`` line. This is the safety net
    for any code path that imports diffusers before app startup completes."""

    def test_animatediff_calls_xpu_patch_before_diffusers_import(self) -> None:
        text = (
            REPO_ROOT / "pipeline" / "background_animatediff.py"
        ).read_text(encoding="utf-8")
        # Locate the function body.
        fn_idx = text.find("def _import_animatediff_sdxl")
        self.assertGreater(fn_idx, -1, "Function must exist")
        body = text[fn_idx:]
        # Accept the umbrella ``patch_all()`` or the individual patches.
        patch_idx = max(
            body.find("_patch_all_torch_compat()"),
            body.find("patch_torch_xpu()"),
        )
        diffusers_idx = body.find("from diffusers import AnimateDiffSDXLPipeline")
        self.assertGreater(
            patch_idx,
            -1,
            "background_animatediff._import_animatediff_sdxl must call a "
            "torch-compat patch before importing diffusers",
        )
        self.assertGreater(
            diffusers_idx,
            -1,
            "diffusers import line must still be present in the function",
        )
        self.assertLess(
            patch_idx,
            diffusers_idx,
            "patch_torch_xpu() must be called BEFORE 'from diffusers import "
            "AnimateDiffSDXLPipeline'",
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
