"""Unit tests for :mod:`pipeline.gpu_memory`.

Hermetic — never imports torch. We stub ``sys.modules["torch"]`` with a
fake that records calls, so CI / laptops without CUDA still exercise the
CUDA branches. This is the lifecycle that keeps VRAM from piling up
between demucs → WhisperX → SDXL stages; it must not regress silently.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

from pipeline.gpu_memory import move_to_cpu, release_cuda_memory


class _FakeTorch:
    """Minimal torch-shaped fake. Records every cuda.* call for assertions."""

    def __init__(self, *, cuda_available: bool = True, fail_on: set[str] | None = None) -> None:
        self._fail_on = fail_on or set()
        self.calls: list[str] = []

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                parent.calls.append("is_available")
                if "is_available" in parent._fail_on:
                    raise RuntimeError("simulated is_available failure")
                return cuda_available

            @staticmethod
            def empty_cache() -> None:
                parent.calls.append("empty_cache")
                if "empty_cache" in parent._fail_on:
                    raise RuntimeError("simulated empty_cache failure")

            @staticmethod
            def ipc_collect() -> None:
                parent.calls.append("ipc_collect")
                if "ipc_collect" in parent._fail_on:
                    raise RuntimeError("simulated ipc_collect failure")

        parent = self
        self.cuda = _Cuda


class TestReleaseCudaMemory(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_torch = sys.modules.get("torch")

    def tearDown(self) -> None:
        if self._saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved_torch

    def _install_fake_torch(self, fake: _FakeTorch) -> None:
        mod = types.ModuleType("torch")
        mod.cuda = fake.cuda  # type: ignore[attr-defined]
        sys.modules["torch"] = mod

    def test_calls_empty_cache_and_ipc_collect_when_cuda_available(self) -> None:
        fake = _FakeTorch(cuda_available=True)
        self._install_fake_torch(fake)
        release_cuda_memory("unit-test")
        self.assertIn("empty_cache", fake.calls)
        self.assertIn("ipc_collect", fake.calls)

    def test_skips_cuda_calls_when_not_available(self) -> None:
        fake = _FakeTorch(cuda_available=False)
        self._install_fake_torch(fake)
        release_cuda_memory("unit-test")
        self.assertIn("is_available", fake.calls)
        self.assertNotIn("empty_cache", fake.calls)
        self.assertNotIn("ipc_collect", fake.calls)

    def test_no_op_when_torch_not_installed(self) -> None:
        """Pure-CPU environments (CI without torch) must not raise."""
        sys.modules["torch"] = None  # import will fail
        try:
            release_cuda_memory("unit-test")  # should just return cleanly
        finally:
            sys.modules.pop("torch", None)

    def test_swallows_empty_cache_failure(self) -> None:
        """A buggy driver must not break cleanup; ipc_collect still runs."""
        fake = _FakeTorch(cuda_available=True, fail_on={"empty_cache"})
        self._install_fake_torch(fake)
        release_cuda_memory("unit-test")
        self.assertIn("empty_cache", fake.calls)
        self.assertIn("ipc_collect", fake.calls)

    def test_swallows_is_available_failure(self) -> None:
        fake = _FakeTorch(cuda_available=True, fail_on={"is_available"})
        self._install_fake_torch(fake)
        release_cuda_memory("unit-test")
        self.assertNotIn("empty_cache", fake.calls)


class TestMoveToCpu(unittest.TestCase):
    def test_prefers_cpu_method(self) -> None:
        obj = MagicMock()
        move_to_cpu(obj)
        obj.cpu.assert_called_once_with()
        obj.to.assert_not_called()

    def test_falls_back_to_to_cpu_when_cpu_missing(self) -> None:
        obj = types.SimpleNamespace(to=MagicMock())
        move_to_cpu(obj)
        obj.to.assert_called_once_with("cpu")

    def test_none_is_no_op(self) -> None:
        move_to_cpu(None)

    def test_object_without_cpu_or_to_is_no_op(self) -> None:
        move_to_cpu(object())  # must not raise

    def test_swallows_cpu_exception_and_tries_to(self) -> None:
        obj = MagicMock()
        obj.cpu.side_effect = RuntimeError("already on cpu")
        move_to_cpu(obj)
        obj.cpu.assert_called_once()
        obj.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
