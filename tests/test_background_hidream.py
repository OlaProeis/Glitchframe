"""
Unit tests for the HiDream backend selector + cache namespacing.

These tests intentionally do **not** spawn the worker subprocess or load any
HiDream weights — they exercise the pure-Python plumbing (env-driven config,
``model_id`` derivation, factory dispatch) so CI without HiDream installed
still passes.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from pipeline.background import (
    IMAGE_BACKEND_HIDREAM,
    IMAGE_BACKEND_SDXL,
    MODE_SDXL_STILLS,
    create_background_source,
    normalize_image_backend,
)
from pipeline.background_stills import BackgroundStills
from pipeline.background_stills_hidream import (
    HiDreamConfig,
    MODEL_ID_PREFIX,
    _compute_model_id,
    load_hidream_config,
)


class NormalizeImageBackendTests(unittest.TestCase):
    def test_default_is_sdxl(self) -> None:
        self.assertEqual(normalize_image_backend(None), IMAGE_BACKEND_SDXL)
        self.assertEqual(normalize_image_backend(""), IMAGE_BACKEND_SDXL)
        self.assertEqual(normalize_image_backend("   "), IMAGE_BACKEND_SDXL)

    def test_canonical_values(self) -> None:
        self.assertEqual(normalize_image_backend("sdxl"), IMAGE_BACKEND_SDXL)
        self.assertEqual(normalize_image_backend("hidream"), IMAGE_BACKEND_HIDREAM)

    def test_aliases(self) -> None:
        self.assertEqual(
            normalize_image_backend("HiDream-O1-Image"), IMAGE_BACKEND_HIDREAM
        )
        self.assertEqual(
            normalize_image_backend("Stable Diffusion XL"), IMAGE_BACKEND_SDXL
        )

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_backend("flux")


class HiDreamModelIdTests(unittest.TestCase):
    def test_namespace_prefix(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "weights"
            path.mkdir()
            mid = _compute_model_id(path, "dev")
            self.assertTrue(mid.startswith(f"{MODEL_ID_PREFIX}:dev:"))

    def test_different_paths_yield_different_ids(self) -> None:
        with TemporaryDirectory() as td:
            a = Path(td) / "a"
            b = Path(td) / "b"
            a.mkdir()
            b.mkdir()
            self.assertNotEqual(
                _compute_model_id(a, "dev"),
                _compute_model_id(b, "dev"),
            )

    def test_different_recipes_yield_different_ids(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "weights"
            p.mkdir()
            self.assertNotEqual(
                _compute_model_id(p, "dev"),
                _compute_model_id(p, "full"),
            )

    def test_does_not_collide_with_sdxl_default(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "weights"
            p.mkdir()
            self.assertNotEqual(
                _compute_model_id(p, "dev"),
                "stabilityai/stable-diffusion-xl-base-1.0",
            )


class HiDreamAutoLayoutTests(unittest.TestCase):
    def test_relaxed_mode_default_paths(self) -> None:
        with TemporaryDirectory() as td:
            fake_root = Path(td) / "m"
            with mock.patch(
                "pipeline.background_stills_hidream._hidream_bundle_root",
                return_value=fake_root,
            ):
                cfg = load_hidream_config(allow_fetch=False, strict_env=False)
            self.assertEqual(cfg.repo, fake_root / "HiDream-O1-Image")
            self.assertEqual(
                cfg.model_path,
                fake_root / "hf--drbaph--HiDream-O1-Image-FP8",
            )


class HiDreamConfigEnvTests(unittest.TestCase):
    def test_missing_required_env_raises(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("GLITCHFRAME_HIDREAM_")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                load_hidream_config(strict_env=True, allow_fetch=False)

    def test_valid_env_resolves_paths(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td)
            py = tdp / "python.exe"
            py.write_text("")
            repo = tdp / "repo"
            repo.mkdir()
            model = tdp / "weights"
            model.mkdir()
            env = {
                "GLITCHFRAME_HIDREAM_PYTHON": str(py),
                "GLITCHFRAME_HIDREAM_REPO": str(repo),
                "GLITCHFRAME_HIDREAM_MODEL_PATH": str(model),
                "GLITCHFRAME_HIDREAM_MODEL_TYPE": "dev",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                cfg = load_hidream_config(strict_env=True, allow_fetch=False)
                self.assertIsInstance(cfg, HiDreamConfig)
                self.assertEqual(cfg.model_type, "dev")
                self.assertEqual(cfg.python, py)
                self.assertEqual(cfg.repo, repo)
                self.assertEqual(cfg.model_path, model)
                self.assertTrue(
                    cfg.model_id().startswith(f"{MODEL_ID_PREFIX}:dev:")
                )

    def test_invalid_model_type_rejected(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td)
            py = tdp / "python.exe"
            py.write_text("")
            repo = tdp / "repo"
            repo.mkdir()
            model = tdp / "weights"
            model.mkdir()
            env = {
                "GLITCHFRAME_HIDREAM_PYTHON": str(py),
                "GLITCHFRAME_HIDREAM_REPO": str(repo),
                "GLITCHFRAME_HIDREAM_MODEL_PATH": str(model),
                "GLITCHFRAME_HIDREAM_MODEL_TYPE": "turbo",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(RuntimeError):
                    load_hidream_config(strict_env=True, allow_fetch=False)


class FactoryDispatchTests(unittest.TestCase):
    def test_sdxl_default_returns_background_stills(self) -> None:
        with TemporaryDirectory() as td:
            cache = Path(td)
            bg = create_background_source(
                MODE_SDXL_STILLS,
                cache,
                preset_id="test",
                preset_prompt="neon synthwave city",
            )
            try:
                self.assertIsInstance(bg, BackgroundStills)
                # The HiDream subclass must NOT match here.
                from pipeline.background_stills_hidream import (
                    BackgroundStillsHiDream,
                )

                self.assertNotIsInstance(bg, BackgroundStillsHiDream)
            finally:
                bg.close()

    def test_hidream_backend_dispatches_to_subclass(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td)
            cache = tdp / "song"
            cache.mkdir()
            py = tdp / "python.exe"
            py.write_text("")
            repo = tdp / "repo"
            repo.mkdir()
            model = tdp / "weights"
            model.mkdir()
            env = {
                "GLITCHFRAME_HIDREAM_PYTHON": str(py),
                "GLITCHFRAME_HIDREAM_REPO": str(repo),
                "GLITCHFRAME_HIDREAM_MODEL_PATH": str(model),
                "GLITCHFRAME_HIDREAM_MODEL_TYPE": "dev",
            }
            cfg = HiDreamConfig(
                python=py,
                repo=repo,
                model_path=model,
                model_type="dev",
                gen_width=1280,
                gen_height=720,
                pipeline_import="auto",
            )
            with mock.patch.dict(os.environ, env, clear=False):
                bg = create_background_source(
                    MODE_SDXL_STILLS,
                    cache,
                    preset_id="test",
                    preset_prompt="neon synthwave city",
                    image_backend=IMAGE_BACKEND_HIDREAM,
                    hidream_config=cfg,
                )
                try:
                    from pipeline.background_stills_hidream import (
                        BackgroundStillsHiDream,
                    )

                    self.assertIsInstance(bg, BackgroundStillsHiDream)
                    # HiDream's manifest model_id MUST be namespaced so the
                    # cache cannot collide with the SDXL bake.
                    self.assertTrue(
                        bg._model_id.startswith(f"{MODEL_ID_PREFIX}:dev:")  # type: ignore[attr-defined]
                    )
                finally:
                    bg.close()


if __name__ == "__main__":
    unittest.main()
