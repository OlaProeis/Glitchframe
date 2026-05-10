"""
HiDream-O1-Image background stills backend.

Mirrors :class:`pipeline.background_stills.BackgroundStills` but generates
keyframes via an out-of-process **worker subprocess** running in HiDream's
own Python venv, so HiDream's heavy CUDA + ``flash-attn`` stack never
mixes with Glitchframe's main environment.

Cache compatibility
-------------------

The generated PNGs land at the same paths as the SDXL backend
(``cache/<hash>/background/keyframe_{i:04d}.png``) and the same
``manifest.json`` schema is used. The cache is namespaced by **model_id**,
so SDXL and HiDream caches never collide:

* SDXL backend → ``model_id = "stabilityai/stable-diffusion-xl-base-1.0"``
* HiDream backend → ``model_id = "hidream:<model_type>:<weights_hash>"``

A different ``model_id`` invalidates the manifest match in
:meth:`pipeline.background_stills.BackgroundManifest.matches_key`, forcing
regeneration with the chosen backend.

Configuration (all read from environment variables, see ``.env.example``):

* ``GLITCHFRAME_HIDREAM_PYTHON`` — path to the HiDream venv's Python
  executable (e.g. Pinokio's ``.../HiDream-O1-Image/env/Scripts/python.exe``).
* ``GLITCHFRAME_HIDREAM_REPO`` — path to a checkout of
  ``HiDream-ai/HiDream-O1-Image``.
* ``GLITCHFRAME_HIDREAM_MODEL_PATH`` — path to the model weights directory.
* ``GLITCHFRAME_HIDREAM_MODEL_TYPE`` — ``dev`` (default, 28 steps) or ``full``
  (50 steps).
* ``GLITCHFRAME_HIDREAM_GEN_WIDTH`` / ``GLITCHFRAME_HIDREAM_GEN_HEIGHT`` —
  override generation resolution (defaults: 1280×720, ~16 GB peak VRAM on
  the Dev FP8 checkpoint; raise these later once stability is verified).
* ``GLITCHFRAME_HIDREAM_PIPELINE_IMPORT`` — override the pipeline import
  spec (default ``models.pipeline:HiDreamImagePipeline``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from pipeline.background_stills import (
    BackgroundManifest,
    BackgroundStills,
    DEFAULT_HEIGHT,
    DEFAULT_KEYFRAME_INTERVAL,
    DEFAULT_WIDTH,
    KEYFRAME_FILENAME_FMT,
    ProgressFn,
    _atomic_write_json,
    _atomic_write_png,
    _background_dir,
    _keyframe_path,
    _manifest_path,
    _resize_rgb,
)

LOGGER = logging.getLogger(__name__)

# Generation resolution defaults. HiDream-O1-Image trained at up to 2048×2048,
# but FP8 Dev fits comfortably at 16:9 720p on a 24 GB card; we Lanczos-upscale
# to the requested output size in :func:`_resize_rgb`. Values override via env.
DEFAULT_GEN_WIDTH = 1280
DEFAULT_GEN_HEIGHT = 720

# Public model_id prefix for the cache key. Concrete model_ids look like
# ``hidream:dev:9f3a…`` (see :func:`_compute_model_id`).
MODEL_ID_PREFIX = "hidream"

# Path to the worker script (a sibling module file). Invoked as a script by
# HiDream's Python interpreter — never imported by Glitchframe.
_WORKER_SCRIPT = Path(__file__).resolve().with_name(
    "background_stills_hidream_worker.py"
)


# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiDreamConfig:
    """Resolved HiDream worker configuration sourced from environment vars."""

    python: Path
    repo: Path
    model_path: Path
    model_type: str
    gen_width: int
    gen_height: int
    pipeline_import: str

    def model_id(self) -> str:
        return _compute_model_id(self.model_path, self.model_type)


def _env(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return raw or None


def _compute_model_id(model_path: Path, model_type: str) -> str:
    """Stable cache id derived from the weights path + chosen recipe.

    Uses a short SHA-256 of the absolute path so different checkpoints
    (e.g. FP8 dev vs BF16 full) never share cached PNGs even when the user
    swaps ``GLITCHFRAME_HIDREAM_MODEL_PATH`` between runs.
    """
    h = hashlib.sha256(str(Path(model_path).resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{MODEL_ID_PREFIX}:{model_type}:{h}"


def load_hidream_config() -> HiDreamConfig:
    """Resolve a :class:`HiDreamConfig` from environment variables.

    Raises
    ------
    RuntimeError
        If a required variable is unset or a path is missing on disk.
    """
    py = _env("GLITCHFRAME_HIDREAM_PYTHON")
    repo = _env("GLITCHFRAME_HIDREAM_REPO")
    model_path = _env("GLITCHFRAME_HIDREAM_MODEL_PATH")
    if not py or not repo or not model_path:
        raise RuntimeError(
            "HiDream backend requires GLITCHFRAME_HIDREAM_PYTHON, "
            "GLITCHFRAME_HIDREAM_REPO and GLITCHFRAME_HIDREAM_MODEL_PATH to be "
            "set (see .env.example)."
        )
    py_path = Path(py)
    repo_path = Path(repo)
    model_path_path = Path(model_path)
    if not py_path.is_file():
        raise RuntimeError(f"GLITCHFRAME_HIDREAM_PYTHON not a file: {py_path}")
    if not repo_path.is_dir():
        raise RuntimeError(f"GLITCHFRAME_HIDREAM_REPO not a directory: {repo_path}")
    if not model_path_path.exists():
        raise RuntimeError(f"GLITCHFRAME_HIDREAM_MODEL_PATH missing: {model_path_path}")

    model_type = (_env("GLITCHFRAME_HIDREAM_MODEL_TYPE") or "dev").lower()
    if model_type not in ("dev", "full"):
        raise RuntimeError(
            f"GLITCHFRAME_HIDREAM_MODEL_TYPE must be 'dev' or 'full', got {model_type!r}"
        )

    gen_w = int(_env("GLITCHFRAME_HIDREAM_GEN_WIDTH") or DEFAULT_GEN_WIDTH)
    gen_h = int(_env("GLITCHFRAME_HIDREAM_GEN_HEIGHT") or DEFAULT_GEN_HEIGHT)
    if gen_w <= 0 or gen_h <= 0:
        raise RuntimeError(
            f"HiDream gen resolution must be positive, got {gen_w}x{gen_h}"
        )

    pipeline_import = (
        _env("GLITCHFRAME_HIDREAM_PIPELINE_IMPORT")
        or "models.pipeline:HiDreamImagePipeline"
    )

    return HiDreamConfig(
        python=py_path,
        repo=repo_path,
        model_path=model_path_path,
        model_type=model_type,
        gen_width=gen_w,
        gen_height=gen_h,
        pipeline_import=pipeline_import,
    )


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------


class _HiDreamWorker:
    """Persistent HiDream subprocess driver (one process for a whole batch).

    Reuses the model load across every keyframe in a batch so an 8B HiDream
    pipeline does not pay a 30-90 s reload per image. Communicates with the
    worker via JSONL on stdin/stdout (see
    :mod:`pipeline.background_stills_hidream_worker`).
    """

    def __init__(self, cfg: HiDreamConfig) -> None:
        self._cfg = cfg
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> "_HiDreamWorker":
        self._spawn()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def _spawn(self) -> None:
        if not _WORKER_SCRIPT.is_file():
            raise RuntimeError(f"HiDream worker script missing: {_WORKER_SCRIPT}")
        cmd = [
            str(self._cfg.python),
            str(_WORKER_SCRIPT),
            "--repo",
            str(self._cfg.repo),
            "--model-path",
            str(self._cfg.model_path),
            "--model-type",
            self._cfg.model_type,
            "--width",
            str(int(self._cfg.gen_width)),
            "--height",
            str(int(self._cfg.gen_height)),
            "--pipeline-import",
            self._cfg.pipeline_import,
        ]
        LOGGER.info("Launching HiDream worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        # Forward worker stderr to our logger so users see HiDream warnings /
        # tracebacks in the Gradio log without polluting stdout (the wire).
        def _drain_stderr() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            for line in self._proc.stderr:
                line = line.rstrip()
                if line:
                    LOGGER.info("hidream-worker: %s", line)

        self._stderr_thread = threading.Thread(
            target=_drain_stderr, name="hidream-worker-stderr", daemon=True
        )
        self._stderr_thread.start()

        # Wait for ``ready`` (or ``fatal``) on the JSONL channel before we
        # start sending jobs so the first keyframe submission does not race
        # the model load.
        evt = self._read_event(timeout_s=None)
        if evt.get("event") == "fatal":
            raise RuntimeError(
                f"HiDream worker failed to start: {evt.get('message', 'unknown error')}"
            )
        if evt.get("event") != "ready":
            raise RuntimeError(
                f"HiDream worker emitted unexpected first event: {evt!r}"
            )

    def _read_event(self, *, timeout_s: float | None) -> dict[str, Any]:
        """Read one JSONL event from the worker's stdout (blocking)."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("HiDream worker is not running")
        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.poll()
            raise RuntimeError(
                f"HiDream worker terminated unexpectedly (rc={rc}); see stderr log"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"HiDream worker emitted non-JSON line: {line!r} ({exc})"
            ) from exc

    def generate(
        self,
        *,
        index: int,
        prompt: str,
        output_path: Path,
        seed: int | None,
        on_step: "Any | None" = None,
    ) -> None:
        """Submit one job; block until the worker emits ``saved`` or ``error``."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("HiDream worker is not running")
        job: dict[str, Any] = {
            "index": int(index),
            "prompt": str(prompt),
            "output_path": str(output_path),
        }
        if seed is not None:
            job["seed"] = int(seed)
        self._proc.stdin.write(json.dumps(job, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()

        while True:
            evt = self._read_event(timeout_s=None)
            kind = evt.get("event")
            if kind == "step":
                if on_step is not None:
                    on_step(
                        int(evt.get("step", 0)),
                        int(evt.get("steps_total", 0)),
                    )
                continue
            if kind == "saved":
                if int(evt.get("index", -1)) != int(index):
                    raise RuntimeError(
                        f"HiDream worker saved wrong keyframe (got "
                        f"{evt.get('index')}, expected {index})"
                    )
                return
            if kind == "error":
                raise RuntimeError(
                    f"HiDream worker error on keyframe {index}: "
                    f"{evt.get('message', 'unknown')}"
                )
            if kind == "fatal":
                raise RuntimeError(
                    f"HiDream worker fatal: {evt.get('message', 'unknown')}"
                )
            raise RuntimeError(f"Unexpected HiDream worker event: {evt!r}")

    def close(self) -> None:
        """Close stdin and wait briefly for the worker to exit."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("HiDream worker stdin close: %s", exc)
            try:
                self._proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                LOGGER.warning("HiDream worker did not exit; killing")
                self._proc.kill()
        finally:
            self._proc = None


# ---------------------------------------------------------------------------
# Public BackgroundSource implementation
# ---------------------------------------------------------------------------


class BackgroundStillsHiDream(BackgroundStills):
    """HiDream-O1-Image variant of :class:`BackgroundStills`.

    Behaves identically to the SDXL backend (manifest schema, RIFE morph,
    Ken Burns, resume on cached PNGs, ``background_frame(t)``) — only the
    *image generation* step is different. Cache PNGs and the manifest live
    at the same paths; the cache is namespaced by ``model_id`` so SDXL and
    HiDream PNGs never collide.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        preset_id: str,
        preset_prompt: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        keyframe_interval: float = DEFAULT_KEYFRAME_INTERVAL,
        seed: int | None = 0,
        ken_burns: bool = False,
        ken_burns_margin: float | None = None,
        rife_morph: bool = False,
        rife_exp: int = 4,
        rife_repo_id: str = "MonsterMMORPG/RIFE_4_26",
        ken_burns_rms_drive_at: Any = None,
        config: HiDreamConfig | None = None,
    ) -> None:
        cfg = config if config is not None else load_hidream_config()
        # Wire HiDream-specific generation resolution + a unique model_id
        # into the parent so the SDXL prompt-hash / cache logic just works.
        super().__init__(
            cache_dir,
            preset_id=preset_id,
            preset_prompt=preset_prompt,
            width=width,
            height=height,
            gen_width=cfg.gen_width,
            gen_height=cfg.gen_height,
            keyframe_interval=keyframe_interval,
            model_id=cfg.model_id(),
            num_inference_steps=28 if cfg.model_type == "dev" else 50,
            guidance_scale=0.0 if cfg.model_type == "dev" else 5.0,
            negative_prompt="",
            seed=seed,
            ken_burns=ken_burns,
            ken_burns_margin=ken_burns_margin,
            rife_morph=rife_morph,
            rife_exp=rife_exp,
            rife_repo_id=rife_repo_id,
            ken_burns_rms_drive_at=ken_burns_rms_drive_at,
        )
        self._hidream_cfg = cfg

    # -- generation override -------------------------------------------------

    def _generate_and_persist(
        self,
        manifest: BackgroundManifest,
        report: ProgressFn,
        *,
        force_regenerate_indices: set[int] | None = None,
    ) -> list[np.ndarray]:
        """Generate any missing keyframes via the HiDream worker subprocess.

        Mirrors :meth:`BackgroundStills._generate_and_persist` resume + write
        semantics so the manifest cache, partial-resume, and editor preview
        flows behave identically across both backends.
        """
        bg_dir = _background_dir(self._cache_dir)
        bg_dir.mkdir(parents=True, exist_ok=True)

        total = max(1, manifest.num_keyframes)
        frames: list[np.ndarray | None] = [None] * total
        prompts = list(manifest.prompts)

        regen = force_regenerate_indices or set()
        missing_indices: list[int] = []
        missing_prompts: list[str] = []
        for i in range(total):
            path = _keyframe_path(self._cache_dir, i)
            if i not in regen and path.is_file():
                try:
                    with Image.open(path) as im:
                        frames[i] = _resize_rgb(
                            im.convert("RGB"), self._width, self._height
                        )
                    continue
                except Exception as exc:  # noqa: BLE001
                    LOGGER.info(
                        "Ignoring unreadable cached keyframe %s (%s); will regenerate",
                        path,
                        exc,
                    )
            missing_indices.append(i)
            missing_prompts.append(prompts[i])

        if not missing_prompts:
            _atomic_write_json(manifest.to_dict(), _manifest_path(self._cache_dir))
            report(1.0, f"Reused {total} cached keyframes")
            return [f for f in frames if f is not None]

        resumed = total - len(missing_prompts)
        if resumed > 0:
            report(
                0.1,
                f"Resuming: {resumed}/{total} keyframes cached, generating "
                f"{len(missing_prompts)} via HiDream",
            )

        report(
            0.05,
            f"Launching HiDream worker ({self._hidream_cfg.model_type}, "
            f"{self._hidream_cfg.gen_width}×{self._hidream_cfg.gen_height})…",
        )

        missing_total = len(missing_prompts)

        def _on_step_for(prompt_idx_local: int) -> Any:
            target_idx = missing_indices[prompt_idx_local]

            def _cb(step: int, steps_total: int) -> None:
                per_prompt = 1.0 / max(1, missing_total)
                frac_in_prompt = min(1.0, step / max(1, steps_total))
                overall = prompt_idx_local * per_prompt + frac_in_prompt * per_prompt
                report(
                    0.1 + 0.85 * overall,
                    f"HiDream keyframe {target_idx + 1}/{total} "
                    f"(step {step}/{steps_total})",
                )

            return _cb

        report(
            0.1,
            f"Generating {missing_total} HiDream keyframes "
            f"({self._num_inference_steps} steps each)",
        )

        with _HiDreamWorker(self._hidream_cfg) as worker:
            for j, target_idx in enumerate(missing_indices):
                seed_j = (
                    self._seed + target_idx if self._seed is not None else None
                )
                out_path = _keyframe_path(self._cache_dir, target_idx)
                worker.generate(
                    index=target_idx,
                    prompt=missing_prompts[j],
                    output_path=out_path,
                    seed=seed_j,
                    on_step=_on_step_for(j),
                )
                # Load the just-written PNG back into memory so the parent's
                # frame cache stays consistent (matches the SDXL path).
                with Image.open(out_path) as im:
                    frames[target_idx] = _resize_rgb(
                        im.convert("RGB"), self._width, self._height
                    )
                report(
                    0.1 + 0.85 * (j + 1) / max(1, missing_total),
                    f"Keyframe {target_idx + 1}/{total} saved",
                )

        if any(f is None for f in frames):
            missing_after = [i for i, f in enumerate(frames) if f is None]
            raise RuntimeError(
                f"HiDream generation left {len(missing_after)} keyframes "
                f"missing: {missing_after}"
            )

        _atomic_write_json(manifest.to_dict(), _manifest_path(self._cache_dir))
        report(1.0, "Background keyframes ready (HiDream)")
        return [f for f in frames if f is not None]  # type: ignore[misc]

    # -- lifecycle -----------------------------------------------------------

    def _dispose_sdxl_pipeline(self) -> None:
        """No SDXL pipeline is loaded for HiDream — override to a clean no-op."""
        return


__all__: Sequence[str] = [
    "BackgroundStillsHiDream",
    "DEFAULT_GEN_HEIGHT",
    "DEFAULT_GEN_WIDTH",
    "HiDreamConfig",
    "MODEL_ID_PREFIX",
    "load_hidream_config",
]

# Silence unused-import warnings for re-exported helpers; they are used via
# the parent class in ``_generate_and_persist``.
_ = (sys,)
