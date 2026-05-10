"""
Standalone HiDream-O1-Image generation worker.

This script is **not** imported by Glitchframe. It is invoked as a subprocess
by :mod:`pipeline.background_stills_hidream` using a *separate* Python
interpreter (typically the HiDream-O1-Image venv installed via Pinokio or
manually) so HiDream's heavy CUDA + flash-attn dependency stack never
contaminates Glitchframe's main environment.

Wire protocol
=============

The worker reads JSONL from stdin, one job per line::

    {"index": 0, "prompt": "...", "output_path": "/abs/path/keyframe_0000.png", "seed": 12345}

For each job it writes JSONL events to stdout, one per line::

    {"event": "ready"}
    {"event": "step", "index": 0, "step": 1, "steps_total": 28}
    ...
    {"event": "saved", "index": 0, "path": "..."}
    {"event": "error", "index": 0, "message": "..."}

Closing stdin (EOF) makes the worker exit cleanly. All non-JSONL diagnostics
go to stderr so they do not corrupt the protocol.

Compatibility
=============

The worker tries to import ``HiDreamImagePipeline`` from the ``models.pipeline``
module of the HiDream-O1-Image repo, mirroring the import that
``inference.py`` performs internally. If your fork of HiDream renames either
the module or the class, set ``--pipeline-import`` (e.g.
``models.pipeline:HiDreamImagePipeline``) to override.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _emit(event: dict) -> None:
    """Write a single JSONL event to stdout and flush."""
    sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    """Diagnostic log to stderr (does not interfere with the JSONL protocol)."""
    sys.stderr.write(f"[hidream-worker] {msg}\n")
    sys.stderr.flush()


def _resolve_pipeline_class(spec: str) -> Any:
    """Import ``module:ClassName`` (e.g. ``models.pipeline:HiDreamImagePipeline``)."""
    if ":" not in spec:
        raise ValueError(
            f"--pipeline-import must look like 'module:ClassName', got {spec!r}"
        )
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, cls_name):
        raise AttributeError(
            f"{mod_name!r} has no attribute {cls_name!r}; check --pipeline-import"
        )
    return getattr(mod, cls_name)


def _build_pipeline(
    *,
    repo: Path,
    model_path: Path,
    model_type: str,
    pipeline_import: str,
) -> Any:
    """Add HiDream repo to ``sys.path``, import its pipeline, load weights to CUDA."""
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is not importable in the HiDream worker venv. Install "
            "HiDream-O1-Image's requirements first (e.g. via the Pinokio app)."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available in the HiDream worker; HiDream-O1-Image "
            "requires a CUDA GPU."
        )

    PipelineCls = _resolve_pipeline_class(pipeline_import)

    _log(
        f"loading HiDream pipeline ({model_type}) from {model_path} "
        f"using {pipeline_import}"
    )
    # Common diffusers-style entry points; we try ``from_pretrained`` first
    # (matches the public API in HiDream-O1-Image's ``inference.py``).
    if hasattr(PipelineCls, "from_pretrained"):
        pipe = PipelineCls.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = PipelineCls(str(model_path))  # type: ignore[call-arg]

    try:
        pipe = pipe.to("cuda")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to move HiDream pipeline to CUDA: {exc}"
        ) from exc

    # If the pipeline exposes a memory saver, use it (mirrors SDXL path).
    for method_name in (
        "enable_vae_slicing",
        "enable_vae_tiling",
        "enable_attention_slicing",
        "enable_xformers_memory_efficient_attention",
    ):
        method = getattr(pipe, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # noqa: BLE001
                _log(f"{method_name} failed: {exc}")

    return pipe


def _generate_one(
    pipe: Any,
    *,
    index: int,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    model_type: str,
) -> None:
    """Run the pipeline for a single prompt and write a PNG to ``output_path``."""
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    generator: Any = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    def _step_cb(_pipe: Any, step_idx: int, _t: Any, kwargs: dict) -> dict:
        # diffusers reports step_idx as 0-based; emit 1-based to match the
        # SDXL UI where users never see "step 0 of N".
        _emit(
            {
                "event": "step",
                "index": int(index),
                "step": int(step_idx) + 1,
                "steps_total": int(steps),
            }
        )
        return kwargs

    call_kwargs: dict[str, Any] = dict(
        prompt=prompt,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )

    out = None
    try:
        out = pipe(**call_kwargs, callback_on_step_end=_step_cb)
    except TypeError:
        # Older / forked pipelines without ``callback_on_step_end`` — fall
        # back silently to a single coarse update per keyframe.
        out = pipe(**call_kwargs)

    images = getattr(out, "images", None)
    if not isinstance(images, list) or not images:
        raise RuntimeError(
            "HiDream pipeline returned no images; check --pipeline-import"
        )
    img: Image.Image = images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    img.save(str(tmp), format="PNG")
    os.replace(tmp, output_path)
    _emit({"event": "saved", "index": int(index), "path": str(output_path)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HiDream-O1-Image generation worker")
    parser.add_argument("--repo", type=Path, required=True, help="Path to HiDream-O1-Image checkout")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model weights")
    parser.add_argument(
        "--model-type",
        choices=["dev", "full"],
        default="dev",
        help="HiDream inference recipe (dev=28 steps, full=50 steps)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override num_inference_steps (default: 28 for dev, 50 for full)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override guidance_scale (default: 0.0 for dev, 5.0 for full)",
    )
    parser.add_argument(
        "--pipeline-import",
        default="models.pipeline:HiDreamImagePipeline",
        help="``module:ClassName`` import path for the HiDream pipeline class",
    )
    args = parser.parse_args(argv)

    if args.steps is None:
        args.steps = 28 if args.model_type == "dev" else 50
    if args.guidance_scale is None:
        args.guidance_scale = 0.0 if args.model_type == "dev" else 5.0

    try:
        pipe = _build_pipeline(
            repo=args.repo,
            model_path=args.model_path,
            model_type=args.model_type,
            pipeline_import=args.pipeline_import,
        )
    except Exception as exc:  # noqa: BLE001
        _log(traceback.format_exc())
        _emit({"event": "fatal", "message": f"{type(exc).__name__}: {exc}"})
        return 2

    _emit({"event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"event": "error", "index": -1, "message": f"bad JSON: {exc}"})
            continue
        try:
            index = int(job["index"])
            prompt = str(job["prompt"])
            output_path = Path(job["output_path"]).resolve()
            seed = job.get("seed")
            seed = int(seed) if seed is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            _emit(
                {
                    "event": "error",
                    "index": int(job.get("index", -1) or -1),
                    "message": f"bad job: {exc}",
                }
            )
            continue
        try:
            _generate_one(
                pipe,
                index=index,
                prompt=prompt,
                output_path=output_path,
                width=int(job.get("width", args.width)),
                height=int(job.get("height", args.height)),
                steps=int(job.get("steps", args.steps)),
                guidance_scale=float(job.get("guidance_scale", args.guidance_scale)),
                seed=seed,
                model_type=args.model_type,
            )
        except Exception as exc:  # noqa: BLE001
            _log(traceback.format_exc())
            _emit({"event": "error", "index": index, "message": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
