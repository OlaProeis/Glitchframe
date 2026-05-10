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
go to stderr so they do not corrupt the JSONL channel.

Compatibility
=============

Upstream ``HiDream-O1-Image`` (May 2026+) replaced the old
``HiDreamImagePipeline`` class with :func:`models.pipeline.generate_image`
plus ``Qwen3VLForConditionalGeneration`` (see the project's ``inference.py``).

With ``--pipeline-import auto`` (default), the worker picks **native**
``generate_image`` when available, else falls back to a diffusers-style class
from ``from_pretrained``. Pass an explicit ``module:Class`` only for forks.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

_JSONL_STDOUT: TextIO | None = None


def _reconfigure_stdio_utf8() -> None:
    """Ensure stdout/stderr accept UTF-8 so Hugging Face import-time prints do not crash on Windows (cp1252).

    ``transformers``' ``auto_docstring`` can ``print()`` messages containing emoji
    (e.g. U+1F6A8) while loading HiDream's ``qwen3_vl_transformers``; that raises
    ``UnicodeEncodeError`` when the console code page is not UTF-8.
    Disable with ``GLITCHFRAME_HIDREAM_WORKER_UTF8_STDIO=0`` if needed.
    """
    if os.environ.get("GLITCHFRAME_HIDREAM_WORKER_UTF8_STDIO", "").strip() == "0":
        return
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
                continue
            except (OSError, ValueError, AttributeError):
                pass
        buf = getattr(stream, "buffer", None)
        if buf is None:
            continue
        try:
            import io

            replacement = io.TextIOWrapper(
                buf,
                encoding="utf-8",
                errors="replace",
                line_buffering=stream is sys.stdout,
                write_through=True,
            )
        except Exception:
            continue
        if stream is sys.stdout:
            sys.stdout = replacement
        else:
            sys.stderr = replacement


_reconfigure_stdio_utf8()


def _wire_jsonl_only_stdout() -> None:
    """Route process ``sys.stdout`` to stderr so Hugging Face / tqdm ``print`` never corrupts JSONL.

    The parent only reads **JSON lines** from the worker's real stdout pipe.
    Recent ``transformers`` builds print docstring reminder lines (with emoji)
    to stdout during imports, which would be parsed as JSON and fail.

    JSON events are written via :func:`_emit` to the saved stream.
    Set ``GLITCHFRAME_HIDREAM_WORKER_ALLOW_LIB_STDOUT=1`` to skip (debug only).
    """
    global _JSONL_STDOUT
    _JSONL_STDOUT = sys.stdout
    if os.environ.get("GLITCHFRAME_HIDREAM_WORKER_ALLOW_LIB_STDOUT", "").strip() == "1":
        return
    sys.stdout = sys.stderr


_wire_jsonl_only_stdout()


def _emit(event: dict) -> None:
    """Write a single JSONL event to the pipe stdout and flush."""
    stream = _JSONL_STDOUT if _JSONL_STDOUT is not None else sys.stdout
    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    stream.flush()


def _log(msg: str) -> None:
    """Diagnostic log to stderr (does not interfere with the JSONL protocol)."""
    sys.stderr.write(f"[hidream-worker] {msg}\n")
    sys.stderr.flush()


def _add_special_tokens(tokenizer: Any) -> None:
    """Attach special-token shortcuts that the pipeline relies on (see ``inference.py``)."""
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor: Any) -> Any:
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


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


@dataclass
class _NativeHiDreamBundle:
    """Weights loaded for :func:`models.pipeline.generate_image`."""

    model: Any
    processor: Any
    model_type: str


def _is_fp8_checkpoint_hint(model_path: Path) -> bool:
    """Heuristic: does this path / repo id look like an FP8 community checkpoint?"""
    hf_id = os.environ.get("GLITCHFRAME_HIDREAM_HF_REPO_ID", "").strip().lower()
    if hf_id and any(h in hf_id for h in ("fp8", "drbaph", "e4m3", "f8e4")):
        return True
    path_s = str(model_path).lower()
    return any(
        hint in path_s
        for hint in (
            "fp8",
            "f8e4m3",
            "float8",
            "drbaph",
            "hidream-o1-image-dev-fp8",
        )
    )


def _native_weights_torch_dtype(model_path: Path) -> Any:
    """Return ``torch.dtype`` for :meth:`Qwen3VLForConditionalGeneration.from_pretrained`.

    Community FP8 checkpoints (e.g. ``drbaph/HiDream-O1-Image-FP8``) keep their main
    weights as ``Float8_e4m3fn`` regardless of ``torch_dtype`` — ``transformers``
    only uses ``torch_dtype`` as a *default for newly-created tensors*; safetensors
    entries that store Float8 stay Float8 on disk and in memory. Upstream
    ``generate_image`` then runs ``torch.autocast(dtype=bfloat16)`` and the BF16 ↔
    Float8 mix raises **Promotion for Float8 Types is not supported** at the
    ``torch.where`` in ``qwen3_vl_transformers._forward_generation``.

    The Float8 tensors are dequantized to BFloat16 *after* load by
    :func:`_dequantize_float8_to_bfloat16` (see that function's docstring), so the
    most economical default for FP8 repos is **bfloat16** — non-Float8 auxiliary
    tensors (LayerNorm scales, embeddings, etc.) load straight into BF16 to match
    ``generate_image``'s autocast and avoid the previous float32 footprint.

    Override: ``GLITCHFRAME_HIDREAM_NATIVE_WEIGHTS_DTYPE=float32|bfloat16``.
    """
    import torch  # type: ignore

    raw = os.environ.get("GLITCHFRAME_HIDREAM_NATIVE_WEIGHTS_DTYPE", "").strip().lower()
    if raw in ("float32", "fp32"):
        return torch.float32
    if raw in ("bfloat16", "bf16"):
        return torch.bfloat16
    if raw in ("auto", ""):
        pass
    else:
        _log(
            f"Ignoring unknown GLITCHFRAME_HIDREAM_NATIVE_WEIGHTS_DTYPE={raw!r}; "
            "using heuristic."
        )

    if _is_fp8_checkpoint_hint(model_path):
        _log(
            "FP8 / low-bit checkpoint detected — loading auxiliary tensors as "
            "bfloat16; Float8 weights will be dequantized to bfloat16 in-place "
            "(see GLITCHFRAME_HIDREAM_DEQUANT_FLOAT8 to disable)."
        )
        return torch.bfloat16
    return torch.bfloat16


def _insert_repo_path(repo: Path) -> None:
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _collect_float8_dtypes() -> set[Any]:
    """Return every Float8 ``torch.dtype`` exposed by the installed PyTorch.

    Builds matter: PyTorch 2.1 added ``float8_e4m3fn`` / ``float8_e5m2``;
    later releases added ``float8_e4m3fnuz`` / ``float8_e5m2fnuz``. This
    function returns whatever the current build advertises so the dequant
    helper does not need to be edited when PyTorch grows new variants.
    """
    import torch  # type: ignore

    out: set[Any] = set()
    for name in dir(torch):
        if not name.startswith("float8_"):
            continue
        attr = getattr(torch, name, None)
        if isinstance(attr, torch.dtype):
            out.add(attr)
    return out


def _audit_model_dtypes(model: Any, *, sample_names: int = 5, label: str = "audit") -> int:
    """Read-only diagnostic: log how parameters / buffers are split by dtype.

    Returns the total number of Float8 tensors (params + buffers) found, so
    callers can decide whether a follow-up dequant pass is needed.

    Set ``GLITCHFRAME_HIDREAM_WORKER_DIAGNOSE_DTYPES=0`` to skip the
    log-line emission (the count return value is still computed since it is
    cheap — one pass over named tensors, no GPU work, no allocations).
    """
    quiet = os.environ.get("GLITCHFRAME_HIDREAM_WORKER_DIAGNOSE_DTYPES", "").strip() == "0"
    try:
        param_counts: dict[str, int] = {}
        buffer_counts: dict[str, int] = {}
        float8_param_names: list[str] = []
        float8_buffer_names: list[str] = []
        float8_total = 0

        for name, p in model.named_parameters():
            key = str(p.dtype)
            param_counts[key] = param_counts.get(key, 0) + 1
            if "float8" in key.lower():
                float8_total += 1
                if len(float8_param_names) < sample_names:
                    float8_param_names.append(name)

        for name, b in model.named_buffers():
            key = str(b.dtype)
            buffer_counts[key] = buffer_counts.get(key, 0) + 1
            if "float8" in key.lower():
                float8_total += 1
                if len(float8_buffer_names) < sample_names:
                    float8_buffer_names.append(name)

        if quiet:
            return float8_total

        def _fmt(counts: dict[str, int]) -> str:
            if not counts:
                return "<empty>"
            return ", ".join(
                f"{dt}={n}" for dt, n in sorted(counts.items(), key=lambda x: -x[1])
            )

        _log(f"dtype {label}: parameters: {_fmt(param_counts)}")
        _log(f"dtype {label}: buffers:    {_fmt(buffer_counts)}")
        if float8_param_names:
            _log(
                f"dtype {label}: Float8 parameters detected (first "
                f"{len(float8_param_names)}): {float8_param_names}"
            )
        if float8_buffer_names:
            _log(
                f"dtype {label}: Float8 buffers detected (first "
                f"{len(float8_buffer_names)}): {float8_buffer_names}"
            )
        if float8_total == 0:
            _log(
                f"dtype {label}: no Float8 tensors — generate_image's bf16 "
                "autocast path is safe."
            )
        return float8_total
    except Exception as exc:  # noqa: BLE001
        _log(f"dtype {label} failed (non-fatal): {type(exc).__name__}: {exc}")
        return 0


def _dequantize_float8_to_bfloat16(model: Any) -> int:
    """Cast every Float8 parameter / buffer in ``model`` to BFloat16 in-place.

    Why this is needed
    ------------------
    Community FP8 checkpoints (e.g. ``drbaph/HiDream-O1-Image-FP8``) ship the
    Qwen3-VL backbone as ``Float8_e4m3fn`` safetensors. ``transformers`` keeps
    those tensors at their on-disk dtype regardless of ``torch_dtype=`` (which
    is only a *default for newly-created tensors*). HiDream's upstream
    ``generate_image`` then runs forward inside ``torch.autocast(dtype=bf16)``
    where ``Qwen3VLModel._forward_generation`` performs::

        inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

    ``inputs_embeds`` is the result of an ``Embedding`` lookup (Float8, because
    the embedding weight is Float8 and autocast does not touch ``Embedding``)
    while ``t_emb_expanded`` is BFloat16 (Linear inside the autocast). PyTorch
    refuses to promote Float8 with BFloat16 → ``RuntimeError: Promotion for
    Float8 Types is not supported``.

    Strategy
    --------
    Cast every Float8 tensor (params + buffers) to BFloat16 once, so the
    autocast path is consistent. Direct ``.to(bfloat16)`` reinterprets each
    Float8 value at full BF16 precision (no scale factors are applied because
    the upstream FP8 release stores plain Float8 weights, not torchao-style
    scaled ``Float8Tensor``s). The model effectively becomes a BF16 build with
    FP8-precision weights — same accuracy as drbaph's intended FP8 inference,
    but compatible with HiDream's BF16 forward path.

    Returns the number of tensors cast. Set
    ``GLITCHFRAME_HIDREAM_DEQUANT_FLOAT8=0`` to opt out (advanced — only useful
    when the env already provides true FP8 ops via ``torchao`` or a fork).
    """
    if os.environ.get("GLITCHFRAME_HIDREAM_DEQUANT_FLOAT8", "").strip() == "0":
        _log("Float8 dequant disabled by GLITCHFRAME_HIDREAM_DEQUANT_FLOAT8=0")
        return 0
    import torch  # type: ignore

    float8_dtypes = _collect_float8_dtypes()
    if not float8_dtypes:
        return 0
    cast = 0
    with torch.no_grad():
        for module in model.modules():
            for pname, p in list(module.named_parameters(recurse=False)):
                if p.dtype in float8_dtypes:
                    new_p = torch.nn.Parameter(
                        p.data.to(torch.bfloat16),
                        requires_grad=p.requires_grad,
                    )
                    setattr(module, pname, new_p)
                    cast += 1
            for bname, b in list(module.named_buffers(recurse=False)):
                if b.dtype in float8_dtypes:
                    persistent = bname not in module._non_persistent_buffers_set
                    module.register_buffer(
                        bname,
                        b.data.to(torch.bfloat16),
                        persistent=persistent,
                    )
                    cast += 1
    return cast


def _load_native_hidream(*, model_path: Path, model_type: str) -> _NativeHiDreamBundle:
    import torch  # type: ignore

    from transformers import AutoProcessor  # type: ignore

    from models.qwen3_vl_transformers import (  # type: ignore  # noqa: E402
        Qwen3VLForConditionalGeneration,
    )

    wdtype = _native_weights_torch_dtype(model_path)
    _log(
        f"loading HiDream native stack ({model_type}) from {model_path} "
        f"(from_pretrained torch_dtype={wdtype})"
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=wdtype,
        device_map="cuda",
    ).eval()
    pre_float8 = _audit_model_dtypes(model, label="audit (post-load)")
    if pre_float8 > 0:
        _log(
            f"dequantizing {pre_float8} Float8 tensors → bfloat16 so "
            "generate_image's BF16 autocast path is consistent (see "
            "_dequantize_float8_to_bfloat16)."
        )
        cast = _dequantize_float8_to_bfloat16(model)
        _log(f"dequant: cast {cast} Float8 tensor(s) to bfloat16")
        post_float8 = _audit_model_dtypes(model, label="audit (post-dequant)")
        if post_float8 > 0:
            _log(
                f"WARNING: {post_float8} Float8 tensor(s) remain after dequant — "
                "generate_image will likely raise Promotion for Float8 Types. "
                "File a bug if this happens with an unmodified worker."
            )
    tokenizer = _get_tokenizer(processor)
    _add_special_tokens(tokenizer)
    return _NativeHiDreamBundle(model=model, processor=processor, model_type=model_type)


def _load_diffusers_style_pipeline(
    *,
    model_path: Path,
    model_type: str,
    pipeline_import: str,
) -> Any:
    import torch  # type: ignore

    PipelineCls = _resolve_pipeline_class(pipeline_import)
    _log(
        f"loading HiDream diffusers-style class ({model_type}) from {model_path} "
        f"using {pipeline_import}"
    )
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
        raise RuntimeError(f"Failed to move HiDream pipeline to CUDA: {exc}") from exc

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


def _load_generation_runtime(
    *,
    repo: Path,
    model_path: Path,
    model_type: str,
    pipeline_import: str,
) -> tuple[str, Any]:
    """Return ``("native", bundle)`` or ``("diffusers", pipe)``."""
    _insert_repo_path(repo)

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

    spec = pipeline_import.strip()
    use_auto = spec.lower() in ("", "auto")

    if not use_auto:
        return (
            "diffusers",
            _load_diffusers_style_pipeline(
                model_path=model_path,
                model_type=model_type,
                pipeline_import=spec,
            ),
        )

    mp = importlib.import_module("models.pipeline")
    if hasattr(mp, "generate_image"):
        return ("native", _load_native_hidream(model_path=model_path, model_type=model_type))
    if hasattr(mp, "HiDreamImagePipeline"):
        _log("auto: using legacy HiDreamImagePipeline.from_pretrained")
        return (
            "diffusers",
            _load_diffusers_style_pipeline(
                model_path=model_path,
                model_type=model_type,
                pipeline_import="models.pipeline:HiDreamImagePipeline",
            ),
        )

    raise RuntimeError(
        "HiDream models.pipeline exposes neither generate_image nor "
        "HiDreamImagePipeline — update HiDream-ai/HiDream-O1-Image or set "
        "--pipeline-import to your fork's pipeline class."
    )


def _generate_one_diffusers(
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
) -> None:
    """Run a diffusers-style ``__call__`` pipeline."""
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    generator: Any = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    def _step_cb(_pipe: Any, step_idx: int, _t: Any, kwargs: dict) -> dict:
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


def _generate_one_native(
    bundle: _NativeHiDreamBundle,
    *,
    index: int,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
) -> None:
    """Call :func:`models.pipeline.generate_image` (current upstream HiDream)."""
    from models.pipeline import DEFAULT_TIMESTEPS, generate_image  # type: ignore  # noqa: E402

    seed_i = int(seed) if seed is not None else 32

    if bundle.model_type == "full":
        shift = 3.0
        timesteps_list = None
        scheduler_name = "default"
        extra: dict[str, Any] = {}
        gscale = float(guidance_scale)
    else:
        shift = 1.0
        timesteps_list = DEFAULT_TIMESTEPS
        scheduler_name = "flash"
        extra = {
            "noise_scale_start": 7.5,
            "noise_scale_end": 7.5,
            "noise_clip_std": 2.5,
        }
        gscale = 0.0

    total_steps = int(steps)

    def _cb(step_idx: int, n_steps: int, _decode: Any) -> None:
        _emit(
            {
                "event": "step",
                "index": int(index),
                "step": int(step_idx) + 1,
                "steps_total": int(n_steps),
            }
        )

    img = generate_image(
        model=bundle.model,
        processor=bundle.processor,
        prompt=prompt,
        ref_image_paths=None,
        height=int(height),
        width=int(width),
        num_inference_steps=total_steps,
        guidance_scale=gscale,
        shift=shift,
        timesteps_list=timesteps_list,
        scheduler_name=scheduler_name,
        seed=seed_i,
        callback=_cb,
        **extra,
    )

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
        default="auto",
        help="``auto`` (default), or ``module:Class`` for a diffusers-style pipeline",
    )
    args = parser.parse_args(argv)

    if args.steps is None:
        args.steps = 28 if args.model_type == "dev" else 50
    if args.guidance_scale is None:
        args.guidance_scale = 0.0 if args.model_type == "dev" else 5.0

    try:
        kind, loaded = _load_generation_runtime(
            repo=args.repo,
            model_path=args.model_path,
            model_type=args.model_type,
            pipeline_import=str(args.pipeline_import),
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
            w = int(job.get("width", args.width))
            h = int(job.get("height", args.height))
            st = int(job.get("steps", args.steps))
            gs = float(job.get("guidance_scale", args.guidance_scale))
            if kind == "native":
                _generate_one_native(
                    loaded,
                    index=index,
                    prompt=prompt,
                    output_path=output_path,
                    width=w,
                    height=h,
                    steps=st,
                    guidance_scale=gs,
                    seed=seed,
                )
            else:
                _generate_one_diffusers(
                    loaded,
                    index=index,
                    prompt=prompt,
                    output_path=output_path,
                    width=w,
                    height=h,
                    steps=st,
                    guidance_scale=gs,
                    seed=seed,
                )
        except Exception as exc:  # noqa: BLE001
            _log(traceback.format_exc())
            _emit({"event": "error", "index": index, "message": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
