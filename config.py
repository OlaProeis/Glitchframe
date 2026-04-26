"""Defaults, paths, and preset registry for Glitchframe."""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pipeline.builtin_shaders import BUILTIN_SHADERS

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_PRESET_STR_FIELDS = frozenset({"prompt", "shader", "typo_style"})

PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(primary: str, legacy: str, default: Path) -> Path:
    """Resolve path from env; *primary* wins, then *legacy* (pre-rename), then *default*."""
    for key in (primary, legacy):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw)
    return default


# Optional overrides via environment (see .env.example). Legacy `MUSICVIDS_*` still works.
CACHE_DIR = _env_path("GLITCHFRAME_CACHE_DIR", "MUSICVIDS_CACHE_DIR", PROJECT_ROOT / "cache")
OUTPUTS_DIR = _env_path("GLITCHFRAME_OUTPUTS_DIR", "MUSICVIDS_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
PRESETS_DIR = _env_path("GLITCHFRAME_PRESETS_DIR", "MUSICVIDS_PRESETS_DIR", PROJECT_ROOT / "presets")
ASSETS_DIR = _env_path("GLITCHFRAME_ASSETS_DIR", "MUSICVIDS_ASSETS_DIR", PROJECT_ROOT / "assets")
SHADERS_DIR = _env_path("GLITCHFRAME_SHADERS_DIR", "MUSICVIDS_SHADERS_DIR", ASSETS_DIR / "shaders")
FONTS_DIR = _env_path("GLITCHFRAME_FONTS_DIR", "MUSICVIDS_FONTS_DIR", ASSETS_DIR / "fonts")
# Bundled UI fonts (SIL Open Font License 1.1); commit these files so renders
# don't fall back to Arial when ``font_path`` is unset. See ``assets/fonts/``.
DEFAULT_UI_FONT = FONTS_DIR / "Inter.ttf"
# Display face for the burned-in title + thumbnail line. Space Grotesk is a
# geometric neo-grotesque designed for display sizes, which reads more
# "modern music video" than Inter at 1080p+. Inter-SemiBold is kept as a
# fallback so renders never crash on hosts that haven't pulled the new font.
DEFAULT_TITLE_FONT = FONTS_DIR / "SpaceGrotesk-SemiBold.ttf"
FALLBACK_TITLE_FONT = FONTS_DIR / "Inter-SemiBold.ttf"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"


def default_ui_font_path() -> Path | None:
    """Return bundled Inter Regular if present, else ``None``."""
    p = DEFAULT_UI_FONT
    return p if p.is_file() else None


def default_title_font_path() -> Path | None:
    """Return the preferred display font for title / thumbnail overlays.

    Preference order:

    1. ``SpaceGrotesk-SemiBold.ttf`` — the bundled display face.
    2. ``Inter-SemiBold.ttf`` — carried over from the previous default so
       repos that haven't pulled the new asset keep rendering.
    3. The body font (Inter Regular) as a last resort.
    4. ``None`` — Skia falls back to a system typeface.
    """
    for candidate in (DEFAULT_TITLE_FONT, FALLBACK_TITLE_FONT):
        if candidate.is_file():
            return candidate
    return default_ui_font_path()

# Hugging Face / model caches (optional; libraries also respect HF_HOME, TORCH_HOME)
MODEL_CACHE_DIR = _env_path("GLITCHFRAME_MODEL_CACHE", "MUSICVIDS_MODEL_CACHE", PROJECT_ROOT / ".cache" / "models")


def ensure_runtime_dirs() -> None:
    """Create standard working directories if missing."""
    for d in (CACHE_DIR, OUTPUTS_DIR, PRESETS_DIR, SHADERS_DIR, FONTS_DIR, MODEL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def song_cache_dir(song_hash: str) -> Path:
    """Per-song workspace under ``CACHE_DIR`` (see ``pipeline/audio_ingest``)."""
    return CACHE_DIR / song_hash


def new_run_id(*, song_hash: str | None = None) -> str:
    """
    Unique folder name under ``OUTPUTS_DIR`` (timestamp + short id; optional
    song hash prefix). Does not create the directory.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if song_hash and str(song_hash).strip():
        h = str(song_hash).strip()[:8]
        return f"{ts}_{h}"
    return f"{ts}_{secrets.token_hex(4)}"


def _validate_preset_dict(stem: str, raw: Any) -> dict[str, Any]:
    """Ensure preset YAML matches the strict schema; raise ``ValueError`` otherwise."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"Preset {stem!r}: root must be a mapping, got {type(raw).__name__}"
        )
    missing = [
        k
        for k in ("prompt", "shader", "typo_style", "colors")
        if k not in raw or raw[k] is None
    ]
    if missing:
        raise ValueError(f"Preset {stem!r}: missing required keys: {', '.join(missing)}")

    out: dict[str, Any] = {}
    for key in _PRESET_STR_FIELDS:
        val = raw[key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"Preset {stem!r}: {key!r} must be a non-empty string, got {val!r}"
            )
        out[key] = val.strip()

    colors_raw = raw["colors"]
    if not isinstance(colors_raw, list) or not colors_raw:
        raise ValueError(
            f"Preset {stem!r}: 'colors' must be a non-empty list of hex strings"
        )
    colors: list[str] = []
    for i, c in enumerate(colors_raw):
        if not isinstance(c, str) or not _HEX_COLOR.fullmatch(c.strip()):
            raise ValueError(
                f"Preset {stem!r}: colors[{i}] must be a #RRGGBB string, got {c!r}"
            )
        colors.append(c.strip().upper())
    out["colors"] = colors

    if out["shader"] not in BUILTIN_SHADERS:
        raise ValueError(
            f"Preset {stem!r}: unknown shader {out['shader']!r}; "
            f"expected one of {list(BUILTIN_SHADERS)}"
        )

    return out


def load_preset_registry(presets_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load all ``*.yaml`` presets from ``presets_dir`` into a dict keyed by stem name
    (e.g. ``neon-synthwave.yaml`` -> ``neon-synthwave``).

    Each entry is validated (``prompt``, ``shader``, ``typo_style``, ``colors``);
    invalid files raise ``ValueError``.
    """
    root = Path(presets_dir) if presets_dir is not None else PRESETS_DIR
    if not root.is_dir():
        return {}

    registry: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.yaml")):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        registry[path.stem] = _validate_preset_dict(path.stem, data)
    return registry


def get_preset_ids() -> list[str]:
    """Sorted list of available preset names (YAML stems)."""
    return sorted(load_preset_registry().keys())


def get_preset(name: str, presets_dir: Path | None = None) -> dict[str, Any]:
    """Return a copy of the preset mapping for ``name`` or raise ``KeyError``."""
    reg = load_preset_registry(presets_dir)
    if name not in reg:
        raise KeyError(f"Unknown preset: {name!r}")
    return dict(reg[name])


if __name__ == "__main__":
    ensure_runtime_dirs()
    reg = load_preset_registry()
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CACHE_DIR:", CACHE_DIR)
    print("OUTPUTS_DIR:", OUTPUTS_DIR)
    print("PRESETS_DIR:", PRESETS_DIR)
    print("Presets loaded:", len(reg), list(reg.keys()))
