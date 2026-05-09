"""
Effects timeline: typed clips (manual or baked) persisted under the song cache.

Each :class:`EffectClip` is a time range over one :class:`EffectKind`, with
per-kind settings validated against a strict allowlist (unknown keys are
rejected with :class:`ValueError`). Load/save uses the same atomic JSON
pattern as the lyrics editor (``.json.tmp`` + :func:`os.replace`).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

EFFECTS_TIMELINE_JSON = "effects_timeline.json"
SCHEMA_VERSION = 1


class EffectKind(str, Enum):
    """V1 effect clip kinds (values match JSON / API)."""

    BEAM = "BEAM"
    LOGO_GLITCH = "LOGO_GLITCH"
    SCREEN_SHAKE = "SCREEN_SHAKE"
    COLOR_INVERT = "COLOR_INVERT"
    CHROMATIC_ABERRATION = "CHROMATIC_ABERRATION"
    SCANLINE_TEAR = "SCANLINE_TEAR"
    FADE = "FADE"
    PIXEL_SMEAR = "PIXEL_SMEAR"
    BLOCK_GLITCH = "BLOCK_GLITCH"


# Kinds that older saves may still reference. They are silently dropped on
# load (rather than raising) so an existing ``effects_timeline.json`` that
# survived from a previous schema version still loads cleanly. New kinds may
# only be added here when retired — never used to introduce alternate spellings.
_DEPRECATED_KIND_NAMES: frozenset[str] = frozenset({"ZOOM_PUNCH"})


# Per-kind allowlists for the ``settings`` dict (unknown keys → ValueError).
# Values must be ``str | int | float | bool | None`` (JSON-scalar, no nesting).
EFFECT_SETTINGS_KEYS: dict[EffectKind, frozenset[str]] = {
    EffectKind.BEAM: frozenset({"color_hex", "strength", "thickness_px"}),
    EffectKind.LOGO_GLITCH: frozenset(
        {
            "strength",
            "jitter",
            "offset_px",
        }
    ),
    EffectKind.SCREEN_SHAKE: frozenset({"amplitude_px", "frequency_hz"}),
    EffectKind.COLOR_INVERT: frozenset({"mix", "intensity"}),
    EffectKind.CHROMATIC_ABERRATION: frozenset(
        {"shift_px", "jitter", "direction_deg"}
    ),
    EffectKind.SCANLINE_TEAR: frozenset(
        {"intensity", "band_count", "band_height_px", "wrap_mode"}
    ),
    # FADE is a single timeline lane that fades **to black**. ``direction_mode``
    # picks "in" (start black, reveal) or "out" (start clear, fade to black).
    # ``peak_alpha`` caps the maximum darkness (1.0 = fully black at the
    # extreme of the ramp). ``ease_mode`` picks "smoothstep" (Hermite, default)
    # or "linear". The clip's ``duration_s`` *is* the ramp length.
    EffectKind.FADE: frozenset({"direction_mode", "peak_alpha", "ease_mode"}),
    # PIXEL_SMEAR — horizontal pixel-streak datamosh. ``intensity`` controls
    # how visible the streaks are, ``density`` how many rows are smeared,
    # ``streak_length_frac`` how far each streak extends across its row.
    EffectKind.PIXEL_SMEAR: frozenset(
        {"intensity", "density", "streak_length_frac"}
    ),
    # BLOCK_GLITCH — JPEG / macroblock displacement. ``intensity`` is the
    # fraction of blocks to shift, ``block_size_px`` the block edge length,
    # ``displace_frac`` the per-axis offset cap as a fraction of the block.
    EffectKind.BLOCK_GLITCH: frozenset(
        {"intensity", "block_size_px", "displace_frac"}
    ),
}


def _is_json_scalar(x: object) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


@dataclass
class EffectClip:
    id: str
    kind: EffectKind
    t_start: float
    duration_s: float
    settings: dict[str, Any] = field(default_factory=dict)
    auto_source: bool = False

    def __post_init__(self) -> None:
        validate_effect_clip(self)


def _default_auto_enabled() -> dict[EffectKind, bool]:
    return {k: True for k in EffectKind}


def _normalize_kb_automation_in_place(
    raw: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not raw:
        return []
    items = sorted((float(t), float(v)) for t, v in raw)
    out: list[tuple[float, float]] = []
    for tt, vv in items:
        if not math.isfinite(tt) or not math.isfinite(vv):
            raise ValueError("ken_burns_rms_automation has non-finite t or v")
        vv = max(0.0, min(2.0, vv))
        if out and abs(out[-1][0] - tt) < 1e-9:
            out[-1] = (tt, vv)
        else:
            out.append((tt, vv))
    return out


@dataclass
class EffectsTimeline:
    clips: list[EffectClip] = field(default_factory=list)
    auto_enabled: dict[EffectKind, bool] = field(
        default_factory=_default_auto_enabled
    )
    auto_reactivity_master: float = 1.0
    # Piecewise-linear automation in [0, 2] (0–200% RMS drive) for SDXL
    # Ken Burns only. Scales analyser RMS before zoom/pan/tilt. Empty → 1.0.
    ken_burns_rms_automation: list[tuple[float, float]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        kb_in = self.ken_burns_rms_automation
        pairs: list[tuple[float, float]] = []
        for i, item in enumerate(kb_in):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((float(item[0]), float(item[1])))
            else:
                raise TypeError(
                    f"ken_burns_rms_automation[{i}] must be a (t, v) pair"
                )
        object.__setattr__(
            self,
            "ken_burns_rms_automation",
            _normalize_kb_automation_in_place(pairs),
        )
        validate_effects_timeline(self)


def validate_settings_for_kind(kind: EffectKind, settings: Mapping[str, Any]) -> None:
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    allow = EFFECT_SETTINGS_KEYS[kind]
    unknown = [k for k in settings if k not in allow]
    if unknown:
        raise ValueError(
            f"Unknown settings keys for {kind.name}: {sorted(unknown)}"
        )
    for key, value in settings.items():
        if not _is_json_scalar(value):
            raise TypeError(
                f"settings[{key!r}] must be JSON-scalar (str, int, float, bool, "
                f"None), got {type(value).__name__}"
            )


def validate_effect_clip(clip: EffectClip) -> None:
    if not clip.id or not isinstance(clip.id, str):
        raise ValueError("EffectClip.id must be a non-empty string")
    if not isinstance(clip.kind, EffectKind):
        raise TypeError("EffectClip.kind must be an EffectKind")
    if not isinstance(clip.t_start, (int, float)) or not isinstance(
        clip.duration_s, (int, float)
    ):
        raise TypeError("t_start and duration_s must be numbers")
    ts, dur = float(clip.t_start), float(clip.duration_s)
    if not (math.isfinite(ts) and math.isfinite(dur)):
        raise ValueError("EffectClip t_start and duration_s must be finite")
    if dur <= 0:
        raise ValueError("EffectClip.duration_s must be > 0")
    validate_settings_for_kind(clip.kind, clip.settings)
    if not isinstance(clip.auto_source, bool):
        raise TypeError("auto_source must be bool")


def validate_effects_timeline(t: EffectsTimeline) -> None:
    if not isinstance(t.clips, list):
        raise TypeError("EffectsTimeline.clips must be a list")
    for c in t.clips:
        if not isinstance(c, EffectClip):
            raise TypeError("Every clip must be an EffectClip")
    if not isinstance(t.ken_burns_rms_automation, list):
        raise TypeError("ken_burns_rms_automation must be a list")
    for i, raw in enumerate(t.ken_burns_rms_automation):
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError(
                f"ken_burns_rms_automation[{i}] must be a (t, v) pair of numbers"
            )
    if not isinstance(t.auto_reactivity_master, (int, float)):
        raise TypeError("auto_reactivity_master must be a number")
    m = float(t.auto_reactivity_master)
    if m != m or m == float("inf") or m == float("-inf"):
        raise ValueError("auto_reactivity_master must be finite")
    if m < 0.0 or m > 2.0:
        raise ValueError("auto_reactivity_master must be in [0.0, 2.0] (0–200%)")
    if not isinstance(t.auto_enabled, dict):
        raise TypeError("auto_enabled must be a dict")
    for k, v in t.auto_enabled.items():
        if not isinstance(k, EffectKind):
            raise TypeError("auto_enabled keys must be EffectKind")
        if not isinstance(v, bool):
            raise TypeError("auto_enabled values must be bool")
    for k in EffectKind:
        if k not in t.auto_enabled:
            raise ValueError(f"auto_enabled missing key {k.name!r}")


def interp_ken_burns_rms_automation(
    points: Sequence[tuple[float, float]], t: float
) -> float:
    """
    Linear interpolation of automation values in ``[0, 2]``.

    * No points → ``1.0`` (100%, neutral envelope).
    * One point → its ``v`` (clamped).
    * Before the first / after the last knot, the value is held (DAW-style).
    """
    if not points:
        return 1.0
    merged: list[tuple[float, float]] = []
    for tt, vv in sorted((float(p[0]), float(p[1])) for p in points):
        vv = max(0.0, min(2.0, float(vv)))
        if merged and abs(merged[-1][0] - tt) < 1e-12:
            merged[-1] = (tt, vv)
        else:
            merged.append((tt, vv))
    if len(merged) == 1:
        return max(0.0, min(2.0, merged[0][1]))
    t0, v0 = merged[0]
    if t <= t0:
        return max(0.0, min(2.0, v0))
    t1, v1 = merged[-1]
    if t >= t1:
        return max(0.0, min(2.0, v1))
    for i in range(len(merged) - 1):
        ta, va = merged[i]
        tb, vb = merged[i + 1]
        if ta <= t <= tb:
            span = tb - ta
            if span < 1e-12:
                return max(0.0, min(2.0, vb))
            u = (t - ta) / span
            return max(0.0, min(2.0, va + (vb - va) * u))
    return max(0.0, min(2.0, v1))


def _kb_auto_to_json(points: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"t": float(tt), "v": float(vv)} for tt, vv in points]


def _kb_auto_from_json(data: object) -> list[tuple[float, float]]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError("ken_burns_rms_automation must be an array")
    out: list[tuple[float, float]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"ken_burns_rms_automation[{i}] must be an object with t, v"
            )
        out.append((float(item["t"]), float(item["v"])))
    return _normalize_kb_automation_in_place(out)


def _auto_enabled_to_json(d: dict[EffectKind, bool]) -> dict[str, bool]:
    return {k.name: d[k] for k in sorted(d, key=lambda x: x.name)}


def _auto_enabled_from_json(
    data: object,
) -> dict[EffectKind, bool]:
    if not isinstance(data, dict):
        raise ValueError("auto_enabled must be a JSON object")
    out: dict[EffectKind, bool] = {}
    for name, val in data.items():
        if isinstance(name, str) and name in _DEPRECATED_KIND_NAMES:
            continue
        if name not in EffectKind.__members__:
            raise ValueError(f"Unknown effect kind in auto_enabled: {name!r}")
        if not isinstance(val, bool):
            raise TypeError(
                f"auto_enabled[{name!r}] must be bool, not {type(val).__name__}"
            )
        out[EffectKind(name)] = val
    # Old saves predate kinds added later (e.g. FADE / PIXEL_SMEAR /
    # BLOCK_GLITCH). Default any missing flag to ``True`` rather than raising
    # so legacy ``effects_timeline.json`` files still load cleanly.
    for k in EffectKind:
        if k not in out:
            out[k] = True
    return out


def _clip_to_json(c: EffectClip) -> dict[str, Any]:
    return {
        "id": c.id,
        "kind": c.kind.name,
        "t_start": c.t_start,
        "duration_s": c.duration_s,
        "settings": dict(c.settings),
        "auto_source": c.auto_source,
    }


class _DeprecatedClip:
    """Sentinel returned by :func:`_clip_from_json` for retired kinds.

    Lets the loader drop legacy clips (e.g. pre-FADE ``ZOOM_PUNCH`` rows)
    silently instead of raising, while still reporting genuinely malformed
    JSON loudly.
    """

    __slots__ = ("kind_name",)

    def __init__(self, kind_name: str) -> None:
        self.kind_name = kind_name


def _clip_from_json(obj: object) -> EffectClip | _DeprecatedClip:
    if not isinstance(obj, dict):
        raise ValueError("Each clip must be a JSON object")
    kind_s = obj.get("kind")
    if isinstance(kind_s, str) and kind_s in _DEPRECATED_KIND_NAMES:
        return _DeprecatedClip(kind_s)
    if kind_s not in EffectKind.__members__:
        raise ValueError(f"Invalid or missing effect kind: {kind_s!r}")
    kind = EffectKind(kind_s)
    raw_settings = obj.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise TypeError("clip.settings must be a JSON object")
    settings: dict[str, Any] = {str(k): v for k, v in raw_settings.items()}
    return EffectClip(
        id=str(obj.get("id") or "").strip() or _raise_clip_id(),
        kind=kind,
        t_start=float(obj["t_start"]),
        duration_s=float(obj["duration_s"]),
        settings=settings,
        auto_source=bool(obj.get("auto_source", False)),
    )


def _raise_clip_id() -> str:
    raise ValueError("EffectClip.id is required and must be non-empty")


def _timeline_to_dict(t: EffectsTimeline) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "auto_reactivity_master": t.auto_reactivity_master,
        "auto_enabled": _auto_enabled_to_json(t.auto_enabled),
        "clips": [_clip_to_json(c) for c in t.clips],
        "ken_burns_rms_automation": _kb_auto_to_json(t.ken_burns_rms_automation),
    }


def _timeline_from_dict(data: object) -> EffectsTimeline:
    if not isinstance(data, dict):
        raise ValueError("Effects timeline must be a JSON object")
    ver = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(ver, int) or ver != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported effects_timeline schema_version: {ver!r} (expected {SCHEMA_VERSION})"
        )
    clips_in = data.get("clips", [])
    if not isinstance(clips_in, list):
        raise TypeError("'clips' must be an array")
    clips: list[EffectClip] = []
    for c in clips_in:
        parsed = _clip_from_json(c)
        if isinstance(parsed, _DeprecatedClip):
            continue
        clips.append(parsed)
    auto = _auto_enabled_from_json(data.get("auto_enabled", {}))
    master = data.get("auto_reactivity_master", 1.0)
    if not isinstance(master, (int, float)):
        raise TypeError("auto_reactivity_master must be a number")
    kb_auto = _kb_auto_from_json(data.get("ken_burns_rms_automation", []))
    return EffectsTimeline(
        clips=clips,
        auto_enabled=auto,
        auto_reactivity_master=float(master),
        ken_burns_rms_automation=kb_auto,
    )


def load(cache_dir: str | Path) -> EffectsTimeline:
    """
    Load ``effects_timeline.json`` from ``cache_dir`` or return a default empty
    timeline if the file is absent. Always validates the shape.
    """
    path = Path(cache_dir) / EFFECTS_TIMELINE_JSON
    if not path.is_file():
        return EffectsTimeline()
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{EFFECTS_TIMELINE_JSON} is not valid JSON: {exc}"
            ) from exc
    return _timeline_from_dict(data)


def save(cache_dir: str | Path, timeline: EffectsTimeline) -> Path:
    """
    Write ``EffectsTimeline`` to ``cache_dir/effects_timeline.json`` using
    ``<name>.json.tmp`` + :func:`os.replace` (atomic on POSIX; replace-in-place
    on Windows). Validates before writing.
    """
    validate_effects_timeline(timeline)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = root / EFFECTS_TIMELINE_JSON
    tmp = out.with_suffix(out.suffix + ".tmp")
    payload = _timeline_to_dict(timeline)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, out)
    return out
