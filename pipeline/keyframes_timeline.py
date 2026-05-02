"""
User-editable SDXL / upload keyframe anchors: timing, prompts, and sources.

State is stored in ``cache/<song_hash>/keyframes_timeline.json``. When present,
:class:`~pipeline.background_stills.BackgroundStills` builds its keyframe plan
from this file instead of auto spacing. Saving edits updates ``manifest.json``
and may reorder ``keyframe_*.png``; RIFE morph caches are cleared so the next
render rebuilds flow interpolations.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.background_stills import (
    MANIFEST_SCHEMA_VERSION,
    BackgroundManifest,
    KeyframePlan,
    _atomic_replace_path,
    _atomic_write_json,
    _background_dir,
    _duration_from_analysis,
    _keyframe_path,
    _load_manifest,
    _manifest_path,
    _rife_manifest_path,
    _rife_timeline_dir,
    _segment_index_for_time,
    _segments_from_analysis,
    prompt_hash,
)

LOGGER = logging.getLogger(__name__)

KEYFRAMES_TIMELINE_FILENAME = "keyframes_timeline.json"
UPLOAD_STAGING_DIRNAME = "upload_staging"
KEYFRAMES_TIMELINE_SCHEMA_VERSION = 1

MIN_ANCHOR_GAP_S = 0.05
# ``analysis.duration_sec`` vs JS ``state.duration`` vs ``t_start`` can disagree
# at the last keyframe by ~1 ULP; treat tiny overflow as end-of-track.
_DURATION_SLACK_ABS_S = 1e-5
_DURATION_SLACK_REL = 1e-12


def _duration_upper_bound(duration_sec: float) -> float:
    d = float(duration_sec)
    return d + max(_DURATION_SLACK_ABS_S, abs(d) * _DURATION_SLACK_REL)


def _upload_staging_entry_ids(
    cache_dir: Path, entries: Sequence[KeyframeTimelineEntry]
) -> set[str]:
    """Entry ids that have ``upload_staging/<id>.png`` (pending crop commit)."""
    ids = {e.id for e in entries}
    staging = _staging_dir(cache_dir)
    if not staging.is_dir() or not ids:
        return set()
    found: set[str] = set()
    for p in staging.glob("*.png"):
        if p.stem in ids:
            found.add(p.stem)
    return found


@dataclass
class KeyframeTimelineEntry:
    id: str
    t_sec: float
    prompt: str
    source: str  # "sdxl" | "upload"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "t_sec": float(self.t_sec),
            "prompt": str(self.prompt),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> KeyframeTimelineEntry:
        src = str(raw.get("source", "sdxl")).strip().lower()
        if src not in ("sdxl", "upload"):
            src = "sdxl"
        return cls(
            id=str(raw["id"]),
            t_sec=float(raw["t_sec"]),
            prompt=str(raw.get("prompt", "")),
            source=src,
        )


@dataclass
class KeyframesTimeline:
    schema_version: int
    manually_edited: bool
    entries: tuple[KeyframeTimelineEntry, ...]
    target_width: int
    target_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "manually_edited": bool(self.manually_edited),
            "entries": [e.to_dict() for e in self.entries],
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KeyframesTimeline":
        ents_raw = raw.get("entries")
        if not isinstance(ents_raw, list):
            raise ValueError("keyframes timeline: entries must be a list")
        entries = tuple(KeyframeTimelineEntry.from_dict(x) for x in ents_raw)
        tw = int(raw.get("target_width", 1920))
        th = int(raw.get("target_height", 1080))
        return cls(
            schema_version=int(raw.get("schema_version", KEYFRAMES_TIMELINE_SCHEMA_VERSION)),
            manually_edited=bool(raw.get("manually_edited", True)),
            entries=entries,
            target_width=tw,
            target_height=th,
        )


def _timeline_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / KEYFRAMES_TIMELINE_FILENAME


def _staging_dir(cache_dir: Path) -> Path:
    return _background_dir(cache_dir) / UPLOAD_STAGING_DIRNAME


def load_keyframes_timeline(cache_dir: Path | str) -> KeyframesTimeline | None:
    p = _timeline_path(Path(cache_dir))
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must be a JSON object")
    return KeyframesTimeline.from_dict(raw)


def save_keyframes_timeline(cache_dir: Path | str, timeline: KeyframesTimeline) -> Path:
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")
    p = _timeline_path(cache)
    _atomic_write_json(timeline.to_dict(), p)
    return p


def clear_rife_morph_cache(cache_dir: Path | str) -> None:
    """Remove RIFE timeline PNGs + manifest so morph rebuilds on next ensure."""
    cache = Path(cache_dir)
    rm = _rife_manifest_path(cache)
    if rm.is_file():
        try:
            rm.unlink()
        except OSError as exc:
            LOGGER.debug("Could not remove RIFE manifest: %s", exc)
    td = _rife_timeline_dir(cache)
    if td.is_dir():
        for child in td.iterdir():
            try:
                if child.is_file():
                    child.unlink()
            except OSError as exc:
                LOGGER.debug("Could not remove %s: %s", child, exc)


def validate_timeline_entries(
    entries: Sequence[KeyframeTimelineEntry],
    *,
    duration_sec: float,
) -> list[KeyframeTimelineEntry]:
    if not entries:
        raise ValueError("At least one keyframe entry is required")
    if duration_sec <= 0.0:
        raise ValueError("Song duration must be positive")
    d = float(duration_sec)
    upper = _duration_upper_bound(d)
    ordered = sorted(entries, key=lambda e: float(e.t_sec))
    normalized: list[KeyframeTimelineEntry] = []
    for e in ordered:
        t = float(e.t_sec)
        if t < 0.0:
            raise ValueError(f"Keyframe time {t} must be >= 0")
        if t > upper:
            raise ValueError(f"Keyframe time {t} out of range [0, {d}]")
        tc = t if t <= d else d
        if tc != t:
            LOGGER.debug("Clamped keyframe time %s to song duration %s", t, d)
        normalized.append(
            KeyframeTimelineEntry(
                id=e.id,
                t_sec=tc,
                prompt=e.prompt,
                source=e.source,
            )
        )
    times = [float(e.t_sec) for e in normalized]
    for a, b in zip(times, times[1:], strict=False):
        if b - a < MIN_ANCHOR_GAP_S:
            raise ValueError(
                f"Keyframes must be at least {MIN_ANCHOR_GAP_S}s apart "
                f"(got {a} and {b})"
            )
    return normalized


def entries_to_keyframe_plans(
    entries: Sequence[KeyframeTimelineEntry],
    analysis: Mapping[str, Any],
) -> list[KeyframePlan]:
    """Build :class:`KeyframePlan` rows matching sorted ``entries`` order (file indices)."""
    duration = _duration_from_analysis(analysis)
    segments = _segments_from_analysis(analysis, duration)
    ordered = sorted(entries, key=lambda e: float(e.t_sec))
    n = len(ordered)
    plans: list[KeyframePlan] = []
    for i, e in enumerate(ordered):
        t = float(e.t_sec)
        seg_idx = _segment_index_for_time(segments, t)
        seg_label = int(segments[seg_idx].get("label", seg_idx))
        plans.append(
            KeyframePlan(
                index=i,
                t_sec=t,
                segment_index=int(seg_idx),
                segment_label=seg_label,
                prompt=str(e.prompt).strip(),
            )
        )
    if n == 0:
        raise ValueError("No keyframe entries")
    return plans


def plans_to_background_manifest(
    plans: Sequence[KeyframePlan],
    *,
    preset_id: str,
    preset_prompt: str,
    analysis: Mapping[str, Any],
    model_id: str,
    gen_width: int,
    gen_height: int,
) -> BackgroundManifest:
    duration = _duration_from_analysis(analysis)
    segments = _segments_from_analysis(analysis, duration)
    prompts = tuple(p.prompt for p in plans)
    ph = prompt_hash(
        preset_prompt=preset_prompt,
        prompts=prompts,
        model_id=model_id,
        width=int(gen_width),
        height=int(gen_height),
    )
    return BackgroundManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        preset_id=str(preset_id),
        prompt_hash=ph,
        section_count=len(segments),
        num_keyframes=len(plans),
        duration_sec=float(duration),
        model_id=str(model_id),
        width=int(gen_width),
        height=int(gen_height),
        keyframe_times=tuple(float(p.t_sec) for p in plans),
        prompts=prompts,
    )


def timeline_identity(entries: Sequence[KeyframeTimelineEntry]) -> tuple[str, ...]:
    """Stable id ordering (sorted by time) for permute vs manifest-only update."""
    ordered = sorted(entries, key=lambda e: float(e.t_sec))
    return tuple(e.id for e in ordered)


def _collect_id_sources(
    cache_dir: Path,
    previous_ids_ordered: tuple[str, ...] | None,
) -> dict[str, Path]:
    """Map entry id to an existing PNG path (disk keyframe index or staging)."""
    out: dict[str, Path] = {}
    bg = _background_dir(cache_dir)
    staging = _staging_dir(cache_dir)
    if previous_ids_ordered:
        for i, eid in enumerate(previous_ids_ordered):
            p = _keyframe_path(cache_dir, i)
            if p.is_file():
                out[eid] = p
    else:
        man = _load_manifest(cache_dir)
        if man:
            for i in range(man.num_keyframes):
                p = _keyframe_path(cache_dir, i)
                if p.is_file():
                    out[f"kf-{i}"] = p
    if staging.is_dir():
        for sp in staging.glob("*.png"):
            eid = sp.stem
            out[eid] = sp
    return out


def rematerialize_keyframe_pngs(
    cache_dir: Path,
    sorted_entries: Sequence[KeyframeTimelineEntry],
    *,
    previous_ids_ordered: tuple[str, ...] | None,
) -> None:
    """
    Rewrite ``keyframe_*.png`` in **sorted-by-time** order to match ``sorted_entries``.

    Each entry id must resolve to an existing keyframe PNG from the previous
    order or to ``upload_staging/<id>.png``.
    """
    bg_dir = _background_dir(cache_dir)
    bg_dir.mkdir(parents=True, exist_ok=True)
    id_src = _collect_id_sources(cache_dir, previous_ids_ordered)
    n = len(sorted_entries)
    tmp_files: list[Path] = []
    for i in range(n):
        tmp_files.append(bg_dir / f".kf_reorder_{i:04d}.png.tmp")

    for i, ent in enumerate(sorted_entries):
        src = id_src.get(ent.id)
        if src is None or not src.is_file():
            raise FileNotFoundError(
                f"No image for keyframe id {ent.id!r} — generate SDXL or apply an upload crop."
            )
        shutil.copy(src, tmp_files[i])

    for i in range(n):
        dst = _keyframe_path(cache_dir, i)
        _atomic_replace_path(tmp_files[i], dst)

    # Drop consumed staging files for ids we just committed
    staging = _staging_dir(cache_dir)
    for ent in sorted_entries:
        sp = staging / f"{ent.id}.png"
        if sp.is_file():
            try:
                sp.unlink()
            except OSError:
                pass


def persist_timeline_and_manifest(
    cache_dir: Path,
    timeline: KeyframesTimeline,
    *,
    preset_id: str,
    preset_prompt: str,
    model_id: str,
    gen_width: int,
    gen_height: int,
    analysis: Mapping[str, Any],
    previous_ids_ordered: tuple[str, ...] | None = None,
) -> None:
    """
    Validate entries, optionally reorder PNGs, write ``manifest.json`` and timeline JSON.

    When no ``keyframe_*.png`` exist yet, only ``keyframes_timeline.json`` is written so
    :meth:`BackgroundStills.ensure_keyframes` can generate stills from the plan.
    """
    duration = _duration_from_analysis(analysis)
    ordered_entries = validate_timeline_entries(timeline.entries, duration_sec=duration)
    upload_staging_ids = _upload_staging_entry_ids(cache_dir, ordered_entries)
    if upload_staging_ids:
        ordered_entries = [
            KeyframeTimelineEntry(
                id=e.id,
                t_sec=e.t_sec,
                prompt=e.prompt,
                source="upload" if e.id in upload_staging_ids else e.source,
            )
            for e in ordered_entries
        ]
    new_ids = timeline_identity(ordered_entries)
    n = len(ordered_entries)
    any_png = any(_keyframe_path(cache_dir, i).is_file() for i in range(n))

    if not any_png:
        updated = KeyframesTimeline(
            schema_version=KEYFRAMES_TIMELINE_SCHEMA_VERSION,
            manually_edited=True,
            entries=tuple(ordered_entries),
            target_width=timeline.target_width,
            target_height=timeline.target_height,
        )
        save_keyframes_timeline(cache_dir, updated)
        return

    needs_rematerialize = True
    if previous_ids_ordered is not None and previous_ids_ordered == new_ids:
        needs_rematerialize = False
    elif previous_ids_ordered is None:
        man = _load_manifest(cache_dir)
        if man is not None and man.num_keyframes == len(new_ids):
            # First edit with synthetic kf-i ids matching manifest count
            synthetic = tuple(f"kf-{i}" for i in range(man.num_keyframes))
            if synthetic == new_ids:
                needs_rematerialize = False

    if needs_rematerialize or upload_staging_ids:
        rematerialize_keyframe_pngs(
            cache_dir,
            ordered_entries,
            previous_ids_ordered=previous_ids_ordered,
        )

    plans = entries_to_keyframe_plans(ordered_entries, analysis)
    manifest = plans_to_background_manifest(
        plans,
        preset_id=preset_id,
        preset_prompt=preset_prompt,
        analysis=analysis,
        model_id=model_id,
        gen_width=gen_width,
        gen_height=gen_height,
    )
    _atomic_write_json(manifest.to_dict(), _manifest_path(cache_dir))

    updated = KeyframesTimeline(
        schema_version=KEYFRAMES_TIMELINE_SCHEMA_VERSION,
        manually_edited=True,
        entries=tuple(ordered_entries),
        target_width=timeline.target_width,
        target_height=timeline.target_height,
    )
    save_keyframes_timeline(cache_dir, updated)
    clear_rife_morph_cache(cache_dir)


def _load_analysis_dict(cache_dir: Path) -> dict[str, Any]:
    p = cache_dir / ANALYSIS_JSON_NAME
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def set_keyframe_entry_prompt(
    cache_dir: Path | str,
    entry_id: str,
    new_prompt: str,
    *,
    source: str | None = None,
) -> None:
    """Update one entry's prompt (and optionally ``source``) in ``keyframes_timeline.json``."""
    cache = Path(cache_dir)
    kt = load_keyframes_timeline(cache)
    if kt is None:
        raise FileNotFoundError(
            f"No {KEYFRAMES_TIMELINE_FILENAME}; save the timeline from the editor first."
        )
    eid = str(entry_id).strip()
    new_src: str | None = None
    if source is not None:
        new_src = "upload" if str(source).lower() == "upload" else "sdxl"
    new_entries: list[KeyframeTimelineEntry] = []
    found = False
    for e in kt.entries:
        if e.id == eid:
            found = True
            use_src = new_src if new_src is not None else e.source
            new_entries.append(
                KeyframeTimelineEntry(
                    id=e.id,
                    t_sec=e.t_sec,
                    prompt=str(new_prompt).strip(),
                    source=use_src,
                )
            )
        else:
            new_entries.append(e)
    if not found:
        raise ValueError(f"Unknown keyframe id {eid!r}")
    save_keyframes_timeline(
        cache,
        KeyframesTimeline(
            schema_version=kt.schema_version,
            manually_edited=True,
            entries=tuple(new_entries),
            target_width=kt.target_width,
            target_height=kt.target_height,
        ),
    )


def refresh_manifest_from_timeline(
    cache_dir: Path | str,
    *,
    preset_id: str,
    preset_prompt: str,
    model_id: str,
    gen_width: int,
    gen_height: int,
) -> BackgroundManifest:
    """Rewrite ``manifest.json`` from the current timeline (no PNG motion). Clears RIFE cache."""
    cache = Path(cache_dir)
    kt = load_keyframes_timeline(cache)
    if kt is None or not kt.entries:
        raise FileNotFoundError("No keyframes timeline on disk")
    analysis = _load_analysis_dict(cache)
    if not analysis:
        raise FileNotFoundError(f"{ANALYSIS_JSON_NAME} missing in cache")
    duration = _duration_from_analysis(analysis)
    ordered = validate_timeline_entries(list(kt.entries), duration_sec=duration)
    plans = entries_to_keyframe_plans(ordered, analysis)
    manifest = plans_to_background_manifest(
        plans,
        preset_id=preset_id,
        preset_prompt=preset_prompt,
        analysis=analysis,
        model_id=model_id,
        gen_width=int(gen_width),
        gen_height=int(gen_height),
    )
    _atomic_write_json(manifest.to_dict(), _manifest_path(cache))
    clear_rife_morph_cache(cache)
    return manifest


__all__ = [
    "KEYFRAMES_TIMELINE_FILENAME",
    "KEYFRAMES_TIMELINE_SCHEMA_VERSION",
    "KeyframeTimelineEntry",
    "KeyframesTimeline",
    "clear_rife_morph_cache",
    "entries_to_keyframe_plans",
    "load_keyframes_timeline",
    "persist_timeline_and_manifest",
    "plans_to_background_manifest",
    "refresh_manifest_from_timeline",
    "save_keyframes_timeline",
    "set_keyframe_entry_prompt",
    "timeline_identity",
    "validate_timeline_entries",
    "rematerialize_keyframe_pngs",
]
