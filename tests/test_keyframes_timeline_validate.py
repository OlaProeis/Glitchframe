"""Keyframes timeline validation and upload-staging commit behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.keyframes_timeline import (
    KEYFRAMES_TIMELINE_SCHEMA_VERSION,
    KeyframeTimelineEntry,
    KeyframesTimeline,
    persist_timeline_and_manifest,
    validate_timeline_entries,
)


def test_validate_timeline_clamps_last_anchor_slightly_past_duration() -> None:
    d = 174.68024943310658
    t_last = 174.6802494331066
    entries = [
        KeyframeTimelineEntry(id="kf-0", t_sec=0.0, prompt="a", source="sdxl"),
        KeyframeTimelineEntry(id="kf-1", t_sec=2.0, prompt="b", source="sdxl"),
        KeyframeTimelineEntry(id="kf-2", t_sec=t_last, prompt="c", source="sdxl"),
    ]
    out = validate_timeline_entries(entries, duration_sec=d)
    assert out[-1].t_sec == d


def test_validate_timeline_rejects_far_past_duration() -> None:
    d = 100.0
    entries = [
        KeyframeTimelineEntry(id="kf-0", t_sec=0.0, prompt="a", source="sdxl"),
        KeyframeTimelineEntry(id="kf-1", t_sec=101.0, prompt="b", source="sdxl"),
    ]
    with pytest.raises(ValueError, match="out of range"):
        validate_timeline_entries(entries, duration_sec=d)


def test_persist_rematerializes_when_upload_staging_present_same_ids(
    tmp_path: Path,
) -> None:
    """Save timeline must merge ``upload_staging/<id>.png`` even when order unchanged."""
    cache = tmp_path / "c1"
    bg = cache / "background"
    bg.mkdir(parents=True)
    analysis = {"duration_sec": 10.0, "segments": []}
    (cache / ANALYSIS_JSON_NAME).write_text(
        json.dumps(analysis), encoding="utf-8"
    )

    def _write_kf(i: int, rgb: tuple[int, int, int]) -> None:
        img = Image.new("RGB", (8, 8), rgb)
        img.save(bg / f"keyframe_{i:04d}.png", format="PNG")

    _write_kf(0, (255, 0, 0))
    _write_kf(1, (0, 255, 0))

    staging = bg / "upload_staging"
    staging.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(
        staging / "kf-1.png", format="PNG"
    )

    manifest = {
        "schema_version": 1,
        "preset_id": "default",
        "prompt_hash": "x",
        "section_count": 1,
        "num_keyframes": 2,
        "duration_sec": 10.0,
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "width": 1920,
        "height": 1080,
        "keyframe_times": [0.0, 10.0],
        "prompts": ["a", "b"],
    }
    (bg / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    tl = KeyframesTimeline(
        schema_version=KEYFRAMES_TIMELINE_SCHEMA_VERSION,
        manually_edited=True,
        entries=(
            KeyframeTimelineEntry(
                id="kf-0", t_sec=0.0, prompt="a", source="sdxl"
            ),
            KeyframeTimelineEntry(
                id="kf-1", t_sec=10.0, prompt="b", source="sdxl"
            ),
        ),
        target_width=1920,
        target_height=1080,
    )
    prev_ids = ("kf-0", "kf-1")

    persist_timeline_and_manifest(
        cache,
        tl,
        preset_id="default",
        preset_prompt="base",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        gen_width=1344,
        gen_height=768,
        analysis=analysis,
        previous_ids_ordered=prev_ids,
    )

    slot1 = Image.open(bg / "keyframe_0001.png").convert("RGB")
    assert slot1.getpixel((0, 0)) == (0, 0, 255)
    assert not (staging / "kf-1.png").is_file()

    saved = json.loads((cache / "keyframes_timeline.json").read_text(encoding="utf-8"))
    ents = saved.get("entries") or []
    by_id = {e["id"]: e for e in ents}
    assert by_id["kf-1"]["source"] == "upload"
