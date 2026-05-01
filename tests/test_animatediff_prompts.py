"""Tests for the AnimateDiff motion-prompt builder and per-preset flavors.

These cover the Phase 4 split between SDXL-still keyframe prompts (which
want structural hints) and AnimateDiff motion prompts (which want motion
language). The builder is ``diffusers``-free so these tests run without
the AnimateDiff pipeline.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image

from pipeline.background_animatediff import (
    ANIMATEDIFF_NEGATIVE_PROMPT,
    DEFAULT_FP16_VAE_ID,
    DEFAULT_MOTION_FLAVOR,
    DEFAULT_NUM_INFERENCE_STEPS,
    MANIFEST_SCHEMA_VERSION,
    MOTION_FLAVORS,
    AnimateDiffBackground,
    _build_motion_prompt,
    _load_stills_keyframe_pil,
    _pacing_cue,
    _pick_stills_keyframe_index,
    _prompt_2_for_index,
    _prompt_hash_segments,
    _read_stills_keyframe_times,
)
from pipeline.background_stills import (
    BACKGROUND_DIRNAME,
    DEFAULT_NEGATIVE_PROMPT,
    KEYFRAME_FILENAME_FMT,
    MANIFEST_FILENAME as STILLS_MANIFEST_FILENAME,
)


class TestPacingCue(unittest.TestCase):
    def test_single_segment_returns_steady(self) -> None:
        self.assertEqual(_pacing_cue(0, 1), "steady motion")

    def test_first_quartile_is_establishing(self) -> None:
        self.assertEqual(
            _pacing_cue(0, 8),
            "establishing shot, slow motion",
        )

    def test_last_quartile_is_fade_out(self) -> None:
        self.assertEqual(_pacing_cue(7, 8), "slower fade-out motion")

    def test_middle_is_steady(self) -> None:
        self.assertEqual(_pacing_cue(4, 8), "steady motion")

    def test_two_segment_song_stays_steady(self) -> None:
        """With only two segments, every segment lands exactly on the
        quartile boundary (rel=0.25 and rel=0.75), so ``_pacing_cue`` keeps
        both on 'steady motion'. This is intentional — a two-segment song
        is too short to warrant an intro/outro pacing split.
        """
        self.assertEqual(_pacing_cue(0, 2), "steady motion")
        self.assertEqual(_pacing_cue(1, 2), "steady motion")

    def test_four_segments_split_into_intro_steady_outro(self) -> None:
        self.assertIn("establishing", _pacing_cue(0, 4))
        self.assertEqual(_pacing_cue(1, 4), "steady motion")
        self.assertEqual(_pacing_cue(2, 4), "steady motion")
        self.assertIn("fade-out", _pacing_cue(3, 4))


class TestBuildMotionPrompt(unittest.TestCase):
    def test_includes_preset_prompt_and_flavor(self) -> None:
        p = _build_motion_prompt(
            "Deep space nebula, violet dust",
            preset_id="cosmic-flow",
            index=2,
            total=5,
        )
        self.assertIn("Deep space nebula", p)
        self.assertIn("cosmic drift", p)

    def test_unknown_preset_falls_back_to_default_flavor(self) -> None:
        p = _build_motion_prompt(
            "some art",
            preset_id="does-not-exist",
            index=1,
            total=3,
        )
        self.assertIn(DEFAULT_MOTION_FLAVOR.split(",", 1)[0], p)

    def test_no_scene_index_or_timestamp_hints(self) -> None:
        """The old keyframe builder appended 'scene N of M / t=X.Xs' — the
        motion builder must not, since those hints drag AnimateDiff off-topic.
        """
        p = _build_motion_prompt(
            "cozy lofi room",
            preset_id="lofi-warm",
            index=3,
            total=7,
            )
        self.assertNotIn("scene ", p)
        self.assertNotIn("t=", p)
        self.assertNotIn("of 7", p)

    def test_pacing_cue_changes_with_position(self) -> None:
        early = _build_motion_prompt(
            "x", preset_id="cosmic-flow", index=0, total=8
        )
        mid = _build_motion_prompt(
            "x", preset_id="cosmic-flow", index=4, total=8
        )
        late = _build_motion_prompt(
            "x", preset_id="cosmic-flow", index=7, total=8
        )
        self.assertIn("establishing", early)
        self.assertIn("steady motion", mid)
        self.assertIn("fade-out", late)

    def test_empty_preset_prompt_skipped_not_crashed(self) -> None:
        p = _build_motion_prompt(
            "", preset_id="cosmic-flow", index=0, total=1
        )
        self.assertFalse(p.startswith(","))
        self.assertIn("cosmic drift", p)


class TestMotionFlavors(unittest.TestCase):
    def test_all_builtin_presets_have_flavors(self) -> None:
        expected = {
            "cosmic-flow",
            "glitch-vhs",
            "lofi-warm",
            "minimal-mono",
            "neon-synthwave",
            "organic-liquid",
        }
        missing = expected - set(MOTION_FLAVORS.keys())
        self.assertFalse(
            missing,
            f"Presets missing motion flavors: {sorted(missing)}",
        )

    def test_flavors_are_nonempty_strings(self) -> None:
        for pid, flavor in MOTION_FLAVORS.items():
            self.assertIsInstance(flavor, str, f"{pid} flavor not a string")
            self.assertTrue(
                flavor.strip(), f"{pid} flavor is blank"
            )


class TestConstants(unittest.TestCase):
    def test_inference_steps_bumped_for_motion(self) -> None:
        self.assertGreaterEqual(DEFAULT_NUM_INFERENCE_STEPS, 35)

    def test_schema_version_bumped_to_two(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 2)

    def test_negative_prompt_extends_stills_default(self) -> None:
        self.assertTrue(
            ANIMATEDIFF_NEGATIVE_PROMPT.startswith(DEFAULT_NEGATIVE_PROMPT),
            "motion negative prompt must extend the stills default, not replace it",
        )
        for term in ("static frame", "frozen motion", "hard cut"):
            self.assertIn(term, ANIMATEDIFF_NEGATIVE_PROMPT)


class TestPromptTwoForIndex(unittest.TestCase):
    """Cross-segment prompt morph relies on each segment passing the *next*
    segment's prompt to SDXL's second text encoder. The last segment has no
    "next" and falls back to its own prompt so it doesn't morph past the song.
    """

    def test_first_segment_returns_second_prompt(self) -> None:
        prompts = ["a", "b", "c", "d"]
        self.assertEqual(_prompt_2_for_index(prompts, 0), "b")

    def test_middle_segment_returns_next_prompt(self) -> None:
        prompts = ["a", "b", "c", "d"]
        self.assertEqual(_prompt_2_for_index(prompts, 2), "d")

    def test_last_segment_falls_back_to_self(self) -> None:
        prompts = ["a", "b", "c"]
        self.assertEqual(_prompt_2_for_index(prompts, 2), "c")

    def test_single_segment_song_returns_self(self) -> None:
        self.assertEqual(_prompt_2_for_index(["only"], 0), "only")

    def test_empty_prompts_raises(self) -> None:
        with self.assertRaises(ValueError):
            _prompt_2_for_index([], 0)

    def test_index_out_of_range_raises(self) -> None:
        with self.assertRaises(IndexError):
            _prompt_2_for_index(["a", "b"], 5)
        with self.assertRaises(IndexError):
            _prompt_2_for_index(["a", "b"], -1)


class TestPromptHashIncludesPromptTwo(unittest.TestCase):
    """Switching the prompt-2 mapping must invalidate the cache so old loops
    rendered without cross-segment morph don't false-match.
    """

    def test_hash_changes_when_pair_separator_appears(self) -> None:
        # The build_expected_manifest builder pairs prompts as
        # ``"<prompt>|||<prompt_2>"``. We mirror that here to confirm the
        # hash flips when the prompt_2 mapping changes.
        without_morph = ["a|||a", "b|||b", "c|||c"]
        with_morph = ["a|||b", "b|||c", "c|||c"]
        h_without = _prompt_hash_segments(
            without_morph,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
        )
        h_with = _prompt_hash_segments(
            with_morph,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
        )
        self.assertNotEqual(h_without, h_with)


class _StubInitImageSource:
    """In-memory ``BackgroundSource`` for init-image lifecycle tests.

    Tracks ``ensure`` / ``close`` so we can assert AnimateDiff drives the
    expected lifecycle (run stills -> close stills -> generate AnimateDiff).
    The ``ensure`` callback writes a tiny ``manifest.json`` + a single PNG
    keyframe under ``cache_dir/background/`` so the disk readers in
    ``background_animatediff`` see real files.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        keyframe_times: tuple[float, ...] = (0.0,),
        color: int = 123,
        width: int = 64,
        height: int = 64,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._times = tuple(float(t) for t in keyframe_times)
        self._color = int(color)
        self._w = int(width)
        self._h = int(height)
        self.ensure_calls: int = 0
        self.close_calls: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return (self._w, self._h)

    def ensure(self, *, force: bool = False, progress: Any = None) -> Any:
        del force
        self.ensure_calls += 1
        bg = self._cache_dir / BACKGROUND_DIRNAME
        bg.mkdir(parents=True, exist_ok=True)
        manifest = {"keyframe_times": list(self._times)}
        (bg / STILLS_MANIFEST_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        for i in range(len(self._times)):
            img = Image.new("RGB", (self._w, self._h), color=(self._color, 0, 0))
            img.save(bg / KEYFRAME_FILENAME_FMT.format(index=i))
        if progress is not None:
            progress(0.5, "stub stills halfway")
            progress(1.0, "stub stills done")
        return {"stub": True}

    def background_frame(self, t: float) -> np.ndarray:
        del t
        return np.full((self._h, self._w, 3), self._color, dtype=np.uint8)

    def close(self) -> None:
        self.close_calls += 1


class TestStillsKeyframeReader(unittest.TestCase):
    """The disk-side bridge between BackgroundStills and AnimateDiff: read
    keyframe times from the stills manifest, pick the closest one for a
    target time, and load the PNG resized to the AnimateDiff gen size.
    """

    def test_read_times_returns_empty_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_read_stills_keyframe_times(Path(tmp)), ())

    def test_read_times_round_trips_floats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / BACKGROUND_DIRNAME).mkdir(parents=True, exist_ok=True)
            (cache / BACKGROUND_DIRNAME / STILLS_MANIFEST_FILENAME).write_text(
                json.dumps({"keyframe_times": [0.0, 4.5, 9.25]}),
                encoding="utf-8",
            )
            self.assertEqual(
                _read_stills_keyframe_times(cache),
                (0.0, 4.5, 9.25),
            )

    def test_pick_closest_returns_none_for_empty(self) -> None:
        self.assertIsNone(_pick_stills_keyframe_index((), 1.0))

    def test_pick_closest_chooses_nearest(self) -> None:
        times = (0.0, 4.0, 8.0, 12.0)
        self.assertEqual(_pick_stills_keyframe_index(times, 0.0), 0)
        self.assertEqual(_pick_stills_keyframe_index(times, 3.0), 1)
        self.assertEqual(_pick_stills_keyframe_index(times, 7.0), 2)
        self.assertEqual(_pick_stills_keyframe_index(times, 100.0), 3)

    def test_load_keyframe_returns_resized_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / BACKGROUND_DIRNAME).mkdir(parents=True, exist_ok=True)
            src = Image.new("RGB", (32, 32), color=(10, 20, 30))
            src.save(cache / BACKGROUND_DIRNAME / KEYFRAME_FILENAME_FMT.format(index=0))
            img = _load_stills_keyframe_pil(cache, 0, target_size=(64, 64))
            self.assertIsNotNone(img)
            assert img is not None
            self.assertEqual(img.size, (64, 64))

    def test_load_keyframe_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                _load_stills_keyframe_pil(Path(tmp), 0, target_size=(64, 64))
            )


class TestInitImageForSegment(unittest.TestCase):
    """``_init_image_for_segment`` returns the SDXL keyframe whose time is
    closest to the segment start. With no SDXL cache on disk it returns
    ``None`` so AnimateDiff falls back to text-to-video for that segment.
    """

    def _make(
        self, cache_dir: Path, *, times: tuple[float, ...]
    ) -> AnimateDiffBackground:
        ad = AnimateDiffBackground(
            cache_dir,
            preset_id="test",
            preset_prompt="x",
            width=8,
            height=8,
            gen_width=64,
            gen_height=64,
            num_frames=2,
        )
        ad._init_keyframe_times = times
        ad._segments = [
            {"t_start": 0.0, "t_end": 5.0, "label": 0},
            {"t_start": 5.0, "t_end": 10.0, "label": 1},
        ]
        return ad

    def test_returns_none_without_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ad = self._make(Path(tmp), times=())
            self.assertIsNone(ad._init_image_for_segment(0))

    def test_picks_correct_keyframe_per_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / BACKGROUND_DIRNAME).mkdir(parents=True, exist_ok=True)
            colors = [(255, 0, 0), (0, 255, 0)]
            for i, c in enumerate(colors):
                Image.new("RGB", (32, 32), color=c).save(
                    cache / BACKGROUND_DIRNAME / KEYFRAME_FILENAME_FMT.format(index=i)
                )
            ad = self._make(cache, times=(0.0, 6.0))
            img0 = ad._init_image_for_segment(0)
            img1 = ad._init_image_for_segment(1)
            self.assertIsNotNone(img0)
            self.assertIsNotNone(img1)
            assert img0 is not None and img1 is not None
            self.assertEqual(img0.size, (64, 64))
            self.assertEqual(img1.size, (64, 64))
            r0 = np.asarray(img0)[16, 16, 0]
            r1 = np.asarray(img1)[16, 16, 0]
            self.assertGreater(int(r0), 200)
            self.assertLess(int(r1), 50)


class TestInitImageSourceLifecycle(unittest.TestCase):
    """The init-image source is closed by ``close()`` if ``ensure`` was
    never called. The normal ``ensure`` flow closes it inline (to free SDXL
    VRAM before AnimateDiff loads) and is exercised in integration tests
    that need GPU; here we cover the early-exit safety net.
    """

    def test_close_releases_init_source_when_ensure_never_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubInitImageSource(Path(tmp))
            ad = AnimateDiffBackground(
                Path(tmp),
                preset_id="test",
                preset_prompt="x",
                width=8,
                height=8,
                gen_width=64,
                gen_height=64,
                num_frames=2,
                init_image_source=stub,
            )
            ad.close()
            self.assertEqual(stub.close_calls, 1)


class TestInitImageStrengthValidation(unittest.TestCase):
    def test_strength_below_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                AnimateDiffBackground(
                    Path(tmp),
                    preset_id="test",
                    preset_prompt="x",
                    width=8,
                    height=8,
                    gen_width=64,
                    gen_height=64,
                    num_frames=2,
                    init_image_strength=0.02,
                )

    def test_strength_above_one_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                AnimateDiffBackground(
                    Path(tmp),
                    preset_id="test",
                    preset_prompt="x",
                    width=8,
                    height=8,
                    gen_width=64,
                    gen_height=64,
                    num_frames=2,
                    init_image_strength=1.2,
                )


class TestPromptHashIncludesInitKey(unittest.TestCase):
    """Toggling SDXL init images on/off must invalidate the cache so old
    text-to-video loops don't get reused for an init-image render (and vice
    versa). The hash mixes a stable string built from the keyframe times.
    """

    def test_hash_changes_when_init_key_changes(self) -> None:
        prompts = ("a|||b", "b|||b")
        h_no_init = _prompt_hash_segments(
            prompts,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
            init_key="init=none",
        )
        h_with_init = _prompt_hash_segments(
            prompts,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
            init_key="img2img-v1|s=0.3800|t=0.000,8.000",
        )
        h_stronger = _prompt_hash_segments(
            prompts,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
            init_key="img2img-v1|s=0.6000|t=0.000,8.000",
        )
        self.assertNotEqual(h_no_init, h_with_init)
        self.assertNotEqual(h_with_init, h_stronger)


class TestFp16SafeVae(unittest.TestCase):
    """The fp16-safe VAE swap is the fix for the all-black-frames bug.

    These tests pin the contract without loading diffusers / torch:
    1. The default VAE id points at the community fp16-safe checkpoint.
    2. Changing ``vae_id`` invalidates the cache hash so any black-frame
       PNGs from the previous (broken) run are regenerated automatically.
    """

    def test_default_vae_id_is_madebyollin_fp16_fix(self) -> None:
        self.assertEqual(DEFAULT_FP16_VAE_ID, "madebyollin/sdxl-vae-fp16-fix")

    def test_prompt_hash_changes_when_vae_id_changes(self) -> None:
        prompts = ("a", "b")
        h_default = _prompt_hash_segments(
            prompts,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
        )
        h_other = _prompt_hash_segments(
            prompts,
            model_id="m",
            motion_adapter_id="ma",
            vae_id="some/other-vae",
        )
        self.assertNotEqual(
            h_default,
            h_other,
            "vae_id must participate in the cache hash so old black-frame "
            "outputs invalidate when the VAE is swapped",
        )

    def test_prompt_hash_changes_vs_legacy_no_vae(self) -> None:
        """Existing v2 caches were hashed without ``vae_id``; their hash must
        differ from the new (fp16-safe-VAE) hash so they don't false-match."""
        prompts = ("a", "b")
        legacy = _prompt_hash_segments(
            prompts, model_id="m", motion_adapter_id="ma"
        )
        new = _prompt_hash_segments(
            prompts,
            model_id="m",
            motion_adapter_id="ma",
            vae_id=DEFAULT_FP16_VAE_ID,
        )
        self.assertNotEqual(legacy, new)


if __name__ == "__main__":
    unittest.main()
