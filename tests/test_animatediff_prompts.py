"""Tests for the AnimateDiff motion-prompt builder and per-preset flavors.

These cover the Phase 4 split between SDXL-still keyframe prompts (which
want structural hints) and AnimateDiff motion prompts (which want motion
language). The builder is ``diffusers``-free so these tests run without
the AnimateDiff pipeline.
"""

from __future__ import annotations

import unittest

from pipeline.background_animatediff import (
    ANIMATEDIFF_NEGATIVE_PROMPT,
    DEFAULT_MOTION_FLAVOR,
    DEFAULT_NUM_INFERENCE_STEPS,
    MANIFEST_SCHEMA_VERSION,
    MOTION_FLAVORS,
    _build_motion_prompt,
    _pacing_cue,
)
from pipeline.background_stills import DEFAULT_NEGATIVE_PROMPT


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
            preset_id="cosmic",
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
            "x", preset_id="cosmic", index=0, total=8
        )
        mid = _build_motion_prompt(
            "x", preset_id="cosmic", index=4, total=8
        )
        late = _build_motion_prompt(
            "x", preset_id="cosmic", index=7, total=8
        )
        self.assertIn("establishing", early)
        self.assertIn("steady motion", mid)
        self.assertIn("fade-out", late)

    def test_empty_preset_prompt_skipped_not_crashed(self) -> None:
        p = _build_motion_prompt(
            "", preset_id="cosmic", index=0, total=1
        )
        self.assertFalse(p.startswith(","))
        self.assertIn("cosmic drift", p)


class TestMotionFlavors(unittest.TestCase):
    def test_all_builtin_presets_have_flavors(self) -> None:
        expected = {
            "cosmic",
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


if __name__ == "__main__":
    unittest.main()
