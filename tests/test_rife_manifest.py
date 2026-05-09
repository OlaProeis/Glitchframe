"""RIFE morph manifest cache-key behaviour."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.background_stills import (
    RifeMorphManifest,
    RIFE_MANIFEST_SCHEMA_VERSION,
    _hash_keyframe_rgb_sequence,
)


class RifeMorphManifestTests(unittest.TestCase):
    def test_matches_key_requires_times_length(self) -> None:
        m = RifeMorphManifest(
            schema_version=RIFE_MANIFEST_SCHEMA_VERSION,
            base_prompt_hash="ab",
            preset_id="p",
            section_count=3,
            num_keyframes=4,
            duration_sec=120.0,
            rife_exp=4,
            rife_repo_id="MonsterMMORPG/RIFE_4_26",
            width=1920,
            height=1080,
            frame_count=3,
            times=(0.0, 60.0, 120.0),
            keyframes_content_hash="kfhash",
        )
        self.assertTrue(
            m.matches_key(
                base_prompt_hash="ab",
                preset_id="p",
                section_count=3,
                num_keyframes=4,
                duration_sec=120.0,
                rife_exp=4,
                rife_repo_id="MonsterMMORPG/RIFE_4_26",
                width=1920,
                height=1080,
                keyframes_content_hash="kfhash",
            )
        )

    def test_matches_key_requires_keyframes_content_hash(self) -> None:
        m = RifeMorphManifest(
            schema_version=RIFE_MANIFEST_SCHEMA_VERSION,
            base_prompt_hash="ab",
            preset_id="p",
            section_count=3,
            num_keyframes=4,
            duration_sec=120.0,
            rife_exp=4,
            rife_repo_id="MonsterMMORPG/RIFE_4_26",
            width=1920,
            height=1080,
            frame_count=3,
            times=(0.0, 60.0, 120.0),
            keyframes_content_hash="aaa",
        )
        self.assertFalse(
            m.matches_key(
                base_prompt_hash="ab",
                preset_id="p",
                section_count=3,
                num_keyframes=4,
                duration_sec=120.0,
                rife_exp=4,
                rife_repo_id="MonsterMMORPG/RIFE_4_26",
                width=1920,
                height=1080,
                keyframes_content_hash="bbb",
            )
        )

    def test_roundtrip_dict(self) -> None:
        m = RifeMorphManifest(
            schema_version=RIFE_MANIFEST_SCHEMA_VERSION,
            base_prompt_hash="h",
            preset_id="neon",
            section_count=1,
            num_keyframes=2,
            duration_sec=10.0,
            rife_exp=3,
            rife_repo_id="MonsterMMORPG/RIFE_4_26",
            width=1280,
            height=720,
            frame_count=2,
            times=(0.0, 10.0),
            keyframes_content_hash="deadbeef",
        )
        m2 = RifeMorphManifest.from_dict(m.to_dict())
        self.assertEqual(m, m2)

    def test_from_dict_v2_missing_hash_gets_empty_string(self) -> None:
        raw = {
            "schema_version": 2,
            "base_prompt_hash": "x",
            "preset_id": "p",
            "section_count": 1,
            "num_keyframes": 2,
            "duration_sec": 5.0,
            "rife_exp": 4,
            "rife_repo_id": "MonsterMMORPG/RIFE_4_26",
            "width": 640,
            "height": 360,
            "frame_count": 2,
            "times": [0.0, 5.0],
        }
        m = RifeMorphManifest.from_dict(raw)
        self.assertEqual(m.keyframes_content_hash, "")

    def test_hash_keyframe_rgb_sequence_order_sensitive(self) -> None:
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = np.ones((2, 2, 3), dtype=np.uint8) * 255
        h1 = _hash_keyframe_rgb_sequence((a, b))
        h2 = _hash_keyframe_rgb_sequence((b, a))
        self.assertNotEqual(h1, h2)

    def test_hash_keyframe_rgb_sequence_pixel_sensitive(self) -> None:
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, 0] = 1
        self.assertNotEqual(
            _hash_keyframe_rgb_sequence((a,)),
            _hash_keyframe_rgb_sequence((b,)),
        )


if __name__ == "__main__":
    unittest.main()
