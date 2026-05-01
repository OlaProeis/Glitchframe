"""RIFE morph manifest cache-key behaviour."""

from __future__ import annotations

import unittest

from pipeline.background_stills import RifeMorphManifest, RIFE_MANIFEST_SCHEMA_VERSION


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
        )
        m2 = RifeMorphManifest.from_dict(m.to_dict())
        self.assertEqual(m, m2)


if __name__ == "__main__":
    unittest.main()
