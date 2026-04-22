"""Unit tests for lyrics-aligner post-processing passes.

Focused on the pure-Python timing massage (NW gap fill, monotonicity,
outro clamp, duration cap). We do not instantiate WhisperX here — that's
covered end-to-end by manual renders — so these tests stay fast and
hermetic.
"""

from __future__ import annotations

import unittest

from pipeline.lyrics_aligner import (
    ADJACENT_WORD_CLAMP_SEC,
    MAX_WORD_DURATION_SEC,
    SECTION_MARKER,
    _assign_user_tokens_to_segments,
    _build_forced_alignment_segments,
    _char_weight,
    _enforce_monotonic_per_line,
    _extract_whisper_words,
    _fill_gap_timings,
    _flatten_transcribe_segments,
    _lyrics_cache_key,
    _polish_timings,
    _proportional_segment_assignment,
    _split_user_lyrics,
    _timings_for_user_tokens,
    _WhisperSegment,
    _WhisperWord,
)


class TestCharWeightedGapFill(unittest.TestCase):
    """``_fill_gap_timings`` should distribute time by character weight."""

    def test_all_unknown_returns_early(self) -> None:
        starts: list[float | None] = [None, None, None]
        ends: list[float | None] = [None, None, None]
        _fill_gap_timings(starts, ends, ["a", "b", "c"])
        self.assertEqual(starts, [None, None, None])
        self.assertEqual(ends, [None, None, None])

    def test_leading_and_trailing_gaps_snap_to_anchor(self) -> None:
        starts: list[float | None] = [None, 1.0, None]
        ends: list[float | None] = [None, 1.2, None]
        _fill_gap_timings(starts, ends, ["x", "y", "z"])
        self.assertEqual(starts, [1.0, 1.0, 1.2])
        self.assertEqual(ends, [1.0, 1.2, 1.2])

    def test_middle_gap_weighted_by_chars(self) -> None:
        # Anchor A end = 1.0, anchor B start = 2.0 → 1 second gap.
        # Gap words: "a" (1) + "tremendously" (12) = 13 total weight.
        # Short word should claim ~1/13 ≈ 0.077s; long word should claim ~12/13.
        starts: list[float | None] = [1.0, None, None, None]
        ends: list[float | None] = [1.0, None, None, 2.0]
        # Seed the anchor's end (index 0) and the last anchor's start (index 3).
        # We treat 0 and 3 as known (both start and end populated for both).
        starts[3] = 2.0
        _fill_gap_timings(starts, ends, ["anchor", "a", "tremendously", "bye"])
        self.assertIsNotNone(starts[1])
        self.assertIsNotNone(starts[2])
        # "a" (short) should occupy a small slice near v0; "tremendously" (long)
        # should consume most of the gap before v1.
        self.assertAlmostEqual(float(starts[1]), 1.0, places=3)
        # "a" end ≈ 1.0 + (1/13) * 1.0
        self.assertAlmostEqual(float(ends[1]), 1.0 + 1.0 / 13.0, places=3)
        self.assertAlmostEqual(float(starts[2]), 1.0 + 1.0 / 13.0, places=3)
        self.assertAlmostEqual(float(ends[2]), 2.0, places=3)

    def test_overlapping_anchors_fall_back_to_uniform(self) -> None:
        # Anchor A end = 2.0, anchor B start = 1.5 (overlap) → uniform fallback.
        starts: list[float | None] = [1.0, None, None, 1.5]
        ends: list[float | None] = [2.0, None, None, 1.7]
        _fill_gap_timings(starts, ends, ["a", "b", "c", "d"])
        # Gap words should sit between 2.0 and 1.5 linearly (decreasing, but
        # at least not negative durations). Main invariant: they're filled.
        self.assertIsNotNone(starts[1])
        self.assertIsNotNone(starts[2])


class TestPolishTimings(unittest.TestCase):
    """``_polish_timings`` clips outro bleed and caps word duration."""

    def test_clips_outro_bleed_in_same_line(self) -> None:
        # Word 0 extends past word 1's start → clip to just before it.
        tokens = [("hello", 0), ("world", 0)]
        timings = [(1.0, 2.0), (1.5, 1.8)]
        out = _polish_timings(tokens, timings)
        self.assertAlmostEqual(out[0][1], 1.5 - ADJACENT_WORD_CLAMP_SEC, places=6)
        self.assertGreaterEqual(out[0][1], out[0][0])
        self.assertEqual(out[1], (1.5, 1.8))

    def test_keeps_outro_across_line_boundary(self) -> None:
        # Different lines: outro leak is fine (cross-line fade handles it).
        tokens = [("end", 0), ("next", 1)]
        timings = [(1.0, 2.0), (1.5, 1.8)]
        out = _polish_timings(tokens, timings)
        self.assertEqual(out[0], (1.0, 2.0))

    def test_caps_long_word_duration(self) -> None:
        tokens = [("soooo", 0)]
        timings = [(1.0, 10.0)]  # 9 s sustained — impossible in sung music.
        out = _polish_timings(tokens, timings)
        self.assertAlmostEqual(out[0][0], 1.0, places=6)
        self.assertAlmostEqual(out[0][1], 1.0 + MAX_WORD_DURATION_SEC, places=6)

    def test_short_words_untouched(self) -> None:
        tokens = [("hey", 0), ("you", 0)]
        timings = [(1.0, 1.3), (1.5, 1.7)]
        out = _polish_timings(tokens, timings)
        self.assertEqual(out, timings)


class TestEndToEndTimingPath(unittest.TestCase):
    """Sanity: the full ``_timings_for_user_tokens`` pipeline with a simple case."""

    def test_matched_words_get_whisper_timings(self) -> None:
        user = [("hello", 0), ("world", 0)]
        whisper = [
            _WhisperWord(word="hello", t_start=1.0, t_end=1.4),
            _WhisperWord(word="world", t_start=1.5, t_end=1.9),
        ]
        out = _timings_for_user_tokens(user, whisper)
        self.assertEqual(out[0], (1.0, 1.4))
        self.assertEqual(out[1], (1.5, 1.9))

    def test_unmatched_middle_word_gets_weighted_share(self) -> None:
        # "the" is not in whisper output → NW gap of 1 word.
        # Whisper saw "hello <silence> world" from 1.0 to 3.0; "the" should
        # fit between whisper's "hello" (end 1.4) and "world" (start 2.5).
        user = [("hello", 0), ("the", 0), ("world", 0)]
        whisper = [
            _WhisperWord(word="hello", t_start=1.0, t_end=1.4),
            _WhisperWord(word="world", t_start=2.5, t_end=2.9),
        ]
        out = _timings_for_user_tokens(user, whisper)
        # Anchors untouched.
        self.assertEqual(out[0], (1.0, 1.4))
        self.assertEqual(out[2], (2.5, 2.9))
        # Gap word lives inside the gap.
        self.assertGreaterEqual(out[1][0], 1.4)
        self.assertLessEqual(out[1][1], 2.5)

    def test_no_whisper_spreads_uniform(self) -> None:
        user = [("a", 0), ("b", 0)]
        whisper: list[_WhisperWord] = []
        out = _timings_for_user_tokens(user, whisper)
        self.assertEqual(len(out), 2)
        self.assertEqual(out, [(0.0, 0.0), (0.0, 0.0)])


class TestCharWeight(unittest.TestCase):
    def test_weight_floor_is_one(self) -> None:
        self.assertEqual(_char_weight(""), 1.0)
        self.assertEqual(_char_weight("x"), 1.0)

    def test_weight_scales_with_length(self) -> None:
        self.assertEqual(_char_weight("hello"), 5.0)
        self.assertEqual(_char_weight("tremendously"), 12.0)


class TestMonotonicEnforcement(unittest.TestCase):
    """Guard the existing behaviour so the polish pass doesn't regress it."""

    def test_pushes_backward_overlap_forward(self) -> None:
        tokens = [("a", 0), ("b", 0), ("c", 0)]
        timings = [(1.0, 1.5), (1.2, 1.3), (1.6, 1.8)]
        out = _enforce_monotonic_per_line(tokens, timings)
        self.assertGreaterEqual(out[1][0], out[0][1])

    def test_per_line_independent(self) -> None:
        tokens = [("a", 0), ("b", 1)]
        timings = [(2.0, 2.5), (1.0, 1.2)]
        out = _enforce_monotonic_per_line(tokens, timings)
        # Different lines — earlier absolute time on line 1 is fine.
        self.assertEqual(out, [(2.0, 2.5), (1.0, 1.2)])


class TestFlattenTranscribeSegments(unittest.TestCase):
    """``_flatten_transcribe_segments`` normalises whisper's transcribe() shape."""

    def test_drops_segments_with_bad_times(self) -> None:
        result = {
            "segments": [
                {"start": "not-a-number", "end": 1.0, "words": []},
                {"start": 1.0, "end": 2.0, "words": [{"word": "hi"}]},
            ]
        }
        segs = _flatten_transcribe_segments(result)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].words, ("hi",))

    def test_clamps_inverted_times_to_zero_duration(self) -> None:
        result = {"segments": [{"start": 5.0, "end": 3.0, "words": []}]}
        segs = _flatten_transcribe_segments(result)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].t_start, 5.0)
        self.assertEqual(segs[0].t_end, 5.0)

    def test_skips_empty_word_strings(self) -> None:
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "  "},
                        {"word": "hello"},
                        {"word": ""},
                    ],
                }
            ]
        }
        segs = _flatten_transcribe_segments(result)
        self.assertEqual(segs[0].words, ("hello",))

    def test_falls_back_to_segment_text_when_no_words(self) -> None:
        """WhisperX transcribe() emits ``text`` but not per-word detail; we
        must still populate ``words`` or the downstream NW collapses.
        """
        result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "  Hello, world! "},
                {"start": 2.5, "end": 4.0, "text": "Foo  bar"},
            ]
        }
        segs = _flatten_transcribe_segments(result)
        self.assertEqual(segs[0].words, ("Hello,", "world!"))
        self.assertEqual(segs[1].words, ("Foo", "bar"))

    def test_empty_text_and_no_words_leaves_empty_tuple(self) -> None:
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": ""}]}
        segs = _flatten_transcribe_segments(result)
        self.assertEqual(segs[0].words, ())


class TestAssignUserTokensToSegments(unittest.TestCase):
    """NW-based mapping from user tokens to whisper segments."""

    def _seg(self, idx: int, t0: float, t1: float, words: tuple[str, ...]) -> _WhisperSegment:
        return _WhisperSegment(idx=idx, t_start=t0, t_end=t1, words=words)

    def test_exact_match_assigns_correct_segments(self) -> None:
        user = [("hello", 0), ("world", 0), ("foo", 1), ("bar", 1)]
        segments = [
            self._seg(0, 0.0, 1.0, ("hello", "world")),
            self._seg(1, 1.0, 2.0, ("foo", "bar")),
        ]
        assignment = _assign_user_tokens_to_segments(user, segments)
        self.assertEqual(assignment, [0, 0, 1, 1])

    def test_unmatched_user_words_inherit_neighbour_segment(self) -> None:
        user = [("hello", 0), ("there", 0), ("world", 0)]
        segments = [
            self._seg(0, 0.0, 1.0, ("hello",)),
            self._seg(1, 1.0, 2.0, ("world",)),
        ]
        assignment = _assign_user_tokens_to_segments(user, segments)
        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[2], 1)
        self.assertIn(assignment[1], (0, 1))

    def test_empty_whisper_words_distributes_proportionally_by_duration(self) -> None:
        """When whisper segments exist but have no recognised words,
        user tokens are spread proportionally to segment duration instead
        of all collapsing into segment 0 (which would shove every word
        into whisper's first 5 s window and break alignment)."""
        user = [("a", 0), ("b", 0), ("c", 0), ("d", 0)]
        segments = [
            self._seg(0, 0.0, 1.0, ()),
            self._seg(1, 1.0, 3.0, ()),
        ]
        result = _assign_user_tokens_to_segments(user, segments)
        self.assertEqual(len(result), 4)
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_no_nw_matches_falls_back_to_proportional(self) -> None:
        """User lyrics that share no normalised tokens with whisper should
        still span the whole stem, not collapse into segment 0."""
        user = [("cat", 0), ("dog", 0), ("fish", 0)]
        segments = [
            self._seg(0, 0.0, 1.0, ("alpha",)),
            self._seg(1, 1.0, 2.0, ("beta",)),
            self._seg(2, 2.0, 3.0, ("gamma",)),
        ]
        result = _assign_user_tokens_to_segments(user, segments)
        self.assertEqual(set(result), {0, 1, 2})

    def test_no_segments_returns_zeros(self) -> None:
        user = [("a", 0), ("b", 0)]
        self.assertEqual(_assign_user_tokens_to_segments(user, []), [0, 0])


class TestProportionalSegmentAssignment(unittest.TestCase):
    def _seg(self, idx: int, t0: float, t1: float, words: tuple[str, ...]) -> _WhisperSegment:
        return _WhisperSegment(idx=idx, t_start=t0, t_end=t1, words=words)

    def test_distributes_by_word_count(self) -> None:
        segments = [
            self._seg(0, 0.0, 1.0, ("a",)),      # weight 1
            self._seg(1, 1.0, 2.0, ("b", "c", "d")),  # weight 3
        ]
        result = _proportional_segment_assignment(4, segments)
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 3)

    def test_distributes_by_duration_when_no_words(self) -> None:
        segments = [
            self._seg(0, 0.0, 1.0, ()),  # 1 s
            self._seg(1, 1.0, 4.0, ()),  # 3 s
        ]
        result = _proportional_segment_assignment(4, segments)
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 3)

    def test_preserves_temporal_order(self) -> None:
        segments = [
            self._seg(0, 0.0, 1.0, ("a",)),
            self._seg(1, 1.0, 2.0, ("b",)),
        ]
        result = _proportional_segment_assignment(4, segments)
        self.assertEqual(result, sorted(result))

    def test_empty_inputs(self) -> None:
        self.assertEqual(_proportional_segment_assignment(0, []), [])
        self.assertEqual(
            _proportional_segment_assignment(
                0, [self._seg(0, 0.0, 1.0, ())]
            ),
            [],
        )
        self.assertEqual(_proportional_segment_assignment(3, []), [0, 0, 0])


class TestBuildForcedAlignmentSegments(unittest.TestCase):
    def _seg(self, idx: int, t0: float, t1: float) -> _WhisperSegment:
        return _WhisperSegment(idx=idx, t_start=t0, t_end=t1, words=())

    def test_groups_user_tokens_by_assignment(self) -> None:
        user = [("hello", 0), ("world", 0), ("foo", 1)]
        segments = [self._seg(0, 0.0, 1.0), self._seg(1, 1.5, 2.5)]
        out = _build_forced_alignment_segments(user, [0, 0, 1], segments, 3.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "hello world")
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[1]["text"], "foo")
        self.assertEqual(out[1]["start"], 1.5)

    def test_drops_empty_segments(self) -> None:
        user = [("a", 0)]
        segments = [self._seg(0, 0.0, 1.0), self._seg(1, 1.0, 2.0)]
        out = _build_forced_alignment_segments(user, [0], segments, 2.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "a")

    def test_widens_sub_100ms_windows(self) -> None:
        user = [("a", 0)]
        segments = [self._seg(0, 5.0, 5.02)]
        out = _build_forced_alignment_segments(user, [0], segments, 6.0)
        self.assertAlmostEqual(out[0]["end"] - out[0]["start"], 0.1, places=6)

    def test_full_audio_fallback_when_all_empty(self) -> None:
        user = [("a", 0), ("b", 0)]
        segments: list[_WhisperSegment] = []
        out = _build_forced_alignment_segments(user, [], segments, 42.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "a b")
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 42.0)


class TestExtractWhisperWords(unittest.TestCase):
    """Pin the contract with :func:`whisperx.align`'s return shape.

    WhisperX 3.x returns ``{"segments": [...], "word_segments": [...]}``
    where each word dict has ``word``/``start``/``end``/``score``; if a
    segment's CTC backtrack fails it still appears in ``segments`` but
    with an empty ``words`` list and is omitted from ``word_segments``.
    These tests fail loudly if a newer WhisperX changes the shape, so we
    notice before the whole align flow silently returns zero timings.
    """

    def test_parses_whisperx_3x_word_segments(self) -> None:
        aligned = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.1, "end": 0.4, "score": 0.9},
                        {"word": "world", "start": 0.5, "end": 0.9, "score": 0.8},
                    ],
                    "chars": None,
                },
            ],
            "word_segments": [
                {"word": "hello", "start": 0.1, "end": 0.4, "score": 0.9},
                {"word": "world", "start": 0.5, "end": 0.9, "score": 0.8},
            ],
        }
        words = _extract_whisper_words(aligned)
        self.assertEqual(
            [(w.word, w.t_start, w.t_end) for w in words],
            [("hello", 0.1, 0.4), ("world", 0.5, 0.9)],
        )

    def test_falls_back_to_segments_when_word_segments_missing(self) -> None:
        """Some versions only populate ``segments[*].words``; we must cope."""
        aligned = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "words": [{"word": "hey", "start": 0.1, "end": 0.3}],
                },
            ],
        }
        words = _extract_whisper_words(aligned)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].word, "hey")

    def test_empty_word_segments_returns_empty(self) -> None:
        """The bug surface: align() returned but emitted zero words.
        Parser must return ``[]`` so the caller can raise a diagnostic."""
        self.assertEqual(_extract_whisper_words({"word_segments": [], "segments": []}), [])
        self.assertEqual(
            _extract_whisper_words(
                {"segments": [{"start": 0.0, "end": 1.0, "words": []}]}
            ),
            [],
        )

    def test_interpolates_missing_start_end_from_neighbours(self) -> None:
        """WhisperX drops ``start``/``end`` keys when a word's characters had
        no alignable timings; we linearly fill from known neighbours."""
        aligned = {
            "word_segments": [
                {"word": "a", "start": 1.0, "end": 1.2},
                {"word": "b"},  # missing timings
                {"word": "c", "start": 1.8, "end": 2.0},
            ]
        }
        words = _extract_whisper_words(aligned)
        self.assertEqual(len(words), 3)
        self.assertGreater(words[1].t_start, words[0].t_end - 1e-6)
        self.assertLess(words[1].t_end, words[2].t_start + 1e-6)

    def test_strips_whitespace_and_skips_empty_words(self) -> None:
        aligned = {
            "word_segments": [
                {"word": " hi ", "start": 0.0, "end": 0.1},
                {"word": "   ", "start": 0.2, "end": 0.3},
                {"word": "", "start": 0.4, "end": 0.5},
                {"word": "there", "start": 0.6, "end": 0.9},
            ]
        }
        words = _extract_whisper_words(aligned)
        self.assertEqual([w.word for w in words], ["hi", "there"])


class TestSplitUserLyricsSectionMarkers(unittest.TestCase):
    """``_split_user_lyrics`` recognises ``---`` section markers.

    The markers exist so the user can hint the aligner where repeated
    sections (pre-chorus / chorus) begin. They must not leak into the
    rendered lyric lines but must be visible in ``section_starts`` as
    token-index boundaries the forced-alignment step can use.
    """

    def test_no_markers_yields_single_section(self) -> None:
        lines, tokens, starts, anchors = _split_user_lyrics("hello world\nfoo bar")
        self.assertEqual(lines, ["hello world", "foo bar"])
        self.assertEqual([w for w, _ in tokens], ["hello", "world", "foo", "bar"])
        self.assertEqual(starts, [0])
        self.assertEqual(anchors, {})

    def test_single_marker_splits_two_sections(self) -> None:
        text = f"a b\n{SECTION_MARKER}\nc d"
        lines, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual(lines, ["a b", "c d"])
        self.assertEqual([w for w, _ in tokens], ["a", "b", "c", "d"])
        self.assertEqual(starts, [0, 2])

    def test_multiple_markers_split_multiple_sections(self) -> None:
        text = (
            f"one\n{SECTION_MARKER}\ntwo three\n{SECTION_MARKER}\nfour five six"
        )
        _, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual([w for w, _ in tokens], ["one", "two", "three", "four", "five", "six"])
        self.assertEqual(starts, [0, 1, 3])

    def test_leading_marker_is_collapsed(self) -> None:
        text = f"{SECTION_MARKER}\na b\n{SECTION_MARKER}\nc"
        _, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual([w for w, _ in tokens], ["a", "b", "c"])
        self.assertEqual(starts, [0, 2])

    def test_trailing_marker_is_collapsed(self) -> None:
        text = f"a\n{SECTION_MARKER}\nb c\n{SECTION_MARKER}\n"
        _, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual([w for w, _ in tokens], ["a", "b", "c"])
        self.assertEqual(starts, [0, 1])

    def test_consecutive_markers_collapse_to_one_break(self) -> None:
        text = f"a\n{SECTION_MARKER}\n{SECTION_MARKER}\n{SECTION_MARKER}\nb"
        _, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual([w for w, _ in tokens], ["a", "b"])
        self.assertEqual(starts, [0, 1])

    def test_typed_markers_behave_like_plain_markers(self) -> None:
        """``--- instrumental`` is a section break like plain ``---``; the
        tag after the hyphens is documentation only and must not leak into
        rendered lines or tokens."""
        text = "a b\n--- instrumental\nc d\n--- chorus\ne"
        lines, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual(lines, ["a b", "c d", "e"])
        self.assertEqual([w for w, _ in tokens], ["a", "b", "c", "d", "e"])
        self.assertEqual(starts, [0, 2, 4])

    def test_marker_must_start_with_three_hyphens(self) -> None:
        # "----" (four hyphens) or "-- -" are NOT markers; only lines
        # matching ``---`` (optionally followed by free text) count.
        text = f"a\n----\nb\n-- -\nc\n{SECTION_MARKER}\nd"
        lines, tokens, starts, _anchors = _split_user_lyrics(text)
        self.assertEqual(lines, ["a", "----", "b", "-- -", "c", "d"])
        self.assertEqual(len(tokens), 7)
        self.assertEqual(starts, [0, 6])


class TestLyricsCacheKeyMarkerSensitivity(unittest.TestCase):
    """``_lyrics_cache_key`` must invalidate the cache when markers appear.

    Adding a ``---`` to the pasted lyrics means a different forced-alignment
    segmentation → different timings → cache miss. We also guarantee the
    unmarked hash is backwards compatible with the pre-marker schema, so
    users who don't use markers keep their cached alignments.
    """

    def test_unmarked_lyrics_hash_is_stable(self) -> None:
        # Two texts with only whitespace differences produce the same hash.
        a = _lyrics_cache_key("hello world\n\nfoo bar\n")
        b = _lyrics_cache_key("hello world\nfoo bar")
        self.assertEqual(a, b)

    def test_adding_marker_changes_hash(self) -> None:
        unmarked = _lyrics_cache_key("a b\nc d")
        marked = _lyrics_cache_key(f"a b\n{SECTION_MARKER}\nc d")
        self.assertNotEqual(unmarked, marked)

    def test_marker_position_matters(self) -> None:
        # Same words, marker in different place → different hash.
        a = _lyrics_cache_key(f"w x\n{SECTION_MARKER}\ny z")
        b = _lyrics_cache_key(f"w\n{SECTION_MARKER}\nx y z")
        self.assertNotEqual(a, b)


class TestBuildForcedAlignmentSegmentsWithMarkers(unittest.TestCase):
    """Marker-driven variant of ``_build_forced_alignment_segments``.

    The key correctness properties:

    1. One output segment per user-marked section (not per whisper segment).
    2. When two user sections NW-landed in the same whisper segment
       (whisper merged them), the shared time range is split proportionally
       by token count so CTC can't bleed across the user boundary.
    3. User sections still get NW-derived windows when whisper segments
       split them cleanly (the happy path).
    """

    def _seg(self, idx: int, t0: float, t1: float) -> _WhisperSegment:
        return _WhisperSegment(idx=idx, t_start=t0, t_end=t1, words=())

    def test_two_user_sections_merged_into_one_whisper_segment_are_split(self) -> None:
        """The repeated-chorus bug: both user sections land in the same
        whisper segment. Marker mode must split the shared window
        proportionally by token count instead of stacking them."""
        user = [("a", 0), ("b", 0), ("c", 1), ("d", 1), ("e", 1)]
        # Both sections assigned to whisper segment 0 (10s–50s, 40s span).
        assignment = [0, 0, 0, 0, 0]
        segments = [self._seg(0, 10.0, 50.0)]
        out = _build_forced_alignment_segments(
            user, assignment, segments, 60.0,
            user_section_starts=[0, 2],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "a b")
        self.assertEqual(out[1]["text"], "c d e")
        # Section 0: 2 tokens / 5 total = 40% of [10, 50] = [10, 26].
        self.assertAlmostEqual(out[0]["start"], 10.0, places=3)
        self.assertAlmostEqual(out[0]["end"], 26.0, places=3)
        # Section 1: 3 tokens / 5 total = 60% of [10, 50] = [26, 50].
        self.assertAlmostEqual(out[1]["start"], 26.0, places=3)
        self.assertAlmostEqual(out[1]["end"], 50.0, places=3)
        # Adjacent sections must not overlap.
        self.assertAlmostEqual(out[0]["end"], out[1]["start"], places=3)

    def test_three_user_sections_split_one_whisper_segment_by_token_count(self) -> None:
        user = (
            [("p", 0)] * 10  # section 0: 10 tokens
            + [("q", 1)] * 20  # section 1: 20 tokens
            + [("r", 2)] * 10  # section 2: 10 tokens
        )
        assignment = [0] * 40
        segments = [self._seg(0, 0.0, 40.0)]
        out = _build_forced_alignment_segments(
            user, assignment, segments, 50.0,
            user_section_starts=[0, 10, 30],
        )
        self.assertEqual(len(out), 3)
        # 10/40, 20/40, 10/40 of 40s span.
        self.assertAlmostEqual(out[0]["end"] - out[0]["start"], 10.0, places=3)
        self.assertAlmostEqual(out[1]["end"] - out[1]["start"], 20.0, places=3)
        self.assertAlmostEqual(out[2]["end"] - out[2]["start"], 10.0, places=3)

    def test_user_sections_mapped_cleanly_get_whisper_windows(self) -> None:
        """Happy path: each user section's tokens NW-land in a distinct
        whisper segment. Marker mode should use those windows as-is."""
        user = [("a", 0), ("b", 0), ("c", 1), ("d", 1)]
        assignment = [0, 0, 1, 1]
        segments = [self._seg(0, 0.0, 5.0), self._seg(1, 10.0, 15.0)]
        out = _build_forced_alignment_segments(
            user, assignment, segments, 20.0,
            user_section_starts=[0, 2],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 5.0)
        self.assertEqual(out[1]["start"], 10.0)
        self.assertEqual(out[1]["end"], 15.0)

    def test_user_section_without_anchors_fills_from_neighbours(self) -> None:
        """If a middle user section has no NW anchors (its whisper segment
        is missing from the lookup), it must still land between its
        anchored neighbours rather than collapsing to 0-duration."""
        user = [("a", 0), ("b", 1), ("c", 2)]
        # Middle token's assignment points at a segment idx we don't supply.
        assignment = [0, 99, 1]
        segments = [self._seg(0, 0.0, 2.0), self._seg(1, 8.0, 10.0)]
        out = _build_forced_alignment_segments(
            user, assignment, segments, 12.0,
            user_section_starts=[0, 1, 2],
        )
        self.assertEqual(len(out), 3)
        self.assertGreaterEqual(out[1]["start"], out[0]["end"])
        self.assertLessEqual(out[1]["end"], out[2]["start"])

    def test_single_section_marker_list_falls_back_to_whisper_bucketing(self) -> None:
        """``user_section_starts = [0]`` is semantically "no markers" —
        identical to the pre-marker code path."""
        user = [("a", 0), ("b", 0)]
        segments = [self._seg(0, 0.0, 1.0), self._seg(1, 1.5, 2.5)]
        with_marker = _build_forced_alignment_segments(
            user, [0, 1], segments, 3.0, user_section_starts=[0]
        )
        without_marker = _build_forced_alignment_segments(
            user, [0, 1], segments, 3.0
        )
        self.assertEqual(with_marker, without_marker)


class TestInlineLineAnchors(unittest.TestCase):
    """``[m:ss]`` timestamps at line start pin that line's first word."""

    def test_anchor_strips_bracket_from_rendered_line(self) -> None:
        lines, tokens, _starts, anchors = _split_user_lyrics("[1:23] hello world")
        self.assertEqual(lines, ["hello world"])
        self.assertEqual([w for w, _ in tokens], ["hello", "world"])
        self.assertEqual(anchors, {0: 83.0})

    def test_anchor_accepts_milliseconds(self) -> None:
        _l, _t, _s, anchors = _split_user_lyrics("[0:01.500] go")
        self.assertAlmostEqual(anchors[0], 1.5, places=6)

    def test_anchor_accepts_hours(self) -> None:
        _l, _t, _s, anchors = _split_user_lyrics("[1:02:03] deep cut")
        self.assertAlmostEqual(anchors[0], 3723.0, places=6)

    def test_multiple_anchors_keyed_by_line_index(self) -> None:
        text = "[0:10] first line\nregular line\n[0:30] third line"
        lines, _tokens, _starts, anchors = _split_user_lyrics(text)
        self.assertEqual(lines, ["first line", "regular line", "third line"])
        self.assertEqual(anchors, {0: 10.0, 2: 30.0})

    def test_bracket_only_line_anchors_next_lyric(self) -> None:
        text = "[0:42]\nhello world"
        lines, tokens, _starts, anchors = _split_user_lyrics(text)
        self.assertEqual(lines, ["hello world"])
        self.assertEqual([w for w, _ in tokens], ["hello", "world"])
        self.assertEqual(anchors, {0: 42.0})

    def test_square_brackets_in_mid_line_are_not_anchors(self) -> None:
        lines, _t, _s, anchors = _split_user_lyrics("hello [whispered] world")
        self.assertEqual(lines, ["hello [whispered] world"])
        self.assertEqual(anchors, {})

    def test_anchor_inside_section_affects_hash(self) -> None:
        a = _lyrics_cache_key("hello world")
        b = _lyrics_cache_key("[0:05] hello world")
        self.assertNotEqual(a, b)

    def test_section_starts_from_anchors_splits_at_anchored_lines(self) -> None:
        from pipeline.lyrics_aligner import _section_starts_from_anchors

        user = [("a", 0), ("b", 0), ("c", 1), ("d", 1), ("e", 2)]
        starts = _section_starts_from_anchors(user, {1: 10.0, 2: 20.0})
        self.assertEqual(starts, [0, 2, 4])

    def test_section_starts_from_anchors_ignores_first_line_anchor(self) -> None:
        """An anchor on line 0 is a *start-of-song* hint, not a break."""
        from pipeline.lyrics_aligner import _section_starts_from_anchors

        user = [("a", 0), ("b", 0), ("c", 1)]
        starts = _section_starts_from_anchors(user, {0: 5.0})
        self.assertEqual(starts, [0])


class TestBuildForcedAlignmentWithAnchors(unittest.TestCase):
    """Anchors override whisper-derived section starts."""

    def _seg(self, idx: int, t0: float, t1: float) -> _WhisperSegment:
        return _WhisperSegment(idx=idx, t_start=t0, t_end=t1, words=())

    def test_anchor_overrides_section_start(self) -> None:
        # Two user sections; whisper merged both into segment 0 [10, 50].
        # The second section is anchored at 40s → section 0 must end at 40,
        # section 1 must start at 40 (instead of the proportional 26s).
        user = [("a", 0), ("b", 0), ("c", 1), ("d", 1), ("e", 1)]
        assignment = [0, 0, 0, 0, 0]
        segments = [self._seg(0, 10.0, 50.0)]
        out = _build_forced_alignment_segments(
            user,
            assignment,
            segments,
            60.0,
            user_section_starts=[0, 2],
            line_anchors={1: 40.0},
        )
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[1]["start"], 40.0, places=3)
        self.assertLessEqual(out[0]["end"], 40.0 + 1e-6)

    def test_anchor_without_marker_synthesises_section_break(self) -> None:
        """With no ``---`` markers, an anchor on a middle line still
        pins that line — the builder auto-creates a section break."""
        user = [("a", 0), ("b", 0), ("c", 1), ("d", 1)]
        assignment = [0, 0, 0, 0]
        segments = [self._seg(0, 0.0, 20.0)]
        out = _build_forced_alignment_segments(
            user,
            assignment,
            segments,
            20.0,
            line_anchors={1: 12.0},
        )
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[1]["start"], 12.0, places=3)


class TestSnapToVocalOnsets(unittest.TestCase):
    from pipeline.lyrics_aligner import _snap_to_vocal_onsets as _snap

    def test_shifts_word_to_onset_within_window(self) -> None:
        from pipeline.lyrics_aligner import _snap_to_vocal_onsets

        tokens = [("a", 0)]
        timings = [(1.00, 1.30)]
        # Onset at 1.04, well within 80 ms window.
        out = _snap_to_vocal_onsets(tokens, timings, [1.04])
        self.assertAlmostEqual(out[0][0], 1.04, places=6)
        # Duration preserved (1.34 - 1.04 == 1.30 - 1.00).
        self.assertAlmostEqual(out[0][1] - out[0][0], 0.30, places=6)

    def test_ignores_distant_onsets(self) -> None:
        from pipeline.lyrics_aligner import _snap_to_vocal_onsets

        tokens = [("a", 0)]
        timings = [(1.00, 1.30)]
        out = _snap_to_vocal_onsets(tokens, timings, [2.00])
        self.assertEqual(out, [(1.00, 1.30)])

    def test_never_crosses_previous_same_line_word(self) -> None:
        from pipeline.lyrics_aligner import _snap_to_vocal_onsets

        tokens = [("a", 0), ("b", 0)]
        # Word1 end = 1.30; word2 nominally starts 1.32. Onset at 1.25
        # would pull word2 backwards across word1.end → must be refused.
        timings = [(1.00, 1.30), (1.32, 1.50)]
        out = _snap_to_vocal_onsets(tokens, timings, [1.25])
        self.assertEqual(out[0], (1.00, 1.30))
        self.assertEqual(out[1], (1.32, 1.50))

    def test_no_onsets_is_identity(self) -> None:
        from pipeline.lyrics_aligner import _snap_to_vocal_onsets

        tokens = [("a", 0)]
        timings = [(1.0, 1.2)]
        self.assertEqual(
            _snap_to_vocal_onsets(tokens, timings, []), timings
        )


class TestSnapToLineAnchors(unittest.TestCase):
    def test_anchors_pin_first_word_and_shift_subsequent(self) -> None:
        from pipeline.lyrics_aligner import _snap_to_line_anchors

        tokens = [("hello", 0), ("world", 0), ("foo", 1), ("bar", 1)]
        timings = [(1.0, 1.3), (1.4, 1.7), (2.0, 2.3), (2.4, 2.7)]
        # Pin line 1 to 3.0 → both "foo" and "bar" shift by +1.0s.
        out = _snap_to_line_anchors(tokens, timings, {1: 3.0})
        self.assertEqual(out[0], (1.0, 1.3))
        self.assertEqual(out[1], (1.4, 1.7))
        self.assertAlmostEqual(out[2][0], 3.0, places=6)
        self.assertAlmostEqual(out[3][0], 3.4, places=6)

    def test_no_anchors_is_identity(self) -> None:
        from pipeline.lyrics_aligner import _snap_to_line_anchors

        tokens = [("a", 0)]
        timings = [(1.0, 1.2)]
        self.assertEqual(_snap_to_line_anchors(tokens, timings, {}), timings)


class TestSchemaV3SerialisesScoreAndManualFlag(unittest.TestCase):
    from pipeline.lyrics_aligner import AlignedWord, LYRICS_ALIGNED_SCHEMA_VERSION

    def test_schema_version_is_3(self) -> None:
        from pipeline.lyrics_aligner import LYRICS_ALIGNED_SCHEMA_VERSION

        self.assertEqual(LYRICS_ALIGNED_SCHEMA_VERSION, 3)

    def test_aligned_word_serialises_score_when_present(self) -> None:
        from pipeline.lyrics_aligner import AlignedWord

        w = AlignedWord(
            word="hi", line_idx=0, t_start=1.0, t_end=1.2, score=0.87
        )
        self.assertEqual(
            w.to_dict(),
            {"word": "hi", "line_idx": 0, "t_start": 1.0, "t_end": 1.2, "score": 0.87},
        )

    def test_aligned_word_omits_score_when_none(self) -> None:
        from pipeline.lyrics_aligner import AlignedWord

        w = AlignedWord(word="hi", line_idx=0, t_start=1.0, t_end=1.2)
        self.assertNotIn("score", w.to_dict())


class TestDeriveLineAnchorsFromTranscription(unittest.TestCase):
    """``_derive_line_anchors_from_transcription`` NW-matches whisper-
    transcribed words to user tokens. Anchors come from the first
    matching token on the line (exact time when it's the line's first
    token; back-estimated otherwise)."""

    def _helper(self):
        from pipeline.lyrics_aligner import _derive_line_anchors_from_transcription

        return _derive_line_anchors_from_transcription

    def test_empty_inputs_return_empty_dict(self) -> None:
        fn = self._helper()
        self.assertEqual(fn([], []), {})
        self.assertEqual(
            fn([("hello", 0)], []), {}
        )
        self.assertEqual(
            fn([], [_WhisperWord(word="hi", t_start=0.0, t_end=0.1)]),
            {},
        )

    def test_first_word_match_produces_anchor(self) -> None:
        fn = self._helper()
        user_tokens = [
            ("Hello", 0), ("world", 0),
            ("goodbye", 1), ("friend", 1),
        ]
        whisper = [
            _WhisperWord(word="hello", t_start=1.2, t_end=1.4),
            _WhisperWord(word="world", t_start=1.5, t_end=1.8),
            _WhisperWord(word="goodbye", t_start=5.1, t_end=5.4),
            _WhisperWord(word="friend", t_start=5.5, t_end=5.9),
        ]
        anchors = fn(user_tokens, whisper)
        self.assertAlmostEqual(anchors[0], 1.2)
        self.assertAlmostEqual(anchors[1], 5.1)

    def test_user_supplied_anchors_are_never_overwritten(self) -> None:
        fn = self._helper()
        user_tokens = [("Hello", 0), ("world", 0)]
        whisper = [
            _WhisperWord(word="hello", t_start=1.2, t_end=1.4),
            _WhisperWord(word="world", t_start=1.5, t_end=1.8),
        ]
        existing = {0: 99.0}  # user already pinned this line elsewhere
        anchors = fn(user_tokens, whisper, existing_anchors=existing)
        # Line 0 is user-pinned — auto pass must skip it so the user wins.
        self.assertNotIn(0, anchors)

    def test_non_first_word_match_back_estimates_line_start(self) -> None:
        """When the first user token has no match but a later token on
        the same line does, we back-estimate the line start by
        subtracting ``local_pos * per_word_duration`` from the matched
        whisper word's ``t_start``. This gives us many more anchors on
        chopped vocals (where whisper drops/mangles the first syllable)
        while still keeping the math bounded — per-word duration is
        clamped to [0.18s, 0.60s] so a 2 s melisma on a later whisper
        word can't push the anchor catastrophically negative."""
        fn = self._helper()
        user_tokens = [
            ("prelude", 0), ("hello", 0), ("world", 0),
        ]
        whisper = [
            _WhisperWord(word="hello", t_start=10.0, t_end=10.3),
            _WhisperWord(word="world", t_start=10.4, t_end=10.7),
        ]
        anchors = fn(user_tokens, whisper)
        self.assertIn(0, anchors)
        # hello is local_pos=1 on line 0; per-word dur clamped to 0.3
        # (actual whisper dur) so anchor ≈ 10.0 - 1 * 0.3 = 9.7s.
        self.assertAlmostEqual(anchors[0], 9.7, places=2)

    def test_back_estimate_clamped_to_zero(self) -> None:
        """A near-start match whose back-estimate would go negative must
        clamp to 0, not produce a bogus time before the song starts."""
        fn = self._helper()
        user_tokens = [
            ("the", 0), ("hello", 0),
        ]
        whisper = [_WhisperWord(word="hello", t_start=0.05, t_end=0.25)]
        anchors = fn(user_tokens, whisper)
        self.assertIn(0, anchors)
        self.assertGreaterEqual(anchors[0], 0.0)

    def test_misheard_words_dont_produce_false_anchors(self) -> None:
        fn = self._helper()
        user_tokens = [("foo", 0), ("bar", 0)]
        # Whisper misheard both (and "cat"/"dog" are too different from
        # "foo"/"bar" for fuzzy matching) → no anchors.
        whisper = [
            _WhisperWord(word="cat", t_start=1.0, t_end=1.2),
            _WhisperWord(word="dog", t_start=1.3, t_end=1.6),
        ]
        self.assertEqual(fn(user_tokens, whisper), {})

    def test_punctuation_and_case_folded_for_matching(self) -> None:
        fn = self._helper()
        user_tokens = [("Hello,", 0)]
        whisper = [_WhisperWord(word="HELLO", t_start=4.2, t_end=4.5)]
        anchors = fn(user_tokens, whisper)
        self.assertAlmostEqual(anchors[0], 4.2)

    def test_fuzzy_match_rescues_chopped_ending(self) -> None:
        """Chopped / sung vocals frequently drop final consonants —
        whisper writes 'comin' for "coming", 'runnin' for "running",
        'til' for "till". Common-prefix matching must rescue these so
        the line gets anchored; without fuzzy matching this whole line
        would silently go unanchored on sung material."""
        fn = self._helper()
        user_tokens = [
            ("coming", 0), ("home", 0),
        ]
        whisper = [
            _WhisperWord(word="comin", t_start=2.0, t_end=2.3),
            _WhisperWord(word="home", t_start=2.4, t_end=2.8),
        ]
        anchors = fn(user_tokens, whisper)
        self.assertIn(0, anchors)
        # "coming" ↔ "comin" is a non-exact match; the fuzzy pathway
        # still pins the line at whisper's t_start for the first token.
        self.assertAlmostEqual(anchors[0], 2.0, places=2)


class TestFilterConfidentWhisperWords(unittest.TestCase):
    """Confidence filter dropped before either (a) deriving line
    anchors from whisper's transcription or (b) persisting ``whisper_words``
    to ``lyrics.aligned.json`` for the editor overlay. Forced alignment
    always places tokens somewhere in its audio window even when the
    token wasn't really spoken — the resulting low-score timings used
    to drift onto unrelated audio and make the ghost-text overlay
    unreadable, so we drop them upstream."""

    def _fn(self):
        from pipeline.lyrics_aligner import _filter_confident_whisper_words

        return _filter_confident_whisper_words

    def test_drops_none_score_words(self) -> None:
        """``_extract_whisper_words`` sets ``score=None`` on any word
        whose timing was linearly interpolated because CTC didn't emit
        one. On long segments that position is basically random, so the
        filter must drop these entries."""
        fn = self._fn()
        ws = [
            _WhisperWord(word="ok", t_start=1.0, t_end=1.2, score=0.9),
            _WhisperWord(word="interp", t_start=1.3, t_end=1.5, score=None),
            _WhisperWord(word="good", t_start=1.6, t_end=1.9, score=0.7),
        ]
        kept = fn(ws)
        self.assertEqual([w.word for w in kept], ["ok", "good"])

    def test_drops_below_threshold(self) -> None:
        """Hallucinated words that CTC still technically aligns but with
        very low confidence are not real signal — drop them so they
        don't end up as ghost labels on chorus / instrumental audio."""
        fn = self._fn()
        ws = [
            _WhisperWord(word="real", t_start=0.0, t_end=0.3, score=0.80),
            _WhisperWord(word="junk", t_start=0.4, t_end=0.7, score=0.10),
            _WhisperWord(word="edge", t_start=0.8, t_end=1.0, score=0.25),
        ]
        kept = fn(ws)
        self.assertEqual([w.word for w in kept], ["real", "edge"])

    def test_threshold_is_parameterised(self) -> None:
        fn = self._fn()
        ws = [
            _WhisperWord(word="a", t_start=0.0, t_end=0.1, score=0.40),
            _WhisperWord(word="b", t_start=0.2, t_end=0.3, score=0.60),
        ]
        self.assertEqual([w.word for w in fn(ws, min_score=0.5)], ["b"])

    def test_empty_input(self) -> None:
        self.assertEqual(self._fn()([]), [])


class TestTokensAreFuzzyEqual(unittest.TestCase):
    """Fuzzy-equality predicate used when NW-matching user tokens against
    whisper's transcription. Has to be aggressive enough to catch typical
    sung-vocal near-misses ("were"↔"we're", "yo"↔"you", "comin"↔"coming")
    but conservative enough that short unrelated words ("a" vs "i",
    "the" vs "to") still don't match."""

    def _fn(self):
        from pipeline.lyrics_aligner import _tokens_are_fuzzy_equal

        return _tokens_are_fuzzy_equal

    def test_exact_match(self) -> None:
        self.assertTrue(self._fn()("hello", "hello"))

    def test_empty_strings_never_match(self) -> None:
        self.assertFalse(self._fn()("", ""))
        self.assertFalse(self._fn()("hello", ""))

    def test_levenshtein_1_same_length(self) -> None:
        fn = self._fn()
        self.assertTrue(fn("coming", "comint"))   # single substitution
        self.assertTrue(fn("hello", "hella"))

    def test_levenshtein_1_insertion(self) -> None:
        fn = self._fn()
        self.assertTrue(fn("were", "weren"))
        self.assertTrue(fn("coming", "comings"))

    def test_common_prefix_of_three_or_more(self) -> None:
        fn = self._fn()
        self.assertTrue(fn("comin", "coming"))
        self.assertTrue(fn("runnin", "running"))

    def test_short_tokens_require_exact_match(self) -> None:
        """Short words are high risk for false-positive matches — "a"
        and "i" would otherwise be Levenshtein-distance 1. We rule them
        out so unrelated filler words don't fuse."""
        fn = self._fn()
        self.assertFalse(fn("a", "i"))
        self.assertFalse(fn("to", "so"))
        self.assertFalse(fn("he", "we"))

    def test_large_distance_rejected(self) -> None:
        fn = self._fn()
        self.assertFalse(fn("hello", "world"))
        self.assertFalse(fn("friend", "cat"))


class TestAlignLyricsPolishFlagsAreDefaultOff(unittest.TestCase):
    """Regression guard: the two aggressive polish passes that degrade
    sung-vocal alignment (silero VAD segment refinement, onset snap) MUST
    be opt-in. Flipping them on-by-default regressed quality on the test
    track; this test pins them off so a future refactor can't silently
    re-enable them."""

    def test_align_lyrics_flags_default_false(self) -> None:
        import inspect

        from pipeline.lyrics_aligner import align_lyrics

        sig = inspect.signature(align_lyrics)
        self.assertIs(sig.parameters["use_silero_vad"].default, False)
        self.assertIs(sig.parameters["use_onset_snap"].default, False)
        # Transcription anchors, by contrast, are a strict quality win
        # (auto line pins derived from whisper's own transcription), so
        # they ARE on by default. Flipping this to False would erase the
        # main reason we do two CTC passes.
        self.assertIs(sig.parameters["use_transcription_anchors"].default, True)


class TestExtractSectionFingerprints(unittest.TestCase):
    """``_extract_section_fingerprints`` picks up to 3 characteristic
    phrases per ``---`` section. Unique phrases beat repeated ones so
    choruses can still be placed (via their shared hook appearing
    multiple times) but verses pin themselves on lines only they own."""

    def test_unique_phrase_wins_over_shared_one(self) -> None:
        from pipeline.lyrics_aligner import _extract_section_fingerprints

        def _mk(line_words: list[tuple[int, list[str]]]) -> tuple[
            list[tuple[str, int]], list[int]
        ]:
            # ``line_words`` is ``[(section_idx, [word, ...]), ...]``.
            tokens: list[tuple[str, int]] = []
            section_starts: list[int] = []
            prev_section = -1
            for sec, ws in line_words:
                if sec != prev_section:
                    section_starts.append(len(tokens))
                    prev_section = sec
                for w in ws:
                    tokens.append((w, sec))
            return tokens, section_starts

        tokens, section_starts = _mk(
            [
                (0, "now you really know me".split()),
                (1, "but now its frozen over".split()),
                (2, "now you really know me".split()),
            ]
        )
        per_section = _extract_section_fingerprints(tokens, section_starts)
        self.assertEqual(len(per_section), 3)
        # Section 1 ("but now its frozen over") must have at least one
        # uniquely-owned fingerprint (other_section_count == 0).
        self.assertTrue(
            any(fp.other_section_count == 0 for fp in per_section[1]),
            f"section 1 should have a unique fingerprint: {per_section[1]}",
        )
        # Section 0 and 2 share the same hook; their fingerprints must
        # report other_section_count >= 1 (the top picks are shared).
        shared_0 = [fp.other_section_count for fp in per_section[0]]
        shared_2 = [fp.other_section_count for fp in per_section[2]]
        self.assertGreaterEqual(min(shared_0), 1)
        self.assertGreaterEqual(min(shared_2), 1)

    def test_short_section_returns_empty(self) -> None:
        from pipeline.lyrics_aligner import _extract_section_fingerprints

        tokens = [("hi", 0), ("there", 0)]
        out = _extract_section_fingerprints(tokens, [0])
        # Only 2 tokens < _FINGERPRINT_MIN_LEN (3) → no phrases possible.
        self.assertEqual(out, [[]])

    def test_caps_at_three_fingerprints_per_section(self) -> None:
        from pipeline.lyrics_aligner import (
            _extract_section_fingerprints,
            _FINGERPRINT_MAX_PER_SECTION,
        )

        tokens = [(f"w{i}", 0) for i in range(20)]
        out = _extract_section_fingerprints(tokens, [0])
        self.assertEqual(len(out), 1)
        self.assertLessEqual(len(out[0]), _FINGERPRINT_MAX_PER_SECTION)

    def test_prefers_longer_phrases_at_equal_uniqueness(self) -> None:
        from pipeline.lyrics_aligner import _extract_section_fingerprints

        tokens = [(w, 0) for w in ["alpha", "beta", "gamma", "delta", "epsilon"]]
        out = _extract_section_fingerprints(tokens, [0])
        self.assertEqual(len(out), 1)
        # With a single section every phrase is unique; the first pick
        # should be the longest (5-gram) one.
        top = out[0][0]
        self.assertEqual(len(top.phrase), 5)


class TestFlattenWhisperWordsForFingerprints(unittest.TestCase):
    """``_flatten_whisper_words_for_fingerprints`` must prefer precise
    CTC timings over segment-level ones, and strip punctuation via
    ``_normalise_token`` before emitting."""

    def test_prefers_transcription_words_when_available(self) -> None:
        from pipeline.lyrics_aligner import (
            _flatten_whisper_words_for_fingerprints,
            _WhisperWord,
        )

        tx = [
            _WhisperWord(word="Hello,", t_start=1.0, t_end=1.4, score=0.9),
            _WhisperWord(word="World!", t_start=1.5, t_end=1.9, score=0.95),
        ]
        out = _flatten_whisper_words_for_fingerprints([], tx)
        self.assertEqual(out, [("hello", 1.0), ("world", 1.5)])

    def test_falls_back_to_segment_words_with_interpolated_times(self) -> None:
        from pipeline.lyrics_aligner import (
            _flatten_whisper_words_for_fingerprints,
            _WhisperSegment,
        )

        segs = [
            _WhisperSegment(
                idx=0, t_start=10.0, t_end=12.0, words=("one", "two", "three", "four")
            ),
        ]
        out = _flatten_whisper_words_for_fingerprints(segs, [])
        # 4 words spread uniformly over a 2 s segment → 0.5 s apart.
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[0][1], 10.0)
        self.assertAlmostEqual(out[1][1], 10.5)
        self.assertAlmostEqual(out[2][1], 11.0)
        self.assertAlmostEqual(out[3][1], 11.5)


class TestFindFingerprintMatchesInTranscript(unittest.TestCase):
    """Fuzzy-matching fingerprint phrases against a flat whisper word
    stream. Must return plausible hits with timestamps at the first
    matched whisper word."""

    def _fps(self, section_idx: int, *phrases: str):
        from pipeline.lyrics_aligner import _SectionFingerprint

        return [
            _SectionFingerprint(
                section_idx=section_idx,
                phrase=tuple(p.split()),
                other_section_count=0,
            )
            for p in phrases
        ]

    def test_exact_phrase_matches_at_first_word_time(self) -> None:
        from pipeline.lyrics_aligner import (
            _find_fingerprint_matches_in_transcript,
        )

        whisper = [
            ("intro", 0.0), ("noise", 0.5), ("now", 5.0), ("you", 5.3),
            ("really", 5.7), ("know", 6.0), ("me", 6.3),
        ]
        fps = self._fps(0, "now you really")
        hits = _find_fingerprint_matches_in_transcript(fps, whisper)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].section_idx, 0)
        self.assertAlmostEqual(hits[0].t_start, 5.0)
        self.assertEqual(hits[0].score, 1.0)

    def test_tolerates_one_missed_word_in_a_three_gram(self) -> None:
        from pipeline.lyrics_aligner import (
            _find_fingerprint_matches_in_transcript,
        )

        whisper = [
            ("now", 1.0), ("blarg", 1.3), ("really", 1.6),
        ]
        fps = self._fps(0, "now you really")
        hits = _find_fingerprint_matches_in_transcript(fps, whisper)
        # 2/3 fuzzy matches = 0.666 >= 0.6 threshold.
        self.assertEqual(len(hits), 1)
        self.assertGreaterEqual(hits[0].score, 0.6)

    def test_returns_multiple_hits_for_repeated_chorus(self) -> None:
        from pipeline.lyrics_aligner import (
            _find_fingerprint_matches_in_transcript,
        )

        whisper = [
            # Chorus 1 at t=10
            ("now", 10.0), ("you", 10.3), ("really", 10.6),
            ("know", 10.9), ("me", 11.2),
            # Gap
            ("filler", 20.0),
            # Chorus 2 at t=30
            ("now", 30.0), ("you", 30.3), ("really", 30.6),
            ("know", 30.9), ("me", 31.2),
        ]
        fps = self._fps(0, "now you really know me")
        hits = _find_fingerprint_matches_in_transcript(fps, whisper)
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(hits[0].t_start, 10.0)
        self.assertAlmostEqual(hits[1].t_start, 30.0)

    def test_merges_overlapping_matches_within_half_second(self) -> None:
        from pipeline.lyrics_aligner import (
            _find_fingerprint_matches_in_transcript,
        )

        whisper = [
            ("now", 5.0), ("you", 5.2), ("really", 5.4),
            ("know", 5.6), ("me", 5.8),
        ]
        # Two different fingerprints from the SAME section that both
        # hit this region — the merge pass should collapse them.
        fps = self._fps(2, "now you really", "you really know")
        hits = _find_fingerprint_matches_in_transcript(fps, whisper)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].section_idx, 2)

    def test_rejects_match_spanning_too_much_audio_time(self) -> None:
        """Regression: CTC sometimes pins one phrase word to intro
        noise at t≈0 and the rest to the real vocal 20-30 s later.
        The sparse "match" spans half a minute and is NOT a real
        occurrence of the phrase. The matcher must reject it rather
        than anchor a section at t=0 on the bogus hit (this was the
        pathology in ``terminals/1.txt`` on the ``We were up till
        three…`` song, where section 0 got dragged to t=0.59 s)."""
        from pipeline.lyrics_aligner import (
            _find_fingerprint_matches_in_transcript,
        )

        whisper = [
            # First phrase word pinned by CTC onto intro noise at t=0.2.
            ("we", 0.2),
            # Real phrase starts much later — 25 s away.
            ("were", 25.1), ("up", 25.4), ("till", 25.7), ("three", 26.0),
        ]
        fps = self._fps(0, "we were up")
        hits = _find_fingerprint_matches_in_transcript(fps, whisper)
        # A valid match on ``we were up`` exists starting at t=25.1 s
        # (the CTC-split first "we" and the real trailing "were up"
        # are array-adjacent but 25 s apart — that spans far more
        # than the ``max(3.0, 3*2.0)=6.0 s`` budget for a 3-gram, so
        # the leading bogus hit must be discarded). What the matcher
        # *should* produce instead: a normal 2/3 fuzzy hit on the
        # real ``were up`` pair at t=25.1 that fits comfortably in
        # the span budget.
        # The anchor time (if any) must be within the real phrase.
        for h in hits:
            self.assertGreater(
                h.t_start, 10.0,
                f"match anchored on intro noise instead of real phrase: {h}",
            )

    def test_accepts_match_within_plausible_span_budget(self) -> None:
        """Sanity: phrases that span a few seconds (normal sung
        rate) still match after the span constraint is added."""
        from pipeline.lyrics_aligner import (
            _find_fingerprint_matches_in_transcript,
        )

        # 3-gram spanning ~2 s — well within the 6 s budget.
        whisper = [
            ("now", 10.0), ("you", 11.0), ("really", 12.0),
        ]
        fps = self._fps(0, "now you really")
        hits = _find_fingerprint_matches_in_transcript(fps, whisper)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].t_start, 10.0)


class TestAssignSectionsViaTemporalDP(unittest.TestCase):
    """DP assignment picks at most one time per section, in order,
    honouring user forced starts and maximising total match score."""

    def _match(self, section_idx: int, t: float, score: float = 1.0):
        from pipeline.lyrics_aligner import _FingerprintMatch

        return _FingerprintMatch(
            section_idx=section_idx, t_start=t, score=score
        )

    def test_no_matches_returns_all_none(self) -> None:
        from pipeline.lyrics_aligner import _assign_sections_via_temporal_dp

        out = _assign_sections_via_temporal_dp(3, [])
        self.assertEqual(out, [None, None, None])

    def test_picks_temporally_ordered_candidates(self) -> None:
        from pipeline.lyrics_aligner import _assign_sections_via_temporal_dp

        matches = [
            self._match(0, 5.0),
            self._match(1, 20.0),
            self._match(2, 60.0),
        ]
        out = _assign_sections_via_temporal_dp(3, matches)
        self.assertEqual(out, [5.0, 20.0, 60.0])

    def test_disambiguates_repeated_chorus_via_ordering(self) -> None:
        from pipeline.lyrics_aligner import _assign_sections_via_temporal_dp

        # Sections 0 and 2 are identical choruses with matches at both
        # chorus timestamps. The DP must place 0 at the earlier hit and
        # 2 at the later one — the opposite would violate monotonicity.
        matches = [
            self._match(0, 10.0, 1.0),
            self._match(0, 50.0, 1.0),
            self._match(1, 30.0, 1.0),
            self._match(2, 10.0, 1.0),
            self._match(2, 50.0, 1.0),
        ]
        out = _assign_sections_via_temporal_dp(3, matches)
        self.assertEqual(out, [10.0, 30.0, 50.0])

    def test_skips_section_with_only_low_score_matches(self) -> None:
        from pipeline.lyrics_aligner import _assign_sections_via_temporal_dp

        # Section 1 only has a sub-threshold match (below 0.5 floor).
        matches = [
            self._match(0, 5.0, 1.0),
            self._match(1, 10.0, 0.3),
            self._match(2, 20.0, 1.0),
        ]
        out = _assign_sections_via_temporal_dp(3, matches)
        self.assertEqual(out, [5.0, None, 20.0])

    def test_forced_anchor_wins_over_fingerprint_match(self) -> None:
        from pipeline.lyrics_aligner import _assign_sections_via_temporal_dp

        # Section 1 has a fingerprint match at 25s but the user pinned
        # it at 30s via ``[m:ss]``. The DP must honour the pin.
        matches = [
            self._match(0, 5.0),
            self._match(1, 25.0),
            self._match(2, 60.0),
        ]
        out = _assign_sections_via_temporal_dp(
            3, matches, forced_section_starts={1: 30.0}
        )
        self.assertEqual(out, [5.0, 30.0, 60.0])

    def test_forced_anchor_forces_neighbours_to_respect_ordering(self) -> None:
        from pipeline.lyrics_aligner import _assign_sections_via_temporal_dp

        # Section 1 pinned at 20s; section 0 only has a match at 30s
        # (after the pin). Monotonicity → section 0 must be skipped.
        matches = [
            self._match(0, 30.0),
            self._match(2, 50.0),
        ]
        out = _assign_sections_via_temporal_dp(
            3, matches, forced_section_starts={1: 20.0}
        )
        self.assertEqual(out, [None, 20.0, 50.0])


class TestBuildSectionWindowsFromFingerprints(unittest.TestCase):
    """Glue that converts DP-chosen section starts into
    ``whisperx.align``-shaped ``{text, start, end}`` windows, filling
    un-placed sections by token-weighted interpolation."""

    def _tokens(self, section_token_counts: list[int]) -> tuple[
        list[tuple[str, int]], list[int]
    ]:
        tokens: list[tuple[str, int]] = []
        section_starts: list[int] = []
        for k, n in enumerate(section_token_counts):
            section_starts.append(len(tokens))
            for i in range(n):
                tokens.append((f"s{k}w{i}", k))
        return tokens, section_starts

    def test_placed_sections_get_chosen_starts_and_next_start_as_end(self) -> None:
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        tokens, starts = self._tokens([3, 4, 2])
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 100.0, chosen_starts=[10.0, 30.0, 70.0]
        )
        self.assertEqual(len(windows), 3)
        self.assertAlmostEqual(windows[0]["start"], 10.0)
        self.assertAlmostEqual(windows[0]["end"], 30.0)
        self.assertAlmostEqual(windows[1]["start"], 30.0)
        self.assertAlmostEqual(windows[1]["end"], 70.0)
        self.assertAlmostEqual(windows[2]["start"], 70.0)
        self.assertAlmostEqual(windows[2]["end"], 100.0)
        # Text per window should be that section's joined words.
        self.assertEqual(windows[0]["text"], "s0w0 s0w1 s0w2")
        self.assertEqual(windows[1]["text"], "s1w0 s1w1 s1w2 s1w3")
        self.assertEqual(windows[2]["text"], "s2w0 s2w1")

    def test_unplaced_middle_section_interpolates_by_tokens(self) -> None:
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        # Three sections of 4 tokens each; section 1 left un-placed.
        tokens, starts = self._tokens([4, 4, 4])
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 120.0, chosen_starts=[0.0, None, 60.0]
        )
        # Section 1 should start somewhere strictly between 0 and 60.
        self.assertGreater(windows[1]["start"], 0.0)
        self.assertLess(windows[1]["start"], 60.0)
        # Monotonic.
        self.assertLess(windows[0]["end"], windows[1]["end"])
        self.assertLess(windows[1]["end"], windows[2]["end"])

    def test_user_line_anchor_overrides_chosen_start(self) -> None:
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        tokens, starts = self._tokens([3, 3, 3])
        # Each section's tokens share its own line_idx (see _tokens helper),
        # so line 1 is the first line of section 1.
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 60.0,
            chosen_starts=[0.0, 20.0, 40.0],
            line_anchors={1: 25.0},
        )
        self.assertAlmostEqual(windows[1]["start"], 25.0)

    def test_all_unplaced_distributes_proportionally(self) -> None:
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        # No matches / no anchors → proportional fallback.
        tokens, starts = self._tokens([1, 2, 1])
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 100.0, chosen_starts=[None, None, None]
        )
        # Section 1 is twice as long as 0 / 2, so it should span 50 s
        # of the 100 s audio, starting at 25 s.
        self.assertAlmostEqual(windows[0]["start"], 0.0, places=3)
        self.assertAlmostEqual(windows[1]["start"], 25.0, places=3)
        self.assertAlmostEqual(windows[2]["start"], 75.0, places=3)

    def test_enforces_monotonic_starts_when_chosen_violates(self) -> None:
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        tokens, starts = self._tokens([2, 2, 2])
        # Pathological input: section 2's chosen start is before 1's.
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 100.0, chosen_starts=[0.0, 50.0, 40.0]
        )
        self.assertLessEqual(windows[0]["start"], windows[1]["start"])
        self.assertLessEqual(windows[1]["start"], windows[2]["start"])

    def test_leading_unplaced_section_respects_token_budget(self) -> None:
        """Regression: when section 0 has no fingerprint match and no
        user anchor but section 1 is pinned, section 0 should NOT
        span the whole [0, section_1_start] range. It should sit in
        a budget-capped window immediately before section 1 so CTC
        isn't asked to stretch its text across tens of seconds of
        intro silence (see ``terminals/1.txt`` regression where
        section 0 with 4 tokens ended up with a 22 s window)."""
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        # Section 0 has 4 tokens → budget ≈ 4 * 0.6 = 2.4 s.
        # Section 1 is forced at t=22 via chosen_starts.
        tokens, starts = self._tokens([4, 3, 3])
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 120.0,
            chosen_starts=[None, 22.0, 60.0],
        )
        # Section 0 must start close to section 1, NOT near t=0.
        self.assertGreater(windows[0]["start"], 15.0)
        self.assertLess(windows[0]["start"], 22.0)
        # And section 1 is still exactly where it was pinned.
        self.assertAlmostEqual(windows[1]["start"], 22.0)

    def test_trailing_unplaced_sections_respect_token_budget(self) -> None:
        """Mirror of the leading case: pinning mid-song and leaving
        the last sections unplaced shouldn't stretch them across the
        entire remaining audio — their window is bounded by the
        total tokens they cover × the per-token budget."""
        from pipeline.lyrics_aligner import _build_section_windows_from_fingerprints

        tokens, starts = self._tokens([3, 3, 2, 2])
        # Section 0 at 0, section 1 at 60. Sections 2 and 3 unplaced.
        # Total trailing tokens = 4 → budget ≈ 4 * 0.6 = 2.4 s,
        # plus section-1's own contribution. Windows must fit inside
        # that budget, not span 120..240.
        windows = _build_section_windows_from_fingerprints(
            tokens, starts, 240.0,
            chosen_starts=[0.0, 60.0, None, None],
        )
        # Section 3's start must be well short of the audio end.
        self.assertGreater(windows[2]["start"], 60.0)
        self.assertLess(windows[3]["start"], 75.0)


class TestWidenSegmentsForCtc(unittest.TestCase):
    """Stretch each whisper segment's ``t_end`` into the silence
    before the next segment so wav2vec2 CTC has breathing room to
    place per-word timings at real phoneme positions instead of
    compressing them into faster-whisper's aggressively-truncated
    end timestamps. Without this, a chorus like "Roll with life"
    repeated three times stacks on top of itself instead of spreading
    across the audio it actually sings over (see editor ghost-text
    clustering bug)."""

    def _seg(self, idx: int, t_start: float, t_end: float, words=()):
        from pipeline.lyrics_aligner import _WhisperSegment
        return _WhisperSegment(
            idx=idx, t_start=t_start, t_end=t_end, words=tuple(words)
        )

    def test_stretches_truncated_end_up_to_next_segment_start(self) -> None:
        """The core case: whisper truncated segment 0 at t=18 even
        though the chorus keeps singing until the next pyannote
        region starts at t=35. CTC needs access to that 17 s of
        audio."""
        from pipeline.lyrics_aligner import _widen_segments_for_ctc
        segments = [
            self._seg(0, 10.0, 18.0, ("roll", "with", "life",
                                      "roll", "with", "life",
                                      "roll", "with", "life")),
            self._seg(1, 35.0, 40.0, ("a", "night", "like", "this")),
        ]
        out = _widen_segments_for_ctc(segments, audio_duration=60.0)
        self.assertAlmostEqual(out[0].t_start, 10.0)
        # Ends right before segment 1's start (minus tiny gap).
        self.assertAlmostEqual(out[0].t_end, 35.0 - 0.05, places=3)
        self.assertAlmostEqual(out[1].t_start, 35.0)

    def test_last_segment_extended_to_audio_end(self) -> None:
        from pipeline.lyrics_aligner import _widen_segments_for_ctc
        segments = [self._seg(0, 10.0, 12.0, ("hello",))]
        out = _widen_segments_for_ctc(segments, audio_duration=60.0)
        self.assertAlmostEqual(out[0].t_end, 60.0)

    def test_does_not_shrink_already_wide_segments(self) -> None:
        """If faster-whisper gave a generous ``end``, don't pull it
        back — CTC is happier with more audio than less."""
        from pipeline.lyrics_aligner import _widen_segments_for_ctc
        segments = [
            self._seg(0, 10.0, 30.0, ("x",)),  # wider than next.start
            self._seg(1, 20.0, 25.0, ("y",)),  # overlaps segment 0
        ]
        out = _widen_segments_for_ctc(segments, audio_duration=60.0)
        # Should not be shrunk from 30 back to 20-eps.
        self.assertGreaterEqual(out[0].t_end, 30.0)

    def test_never_overlaps_next_segment_when_gap_available(self) -> None:
        from pipeline.lyrics_aligner import _widen_segments_for_ctc
        segments = [
            self._seg(0, 10.0, 12.0, ("a",)),
            self._seg(1, 20.0, 25.0, ("b",)),
        ]
        out = _widen_segments_for_ctc(
            segments, audio_duration=60.0, gap_before_next=0.1,
        )
        self.assertLess(out[0].t_end, out[1].t_start)
        self.assertAlmostEqual(out[0].t_end, 19.9, places=3)

    def test_preserves_segment_text_and_index(self) -> None:
        from pipeline.lyrics_aligner import _widen_segments_for_ctc
        segments = [
            self._seg(0, 10.0, 12.0, ("hello", "world")),
            self._seg(1, 20.0, 25.0, ("foo",)),
        ]
        out = _widen_segments_for_ctc(segments, audio_duration=60.0)
        self.assertEqual(out[0].idx, 0)
        self.assertEqual(out[0].words, ("hello", "world"))
        self.assertEqual(out[1].idx, 1)
        self.assertEqual(out[1].words, ("foo",))

    def test_empty_input_returns_empty(self) -> None:
        from pipeline.lyrics_aligner import _widen_segments_for_ctc
        self.assertEqual(
            _widen_segments_for_ctc([], audio_duration=60.0), [],
        )


class TestFirstDenseVocalActivityTime(unittest.TestCase):
    """Density-based detector for where real singing starts in the
    whisper transcript. Used as the floor for leading
    forced-alignment windows to stop user lyrics from being placed on
    silent intros (see ``_apply_vocal_activity_floor``)."""

    def _w(self, word: str, t: float, score: float = 0.8):
        from pipeline.lyrics_aligner import _WhisperWord
        return _WhisperWord(word=word, t_start=t, t_end=t + 0.2, score=score)

    def test_empty_input_returns_none(self) -> None:
        from pipeline.lyrics_aligner import _first_dense_vocal_activity_time
        self.assertIsNone(_first_dense_vocal_activity_time([]))

    def test_fewer_words_than_min_returns_none(self) -> None:
        from pipeline.lyrics_aligner import _first_dense_vocal_activity_time
        words = [self._w("hello", 10.0), self._w("world", 10.5)]
        self.assertIsNone(_first_dense_vocal_activity_time(words, min_words=3))

    def test_skips_isolated_stray_word_at_intro(self) -> None:
        """The exact pathology from the user's screenshot: whisper
        heard a stray "We" at t=0 from a vocal chop, then nothing
        until real singing starts at t=50. Must skip the chop."""
        from pipeline.lyrics_aligner import _first_dense_vocal_activity_time
        words = [
            self._w("we", 0.2),
            # Long silence until real vocals.
            self._w("were", 50.0),
            self._w("up", 50.3),
            self._w("till", 50.7),
            self._w("three", 51.0),
        ]
        t = _first_dense_vocal_activity_time(words)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 50.0, places=2)

    def test_dense_intro_is_accepted_as_true_start(self) -> None:
        """If whisper genuinely heard 3+ words in the first 5 s,
        that's real singing — floor must not skip past it."""
        from pipeline.lyrics_aligner import _first_dense_vocal_activity_time
        words = [
            self._w("oh", 0.5),
            self._w("hello", 1.0),
            self._w("world", 1.5),
            self._w("this", 2.0),
        ]
        self.assertAlmostEqual(
            _first_dense_vocal_activity_time(words), 0.5, places=2,
        )

    def test_low_confidence_words_excluded_by_default(self) -> None:
        """A word with CTC score < 0.25 is probably a hallucination;
        the density check must not count it."""
        from pipeline.lyrics_aligner import _first_dense_vocal_activity_time
        words = [
            self._w("garbage", 0.2, score=0.1),
            self._w("noise", 1.0, score=0.1),
            self._w("junk", 2.0, score=0.1),
            self._w("real", 50.0, score=0.8),
            self._w("vocals", 50.5, score=0.8),
            self._w("now", 51.0, score=0.8),
        ]
        self.assertAlmostEqual(
            _first_dense_vocal_activity_time(words), 50.0, places=2,
        )


class TestApplyVocalActivityFloor(unittest.TestCase):
    """Post-build guard on ``forced_segments``: shift segment 0's
    ``start`` forward to the first-dense-vocal-run time when the
    user hasn't anchored anything earlier. Regression target: user
    loads a song with an instrumental intro and whisper only picks
    up a single stray word near t=0, but the aligner still placed
    30+ user words onto that silent intro."""

    def _w(self, word: str, t: float, score: float = 0.8):
        from pipeline.lyrics_aligner import _WhisperWord
        return _WhisperWord(word=word, t_start=t, t_end=t + 0.2, score=score)

    def test_shifts_segment_zero_forward_to_floor_when_no_user_anchor(self) -> None:
        from pipeline.lyrics_aligner import _apply_vocal_activity_floor
        # Whisper: stray word at 0.2, real run at 50+.
        words = [
            self._w("we", 0.2),
            self._w("were", 50.0), self._w("up", 50.3),
            self._w("till", 50.7), self._w("three", 51.0),
        ]
        # Forced segment 0 naively starts at t=0.5.
        segments = [{"text": "we were up till three", "start": 0.5, "end": 60.0}]
        out, floor, shift = _apply_vocal_activity_floor(
            segments, transcription_words=words, line_anchors={}
        )
        self.assertIsNotNone(floor)
        self.assertIsNotNone(shift)
        assert floor is not None and shift is not None
        self.assertGreater(shift, 40.0)
        # New start is at or slightly before the floor (buffer_sec).
        self.assertGreater(out[0]["start"], 45.0)
        self.assertLessEqual(out[0]["start"], 50.0)

    def test_respects_user_anchor_earlier_than_floor(self) -> None:
        """If the user pinned line 0 at t=1.0 s via ``[0:01]``, they
        believe something sings there. The heuristic must stand down
        even if whisper didn't pick anything up until t=50 s."""
        from pipeline.lyrics_aligner import _apply_vocal_activity_floor
        words = [
            self._w("real", 50.0), self._w("vocals", 50.5),
            self._w("later", 51.0),
        ]
        segments = [{"text": "I whisper softly", "start": 0.5, "end": 60.0}]
        out, _floor, shift = _apply_vocal_activity_floor(
            segments,
            transcription_words=words,
            line_anchors={0: 1.0},
        )
        self.assertIsNone(shift)
        self.assertAlmostEqual(out[0]["start"], 0.5)

    def test_no_op_when_transcript_too_sparse_to_judge(self) -> None:
        from pipeline.lyrics_aligner import _apply_vocal_activity_floor
        # Only two words total → no dense run detectable.
        words = [self._w("hello", 1.0), self._w("world", 30.0)]
        segments = [{"text": "whatever", "start": 0.5, "end": 60.0}]
        out, floor, shift = _apply_vocal_activity_floor(
            segments, transcription_words=words, line_anchors={}
        )
        self.assertIsNone(floor)
        self.assertIsNone(shift)
        self.assertAlmostEqual(out[0]["start"], 0.5)

    def test_no_shift_when_current_start_close_to_floor(self) -> None:
        """If segment 0 already starts near real vocals, don't
        fiddle with it — CTC will do a better job than our heuristic
        within the first couple of seconds of noise."""
        from pipeline.lyrics_aligner import _apply_vocal_activity_floor
        words = [
            self._w("oh", 15.0), self._w("hello", 15.5),
            self._w("world", 16.0),
        ]
        segments = [{"text": "oh hello world", "start": 14.5, "end": 30.0}]
        out, _floor, shift = _apply_vocal_activity_floor(
            segments, transcription_words=words, line_anchors={}
        )
        self.assertIsNone(shift)
        self.assertAlmostEqual(out[0]["start"], 14.5)

    def test_does_not_invert_segment_when_end_too_close_to_floor(self) -> None:
        """Pathological case: user pinned segment 1 right after
        segment 0 (so segment 0's end is, say, t=3.0) but whisper
        says real singing only starts at t=50. Segment 0 can't be
        shifted to t=50 without inverting; leave it alone rather
        than producing negative-length nonsense."""
        from pipeline.lyrics_aligner import _apply_vocal_activity_floor
        words = [
            self._w("real", 50.0), self._w("vocals", 50.5),
            self._w("here", 51.0),
        ]
        segments = [{"text": "first", "start": 0.5, "end": 3.0}]
        out, _floor, shift = _apply_vocal_activity_floor(
            segments, transcription_words=words, line_anchors={}
        )
        self.assertIsNone(shift)
        self.assertAlmostEqual(out[0]["start"], 0.5)
        # Segment untouched.
        self.assertAlmostEqual(out[0]["end"], 3.0)


class TestSectionFingerprintEndToEnd(unittest.TestCase):
    """End-to-end test of the fingerprint flow on the pathology that
    motivated the rewrite: chorus repeats + a noisy intro where
    whisper hallucinates chorus-shaped tokens at ``t=0``. The
    fingerprint DP must NOT anchor the user's first section at
    ``t=0`` just because the intro noise looks chorus-ish."""

    def test_intro_hallucination_does_not_drag_first_section_to_zero(self) -> None:
        from pipeline.lyrics_aligner import (
            _assign_sections_via_temporal_dp,
            _extract_section_fingerprints,
            _find_fingerprint_matches_in_transcript,
            _flatten_whisper_words_for_fingerprints,
            _WhisperWord,
        )

        # Three user sections: verse → pre-chorus → chorus.
        user_tokens: list[tuple[str, int]] = []
        lines = [
            # Section 0: verse (unique phrasing)
            "they told me not to go so now im really going",
            # Section 1: pre-chorus
            "my blood was never cold but now its frozen over",
            # Section 2: chorus
            "and you dont have to know but now you really know me",
        ]
        section_starts = []
        for li, line in enumerate(lines):
            section_starts.append(len(user_tokens))
            for w in line.split():
                user_tokens.append((w, li))
        # Compress to just three section boundaries on the token indices.
        section_starts = [0, len(lines[0].split()), len(lines[0].split()) + len(lines[1].split())]

        # Whisper transcript:
        #   * intro noise at t=0..4 that happens to contain "now" and
        #     "really" — weak chorus-shaped hallucination.
        #   * real verse content starting at t=15
        #   * real pre-chorus at t=40
        #   * real chorus at t=70
        def _words(pairs: list[tuple[str, float]]) -> list[_WhisperWord]:
            return [_WhisperWord(word=w, t_start=t, t_end=t + 0.3, score=0.8) for w, t in pairs]

        tx = _words(
            [
                ("now", 0.2), ("really", 1.0), ("nah", 2.0), ("uh", 3.5),
                ("they", 15.0), ("told", 15.3), ("me", 15.6), ("not", 15.9),
                ("to", 16.2), ("go", 16.5),
                ("my", 40.0), ("blood", 40.3), ("was", 40.6), ("never", 40.9),
                ("cold", 41.2),
                ("and", 70.0), ("you", 70.3), ("dont", 70.6), ("have", 70.9),
                ("to", 71.2), ("know", 71.5),
            ]
        )

        fingerprints = _extract_section_fingerprints(user_tokens, section_starts)
        flat = _flatten_whisper_words_for_fingerprints([], tx)
        flat_fps = [fp for sec in fingerprints for fp in sec]
        matches = _find_fingerprint_matches_in_transcript(flat_fps, flat)
        chosen = _assign_sections_via_temporal_dp(3, matches)

        # Section 0 (verse) must anchor on the real verse-start at ~15s,
        # NOT on the intro noise at t=0. Section 1 ~40s; section 2 ~70s.
        self.assertIsNotNone(chosen[0])
        assert chosen[0] is not None
        self.assertGreaterEqual(
            chosen[0], 10.0,
            f"verse anchored too early — probably chose intro noise: {chosen}",
        )
        # Strict temporal order.
        times = [t for t in chosen if t is not None]
        self.assertEqual(times, sorted(times))

    def test_repeated_chorus_sections_are_assigned_different_times(self) -> None:
        from pipeline.lyrics_aligner import (
            _assign_sections_via_temporal_dp,
            _extract_section_fingerprints,
            _find_fingerprint_matches_in_transcript,
            _flatten_whisper_words_for_fingerprints,
            _WhisperWord,
        )

        # Two identical chorus sections separated by a pre-chorus —
        # the classic "Now you really know me" repeat.
        chorus = "now you really know me now you really know me"
        prechorus = "but now its frozen over frozen over"
        user_tokens: list[tuple[str, int]] = []
        section_starts = [0]
        for li, line in enumerate([chorus, prechorus, chorus]):
            section_starts.append(section_starts[-1] + len(line.split()) if li < 2 else section_starts[-1])
            # Rebuild section_starts cleanly below.
        # Rebuild cleanly.
        user_tokens = []
        section_starts = []
        for li, line in enumerate([chorus, prechorus, chorus]):
            section_starts.append(len(user_tokens))
            for w in line.split():
                user_tokens.append((w, li))

        tx = [
            _WhisperWord(word=w, t_start=t, t_end=t + 0.3, score=0.8)
            for w, t in [
                # Chorus 1 around t=10
                ("now", 10.0), ("you", 10.2), ("really", 10.4), ("know", 10.6), ("me", 10.8),
                # Pre-chorus around t=30
                ("but", 30.0), ("now", 30.2), ("its", 30.4),
                ("frozen", 30.6), ("over", 30.8),
                # Chorus 2 around t=60
                ("now", 60.0), ("you", 60.2), ("really", 60.4), ("know", 60.6), ("me", 60.8),
            ]
        ]

        fingerprints = _extract_section_fingerprints(user_tokens, section_starts)
        flat = _flatten_whisper_words_for_fingerprints([], tx)
        flat_fps = [fp for sec in fingerprints for fp in sec]
        matches = _find_fingerprint_matches_in_transcript(flat_fps, flat)
        chosen = _assign_sections_via_temporal_dp(3, matches)

        # All three sections placed; first chorus early, repeat late.
        self.assertIsNotNone(chosen[0])
        self.assertIsNotNone(chosen[2])
        assert chosen[0] is not None and chosen[2] is not None
        self.assertLess(chosen[0], 20.0)
        self.assertGreater(chosen[2], 50.0)

    def test_ctc_split_phrase_does_not_drag_section_zero_to_intro(self) -> None:
        """Regression reproducing the ``terminals/1.txt`` pathology:
        CTC aligned the first word of section 0's phrase onto intro
        noise at t≈0 and the rest onto the real vocal at t≈25 s. The
        matcher's temporal-span guard must reject the 25 s-wide
        "match", and the window builder's token-budget fallback must
        then place section 0 just before the user-pinned section 1
        (t=22 s) instead of defaulting to t=0."""
        from pipeline.lyrics_aligner import (
            _assign_sections_via_temporal_dp,
            _build_section_windows_from_fingerprints,
            _extract_section_fingerprints,
            _find_fingerprint_matches_in_transcript,
            _flatten_whisper_words_for_fingerprints,
            _WhisperWord,
        )

        # User lyrics: section 0 has 4 tokens (short intro phrase),
        # section 1 is a longer chorus pinned by the user at 22 s.
        user_tokens: list[tuple[str, int]] = []
        section_starts: list[int] = []
        section_lines = [
            ["we were up till"],           # section 0 (4 tokens)
            ["my blood was never cold"],   # section 1 (5 tokens, pinned)
        ]
        line_idx = 0
        for sec in section_lines:
            section_starts.append(len(user_tokens))
            for line in sec:
                for w in line.split():
                    user_tokens.append((w, line_idx))
                line_idx += 1

        # Whisper + CTC — first "we" stuck on intro noise at t≈0,
        # rest of the phrase at t≈25 s where the vocal really starts.
        tx = [
            _WhisperWord(word=w, t_start=t, t_end=t + 0.2, score=0.6)
            for w, t in [
                ("we", 0.2),
                ("were", 25.1), ("up", 25.4), ("till", 25.7),
                # Section 1 content at t=40 (past the user's 22 s
                # pin because CTC also mis-located it — irrelevant
                # to the test, but realistic).
                ("my", 40.0), ("blood", 40.3), ("was", 40.6),
                ("never", 40.9), ("cold", 41.2),
            ]
        ]

        fingerprints = _extract_section_fingerprints(user_tokens, section_starts)
        flat = _flatten_whisper_words_for_fingerprints([], tx)
        flat_fps = [fp for sec in fingerprints for fp in sec]
        matches = _find_fingerprint_matches_in_transcript(flat_fps, flat)

        # Under the old matcher this produced a match at t≈0.2 for
        # section 0. After the span fix, any section-0 hit must be
        # on the *real* phrase near t=25.
        for m in matches:
            if m.section_idx == 0:
                self.assertGreater(
                    m.t_start, 10.0,
                    f"section-0 match anchored on intro noise: {m}",
                )

        # User pinned section 1 at 22 s via a line anchor.
        chosen = _assign_sections_via_temporal_dp(
            2, matches, forced_section_starts={1: 22.0}
        )
        # Section 1 must honour the pin.
        self.assertEqual(chosen[1], 22.0)
        # Section 0 is either unplaced (monotonicity clash with the
        # 22 s pin — its real phrase is at 25 s > 22 s) or placed at
        # an impossible time. Either way, the window builder is the
        # one that must produce a sane answer.
        windows = _build_section_windows_from_fingerprints(
            user_tokens, section_starts, 120.0, chosen,
        )
        # Section 0's window must NOT start anywhere near t=0 —
        # the token budget caps it to a few seconds before 22 s.
        self.assertGreater(
            windows[0]["start"], 15.0,
            f"section 0 window still drifts to intro: {windows[0]}",
        )
        self.assertLess(windows[0]["start"], 22.0)
        self.assertAlmostEqual(windows[1]["start"], 22.0)


if __name__ == "__main__":
    unittest.main()
