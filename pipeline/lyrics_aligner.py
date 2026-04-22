"""
Lyrics aligner: align user-pasted lyrics to the demucs ``vocals.wav`` produced
by :mod:`pipeline.audio_analyzer` using WhisperX word-level timestamps.

The module runs WhisperX ``large-v3`` on the isolated vocal stem, forces word-
level alignment via a wav2vec2 phoneme model, and then maps the recognised
tokens onto the user's **exact** lyrics through a Needleman-Wunsch pass on a
normalized (NFKD, case-folded, punctuation-stripped) representation. The user's
original spelling / punctuation is preserved in the output; only timestamps
come from whisper.

Output is persisted atomically at ``cache/<song_hash>/lyrics.aligned.json``.
Cache reuse is keyed by ``(song_hash, lyrics_sha256)``: editing the lyrics
text invalidates the cache automatically, and ``force=True`` always re-runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline.audio_analyzer import VOCALS_WAV_NAME
from pipeline.gpu_memory import move_to_cpu, release_cuda_memory

LOGGER = logging.getLogger(__name__)

LYRICS_ALIGNED_JSON_NAME = "lyrics.aligned.json"
# v1: whisper transcription + NW-mapped user lyrics.
# v2: forced alignment against user lyrics (current default for auto-align).
# v3: adds per-word `score` (wav2vec2 CTC confidence, 0..1) and top-level
#     `manually_edited` flag so the visual editor can persist overrides
#     without the next Align click silently regenerating them.
LYRICS_ALIGNED_SCHEMA_VERSION = 3

# A line whose trimmed content *starts with* three hyphens is treated as an
# explicit section break in the pasted lyrics. Any text after the hyphens is
# a human-readable tag (e.g. ``--- chorus``, ``--- instrumental``) for your
# own notes; it does not change alignment behaviour. The marker is used ONLY
# as a forced-alignment segmentation hint: it splits the user's words into
# musician-authored sections so WhisperX cannot bleed words from one repeated
# section into the next when its VAD merges them. The marker lines themselves
# are dropped from ``lines`` and from the rendered typography output.
SECTION_MARKER = "---"
# A line is treated as a section marker if its trimmed content matches this
# regex. Legacy ``---`` with nothing after it keeps working; everything else
# after the hyphens is documentation. Internal whitespace around the tag is
# tolerated.
_SECTION_MARKER_RE = re.compile(r"^---(?:\s+.*)?$")

# Inline line-start timestamp anchor: ``[m:ss]`` / ``[m:ss.mmm]`` /
# ``[h:mm:ss]`` at the very start of a lyric line pins that line's first
# word to the given audio time. Forced alignment uses the anchor as a hard
# window bound so the user can correct misplaced sections without touching
# the editor. The bracket block is stripped from the rendered lyric text.
_LINE_ANCHOR_RE = re.compile(
    r"^\[\s*"
    r"(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)(?:\.(?P<ms>\d{1,3}))?"
    r"\s*\]\s*"
)

DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LANGUAGE = "en"
# Rate WhisperX expects on the audio it loads (:func:`whisperx.load_audio`
# is hard-coded to 16 kHz, regardless of the input sample rate).
WHISPERX_SAMPLE_RATE = 16_000

# Whisper was trained on spoken speech; its default decoding gates (no-speech
# head, compression-ratio heuristic, logprob cutoff) aggressively throw away
# content on *sung* vocals — held vowels read as silence, choruses that
# legitimately repeat "Now you really know me" six times register as
# "too compressed, must be a hallucination" and get dropped wholesale.
# These overrides keep whisper from silently discarding real singing. They
# are passed through :func:`whisperx.load_model` as ``asr_options`` and
# :data:`faster_whisper.transcribe.TranscriptionOptions.update` merges them
# with the library defaults, so we only list the keys we actually override.
_SUNG_VOCALS_ASR_OPTIONS: dict[str, Any] = {
    # Default 0.6: reject segment as silence when the no-speech head scores
    # above this AND avg_logprob is below ``log_prob_threshold``. Sustained
    # vowels / melismatic passages routinely trip the no-speech head on
    # singing even when the vocal is clearly audible; 0.15 leaves the gate
    # for genuine silence (whisperx's VAD already carved those out) while
    # saving held notes.
    "no_speech_threshold": 0.15,
    # Default 2.4: segments with a zlib compression ratio above this are
    # treated as repetitive hallucinations and retried at higher
    # temperature, eventually dropped. A real chorus that legitimately
    # repeats the same phrase 4-6 times has a very high compression ratio;
    # 100.0 disables the gate (we'd rather keep a truthful repeat than
    # false-positive it as a hallucination).
    "compression_ratio_threshold": 100.0,
    # Default -1.0: treat segment as low-quality and fall back when
    # avg_logprob drops below this. Sung vocals have intrinsically lower
    # per-token logprobs than clean speech; -2.0 stops the fallback from
    # blanking slow ballads / soft bridges.
    "log_prob_threshold": -2.0,
    # Default already False in whisperx 3.x, but set explicitly so an
    # upstream bump can't silently re-enable hallucinated-drift between
    # chunks (once whisper loses sync on a chorus, prior-token conditioning
    # makes it drift into wrong-chorus variants).
    "condition_on_previous_text": False,
}

# Pyannote VAD on a near-clean vocal stem: demucs has already isolated the
# vocal, so we bias toward keeping breathy / quiet passages rather than the
# speech-optimised thresholds that discard them. Both knobs are lower than
# the whisperx defaults (0.500 / 0.363).
_SUNG_VOCALS_VAD_OPTIONS: dict[str, Any] = {
    "vad_onset": 0.200,
    "vad_offset": 0.100,
}

# NOTE: We deliberately do NOT pass the user's pasted lyrics as
# whisper's ``initial_prompt``. Biasing the LM that way is tempting —
# it does raise the raw transcribed-word count — but on sung material
# with stuttery / glitchy intro vocals it causes whisper to hallucinate
# chorus-shaped phrases into non-lyrical noise. The resulting bogus
# transcription anchors then drag forced alignment to ``t = 0`` and
# smear every user lyric to the top of the song. Repeated-chorus
# disambiguation is handled downstream by section-level fingerprint
# matching (see ``_extract_section_fingerprints`` and friends) rather
# than LM priming, which is robust to bad transcription within a
# section as long as ONE characteristic phrase is heard anywhere
# inside it.

# Needleman-Wunsch scoring (normalized token comparison).
NW_MATCH_SCORE = 2
NW_MISMATCH_SCORE = -1
NW_GAP_SCORE = -1

# Post-alignment polish limits. WhisperX forced-alignment occasionally
# produces a pathological ``t_end`` on a sustained note or a held syllable
# where the word stays on screen for many seconds past the vocal. Capping
# single-word duration keeps the outro fade from lingering over subsequent
# words while still leaving room for held vocal peaks on ballads.
MAX_WORD_DURATION_SEC = 2.0
# Minimum gap to keep between adjacent words in the same line when clipping
# outro overlap. ~30 ms just prevents back-to-back words from sharing a
# boundary frame; much larger and we visually cut the trailing word short.
ADJACENT_WORD_CLAMP_SEC = 0.03

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class AlignedWord:
    """One user-supplied word with timings borrowed from whisper."""

    word: str
    line_idx: int
    t_start: float
    t_end: float
    # Wav2vec2 CTC confidence carried through from ``whisperx.align``; None
    # when the aligner didn't return a score for this word (e.g. after
    # character-weighted gap fill). Range is roughly 0..1; the visual editor
    # surfaces it as colour-coded confidence so low-confidence words are
    # obvious at a glance.
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "word": self.word,
            "line_idx": self.line_idx,
            "t_start": float(self.t_start),
            "t_end": float(self.t_end),
        }
        if self.score is not None:
            d["score"] = float(self.score)
        return d


@dataclass(frozen=True)
class AlignmentResult:
    """Return payload for :func:`align_lyrics`."""

    song_hash: str
    cache_dir: Path
    aligned_json: Path
    language: str
    model: str
    lines: list[str]
    words: list[AlignedWord]


# ---------------------------------------------------------------------------
# Text handling: line/word tokenisation + normalisation
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
# Keep only letters / digits after NFKD; everything else (punctuation, marks,
# apostrophes, combining diacritics, emoji, etc.) is stripped for comparison.
_NORMALISE_STRIP_RE = re.compile(r"[^0-9a-z]+", re.ASCII)


def _normalise_token(text: str) -> str:
    """Case-fold, strip diacritics and punctuation — for comparison only."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        ch for ch in nfkd if not unicodedata.combining(ch)
    )
    folded = without_marks.casefold()
    return _NORMALISE_STRIP_RE.sub("", folded)


def _parse_anchor_prefix(stripped: str) -> tuple[float | None, str]:
    """Return ``(anchor_seconds, remaining_text)`` after stripping a leading
    ``[m:ss]`` / ``[m:ss.mmm]`` / ``[h:mm:ss]`` block.

    No match → ``(None, stripped)`` unchanged. Matches are always anchored
    at the very start of ``stripped``; times elsewhere in the line are left
    alone so lyric text like ``"[whispered] go home"`` is unaffected.
    """
    match = _LINE_ANCHOR_RE.match(stripped)
    if not match:
        return None, stripped
    h = int(match.group("h") or 0)
    m = int(match.group("m") or 0)
    s = int(match.group("s") or 0)
    ms_group = match.group("ms")
    frac = 0.0
    if ms_group:
        # "[0:12.5]" → 0.5 s; "[0:12.50]" → 0.50 s; "[0:12.500]" → 0.500 s.
        frac = int(ms_group) / (10 ** len(ms_group))
    anchor = float(h * 3600 + m * 60 + s) + frac
    remaining = stripped[match.end():].strip()
    return anchor, remaining


def _is_section_marker(stripped: str) -> bool:
    """``---`` or ``--- <free tag>`` at line start; no other hyphen runs."""
    return bool(_SECTION_MARKER_RE.match(stripped))


def _split_user_lyrics(
    lyrics_text: str,
) -> tuple[list[str], list[tuple[str, int]], list[int], dict[int, float]]:
    """
    Split pasted lyrics into ``(lines, tokens, section_starts, line_anchors)``.

    Blank lines are dropped from ``lines`` but still increment the line gap
    visually; downstream consumers only need ``line_idx`` to group words, so
    we simply renumber non-empty lines 0..N-1 and emit user tokens as
    ``(original_word, line_idx)``.

    A line whose trimmed content matches :data:`_SECTION_MARKER_RE` (``---``
    optionally followed by a free-form tag like ``instrumental`` / ``chorus``)
    acts as a section break: it's stripped from ``lines`` entirely and
    recorded in ``section_starts`` as the token index where the next section
    begins. ``section_starts`` always begins with ``0`` (the first section
    starts at token 0) and contains one entry per non-empty section. A
    lyrics text without any markers yields ``section_starts == [0]`` —
    i.e. "one section covering everything" — which downstream treats as the
    no-marker path.

    Consecutive markers and markers before any tokens / after the last token
    are collapsed so empty sections are never emitted.

    A leading ``[m:ss]`` / ``[m:ss.mmm]`` / ``[h:mm:ss]`` on a lyric line is
    stripped from the rendered text and recorded in ``line_anchors`` mapped
    from the emitted ``line_idx`` to the anchor time in seconds. The
    forced-alignment step uses these as hard window bounds; whisper's VAD
    is ignored within anchored sections.
    """
    lines: list[str] = []
    tokens: list[tuple[str, int]] = []
    section_starts: list[int] = [0]
    line_anchors: dict[int, float] = {}
    for raw_line in lyrics_text.replace("\r\n", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _is_section_marker(stripped):
            next_start = len(tokens)
            if next_start > section_starts[-1]:
                section_starts.append(next_start)
            continue
        anchor, remaining = _parse_anchor_prefix(stripped)
        if not remaining:
            # Bracket-only line (``[1:23]`` by itself) — treat as a hint
            # anchored to the *next* lyric line rather than emitting an
            # empty lyric. We stash it and apply on the next real line.
            if anchor is not None:
                line_anchors[len(lines)] = anchor
            continue
        line_idx = len(lines)
        if anchor is not None:
            # The explicit anchor wins over any previously-stashed
            # bracket-only hint for this slot.
            line_anchors[line_idx] = anchor
        lines.append(remaining)
        for raw_word in _WHITESPACE_RE.split(remaining):
            if raw_word:
                tokens.append((raw_word, line_idx))
    # A trailing marker (``...\n---\n``) would leave ``section_starts[-1] ==
    # len(tokens)``, which downstream treats as an empty final section. Drop
    # it so we never emit a zero-token section.
    while len(section_starts) > 1 and section_starts[-1] >= len(tokens):
        section_starts.pop()
    # Drop dangling bracket-only anchors that point past the last emitted
    # line so we never reference a non-existent line.
    line_anchors = {
        idx: t for idx, t in line_anchors.items() if idx < len(lines)
    }
    return lines, tokens, section_starts, line_anchors


def _lyrics_cache_key(lyrics_text: str) -> str:
    """Stable sha256 of the **canonical** lyric tokens used for cache keying."""
    # We hash a joined token list (not raw text) so cosmetic whitespace /
    # trailing newline differences don't needlessly bust the cache.
    _, tokens, section_starts, line_anchors = _split_user_lyrics(lyrics_text)
    canonical = "\n".join(f"{idx}\t{word}" for word, idx in tokens)
    # Only fold markers / anchors into the canonical form when the user
    # actually used them. Unmarked / un-anchored lyrics hash identically to
    # earlier versions so existing caches stay valid; adding a marker or
    # anchor naturally invalidates the cache because the hashed canonical
    # form changes.
    if len(section_starts) > 1:
        canonical += "\n::sections::\n" + ",".join(str(s) for s in section_starts)
    if line_anchors:
        anchors_part = ",".join(
            f"{idx}={line_anchors[idx]:.3f}" for idx in sorted(line_anchors)
        )
        canonical += "\n::anchors::\n" + anchors_part
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# WhisperX invocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WhisperWord:
    word: str
    t_start: float
    t_end: float
    score: float | None = None


def _default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _pick_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _log_align_shape(aligned: Any) -> None:
    """TEMP: dump the return shape of ``whisperx.align`` for diagnosis.

    Remove this helper (and the call site in :func:`_run_whisperx_forced`)
    once the empty-word-segments bug is root-caused. We only log at INFO
    level so the output shows up in the Gradio run terminal alongside the
    existing ``WhisperX transcribe:`` / ``Forced-alignment input:`` lines
    the user reads while clicking "Align lyrics".
    """
    LOGGER.info("align() return type=%s", type(aligned).__name__)
    if not isinstance(aligned, dict):
        return
    try:
        LOGGER.info("align() return keys=%s", list(aligned.keys()))
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("align() return keys unreadable: %s", exc)

    flat = aligned.get("word_segments")
    if isinstance(flat, list):
        LOGGER.info(
            "align() word_segments: n=%d; first3=%s",
            len(flat),
            flat[:3],
        )
    else:
        LOGGER.info("align() word_segments: missing or non-list (%s)", type(flat).__name__)

    segs = aligned.get("segments")
    if isinstance(segs, list):
        LOGGER.info("align() segments: n=%d", len(segs))
        if segs:
            first = segs[0]
            if isinstance(first, dict):
                LOGGER.info("align() segments[0] keys=%s", list(first.keys()))
                seg_words = first.get("words")
                if isinstance(seg_words, list):
                    LOGGER.info(
                        "align() segments[0].words: n=%d; first3=%s",
                        len(seg_words),
                        seg_words[:3],
                    )
                else:
                    LOGGER.info(
                        "align() segments[0].words: missing or non-list (%s)",
                        type(seg_words).__name__,
                    )
            else:
                LOGGER.info("align() segments[0] type=%s (not dict)", type(first).__name__)
    else:
        LOGGER.info("align() segments: missing or non-list (%s)", type(segs).__name__)


def _extract_whisper_words(align_result: dict[str, Any]) -> list[_WhisperWord]:
    """
    Flatten WhisperX align output into a single ordered list of words with
    valid ``start``/``end`` timestamps. Missing timestamps are filled by
    linear interpolation between neighbours.
    """
    raw_words: list[dict[str, Any]] = []

    # WhisperX returns both ``segments[*].words`` (grouped) and a flat
    # ``word_segments`` list. Prefer the flat one when available; fall back to
    # walking segments.
    flat = align_result.get("word_segments")
    if isinstance(flat, list) and flat:
        raw_words = [w for w in flat if isinstance(w, dict)]
    else:
        for seg in align_result.get("segments", []) or []:
            for w in seg.get("words", []) or []:
                if isinstance(w, dict):
                    raw_words.append(w)

    # Normalise: carry a (maybe-None) start/end, fill gaps. Score is
    # preserved as-is (None when the aligner didn't emit one).
    starts: list[float | None] = []
    ends: list[float | None] = []
    scores: list[float | None] = []
    words: list[str] = []
    for w in raw_words:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        s = w.get("start")
        e = w.get("end")
        sc = w.get("score")
        words.append(text)
        starts.append(float(s) if isinstance(s, (int, float)) else None)
        ends.append(float(e) if isinstance(e, (int, float)) else None)
        scores.append(float(sc) if isinstance(sc, (int, float)) else None)

    if not words:
        return []

    _interp_fill(starts)
    _interp_fill(ends)

    out: list[_WhisperWord] = []
    for text, s, e, sc in zip(words, starts, ends, scores):
        if s is None or e is None:
            continue
        if e < s:
            e = s
        out.append(_WhisperWord(word=text, t_start=s, t_end=e, score=sc))
    return out


def _interp_fill(values: list[float | None]) -> None:
    """In-place linear interpolation of ``None`` entries between known values."""
    n = len(values)
    if n == 0:
        return
    # Find first/last known index.
    known = [i for i, v in enumerate(values) if v is not None]
    if not known:
        for i in range(n):
            values[i] = 0.0
        return

    # Fill leading Nones with first known value.
    first = known[0]
    for i in range(first):
        values[i] = values[first]
    # Fill trailing Nones with last known value.
    last = known[-1]
    for i in range(last + 1, n):
        values[i] = values[last]
    # Interpolate between each adjacent pair of known values.
    for k0, k1 in zip(known, known[1:]):
        if k1 == k0 + 1:
            continue
        v0 = float(values[k0])  # type: ignore[arg-type]
        v1 = float(values[k1])  # type: ignore[arg-type]
        span = k1 - k0
        for j in range(1, span):
            values[k0 + j] = v0 + (v1 - v0) * (j / span)


def _import_whisperx() -> Any:
    try:
        import whisperx  # type: ignore

        return whisperx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "whisperx is not installed in this Python environment. Install into the "
            "same interpreter that runs the app (e.g. `.venv`): "
            f"`{sys.executable} -m pip install -e .[lyrics]` "
            "(not `py -m pip` unless that points at your venv)"
        ) from exc


@dataclass(frozen=True)
class _WhisperSegment:
    """One transcribed segment with its time window and (whisper's) words."""

    idx: int
    t_start: float
    t_end: float
    words: tuple[str, ...]


def _flatten_transcribe_segments(
    transcribe_result: dict[str, Any],
) -> list[_WhisperSegment]:
    """Extract ``[_WhisperSegment, ...]`` from a transcribe() response.

    WhisperX's ``load_model().transcribe(audio)`` does **not** emit per-word
    detail on its segments (that's what the separate :func:`whisperx.align`
    step is for). Relying on ``seg["words"]`` alone would give us an empty
    word list, collapse every user token into segment 0, and produce a
    one-segment forced-alignment input covering only whisper's first
    speech segment — which is then impossible to fit the full lyrics into
    and comes back with no timings.

    We therefore fall back to whitespace-splitting ``seg["text"]`` so NW
    has per-segment word text to match user tokens against. If a future
    WhisperX version starts returning word-level detail from transcribe,
    the ``words`` branch transparently takes over.
    """
    segments: list[_WhisperSegment] = []
    for i, seg in enumerate(transcribe_result.get("segments", []) or []):
        try:
            t0 = float(seg.get("start", 0.0))
            t1 = float(seg.get("end", t0))
        except (TypeError, ValueError):
            continue
        if t1 < t0:
            t1 = t0
        words: list[str] = []
        for w in seg.get("words", []) or []:
            text = str(w.get("word", "")).strip() if isinstance(w, dict) else ""
            if text:
                words.append(text)
        if not words:
            seg_text = str(seg.get("text", "") or "").strip()
            if seg_text:
                words = [tok for tok in seg_text.split() if tok]
        segments.append(
            _WhisperSegment(
                idx=i, t_start=t0, t_end=t1, words=tuple(words)
            )
        )
    return segments


# Faster-whisper's segment ``end`` timestamps aggressively truncate
# right after the last decoded word even when the vocal clearly
# continues. When we later feed those windows into ``whisperx.align``
# (CTC), the aligner is bound to the window and is forced to compress
# every word of the segment onto whatever audio fits — so a chorus
# like "Roll with life" repeated three times inside a 5 s window gets
# stacked on top of itself instead of spread across the actual 12 s
# the phrase sings over. Stretching each segment's ``end`` forward
# into the silence up to the next segment's ``start`` (never
# overlapping neighbours) gives CTC the headroom it needs to place
# words at real phoneme-detected times.
_SEGMENT_WIDEN_GAP_SEC = 0.05


def _widen_segments_for_ctc(
    segments: list[_WhisperSegment],
    *,
    audio_duration: float,
    gap_before_next: float = _SEGMENT_WIDEN_GAP_SEC,
) -> list[_WhisperSegment]:
    """Return a copy of ``segments`` with each segment's ``t_end``
    pushed forward to just before the next segment's ``t_start``
    (or to ``audio_duration`` for the last one).

    Only ends are moved; starts are left alone because pyannote VAD's
    onset detection is far more accurate than its offset detection on
    sung vocals (held notes routinely tail off well past the reported
    ``end``). The small ``gap_before_next`` keeps segments strictly
    non-overlapping so ``whisperx.align`` never sees duplicate audio.
    """
    n = len(segments)
    if n == 0:
        return list(segments)
    widened: list[_WhisperSegment] = []
    for i, seg in enumerate(segments):
        if i + 1 < n:
            ceiling = float(segments[i + 1].t_start) - float(gap_before_next)
        else:
            ceiling = float(audio_duration)
        new_end = max(float(seg.t_end), ceiling)
        if new_end < float(seg.t_start):
            new_end = float(seg.t_start)
        widened.append(
            _WhisperSegment(
                idx=seg.idx,
                t_start=float(seg.t_start),
                t_end=new_end,
                words=seg.words,
            )
        )
    return widened


def _proportional_segment_assignment(
    n_user: int, segments: list[_WhisperSegment]
) -> list[int]:
    """Fallback distribution: spread ``n_user`` tokens across ``segments``
    proportionally to each segment's whisper-word count (or duration, when
    the word counts are all zero). Preserves segment order so token ``i``
    lands in a plausible temporal slot even without any NW matches.
    """
    if n_user <= 0 or not segments:
        return [0] * n_user

    weights = [max(0, len(seg.words)) for seg in segments]
    if sum(weights) == 0:
        weights = [max(1e-6, float(seg.t_end - seg.t_start)) for seg in segments]
    total = float(sum(weights)) or 1.0

    out: list[int] = []
    acc = 0.0
    per_token = total / float(n_user)
    for i in range(n_user):
        target = (i + 0.5) * per_token
        running = 0.0
        chosen = segments[-1].idx
        for seg_pos, w in enumerate(weights):
            running += w
            if target <= running:
                chosen = segments[seg_pos].idx
                break
        out.append(chosen)
        acc += per_token
    return out


def _tokens_are_fuzzy_equal(a: str, b: str) -> bool:
    """Return True when two *already-normalised* tokens should count as a match.

    The strict ``a == b`` test is too conservative on sung / chopped vocals:
    whisper hears "were" for "we're", "yo" for "you", "till" for "til",
    "thru" for "through", etc. A single-edit Levenshtein distance covers
    the vast majority of these near-misses while still rejecting unrelated
    words. We also treat prefixes (either direction) of length ≥ 3 as a
    match, so "yo"↔"you" / "comin"↔"coming" work.

    Both arguments MUST already be run through :func:`_normalise_token`;
    this function does not strip punctuation / case for you.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    la, lb = len(a), len(b)
    # Short tokens are riskier; require an exact match for anything < 3 chars.
    if la < 3 or lb < 3:
        return False
    # Mutual prefix of length ≥ 3: covers contractions / chopped endings.
    if la >= 3 and lb >= 3:
        short, long = (a, b) if la <= lb else (b, a)
        if long.startswith(short):
            return True
    # Fall back to Levenshtein-1 for same-length typos and insert/delete-1
    # cases that didn't clear the prefix test.
    if abs(la - lb) > 1:
        return False
    # Distance-1 DP bounded to 1 (early-out as soon as we exceed).
    if la == lb:
        diffs = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diffs += 1
                if diffs > 1:
                    return False
        return diffs <= 1
    # Insert / delete of one char: longer word minus one char must equal shorter.
    short, long = (a, b) if la < lb else (b, a)
    i = j = mismatches = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        mismatches += 1
        if mismatches > 1:
            return False
        j += 1
    return True


# CTC score threshold below which a transcribed whisper word is treated
# as unreliable — either a hallucination (whisper invented text that
# isn't really in this audio window) or an interpolated-timings gap
# (the word had no CTC match and got its position linearly filled in
# from neighbours, which lands it somewhere essentially random on long
# segments). Wav2vec2 CTC scores real phoneme matches at ~0.7-1.0;
# anything below 0.25 is almost always junk. Tuning note: setting this
# too high starves the anchor pass on sung material (where real vocal
# formants score lower than speech); setting it too low re-introduces
# the "whisper ghost text drifts onto unrelated audio" bug users hit
# on chopped / melismatic tracks.
_MIN_WHISPER_WORD_SCORE = 0.25


def _filter_confident_whisper_words(
    words: list[_WhisperWord],
    *,
    min_score: float = _MIN_WHISPER_WORD_SCORE,
) -> list[_WhisperWord]:
    """Keep only whisper-transcribed words whose CTC score passes the bar.

    Two classes of word are dropped:

    * ``score is None`` — :func:`_extract_whisper_words` sets this on any
      word whose timing was linearly interpolated because CTC didn't
      emit one. On short windows this is a reasonable fallback, but on
      long segments the interpolated position is essentially random.
    * ``score < min_score`` — whisper hallucinated the token, or CTC
      couldn't align the phonemes confidently. Either way, the timing
      is not trustworthy enough to show as a manual-alignment reference.

    Used both for anchor derivation (so we don't pin line-starts on
    hallucinated whisper timings) and for the ``whisper_words`` list
    persisted into ``lyrics.aligned.json`` (so the editor's ghost-text
    overlay doesn't clutter the waveform with labels that will just
    mislead the user).
    """
    return [
        w for w in words
        if w.score is not None and float(w.score) >= float(min_score)
    ]


def _derive_line_anchors_from_transcription(
    user_tokens: list[tuple[str, int]],
    transcription_words: list[_WhisperWord],
    *,
    existing_anchors: dict[int, float] | None = None,
) -> dict[int, float]:
    """Auto-generate ``{line_idx: t_start}`` anchors from transcribed word times.

    Takes the per-word timings produced by running CTC on whisper's **own
    transcribed text** (so they reflect what the language model *heard*,
    not what the user pasted), NW-matches them against the user tokens,
    and for each user line:

    * If the **first** user token has a match, pin the line start to the
      matched whisper word's ``t_start`` (the common case, exact time).
    * If the first doesn't match but some *later* token in the same line
      does, back-estimate the line start by subtracting
      ``(local_position * estimated_word_duration)`` from the matched
      whisper word's ``t_start``. Estimated word duration is bounded to
      ``[0.18s, 0.60s]`` which covers both fast rap and slow ballads
      without producing wildly negative anchor times.

    Matching itself is loosened past strict string equality via
    :func:`_tokens_are_fuzzy_equal` — this is the biggest quality win on
    chopped / sung vocals where whisper drops final consonants ("were"
    for "we're"), clips syllables ("yo" for "you"), or mishears short
    function words. Without fuzzy matching most line-starts on sung
    material never get anchored and the forced-alignment pass drifts by
    many seconds.

    Lines for which the user already supplied a ``[m:ss]`` anchor are
    never overwritten — explicit user pins always win.

    Returns an empty dict if no confident matches were found or the
    inputs are empty. Never raises.
    """
    existing = existing_anchors or {}
    if not user_tokens or not transcription_words:
        return {}

    user_norm = [_normalise_token(w) for w, _ in user_tokens]
    whisper_norm = [_normalise_token(w.word) for w in transcription_words]
    if not any(user_norm) or not any(whisper_norm):
        return {}

    pairs = _needleman_wunsch(user_norm, whisper_norm)
    user_to_whisper: dict[int, int] = {}
    for ui, wj in pairs:
        if ui is None or wj is None:
            continue
        if not _tokens_are_fuzzy_equal(user_norm[ui], whisper_norm[wj]):
            continue
        user_to_whisper[ui] = wj

    if not user_to_whisper:
        return {}

    # Group global user-token indices by line, in source order, along with
    # each token's local position inside its line (0 = first token).
    tokens_by_line: dict[int, list[tuple[int, int]]] = {}
    for idx, (_, line_idx) in enumerate(user_tokens):
        bucket = tokens_by_line.setdefault(line_idx, [])
        bucket.append((len(bucket), idx))

    anchors: dict[int, float] = {}
    for line_idx, tok_list in tokens_by_line.items():
        if line_idx in existing:
            continue
        for local_pos, global_idx in tok_list:
            wj = user_to_whisper.get(global_idx)
            if wj is None:
                continue
            ww = transcription_words[wj]
            if local_pos == 0:
                anchor_time = float(ww.t_start)
            else:
                # Back-estimate to the line's first word. Clamp the per-word
                # duration so a weirdly-long whisper word (e.g. a 2 s melisma)
                # doesn't create a grossly-negative anchor.
                word_dur = max(0.18, min(0.60, float(ww.t_end - ww.t_start)))
                anchor_time = float(ww.t_start) - local_pos * word_dur
            anchors[line_idx] = max(0.0, anchor_time)
            break
    return anchors


def _derive_transcription_anchors_via_align(
    *,
    whisper_segments: list[_WhisperSegment],
    audio: Any,
    whisperx: Any,
    align_model: Any,
    metadata: Any,
    device: str,
    user_tokens: list[tuple[str, int]],
    existing_anchors: dict[int, float],
) -> tuple[dict[int, float], list[_WhisperWord]]:
    """Run CTC on whisper's own transcribed text and derive line anchors.

    Returns ``(anchors, transcription_words)`` — the anchors used by the
    forced-alignment pass, *and* the per-word CTC timings for whisper's
    own transcribed text. We persist ``transcription_words`` alongside
    the user alignment so the visual editor can display "what whisper
    heard here" as ghost text above each user word; this lets the user
    re-align manually from the waveform without listening back to every
    phrase.

    Builds ``whisperx.align``-shaped input from the transcribed segments
    (text = whitespace-joined whisper words per segment), runs alignment
    to get per-word timings for *what whisper heard*, then delegates to
    :func:`_derive_line_anchors_from_transcription` for NW matching.

    The caller is responsible for loading / releasing ``align_model`` —
    we only borrow it. This function never raises; on any failure it
    logs and returns ``({}, [])`` so the outer flow falls back to the
    user-text alignment with whatever manual anchors exist.
    """
    segments_for_align: list[dict[str, Any]] = []
    for seg in whisper_segments:
        if not seg.words:
            continue
        text = " ".join(seg.words).strip()
        if not text:
            continue
        segments_for_align.append(
            {"text": text, "start": float(seg.t_start), "end": float(seg.t_end)}
        )
    if not segments_for_align:
        return {}, []

    try:
        aligned = whisperx.align(
            segments_for_align,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
            print_progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Whisper-transcript CTC align failed: %s — auto anchors skipped.",
            exc,
        )
        return {}, []

    raw_transcription_words = _extract_whisper_words(aligned)
    if not raw_transcription_words:
        LOGGER.info(
            "Whisper-transcript CTC returned no per-word timings; "
            "auto anchors skipped."
        )
        return {}, []

    # Drop interpolated / low-confidence CTC placements before doing
    # anything with them. On sung / chopped material CTC often fails
    # to align whisper's hallucinated tokens, then either marks them
    # with a low score or omits the timing entirely (which we fill by
    # interpolation above). Both cases produce labels that land on
    # unrelated audio — using them to pin line anchors, or displaying
    # them in the editor, actively misleads the user.
    transcription_words = _filter_confident_whisper_words(
        raw_transcription_words
    )
    dropped = len(raw_transcription_words) - len(transcription_words)
    if dropped > 0:
        LOGGER.info(
            "Whisper-transcript CTC: dropped %d/%d low-confidence or "
            "interpolated words (score < %.2f or None)",
            dropped,
            len(raw_transcription_words),
            _MIN_WHISPER_WORD_SCORE,
        )
    if not transcription_words:
        return {}, []

    anchors = _derive_line_anchors_from_transcription(
        user_tokens,
        transcription_words,
        existing_anchors=existing_anchors,
    )
    return anchors, transcription_words


def _assign_user_tokens_to_segments(
    user_tokens: list[tuple[str, int]],
    segments: list[_WhisperSegment],
) -> list[int]:
    """Return ``len(user_tokens)`` ints, one per user token, = segment index.

    The mapping is produced by flattening whisper's recognised words across
    every segment, running Needleman-Wunsch between that flat list and the
    user tokens (normalised), and then carrying each matched user token's
    segment index forward / backward so unmatched neighbours inherit the
    segment of the closest match. This is the critical correspondence step
    that lets us feed the user's pasted lyrics — split at *whisper's* VAD
    boundaries — into forced alignment.

    When whisper returned no usable words at all (totally instrumental
    stretch, very short clip), every user token is assigned to segment 0
    so the caller can still build a single fallback segment.

    When NW produced zero matches but multiple whisper segments exist
    (mishears everywhere), we fall back to a proportional distribution
    instead of collapsing everything into segment 0 — aligning hundreds
    of words into a single 5 s window is guaranteed to emit nothing.
    """
    n = len(user_tokens)
    if n == 0 or not segments:
        return [0] * n

    # Flatten whisper words + remember which segment each one came from.
    flat_words: list[str] = []
    flat_seg: list[int] = []
    for seg in segments:
        for w in seg.words:
            flat_words.append(w)
            flat_seg.append(seg.idx)

    if not flat_words:
        return _proportional_segment_assignment(n, segments)

    user_norm = [_normalise_token(word) for word, _ in user_tokens]
    whisper_norm = [_normalise_token(w) for w in flat_words]

    pairs = _needleman_wunsch(user_norm, whisper_norm)

    assignment: list[int | None] = [None] * n
    match_count = 0
    for ui, wj in pairs:
        if ui is None or wj is None:
            continue
        if not user_norm[ui] or user_norm[ui] != whisper_norm[wj]:
            continue
        assignment[ui] = flat_seg[wj]
        match_count += 1

    if match_count == 0:
        # NW found no anchors (whisper's mishears don't overlap user lyrics at
        # all). Anchored fill-from-neighbour is meaningless here; go straight
        # to proportional distribution so we still span the whole stem.
        return _proportional_segment_assignment(n, segments)

    # Forward fill: unmatched user token inherits the prior match's segment.
    last: int | None = None
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = last
        else:
            last = assignment[i]
    # Backward fill for leading unmatched tokens (no prior anchor).
    next_seen: int | None = None
    for i in range(n - 1, -1, -1):
        if assignment[i] is None:
            assignment[i] = next_seen
        else:
            next_seen = assignment[i]

    return [int(v) if v is not None else 0 for v in assignment]


def _build_forced_alignment_segments(
    user_tokens: list[tuple[str, int]],
    segment_assignment: list[int],
    segments: list[_WhisperSegment],
    audio_duration_sec: float,
    *,
    user_section_starts: list[int] | None = None,
    line_anchors: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Build ``whisperx.align`` input dicts, optionally partitioned by user markers.

    Two behaviours, selected by ``user_section_starts``:

    * **No markers** (``user_section_starts`` is ``None`` or has ``<= 1``
      entries): group user tokens by their NW-assigned whisper segment index.
      One dict per non-empty whisper bucket. This is the pre-marker default
      and keeps existing behaviour for lyrics that don't use ``---``.
    * **User markers present**: group user tokens by user-authored section
      boundaries (``section_starts``) and derive each section's time window
      from the union of whisper segments its tokens were NW-assigned to.
      When whisper's VAD merged two or more user sections into a single
      segment (the repeated-chorus bug), the shared span is split between
      the colliding sections proportionally to their token counts. Sections
      with no NW anchors at all are filled from neighbouring sections'
      boundaries.

    Each dict has the exact shape ``whisperx.align`` expects
    (``{"text": ..., "start": ..., "end": ...}``). When every bucket ends up
    empty (degenerate case where whisper's transcribe returned nothing we
    could use), we return one big segment spanning the whole stem with the
    full user text so alignment still runs on *something* rather than
    crashing.
    """
    # An anchor on a line in the middle of the song is itself a hard section
    # boundary; synthesise implicit section starts at each anchored line
    # token so the marker-mode path can apply the anchors as hard window
    # bounds. Only takes effect if the user has no explicit ``---`` markers.
    effective_section_starts = user_section_starts
    if (
        (user_section_starts is None or len(user_section_starts) <= 1)
        and line_anchors
    ):
        effective_section_starts = _section_starts_from_anchors(
            user_tokens, line_anchors
        )

    if effective_section_starts is not None and len(effective_section_starts) > 1:
        out = _build_forced_alignment_segments_by_user_sections(
            user_tokens,
            segment_assignment,
            segments,
            audio_duration_sec,
            effective_section_starts,
            line_anchors=line_anchors or {},
        )
        if out:
            return out
        # Falls through to whisper-bucket behaviour below if the marker path
        # produced nothing (shouldn't happen but we keep a safety net).

    by_seg: dict[int, list[str]] = {}
    for ui, seg_idx in enumerate(segment_assignment):
        by_seg.setdefault(int(seg_idx), []).append(user_tokens[ui][0])

    seg_lookup = {seg.idx: seg for seg in segments}

    out: list[dict[str, Any]] = []
    for seg_idx in sorted(by_seg.keys()):
        words = by_seg[seg_idx]
        if not words:
            continue
        seg = seg_lookup.get(seg_idx)
        if seg is None:
            continue
        # Widen pathologically narrow segments so the forced aligner has
        # breathing room to place the assigned words — a 0.2 s window with
        # 5 user words will push them all onto the same frame.
        t0 = float(seg.t_start)
        t1 = float(seg.t_end)
        if t1 - t0 < 0.1:
            t1 = t0 + 0.1
        out.append(
            {
                "text": " ".join(words),
                "start": t0,
                "end": t1,
            }
        )

    if not out:
        full_text = " ".join(word for word, _ in user_tokens)
        out = [
            {
                "text": full_text,
                "start": 0.0,
                "end": max(0.1, float(audio_duration_sec)),
            }
        ]
    return out


def _section_starts_from_anchors(
    user_tokens: list[tuple[str, int]],
    line_anchors: dict[int, float],
) -> list[int]:
    """Synthesise ``section_starts`` at each anchored line's first token.

    Used when the user has anchors but no explicit ``---`` markers: each
    anchor naturally bounds a section, because forced alignment needs
    disjoint windows to honour the anchor as a hard ``start``. We always
    include ``0`` as the first section start so downstream treats this as
    a multi-section lyric (and skips the anchor path if only one section
    results after filtering).
    """
    if not line_anchors:
        return [0]
    first_token_idx_by_line: dict[int, int] = {}
    for idx, (_word, line_idx) in enumerate(user_tokens):
        first_token_idx_by_line.setdefault(line_idx, idx)
    starts = {0}
    for line_idx in line_anchors:
        tok_idx = first_token_idx_by_line.get(line_idx)
        if tok_idx is not None and tok_idx > 0:
            starts.add(tok_idx)
    return sorted(starts)


def _build_forced_alignment_segments_by_user_sections(
    user_tokens: list[tuple[str, int]],
    segment_assignment: list[int],
    segments: list[_WhisperSegment],
    audio_duration_sec: float,
    user_section_starts: list[int],
    *,
    line_anchors: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Marker-driven variant of :func:`_build_forced_alignment_segments`.

    Each entry in ``user_section_starts`` is the first token index of a
    user-authored section; the final section runs to ``len(user_tokens)``.
    For each section we:

    1. Pick the union of whisper segments that the section's NW-assigned
       tokens landed in → initial ``[start, end]`` window.
    2. Resolve collisions with adjacent sections by splitting the shared
       time range proportionally to each section's token count. This is
       what prevents a repeated chorus from collapsing onto its predecessor
       when whisper's VAD merged them into one big speech segment.
    3. Fill any section with zero NW anchors from its neighbours' bounds
       (leading / trailing sections default to ``0`` / ``audio_duration``).
    4. Guarantee a minimum 0.1 s window so ``whisperx.align`` always has
       breathing room for CTC backtrack.
    """
    n_tokens = len(user_tokens)
    n_sections = len(user_section_starts)
    if n_sections == 0 or n_tokens == 0:
        return []

    # Section k covers tokens [bounds[k], bounds[k + 1]).
    bounds = list(user_section_starts) + [n_tokens]
    seg_lookup = {seg.idx: seg for seg in segments}

    @dataclass
    class _SectionWindow:
        start: float | None
        end: float | None
        token_count: int
        text: str
        # Hard bound from a user anchor. When set, the overlap-resolution
        # pass treats it as immovable so neighbours bend around it instead.
        forced_start: float | None = None
        forced_end: float | None = None

    # Map each anchored line to the section index that contains its first
    # token. We use the anchor to override that section's ``start``; the
    # next anchored section's start (or the audio end) closes the window.
    first_token_idx_by_line: dict[int, int] = {}
    for idx, (_word, line_idx) in enumerate(user_tokens):
        first_token_idx_by_line.setdefault(line_idx, idx)
    anchors_by_section: dict[int, float] = {}
    if line_anchors:
        for line_idx, t in line_anchors.items():
            tok = first_token_idx_by_line.get(line_idx)
            if tok is None:
                continue
            # Find the section containing this token.
            for k in range(n_sections):
                if bounds[k] <= tok < bounds[k + 1]:
                    # First anchor on a section wins (ignore later lines).
                    anchors_by_section.setdefault(k, float(t))
                    break

    windows: list[_SectionWindow] = []
    for k in range(n_sections):
        lo, hi = bounds[k], bounds[k + 1]
        section_tokens = user_tokens[lo:hi]
        text = " ".join(word for word, _ in section_tokens)
        if not text:
            continue

        whisper_ids = {int(segment_assignment[ui]) for ui in range(lo, hi)}
        whisper_segs = [
            seg_lookup[sid] for sid in whisper_ids if sid in seg_lookup
        ]
        if whisper_segs:
            start: float | None = min(float(s.t_start) for s in whisper_segs)
            end: float | None = max(float(s.t_end) for s in whisper_segs)
        else:
            start = end = None

        forced_start = anchors_by_section.get(k)
        if forced_start is not None:
            start = forced_start
        windows.append(
            _SectionWindow(
                start=start,
                end=end,
                token_count=hi - lo,
                text=text,
                forced_start=forced_start,
            )
        )

    if not windows:
        return []

    # If section K+1 has a forced_start, that's K's forced_end (a hard
    # upstream bound). Propagate that so the overlap resolver sees correct
    # neighbours and never bleeds past a user-pinned boundary.
    for i in range(len(windows) - 1):
        next_forced = windows[i + 1].forced_start
        if next_forced is not None:
            windows[i].forced_end = next_forced
            # Also shrink ``end`` if whisper gave us something larger.
            if windows[i].end is None or windows[i].end > next_forced:
                windows[i].end = next_forced

    # Resolve overlap clusters. Walking in order, any run of sections whose
    # windows overlap (or whose left neighbour has no window yet) is split
    # proportionally by token count across the run's union span. This is the
    # critical fix for "whisper merged sections A and B into one big VAD
    # segment and both user-sections inherited the same window".
    _split_overlapping_section_windows(windows, audio_duration_sec)

    out: list[dict[str, Any]] = []
    min_window_sec = 0.1
    for w in windows:
        if w.start is None or w.end is None:
            # Should have been filled by _split_overlapping_section_windows;
            # this is a belt-and-braces clamp for corrupt inputs.
            continue
        t0 = float(w.start)
        t1 = float(w.end)
        if t1 - t0 < min_window_sec:
            t1 = t0 + min_window_sec
        out.append({"text": w.text, "start": t0, "end": t1})

    return out


def _split_overlapping_section_windows(
    windows: list[Any],
    audio_duration_sec: float,
) -> None:
    """Resolve overlaps + missing windows in-place for user-marker sections.

    Operates on a list of mutable objects exposing ``start``, ``end`` and
    ``token_count`` attributes. Objects may optionally expose
    ``forced_start`` / ``forced_end`` (user-supplied anchors or inferred
    hard bounds); these are honoured exactly — the proportional split only
    moves the *free* (non-forced) edges of a cluster so pinned boundaries
    never drift.

    Two passes:

    1. Fill any ``None`` window using neighbouring windows as bounds so every
       section has a provisional ``[start, end]``. Leading / trailing gaps
       default to ``0`` / ``audio_duration_sec``.
    2. Walk the list in order; whenever two or more consecutive sections
       overlap (``w[i].end > w[i + 1].start``), collect the full run and
       split the union span proportionally to each section's token count.

    After this function returns, every window has non-None bounds and
    adjacent windows no longer overlap. Window lengths honour each section's
    relative token weight so a long chorus claims more time than a short
    pre-chorus even when both landed in the same whisper VAD segment.
    """
    n = len(windows)
    if n == 0:
        return

    audio_end = max(0.1, float(audio_duration_sec))

    def _forced_start(w: Any) -> float | None:
        return getattr(w, "forced_start", None)

    def _forced_end(w: Any) -> float | None:
        return getattr(w, "forced_end", None)

    # Pass 1: fill None windows from anchored neighbours.
    for i in range(n):
        if windows[i].start is not None and windows[i].end is not None:
            continue
        # Find the previous anchored section's end (or 0.0).
        prev_end = 0.0
        for j in range(i - 1, -1, -1):
            if windows[j].end is not None:
                prev_end = float(windows[j].end)
                break
        # Find the next anchored section's start (or audio_end).
        next_start = audio_end
        for j in range(i + 1, n):
            if windows[j].start is not None:
                next_start = float(windows[j].start)
                break
        if next_start < prev_end:
            # Shouldn't happen but avoid negative durations.
            next_start = prev_end
        if windows[i].start is None:
            windows[i].start = prev_end
        if windows[i].end is None:
            windows[i].end = next_start

    # Pass 2: collapse overlapping runs via proportional token split. Any
    # section whose forced_start / forced_end is set is treated as an
    # immovable boundary: the cluster splits around it.
    i = 0
    epsilon = 1e-6
    while i < n:
        j = i
        cluster_start = float(windows[i].start)
        cluster_end = float(windows[i].end)
        while j + 1 < n and float(windows[j + 1].start) < cluster_end - epsilon:
            # A forced_start on the right neighbour is a hard boundary;
            # the cluster ends here.
            if _forced_start(windows[j + 1]) is not None:
                break
            j += 1
            if float(windows[j].end) > cluster_end:
                cluster_end = float(windows[j].end)
        if j > i:
            # Keep forced bounds pinned while splitting the free span among
            # the rest. We anchor at the cluster's left edge (which may be
            # a forced_start on w[i]) and right edge (forced_end on w[j] or
            # the union's cluster_end).
            if _forced_start(windows[i]) is not None:
                cluster_start = float(_forced_start(windows[i]))
            if _forced_end(windows[j]) is not None:
                cluster_end = float(_forced_end(windows[j]))
            total_tokens = sum(
                max(1, int(windows[k].token_count)) for k in range(i, j + 1)
            )
            span = cluster_end - cluster_start
            cursor = cluster_start
            for k in range(i, j + 1):
                frac = max(1, int(windows[k].token_count)) / float(total_tokens)
                share = span * frac
                windows[k].start = cursor
                windows[k].end = cursor + share
                cursor += share
        else:
            # Single section: honour its forced_end if it would otherwise
            # exceed the pin.
            fe = _forced_end(windows[i])
            if fe is not None and float(windows[i].end) > fe:
                windows[i].end = fe
        i = j + 1


# ---------------------------------------------------------------------------
# Section-level fingerprint matching
#
# Word-level Needleman-Wunsch on the whisper transcript is the wrong level
# of abstraction when the user has annotated ``---`` sections and the
# transcription is partially broken. NW is greedy per-token; it can't
# distinguish "Now you really know me" at 1:05 from the same phrase at
# 1:40, so the VAD-driven segment bucketing ends up gambling which
# chorus-repeat each user section lands on. When whisper also hears
# hallucinated intro words in noisy pre-vocal audio, NW happily anchors
# the user's first section at ``t=0``.
#
# The fingerprint flow treats each user section as one atomic unit and
# places it on the timeline in three steps:
#
#   1. Extract 2-3 characteristic phrases per section, preferring those
#      that occur in few other sections (unique hooks beat generic
#      filler).
#   2. Scan whisper's transcribed word stream for fuzzy occurrences of
#      each phrase. Every plausible hit is a ``(t_start, score)``
#      candidate for the section it came from.
#   3. Solve section→time assignment as a monotonic DP: assign one
#      candidate per section, enforcing strictly increasing times and
#      maximising total score. Sections with no candidate are left
#      un-anchored and filled later by interpolation from neighbours.
#
# The DP's temporal-ordering constraint is what disambiguates repeated
# choruses: the second chorus can only borrow a match that's later than
# the first chorus's chosen time. Because each section only needs ONE
# solid fingerprint hit to pin its start, the flow is robust to bad
# transcription within the section — the remaining user words are placed
# by the downstream forced-alignment pass running on a tighter,
# correctly-bounded window.
# ---------------------------------------------------------------------------


# 3-word phrases are long enough to be characteristic in most pop lyrics
# but short enough to still land inside one whisper segment even when VAD
# chops aggressively. We also consider 4- and 5-grams when available —
# longer phrases score strictly higher on uniqueness.
_FINGERPRINT_MIN_LEN = 3
_FINGERPRINT_MAX_LEN = 5
# Keep at most this many fingerprints per section. Two or three is plenty
# for the DP to find a good anchor; letting more through just slows
# candidate search linearly without improving quality (the best hit per
# section is all the DP uses).
_FINGERPRINT_MAX_PER_SECTION = 3
# Fuzzy match acceptance threshold. A candidate is recorded when the
# fraction of phrase tokens that fuzzy-match at consecutive whisper
# positions passes this. 0.6 tolerates ~1 missed/misheard word in a
# 3-gram and ~2 in a 5-gram; lower reintroduces false positives on
# generic filler ("you don't have").
_FINGERPRINT_MIN_MATCH_FRAC = 0.6
# The DP prefers placing a section over skipping it even when the match
# is weak, but below this floor the match is too noisy to use — skipping
# and letting interpolation fill the gap is safer than pinning on junk.
_FINGERPRINT_DP_MIN_ACCEPT = 0.5
# Maximum audio-time span a single fingerprint match is allowed to
# cover: ``max(_FINGERPRINT_MIN_SPAN_SEC, phrase_len *
# _FINGERPRINT_SEC_PER_WORD)``. Matches whose first-matched and
# last-matched whisper words span more than this are rejected.
#
# Why this exists: CTC-aligned whisper text occasionally splits a
# phrase across a hallucinated intro word, so a real 3-gram like
# "we were up" can end up with its first whisper word pinned to
# noise at t=0 and the second + third pinned to the real vocal at
# t=25s. Without a span check my matcher accepted that 25-second
# gap as a 3/3 fuzzy match and anchored the whole section at t=0,
# which then dragged CTC into cram-fitting 50+ user words across a
# mostly-silent intro window (see the regression fixed in
# ``TestFingerprintMatchTemporalSpanConstraint``). Sung vocals at
# any natural rate fit a 3-5 word phrase in at most a few seconds;
# 2 s/word + a 3 s floor covers even slow ballads while filtering
# the pathological split-across-silence false positives.
_FINGERPRINT_SEC_PER_WORD = 2.0
_FINGERPRINT_MIN_SPAN_SEC = 3.0
# Upper-bound budget for unplaced leading/trailing sections in
# ``_build_section_windows_from_fingerprints``. Used to shrink the
# window for a section that has no fingerprint match and no user
# anchor, so it doesn't stretch across tens of seconds of intro/outro
# silence when the adjacent anchor is far away. 0.6 s/token is a
# generous upper bound on sung phrasing (typical pop vocals are
# 2-3 words/sec; slow ballads occasionally dip to ~1.7 w/s ≈ 0.6 s/w).
_SEC_PER_TOKEN_BUDGET = 0.6


@dataclass(frozen=True)
class _SectionFingerprint:
    """One characteristic phrase extracted from a user section.

    ``phrase`` is already ``_normalise_token``-d; ``other_section_count``
    is how many *other* sections contain this exact phrase (drives the
    uniqueness ranking when selecting fingerprints).
    """

    section_idx: int
    phrase: tuple[str, ...]
    other_section_count: int


@dataclass(frozen=True)
class _FingerprintMatch:
    """One candidate hit for a fingerprint against the whisper transcript.

    ``t_start`` is the timestamp of the first fuzzy-matched whisper word;
    ``score`` is ``matched_tokens / len(phrase)`` in the ``0..1`` range.
    """

    section_idx: int
    t_start: float
    score: float


def _normalise_section_tokens(
    user_tokens: list[tuple[str, int]],
    section_starts: list[int],
) -> list[list[str]]:
    """Split ``user_tokens`` into per-section normalised word lists."""
    if not section_starts:
        return []
    bounds = list(section_starts) + [len(user_tokens)]
    out: list[list[str]] = []
    for k in range(len(section_starts)):
        lo, hi = bounds[k], bounds[k + 1]
        out.append(
            [_normalise_token(word) for word, _ in user_tokens[lo:hi]]
        )
    return out


def _extract_section_fingerprints(
    user_tokens: list[tuple[str, int]],
    section_starts: list[int],
) -> list[list[_SectionFingerprint]]:
    """Pick 2-3 characteristic phrases per user section.

    For each section we enumerate every 3-, 4- and 5-gram of normalised
    tokens, count how many *other* sections contain the same n-gram as
    a contiguous substring, and keep the top ``_FINGERPRINT_MAX_PER_SECTION``
    phrases ranked by ``(other_section_count asc, len desc, position asc)``.

    A section shorter than :data:`_FINGERPRINT_MIN_LEN` tokens, or one
    that contains only duplicates of every possible phrase (e.g. a
    single-word repeated hook), returns an empty list — the DP will
    then skip that section and we fall back to interpolation from
    neighbours. Empty normalised tokens (punctuation-only) are dropped
    before n-gramming so "..." / "(...)" markers don't sabotage the
    phrase window.
    """
    per_section = _normalise_section_tokens(user_tokens, section_starts)
    # Drop empty normalised tokens within each section (e.g. a "..."
    # marker) so a 3-gram window isn't silently padded with empty
    # strings that can never match any whisper word.
    cleaned: list[list[str]] = [
        [tok for tok in section if tok] for section in per_section
    ]

    # Build an index of "n-gram → set of section indices that contain it"
    # so we can quickly count how many other sections share any candidate
    # phrase. Using a single multi-n index keeps selection O(total n-grams).
    phrase_sections: dict[tuple[str, ...], set[int]] = {}
    for k, tokens in enumerate(cleaned):
        for n in range(_FINGERPRINT_MIN_LEN, _FINGERPRINT_MAX_LEN + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrase = tuple(tokens[i : i + n])
                phrase_sections.setdefault(phrase, set()).add(k)

    out: list[list[_SectionFingerprint]] = []
    for k, tokens in enumerate(cleaned):
        candidates: list[tuple[int, int, int, tuple[str, ...]]] = []
        # (other_section_count, -len, position, phrase) — sort ascending
        # picks unique-longest-earliest first.
        for n in range(_FINGERPRINT_MIN_LEN, _FINGERPRINT_MAX_LEN + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrase = tuple(tokens[i : i + n])
                sections = phrase_sections.get(phrase, set())
                other = len(sections - {k})
                candidates.append((other, -n, i, phrase))
        candidates.sort()
        seen: set[tuple[str, ...]] = set()
        picks: list[_SectionFingerprint] = []
        for other, _neg_n, _pos, phrase in candidates:
            if phrase in seen:
                continue
            seen.add(phrase)
            picks.append(
                _SectionFingerprint(
                    section_idx=k,
                    phrase=phrase,
                    other_section_count=other,
                )
            )
            if len(picks) >= _FINGERPRINT_MAX_PER_SECTION:
                break
        out.append(picks)
    return out


def _flatten_whisper_words_for_fingerprints(
    whisper_segments: list[_WhisperSegment],
    transcription_words: list[_WhisperWord],
) -> list[tuple[str, float]]:
    """Return ``[(normalised_word, t_start), ...]`` for fingerprint search.

    Prefers the CTC-aligned ``transcription_words`` (real per-word
    timings) when available; falls back to segment-start timings spread
    evenly across each segment's whisper words. The fallback timings are
    only used for matching — they're never written back as authoritative
    word positions.
    """
    if transcription_words:
        out: list[tuple[str, float]] = []
        for w in transcription_words:
            norm = _normalise_token(w.word)
            if norm:
                out.append((norm, float(w.t_start)))
        return out

    out = []
    for seg in whisper_segments:
        if not seg.words:
            continue
        duration = max(0.0, float(seg.t_end - seg.t_start))
        n = len(seg.words)
        per = duration / n if n > 0 else 0.0
        for i, word in enumerate(seg.words):
            norm = _normalise_token(word)
            if norm:
                out.append((norm, float(seg.t_start) + per * i))
    return out


def _find_fingerprint_matches_in_transcript(
    fingerprints: list[_SectionFingerprint],
    whisper_words: list[tuple[str, float]],
) -> list[_FingerprintMatch]:
    """Return all plausible candidate hits across every fingerprint.

    For each ``(phrase, t_start)`` we slide a window of ``len(phrase)``
    whisper tokens over the transcript and accept the position if the
    ratio of :func:`_tokens_are_fuzzy_equal` matches passes
    :data:`_FINGERPRINT_MIN_MATCH_FRAC`. Matches from different
    fingerprints belonging to the same section are merged: when two
    phrases hit within 0.5 s of each other we keep only the stronger
    one, because two overlapping matches shouldn't both get to
    compete for the DP's one pick per section.
    """
    if not fingerprints or not whisper_words:
        return []

    raw: list[_FingerprintMatch] = []
    whisper_norm = [w for w, _ in whisper_words]
    whisper_times = [t for _, t in whisper_words]
    n_whisper = len(whisper_norm)

    for fp in fingerprints:
        phrase = fp.phrase
        plen = len(phrase)
        if plen == 0 or n_whisper < plen:
            continue
        max_span = max(
            _FINGERPRINT_MIN_SPAN_SEC,
            float(plen) * _FINGERPRINT_SEC_PER_WORD,
        )
        for i in range(n_whisper - plen + 1):
            matched_positions: list[int] = []
            for j in range(plen):
                if _tokens_are_fuzzy_equal(phrase[j], whisper_norm[i + j]):
                    matched_positions.append(i + j)
            frac = len(matched_positions) / plen
            if frac < _FINGERPRINT_MIN_MATCH_FRAC:
                continue
            # Temporal-span guard: reject windows where the first and
            # last fuzzy-matched whisper words are implausibly far
            # apart. Without this a CTC-aligned transcript that
            # placed one phrase word on intro noise at t=0 and the
            # rest on the real vocal 20+ seconds later would happily
            # register as a "match" and anchor a section at t=0 (see
            # the pathology log in ``terminals/1.txt`` on the
            # ``We were up till three…`` song).
            first_matched = matched_positions[0]
            last_matched = matched_positions[-1]
            span = float(whisper_times[last_matched] - whisper_times[first_matched])
            if span > max_span:
                continue
            # Anchor the match on the *first* fuzzy-matched whisper
            # word rather than on position ``i``: if the first phrase
            # token didn't match but the second did, we want the
            # candidate time to point at the real overlap's start,
            # not at an unrelated word just before it.
            t_start = float(whisper_times[first_matched])
            raw.append(
                _FingerprintMatch(
                    section_idx=fp.section_idx,
                    t_start=t_start,
                    score=frac,
                )
            )

    # Dedup overlapping matches per section: if two matches lie within
    # 0.5 s of each other (same chorus-repeat seen by two different
    # fingerprint phrases) collapse to the strongest score.
    raw.sort(key=lambda m: (m.section_idx, m.t_start))
    merged: list[_FingerprintMatch] = []
    for m in raw:
        if (
            merged
            and merged[-1].section_idx == m.section_idx
            and abs(merged[-1].t_start - m.t_start) <= 0.5
        ):
            if m.score > merged[-1].score:
                merged[-1] = m
            continue
        merged.append(m)
    return merged


def _assign_sections_via_temporal_dp(
    n_sections: int,
    matches: list[_FingerprintMatch],
    *,
    forced_section_starts: dict[int, float] | None = None,
) -> list[float | None]:
    """Pick at most one ``t_start`` per section, maximising total score.

    Sections are placed in their original order; a pick for section k
    must have ``t_start`` strictly greater than section k-1's pick (or
    its forced anchor). Sections without a usable candidate return
    ``None`` and are filled by the caller via interpolation.

    ``forced_section_starts`` pins specific sections to known times
    (derived from the user's ``[m:ss]`` line anchors). A forced section
    contributes a synthetic high-score candidate at exactly that time,
    so the DP is guaranteed to pick it over any fingerprint match, and
    neighbouring sections are forced to respect the pin's ordering.
    """
    if n_sections <= 0:
        return []

    forced_section_starts = forced_section_starts or {}

    # Per-section candidate list: include all fingerprint matches, plus
    # a synthetic "forced anchor" entry when the user pinned the section.
    by_section: list[list[tuple[float, float]]] = [[] for _ in range(n_sections)]
    for m in matches:
        if 0 <= m.section_idx < n_sections and m.score >= _FINGERPRINT_DP_MIN_ACCEPT:
            by_section[m.section_idx].append((m.t_start, m.score))
    # Synthetic high-score anchor: score 2.0 dominates any real match
    # (which is capped at 1.0) so the DP always picks the forced time.
    _FORCED_SYNTHETIC_SCORE = 2.0
    for k, t in forced_section_starts.items():
        if 0 <= k < n_sections:
            by_section[k] = [(float(t), _FORCED_SYNTHETIC_SCORE)]

    for k in range(n_sections):
        by_section[k].sort(key=lambda p: p[0])

    # DP over (section_idx, candidate_idx). Candidate 0 is always
    # "skip this section"; candidates 1..N correspond to the section's
    # sorted fingerprint matches. Each cell stores the best cumulative
    # score reaching this pick, the last-picked-time carried forward,
    # and a backtrack pointer into the previous layer.
    NEG_INF = float("-inf")

    @dataclass
    class _Cell:
        score: float
        last_time: float
        is_pick: bool  # True when this cell actually picked a time
        prev_j: int    # index into the previous layer; -1 for virtual start

    # Virtual start layer: one cell with score=0 and last_time=-inf.
    prev_layer: list[_Cell] = [
        _Cell(score=0.0, last_time=NEG_INF, is_pick=False, prev_j=-1)
    ]
    layers: list[list[_Cell]] = []

    for k in range(n_sections):
        cands = by_section[k]
        cells: list[_Cell] = []

        # Skip cell (candidate 0): best predecessor carries through
        # unchanged. We track predecessor index for backtracking.
        best_skip_j = max(
            range(len(prev_layer)), key=lambda j: prev_layer[j].score
        )
        best_skip = prev_layer[best_skip_j]
        cells.append(
            _Cell(
                score=best_skip.score,
                last_time=best_skip.last_time,
                is_pick=False,
                prev_j=best_skip_j,
            )
        )

        # Real candidates: predecessor must have last_time strictly
        # less than this candidate's time.
        for (t, sc) in cands:
            best_prev_j = -1
            best_prev_score = NEG_INF
            for pi, pcell in enumerate(prev_layer):
                if pcell.last_time < t and pcell.score > best_prev_score:
                    best_prev_score = pcell.score
                    best_prev_j = pi
            if best_prev_j < 0:
                continue
            cells.append(
                _Cell(
                    score=best_prev_score + sc,
                    last_time=t,
                    is_pick=True,
                    prev_j=best_prev_j,
                )
            )

        layers.append(cells)
        prev_layer = cells

    # Backtrack from the highest-score cell in the last layer.
    out: list[float | None] = [None] * n_sections
    if not layers or not layers[-1]:
        return out

    cur_j = max(range(len(layers[-1])), key=lambda j: layers[-1][j].score)
    for cur_k in range(n_sections - 1, -1, -1):
        cell = layers[cur_k][cur_j]
        out[cur_k] = cell.last_time if cell.is_pick else None
        cur_j = cell.prev_j
        if cur_j < 0:
            break
    return out


def _build_section_windows_from_fingerprints(
    user_tokens: list[tuple[str, int]],
    section_starts: list[int],
    audio_duration_sec: float,
    chosen_starts: list[float | None],
    *,
    line_anchors: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Turn fingerprint-assigned section starts into forced-align windows.

    Each section gets one entry ``{"text": ..., "start": ..., "end": ...}``
    where:

    * ``start`` is the fingerprint-chosen time for that section, or the
      user's ``[m:ss]`` anchor if present (anchors always win), or a
      linearly-interpolated value based on token weight when the DP left
      that section un-placed.
    * ``end`` is the next section's ``start`` (or ``audio_duration_sec``
      for the final section).

    Un-placed sections in the middle of the song are interpolated
    linearly from the nearest anchored/placed neighbours, weighted by
    token count so a long chorus claims more time than a short
    pre-chorus. Leading un-placed sections default to ``start = 0.0``;
    trailing ones extend to ``audio_duration_sec``. The whole result is
    passed through a monotonic-fix-up that enforces strictly increasing
    starts (occasionally fingerprints place section k+1 very slightly
    before section k's end because of the 0.5 s merge window).
    """
    n_sections = len(section_starts)
    if n_sections == 0 or not user_tokens:
        return []

    bounds = list(section_starts) + [len(user_tokens)]
    line_anchors = line_anchors or {}

    # Resolve user ``[m:ss]`` anchors into a per-section "first anchor"
    # lookup. A section inherits the earliest anchor of any line that
    # falls inside its token range.
    first_token_idx_by_line: dict[int, int] = {}
    for idx, (_word, line_idx) in enumerate(user_tokens):
        first_token_idx_by_line.setdefault(line_idx, idx)
    anchors_by_section: dict[int, float] = {}
    for line_idx, t in line_anchors.items():
        tok = first_token_idx_by_line.get(line_idx)
        if tok is None:
            continue
        for k in range(n_sections):
            if bounds[k] <= tok < bounds[k + 1]:
                anchors_by_section.setdefault(k, float(t))
                break

    # Merge chosen starts with user anchors (anchors win).
    merged_starts: list[float | None] = list(chosen_starts)
    for k, t in anchors_by_section.items():
        if 0 <= k < n_sections:
            merged_starts[k] = float(t)

    audio_end = max(0.1, float(audio_duration_sec))
    token_counts = [bounds[k + 1] - bounds[k] for k in range(n_sections)]

    # Fill None entries by linear interpolation between anchored
    # neighbours, weighted by cumulative token counts so sections get
    # proportional shares of the gap rather than equal shares.
    starts: list[float] = [0.0] * n_sections
    # Pre-fill from merged_starts where defined.
    known: list[tuple[int, float]] = [
        (k, float(t)) for k, t in enumerate(merged_starts) if t is not None
    ]
    if not known:
        # No anchors at all — distribute proportionally across the full song.
        total_tok = sum(token_counts) or 1
        cursor = 0.0
        for k in range(n_sections):
            starts[k] = cursor
            cursor += audio_end * (token_counts[k] / total_tok)
    else:
        # Leading gap: budget-capped .. first known. Historically we
        # distributed the whole ``[0, first_t)`` range across unplaced
        # leading sections by token weight, which meant a user pinning
        # section 1 at t=22 s and leaving section 0 un-fingerprinted
        # ended up with section 0 getting a [0, 22] window even when
        # section 0 realistically only sings for the last few seconds.
        # CTC then crammed 50+ user words onto the mostly-silent intro.
        # Instead, compute a token budget (``_SEC_PER_TOKEN_BUDGET``
        # sec/token, a conservative upper bound for sung phrasing) and
        # pull the first unplaced section's start to ``first_t -
        # budget`` rather than all the way back to 0.
        first_k, first_t = known[0]
        if first_k > 0:
            leading_tokens = sum(token_counts[:first_k]) or 1
            budget = _SEC_PER_TOKEN_BUDGET * leading_tokens
            t0 = max(0.0, float(first_t) - budget)
            t1 = first_t
            span = max(0.0, t1 - t0)
            cursor = t0
            for k in range(first_k):
                starts[k] = cursor
                cursor += span * (token_counts[k] / leading_tokens)
        starts[first_k] = first_t

        for (ka, ta), (kb, tb) in zip(known, known[1:]):
            # Section ka is already set to ta; distribute [ta .. tb) across
            # sections (ka + 1 .. kb - 1) then set starts[kb] = tb.
            if kb - ka <= 1:
                starts[kb] = tb
                continue
            middle_tokens = sum(token_counts[ka + 1 : kb]) or 1
            span = max(0.0, tb - ta)
            # Each section's share starts *after* ka's share — ka owns
            # tokens in [ka..ka+1), so the first interior section starts
            # at ta + (ka's_token_share). But since we only need section
            # starts (not ends), we can simply walk forward from ta.
            cursor = ta + span * (token_counts[ka] / (token_counts[ka] + middle_tokens))
            for k in range(ka + 1, kb):
                starts[k] = cursor
                cursor += span * (token_counts[k] / (token_counts[ka] + middle_tokens))
            starts[kb] = tb

        # Trailing gap: last known .. budget-capped end. Mirror of the
        # leading-gap cap. Without this, pinning section 7-of-10 and
        # leaving 8, 9, 10 unplaced spread them across the entire
        # remainder of the track, even if they realistically only
        # account for 20 s of content near the end.
        last_k, last_t = known[-1]
        if last_k < n_sections - 1:
            trailing_tokens = sum(token_counts[last_k + 1 :]) or 1
            budget = _SEC_PER_TOKEN_BUDGET * trailing_tokens
            t_end_budget = min(float(audio_end), float(last_t) + budget + float(token_counts[last_k]) * _SEC_PER_TOKEN_BUDGET)
            span_last = max(0.0, t_end_budget - last_t)
            # Reserve last_k's share first so the first unplaced section
            # starts strictly after ta.
            weight_self = float(token_counts[last_k])
            weight_rest = float(trailing_tokens)
            cursor = last_t + span_last * (weight_self / (weight_self + weight_rest))
            for k in range(last_k + 1, n_sections):
                starts[k] = cursor
                cursor += span_last * (token_counts[k] / (weight_self + weight_rest))

    # Enforce strictly-increasing starts. A monotonic sweep with a small
    # minimum gap keeps forced-alignment happy (zero-length windows
    # confuse CTC backtrack).
    min_gap = 0.1
    for k in range(1, n_sections):
        if starts[k] <= starts[k - 1]:
            starts[k] = starts[k - 1] + min_gap
    # Clamp to audio length.
    for k in range(n_sections):
        if starts[k] > audio_end - min_gap:
            starts[k] = max(0.0, audio_end - min_gap)
    # Re-sweep in case the clamp violated monotonicity at the tail.
    for k in range(1, n_sections):
        if starts[k] <= starts[k - 1]:
            starts[k] = min(audio_end, starts[k - 1] + min_gap)

    out: list[dict[str, Any]] = []
    for k in range(n_sections):
        lo, hi = bounds[k], bounds[k + 1]
        text = " ".join(word for word, _ in user_tokens[lo:hi])
        if not text:
            continue
        t_start = float(starts[k])
        t_end = float(starts[k + 1]) if k + 1 < n_sections else float(audio_end)
        if t_end - t_start < min_gap:
            t_end = t_start + min_gap
        out.append({"text": text, "start": t_start, "end": t_end})
    return out


# Vocal-activity floor: treat the whisper transcript as the source of
# truth for "where does singing actually begin". If whisper only picks
# up a single stray word near ``t=0`` and then goes silent for tens of
# seconds before the first sustained run of transcribed words, any
# forced-alignment window that starts in that silence will cram the
# user's lyrics onto intro noise. A dense-window detector on
# ``transcription_words`` + a post-build shift on ``forced_segments[0]``
# prevents that, without second-guessing the user when they've
# explicitly pinned an early line via ``[m:ss]``.
_VOCAL_FLOOR_MIN_WORDS = 3
_VOCAL_FLOOR_WINDOW_SEC = 5.0
_VOCAL_FLOOR_BUFFER_SEC = 0.5
_VOCAL_FLOOR_MIN_SHIFT_SEC = 2.0


def _first_dense_vocal_activity_time(
    transcription_words: Sequence[_WhisperWord],
    *,
    min_words: int = _VOCAL_FLOOR_MIN_WORDS,
    window_sec: float = _VOCAL_FLOOR_WINDOW_SEC,
    min_score: float = _MIN_WHISPER_WORD_SCORE,
) -> float | None:
    """Earliest time where ``min_words`` confident whisper-transcribed
    words occur within a ``window_sec`` sliding window.

    Interpretation: the first "real run of singing" as whisper heard
    it. One stray hallucinated word on intro noise fails the density
    test (one word ≠ sustained singing), so the function's output
    skips past it to the first place where the transcript actually
    carries content.

    Returns ``None`` when there are not enough confident words to
    identify any dense run — callers should treat that as "no
    opinion" and leave forced-alignment windows alone.
    """
    words = [
        w for w in transcription_words
        if (w.score is None or float(w.score) >= float(min_score))
    ]
    if len(words) < min_words:
        return None
    words_sorted = sorted(words, key=lambda w: float(w.t_start))
    for i, anchor in enumerate(words_sorted):
        t0 = float(anchor.t_start)
        j = i + 1
        count = 1
        while j < len(words_sorted):
            if float(words_sorted[j].t_start) - t0 > window_sec:
                break
            count += 1
            if count >= min_words:
                return t0
            j += 1
    return None


def _apply_vocal_activity_floor(
    forced_segments: list[dict[str, Any]],
    *,
    transcription_words: Sequence[_WhisperWord],
    line_anchors: Mapping[int, float] | None,
    buffer_sec: float = _VOCAL_FLOOR_BUFFER_SEC,
    min_shift_sec: float = _VOCAL_FLOOR_MIN_SHIFT_SEC,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    """Move ``forced_segments[0]['start']`` forward to the first dense
    whisper run when the current start would otherwise place user
    lyrics onto a silent intro.

    The floor only fires when:
      * There are enough confident whisper words to identify a dense
        first run (otherwise the function is a no-op).
      * The user has NOT pinned any line earlier than the floor via
        ``[m:ss]`` — an explicit early anchor wins over the heuristic.
      * The first segment's current start is at least
        ``min_shift_sec`` s before the floor (small gaps are left
        alone; they're within normal CTC noise).

    The shift is capped at ``segment.end - 0.2`` so the segment never
    collapses to zero duration; if the user's own pinning creates a
    conflict (their next segment starts before the floor), we leave
    segment 0 alone and log a warning — that's a user-data problem
    the aligner can't silently paper over.

    Returns ``(segments, floor, shift)`` where ``floor`` is the
    computed floor time (``None`` if unused) and ``shift`` is the
    applied shift in seconds (``None`` if none was applied).
    """
    if not forced_segments:
        return forced_segments, None, None
    floor_raw = _first_dense_vocal_activity_time(transcription_words)
    if floor_raw is None:
        return forced_segments, None, None
    floor = max(0.0, float(floor_raw) - float(buffer_sec))
    # Respect any user anchor earlier than the floor — the user knows
    # something the heuristic doesn't (e.g. a whispered intro word
    # that whisper missed).
    if line_anchors:
        earliest_user = min(float(t) for t in line_anchors.values())
        if earliest_user < floor:
            return forced_segments, floor, None
    first_seg = forced_segments[0]
    cur_start = float(first_seg.get("start", 0.0))
    cur_end = float(first_seg.get("end", cur_start + 0.1))
    if floor - cur_start < float(min_shift_sec):
        return forced_segments, floor, None
    # Cap so the segment retains at least 0.2 s of span.
    capped = min(floor, cur_end - 0.2)
    if capped <= cur_start:
        # Can't move forward at all — segment is inverted-shaped.
        return forced_segments, floor, None
    # If the user pinned a downstream segment early enough that we
    # can only do a *partial* shift that still leaves segment 0
    # starting well before real singing, a partial shift is worse
    # than useless (it just makes the window tighter without
    # fixing the root misalignment). Require the shift to reach
    # within a small tolerance of the true floor or abort.
    if floor - capped > 1.0:
        return forced_segments, floor, None
    shift = capped - cur_start
    first_seg["start"] = capped
    if cur_end - capped < 0.2:
        first_seg["end"] = capped + 0.2
    return forced_segments, floor, shift


def _run_whisperx_forced(
    vocals_wav: Path,
    user_tokens: list[tuple[str, int]],
    *,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    batch_size: int,
    progress: ProgressFn | None,
    user_section_starts: list[int] | None = None,
    line_anchors: dict[int, float] | None = None,
    use_silero_vad: bool = False,
    use_transcription_anchors: bool = True,
) -> tuple[list[_WhisperWord], str, list[_WhisperWord]]:
    """Run WhisperX forced alignment using the **user's pasted lyrics** as ground truth.

    Two-stage flow so every returned word is a user word with a real
    phoneme-level timing — no Needleman-Wunsch fallback interpolation for
    matched words:

    1. Transcribe the vocal stem once. We only use this to recover
       whisper's VAD-style segment boundaries and per-segment word lists;
       we throw away the transcribed **text** after the NW mapping step.
    2. For each whisper segment, assemble the slice of user lyrics that
       Needleman-Wunsch mapped into it, and pass those as the ``text``
       to :func:`whisperx.align`. CTC forced alignment then places each
       user word against the vocal audio phoneme-by-phoneme.

    Returns ``(aligned_words, language, transcription_words)``:

    * ``aligned_words`` — raw CTC-aligned words against the user's lyrics;
      the caller still runs ``_timings_for_user_tokens`` to handle the
      (hopefully tiny) cases where the aligner tokenised differently
      than :func:`_split_user_lyrics`.
    * ``transcription_words`` — per-word CTC timings for whisper's *own*
      transcribed text (i.e. what whisper heard). Empty list when
      ``use_transcription_anchors=False`` or the transcript CTC pass
      produced nothing. Persisted in ``lyrics.aligned.json`` so the
      visual editor can show it as ghost text above the user words.
    """
    whisperx = _import_whisperx()

    def _report(p: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, p)), msg)

    _report(0.20, f"Loading WhisperX {model_name} on {device}…")
    # Sung-vocals ASR/VAD overrides: the whisperx defaults are tuned for
    # speech and routinely drop real singing (see ``_SUNG_VOCALS_ASR_OPTIONS``
    # / ``_SUNG_VOCALS_VAD_OPTIONS`` docstrings). We deliberately do NOT
    # feed the user's pasted lyrics as ``initial_prompt`` — see the
    # comment on the constant definitions; intro hallucination is worse
    # than missed chorus repeats, which the section-fingerprint pass
    # resolves instead.
    asr_options: dict[str, Any] = dict(_SUNG_VOCALS_ASR_OPTIONS)
    vad_options: dict[str, Any] = dict(_SUNG_VOCALS_VAD_OPTIONS)
    try:
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
            vad_options=vad_options,
        )
    except TypeError as exc:
        # Older whisperx builds (< ~3.1) don't accept ``asr_options`` /
        # ``vad_options`` at load time. Fall back to the plain signature so
        # the pipeline still runs; log loudly so we notice when the tuned
        # decoding defaults silently stopped applying (which would look
        # like a regression in transcription word count).
        LOGGER.warning(
            "whisperx.load_model did not accept asr_options/vad_options "
            "(%s); falling back to plain load. Sung-vocals decoding "
            "tweaks disabled on this whisperx build — expect lower "
            "transcribed-word counts.",
            exc,
        )
        model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # The transcribe model (faster-whisper large-v3 FP16 ≈ 3 GB on CUDA)
    # and VAD are only needed through the ``model.transcribe`` call. Free
    # it before the align model is loaded so both don't stack in VRAM,
    # and free it before returning so the next pipeline stage (SDXL
    # background / AnimateDiff) doesn't land on top of resident weights.
    try:
        _report(0.30, "Loading vocals.wav for WhisperX…")
        audio = whisperx.load_audio(str(vocals_wav))
        audio_duration = float(len(audio)) / float(WHISPERX_SAMPLE_RATE)

        _report(0.40, "Transcribing vocals for segment boundaries…")
        try:
            transcribe_result = model.transcribe(
                audio, batch_size=batch_size, language=language
            )
        except TypeError:
            # Older whisperx signatures didn't accept ``language``; fall through
            # and let the model auto-detect. Still fine since the user told us
            # what it is — we just pass it explicitly to the aligner below.
            transcribe_result = model.transcribe(audio, batch_size=batch_size)
    finally:
        move_to_cpu(model)
        del model
        release_cuda_memory("whisperx transcribe model")

    detected_language = str(transcribe_result.get("language") or language)
    # Override whisper's detection with the user-supplied language: we're
    # English-only by product decision, and whisper occasionally misdetects
    # short / sparse vocals (especially ad-libs / melismatic passages).
    resolved_language = language or detected_language

    whisper_segments = _flatten_transcribe_segments(transcribe_result)
    # Widen segment ends into the adjacent silence before CTC runs, so
    # per-word timings aren't compressed inside faster-whisper's
    # aggressively-truncated windows. See ``_widen_segments_for_ctc``
    # for rationale.
    _widened_delta = 0.0
    if whisper_segments:
        orig_total = sum(float(s.t_end - s.t_start) for s in whisper_segments)
        whisper_segments = _widen_segments_for_ctc(
            whisper_segments, audio_duration=audio_duration
        )
        new_total = sum(float(s.t_end - s.t_start) for s in whisper_segments)
        _widened_delta = max(0.0, new_total - orig_total)
    _raw_word_count = sum(len(s.words) for s in whisper_segments)
    LOGGER.info(
        "WhisperX transcribe: %d segments, %d whisper words recognised, "
        "audio=%.1fs, widened +%.1fs of end-slack for CTC",
        len(whisper_segments),
        _raw_word_count,
        audio_duration,
        _widened_delta,
    )
    LOGGER.debug(
        "Post-widen segments (first 3): %s",
        [
            (round(s.t_start, 2), round(s.t_end, 2), len(s.words))
            for s in whisper_segments[:3]
        ],
    )
    # Preview the first handful of transcribed words so the quality of the
    # decode can be sanity-checked from the run log alone (no need to open
    # ``lyrics.aligned.json`` to see whether whisper actually heard the
    # right vocabulary). This is the primary diagnostic for tuning the
    # ``_SUNG_VOCALS_ASR_OPTIONS`` / ``_SUNG_VOCALS_VAD_OPTIONS`` constants.
    _preview: list[str] = []
    for _seg in whisper_segments:
        for _w in _seg.words:
            _preview.append(str(_w))
            if len(_preview) >= 15:
                break
        if len(_preview) >= 15:
            break
    LOGGER.info(
        "WhisperX first-words preview (%d/%d): %s",
        len(_preview),
        _raw_word_count,
        " ".join(_preview) if _preview else "(nothing recognised)",
    )
    # Silero VAD is trained on spoken speech; on *sung* material it tends
    # to reject sustained vowels / melisma / soft vocals, which shrinks
    # whisper's segment bounds and crams user words into too-narrow CTC
    # windows. We keep the refinement available (``use_silero_vad=True``)
    # for speech-style content but default it OFF for alignment quality.
    if use_silero_vad:
        whisper_segments = _refine_segments_with_vocal_vad(
            whisper_segments,
            vocals_wav=vocals_wav,
            audio_duration_sec=audio_duration,
        )
    segment_assignment = _assign_user_tokens_to_segments(
        user_tokens, whisper_segments
    )
    n_user_sections = (
        len(user_section_starts) if user_section_starts is not None else 1
    )
    if n_user_sections > 1:
        LOGGER.info(
            "User-section mode: %d sections from '---' markers; "
            "overriding whisper-VAD bucketing to prevent cross-section bleed",
            n_user_sections,
        )

    _report(0.70, f"Loading align model ({resolved_language})…")
    align_model, metadata = whisperx.load_align_model(
        language_code=resolved_language, device=device
    )

    # Hold the align model strictly for the duration of ``whisperx.align``;
    # wav2vec2 is ~500 MB on GPU and we must release it before returning so
    # the SDXL background stage has room to load cleanly. ``try/finally``
    # guarantees release even if align raises (e.g. CTC backtrack failure).
    #
    # We do up to TWO align passes against the same loaded model:
    #
    #   Pass A (optional, ``use_transcription_anchors``): align whisper's
    #   *own* transcribed text so we get per-word timings for what whisper
    #   heard. NW-matching those to the user's tokens gives us auto line
    #   anchors — robust "you said this line at this time" pins that the
    #   forced-alignment windowing treats exactly like user-supplied
    #   ``[m:ss]`` anchors.
    #
    #   Pass B (always): align the user's exact lyrics against the vocal
    #   audio, within per-section windows that now honour the merged
    #   anchors. This is the canonical per-word timing we persist.
    effective_line_anchors: dict[int, float] = dict(line_anchors or {})
    transcription_words: list[_WhisperWord] = []
    try:
        if use_transcription_anchors:
            try:
                _report(0.75, "Aligning whisper transcript for time anchors…")
                auto_anchors, transcription_words = (
                    _derive_transcription_anchors_via_align(
                        whisper_segments=whisper_segments,
                        audio=audio,
                        whisperx=whisperx,
                        align_model=align_model,
                        metadata=metadata,
                        device=device,
                        user_tokens=user_tokens,
                        existing_anchors=effective_line_anchors,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Transcription-anchor pass failed (continuing without "
                    "auto anchors): %s",
                    exc,
                )
                auto_anchors = {}
                transcription_words = []
            if auto_anchors:
                LOGGER.info(
                    "Transcription anchors: derived %d auto line pins "
                    "(user already pinned %d); %d whisper-transcribed words "
                    "persisted for editor overlay",
                    len(auto_anchors),
                    len(effective_line_anchors),
                    len(transcription_words),
                )
                for line_idx, t in auto_anchors.items():
                    # User anchors always win; only add where user didn't pin.
                    effective_line_anchors.setdefault(line_idx, t)

        # When the user has authored ``---`` sections, prefer the
        # section-fingerprint placement over word-level NW bucketing:
        # choruses that repeat get disambiguated by the DP's strict
        # temporal-ordering constraint, and an intro where whisper
        # hallucinated lyrics can no longer drag the first section's
        # window back to ``t=0``. When there are no markers (or only
        # one section after marker collapse), fall through to the
        # historical NW-bucket path which is still fine for unmarked
        # lyrics.
        forced_segments: list[dict[str, Any]] = []
        if user_section_starts is not None and len(user_section_starts) > 1:
            fingerprints = _extract_section_fingerprints(
                user_tokens, user_section_starts
            )
            flat_for_fp = _flatten_whisper_words_for_fingerprints(
                whisper_segments, transcription_words
            )
            flat_fingerprints: list[_SectionFingerprint] = [
                fp for section_fps in fingerprints for fp in section_fps
            ]
            matches = _find_fingerprint_matches_in_transcript(
                flat_fingerprints, flat_for_fp
            )
            # Translate user ``[m:ss]`` line-anchors into per-section
            # forced starts for the DP so user pins always win over
            # fingerprint picks.
            first_token_idx_by_line: dict[int, int] = {}
            for idx, (_w, line_idx) in enumerate(user_tokens):
                first_token_idx_by_line.setdefault(line_idx, idx)
            bounds = list(user_section_starts) + [len(user_tokens)]
            forced_sec: dict[int, float] = {}
            for line_idx, t in effective_line_anchors.items():
                tok = first_token_idx_by_line.get(line_idx)
                if tok is None:
                    continue
                for k in range(len(user_section_starts)):
                    if bounds[k] <= tok < bounds[k + 1]:
                        forced_sec.setdefault(k, float(t))
                        break
            chosen_starts = _assign_sections_via_temporal_dp(
                len(user_section_starts),
                matches,
                forced_section_starts=forced_sec,
            )
            LOGGER.info(
                "Section-fingerprint placement: n_sections=%d, "
                "fingerprints=%d (flat), matches=%d, forced=%d; "
                "chosen=%s",
                len(user_section_starts),
                len(flat_fingerprints),
                len(matches),
                len(forced_sec),
                [None if t is None else round(float(t), 2) for t in chosen_starts],
            )
            forced_segments = _build_section_windows_from_fingerprints(
                user_tokens,
                user_section_starts,
                audio_duration,
                chosen_starts,
                line_anchors=effective_line_anchors,
            )
        if not forced_segments:
            # No markers, or fingerprint flow produced nothing usable
            # (shouldn't happen — the builder always returns at least
            # one window — but keep the safety net so we never crash
            # forced alignment on a corrupted input).
            forced_segments = _build_forced_alignment_segments(
                user_tokens,
                segment_assignment,
                whisper_segments,
                audio_duration,
                user_section_starts=user_section_starts,
                line_anchors=effective_line_anchors,
            )

        # Transcription-as-source-of-truth floor: regardless of which
        # path built ``forced_segments`` above, if whisper's transcript
        # clearly shows there's no real singing until some time T and
        # the user hasn't pinned anything earlier than T, we refuse to
        # start the leading segment before T. This is the guard that
        # stops user lyrics from being placed on a pure-instrumental
        # intro or a single chopped-up vocal stab that whisper heard
        # as one stray word.
        forced_segments, vocal_floor, vocal_shift = _apply_vocal_activity_floor(
            forced_segments,
            transcription_words=transcription_words,
            line_anchors=effective_line_anchors,
        )
        if vocal_floor is not None:
            if vocal_shift is not None:
                LOGGER.info(
                    "Vocal-activity floor: first dense whisper run at "
                    "~%.2fs; shifted forced-alignment segment[0] start "
                    "forward by %.2fs (new=%.2fs)",
                    vocal_floor,
                    vocal_shift,
                    float(forced_segments[0]["start"]),
                )
            else:
                LOGGER.info(
                    "Vocal-activity floor: first dense whisper run at "
                    "~%.2fs; no shift applied (user anchor earlier, "
                    "gap below threshold, or segment boxed in)",
                    vocal_floor,
                )

        LOGGER.info(
            "Forced-alignment input: %d segments; first=%.2fs end=%.2fs "
            "last=%.2fs end=%.2fs",
            len(forced_segments),
            forced_segments[0]["start"] if forced_segments else 0.0,
            forced_segments[0]["end"] if forced_segments else 0.0,
            forced_segments[-1]["start"] if forced_segments else 0.0,
            forced_segments[-1]["end"] if forced_segments else 0.0,
        )
        _report(
            0.85,
            f"Forced-aligning {sum(len(s['text'].split()) for s in forced_segments)} "
            f"user words against {len(forced_segments)} vocal segments…",
        )
        aligned = whisperx.align(
            forced_segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
            print_progress=False,
        )

        # TEMP DEBUG (remove once the empty-word-segments bug is root-caused):
        # dumps the exact shape WhisperX 3.x is returning so we can see whether
        # word_segments is present, whether segments[*].words is populated, and
        # what the per-word dict keys look like. The existing ``WhisperX
        # transcribe:`` / ``Forced-alignment input:`` INFO logs are the permanent
        # ones; this block is intentionally temporary.
        _log_align_shape(aligned)

        words = _extract_whisper_words(aligned)
        LOGGER.info(
            "_extract_whisper_words: returned=%d words; first5=%s",
            len(words),
            [(w.word, round(w.t_start, 3), round(w.t_end, 3)) for w in words[:5]],
        )
    finally:
        move_to_cpu(align_model)
        del align_model
        release_cuda_memory("whisperx align model")
    if not words:
        # Diagnostic: alignment returned nothing. Raise so the user sees it
        # rather than silently getting a blank typography layer.
        whisper_seg_count = len(whisper_segments)
        whisper_word_count = sum(len(s.words) for s in whisper_segments)
        raise RuntimeError(
            "WhisperX forced alignment returned no word-level timings. "
            f"Input: {len(user_tokens)} user words across {len(forced_segments)} "
            f"alignment segments (whisper saw {whisper_seg_count} speech "
            f"segments / {whisper_word_count} words, audio={audio_duration:.1f}s); "
            f"language={resolved_language}. Check cache/<hash>/vocals.wav "
            "sounds right and that the pasted lyrics match the track."
        )
    return words, resolved_language, transcription_words


# ---------------------------------------------------------------------------
# Needleman-Wunsch alignment of normalised token sequences
# ---------------------------------------------------------------------------


def _needleman_wunsch(
    user_tokens_norm: list[str],
    whisper_tokens_norm: list[str],
) -> list[tuple[int | None, int | None]]:
    """
    Classic Needleman-Wunsch DP. Returns a list of ``(i, j)`` pairs where
    ``i`` indexes ``user_tokens_norm`` (or None for a whisper insertion) and
    ``j`` indexes ``whisper_tokens_norm`` (or None for a user deletion).
    The pairing is in sequence order.
    """
    m, n = len(user_tokens_norm), len(whisper_tokens_norm)
    # DP score matrix (m+1) x (n+1); use ints for speed.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i * NW_GAP_SCORE
    for j in range(1, n + 1):
        dp[0][j] = j * NW_GAP_SCORE
    for i in range(1, m + 1):
        ui = user_tokens_norm[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, n + 1):
            match = NW_MATCH_SCORE if ui == whisper_tokens_norm[j - 1] else NW_MISMATCH_SCORE
            diag = prev_row[j - 1] + match
            up = prev_row[j] + NW_GAP_SCORE
            left = row[j - 1] + NW_GAP_SCORE
            best = diag
            if up > best:
                best = up
            if left > best:
                best = left
            row[j] = best

    # Traceback.
    pairs: list[tuple[int | None, int | None]] = []
    i, j = m, n
    while i > 0 and j > 0:
        ui = user_tokens_norm[i - 1]
        wj = whisper_tokens_norm[j - 1]
        match = NW_MATCH_SCORE if ui == wj else NW_MISMATCH_SCORE
        if dp[i][j] == dp[i - 1][j - 1] + match:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + NW_GAP_SCORE:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    while i > 0:
        pairs.append((i - 1, None))
        i -= 1
    while j > 0:
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def _char_weight(normalised: str) -> float:
    """Return a speaking-duration weight for a normalised token.

    Character count is a cheap, language-neutral proxy for how long a word
    typically takes to sing/say — good enough for distributing a gap's total
    duration among unmatched user words. Min of 1 keeps single-letter / empty
    tokens from collapsing to zero share.
    """
    return float(max(1, len(normalised)))


def _fill_gap_timings(
    start_or_none: list[float | None],
    end_or_none: list[float | None],
    user_norm: list[str],
) -> None:
    """Fill ``None`` entries using character-weighted distribution.

    For each gap between two matched anchors, each unmatched user word gets
    a slice of ``(anchor_B.t_start - anchor_A.t_end)`` proportional to its
    character count. This is a strict improvement over uniform-by-index
    interpolation when WhisperX misses several words in a row: a short word
    like "the" no longer consumes the same timeline slot as "tremendously".

    Leading / trailing gaps (before the first anchor or after the last) fall
    back to the nearest anchor's timestamp — we have no second anchor to
    interpolate against, so this matches the old behaviour there.
    """
    n = len(start_or_none)
    if n == 0 or n != len(end_or_none) or n != len(user_norm):
        return
    known = [
        i for i in range(n)
        if start_or_none[i] is not None and end_or_none[i] is not None
    ]
    if not known:
        return  # caller handles the degenerate "no matches" case

    first = known[0]
    last = known[-1]
    lead_t = float(start_or_none[first])
    trail_t = float(end_or_none[last])
    for i in range(first):
        start_or_none[i] = lead_t
        end_or_none[i] = lead_t
    for i in range(last + 1, n):
        start_or_none[i] = trail_t
        end_or_none[i] = trail_t

    for k0, k1 in zip(known, known[1:]):
        if k1 == k0 + 1:
            continue
        v0 = float(end_or_none[k0])
        v1 = float(start_or_none[k1])
        if v1 <= v0:
            # Anchors touch or overlap — fall back to uniform-by-index so we
            # never produce negative durations.
            span = k1 - k0
            for j in range(1, span):
                t = v0 + (v1 - v0) * (j / span)
                start_or_none[k0 + j] = t
                end_or_none[k0 + j] = t
            continue
        weights = [_char_weight(user_norm[j]) for j in range(k0 + 1, k1)]
        total_w = sum(weights)
        if total_w <= 0.0:
            continue
        gap_duration = v1 - v0
        acc_w = 0.0
        for local, j in enumerate(range(k0 + 1, k1)):
            s = v0 + (acc_w / total_w) * gap_duration
            acc_w += weights[local]
            e = v0 + (acc_w / total_w) * gap_duration
            start_or_none[j] = s
            end_or_none[j] = e


def _timings_for_user_tokens(
    user_tokens: list[tuple[str, int]],
    whisper_words: list[_WhisperWord],
) -> list[tuple[float, float]]:
    """Map each user token to ``(t_start, t_end)`` by NW + gap interpolation."""
    timings, _scores = _timings_and_scores_for_user_tokens(user_tokens, whisper_words)
    return timings


def _timings_and_scores_for_user_tokens(
    user_tokens: list[tuple[str, int]],
    whisper_words: list[_WhisperWord],
) -> tuple[list[tuple[float, float]], list[float | None]]:
    """Map each user token to ``(t_start, t_end)`` plus a parallel list of
    wav2vec2 CTC scores. Scores are ``None`` for tokens whose timings were
    interpolated (unmatched in NW or filled from anchors). The visual editor
    uses the scores to colour-code low-confidence words.
    """
    m = len(user_tokens)
    n = len(whisper_words)

    if m == 0:
        return [], []
    if n == 0:
        return [(0.0, 0.0)] * m, [None] * m

    user_norm = [_normalise_token(word) for word, _ in user_tokens]
    whisper_norm = [_normalise_token(w.word) for w in whisper_words]

    pairs = _needleman_wunsch(user_norm, whisper_norm)

    # Accept a pairing as a "match" only when the normalised tokens are equal
    # AND both are non-empty; otherwise treat it as unaligned and interpolate.
    start_or_none: list[float | None] = [None] * m
    end_or_none: list[float | None] = [None] * m
    scores: list[float | None] = [None] * m
    for ui, wj in pairs:
        if ui is None or wj is None:
            continue
        if not user_norm[ui] or user_norm[ui] != whisper_norm[wj]:
            continue
        w = whisper_words[wj]
        start_or_none[ui] = w.t_start
        end_or_none[ui] = w.t_end
        scores[ui] = w.score

    # Fill gaps with character-weighted distribution so unmatched short words
    # don't hold the screen as long as unmatched long words.
    _fill_gap_timings(start_or_none, end_or_none, user_norm)

    # If nothing at all matched (e.g. totally wrong lyrics), spread whisper
    # timings uniformly across user tokens so timestamps stay within range.
    if all(v is None for v in start_or_none) or all(v is None for v in end_or_none):
        t0 = whisper_words[0].t_start
        t1 = whisper_words[-1].t_end
        if t1 <= t0:
            t1 = t0 + max(1.0, float(m))
        span = (t1 - t0) / max(1, m)
        out_timings: list[tuple[float, float]] = []
        for k in range(m):
            s = t0 + k * span
            e = s + span
            out_timings.append((s, e))
        return out_timings, [None] * m

    timings: list[tuple[float, float]] = []
    for s, e in zip(start_or_none, end_or_none):
        s_f = float(s) if s is not None else 0.0
        e_f = float(e) if e is not None else s_f
        if e_f < s_f:
            e_f = s_f
        timings.append((s_f, e_f))
    return timings, scores


def _enforce_monotonic_per_line(
    user_tokens: list[tuple[str, int]],
    timings: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Within each line, push timings forward so ``t_start`` is non-decreasing."""
    out: list[tuple[float, float]] = []
    last_end_by_line: dict[int, float] = {}
    for (_word, line_idx), (s, e) in zip(user_tokens, timings):
        last_end = last_end_by_line.get(line_idx)
        if last_end is not None and s < last_end:
            s = last_end
        if e < s:
            e = s
        out.append((s, e))
        last_end_by_line[line_idx] = e
    return out


def _polish_timings(
    user_tokens: list[tuple[str, int]],
    timings: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Final clean-up pass over aligned timings.

    * **Clip outro bleed between same-line neighbours.** If whisper assigned a
      word a ``t_end`` that runs past the next word's ``t_start``, the outro
      fade would overlap the next word on screen. Clip to
      ``next.t_start - ADJACENT_WORD_CLAMP_SEC`` so the two envelopes don't
      fight.
    * **Cap pathological durations.** WhisperX occasionally emits
      multi-second ``t_end`` values on sustained notes or fades; cap each
      word's duration at :data:`MAX_WORD_DURATION_SEC` so the word stops
      lingering on screen long after it was sung.

    Run *after* :func:`_enforce_monotonic_per_line` so both passes see the
    same, already-monotonic ``t_start`` sequence.
    """
    n = len(user_tokens)
    if n == 0 or len(timings) != n:
        return list(timings)
    out: list[tuple[float, float]] = [(float(s), float(e)) for s, e in timings]

    # Clip outro overlap with the next same-line word.
    for i in range(n - 1):
        line_i = user_tokens[i][1]
        line_j = user_tokens[i + 1][1]
        if line_i != line_j:
            continue
        s_i, e_i = out[i]
        s_j, _ = out[i + 1]
        limit = s_j - ADJACENT_WORD_CLAMP_SEC
        if e_i > limit:
            out[i] = (s_i, max(s_i, limit))

    # Cap single-word durations.
    for i in range(n):
        s, e = out[i]
        if e - s > MAX_WORD_DURATION_SEC:
            out[i] = (s, s + MAX_WORD_DURATION_SEC)

    return out


# Max distance a vocal onset can pull a word's start when snapping. Larger
# than a typical CTC precision error (20-30 ms) so genuine mis-alignments
# get corrected, but smaller than a syllable so we never walk a word onto
# a neighbour's consonant.
ONSET_SNAP_WINDOW_SEC = 0.08


def _snap_to_vocal_onsets(
    user_tokens: list[tuple[str, int]],
    timings: list[tuple[float, float]],
    onset_times: Sequence[float],
) -> list[tuple[float, float]]:
    """Snap each word's ``t_start`` to the nearest vocal onset within
    :data:`ONSET_SNAP_WINDOW_SEC`.

    Singers hit the consonant *on* the onset; CTC forced alignment often
    lands within 30-50 ms of that target but is consistently biased late
    on sustained vowels. Snapping to a real onset kills that "off by a
    frame" feel. We preserve each word's duration (shifting ``t_end`` by
    the same delta) and never cross a previous word's end within a line.

    ``onset_times`` must be sorted ascending. Words whose nearest onset
    is further than :data:`ONSET_SNAP_WINDOW_SEC` are left alone — we'd
    rather miss a correction than introduce a bigger error.
    """
    if not onset_times or not timings:
        return list(timings)
    import bisect

    onsets = list(onset_times)
    out: list[tuple[float, float]] = []
    last_end_by_line: dict[int, float] = {}
    for (_word, line_idx), (s, e) in zip(user_tokens, timings):
        # Nearest onset via binary search.
        pos = bisect.bisect_left(onsets, s)
        candidates: list[float] = []
        if pos > 0:
            candidates.append(onsets[pos - 1])
        if pos < len(onsets):
            candidates.append(onsets[pos])
        nearest = min(candidates, key=lambda o: abs(o - s)) if candidates else None
        new_s, new_e = s, e
        if nearest is not None and abs(nearest - s) <= ONSET_SNAP_WINDOW_SEC:
            delta = nearest - s
            candidate_s = s + delta
            candidate_e = e + delta
            # Don't cross the previous same-line word's end.
            prev_end = last_end_by_line.get(line_idx)
            if prev_end is None or candidate_s >= prev_end:
                new_s, new_e = candidate_s, candidate_e
        out.append((new_s, new_e))
        last_end_by_line[line_idx] = new_e
    return out


def _snap_to_line_anchors(
    user_tokens: list[tuple[str, int]],
    timings: list[tuple[float, float]],
    line_anchors: dict[int, float],
) -> list[tuple[float, float]]:
    """Hard-snap the first word of each anchored line to its anchor time.

    Honours the user's explicit ``[m:ss]`` pin regardless of what CTC
    produced. Preserves each anchored word's duration; subsequent words on
    the same line are pushed forward by the same delta so relative spacing
    is kept (monotonic enforcement runs afterwards to repair any overlap
    with the next line).
    """
    if not line_anchors or not timings:
        return list(timings)

    first_token_idx_by_line: dict[int, int] = {}
    for idx, (_word, line_idx) in enumerate(user_tokens):
        first_token_idx_by_line.setdefault(line_idx, idx)

    out: list[tuple[float, float]] = [(float(s), float(e)) for s, e in timings]
    for line_idx in sorted(line_anchors):
        tok_idx = first_token_idx_by_line.get(line_idx)
        if tok_idx is None:
            continue
        anchor_t = float(line_anchors[line_idx])
        current_s, _current_e = out[tok_idx]
        delta = anchor_t - current_s
        if abs(delta) < 1e-6:
            continue
        # Shift this word and every subsequent word on the same line by delta.
        for j in range(tok_idx, len(user_tokens)):
            if user_tokens[j][1] != line_idx:
                break
            s, e = out[j]
            out[j] = (s + delta, e + delta)
    return out


def _refine_segments_with_vocal_vad(
    whisper_segments: list[_WhisperSegment],
    *,
    vocals_wav: Path,
    audio_duration_sec: float,
) -> list[_WhisperSegment]:
    """Intersect whisper VAD segments with Silero-VAD speech windows.

    Whisper's VAD is built for general speech; on singing / backing-vocal
    / ad-lib stretches it routinely emits a single big segment that spans
    real vocals AND several seconds of instrumental, which is the main
    source of "typography pastes text during an instrumental break" bugs.
    Silero VAD is a purpose-built voice detector and much tighter.

    When :mod:`pipeline.vocal_vad` (Silero) is not available we return the
    whisper segments unchanged — this feature is strictly a quality win
    when the dep is installed, never a correctness requirement.

    The intersection widens pathologically narrow (<100 ms) sub-segments
    to at least 100 ms so CTC still has breathing room.
    """
    try:
        from pipeline.vocal_vad import detect_vocal_speech_spans
    except Exception as exc:  # noqa: BLE001
        LOGGER.info(
            "silero-vad not available — keeping whisper's VAD bounds "
            "(install silero-vad to enable tighter vocal segmentation): %s",
            exc,
        )
        return whisper_segments
    try:
        speech_spans = detect_vocal_speech_spans(vocals_wav)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "silero-vad failed on %s — keeping whisper VAD bounds: %s",
            vocals_wav,
            exc,
        )
        return whisper_segments
    if not speech_spans:
        LOGGER.info(
            "silero-vad returned no speech — this usually means the vocal "
            "stem is silent; keeping whisper VAD bounds as the safest fallback."
        )
        return whisper_segments

    refined: list[_WhisperSegment] = []
    for seg in whisper_segments:
        # Find every silero span overlapping the whisper segment.
        overlaps: list[tuple[float, float]] = []
        for span_start, span_end in speech_spans:
            if span_end <= seg.t_start or span_start >= seg.t_end:
                continue
            lo = max(seg.t_start, span_start)
            hi = min(seg.t_end, span_end)
            if hi - lo < 1e-3:
                continue
            overlaps.append((lo, hi))
        if not overlaps:
            # Whisper thought there was speech here but silero disagrees —
            # drop the segment. Any user tokens that NW-landed here will
            # either be reassigned (if other whisper segments exist) or
            # fall through to the full-audio fallback inside the builder.
            continue
        # Keep the segment as one unit but shrink its bounds to the
        # smallest-to-largest speech overlap. We don't split here because
        # splitting would require remapping whisper's word lists across the
        # new sub-segments, which is messier than it's worth — one tighter
        # window per original whisper segment already fixes the bleed.
        new_start = overlaps[0][0]
        new_end = overlaps[-1][1]
        if new_end - new_start < 0.1:
            new_end = new_start + 0.1
        refined.append(
            _WhisperSegment(
                idx=seg.idx,
                t_start=float(new_start),
                t_end=float(new_end),
                words=seg.words,
            )
        )
    if not refined:
        LOGGER.warning(
            "silero-vad rejected every whisper segment on %s; keeping "
            "originals so alignment still runs.",
            vocals_wav,
        )
        return whisper_segments
    total_orig = sum(s.t_end - s.t_start for s in whisper_segments)
    total_refined = sum(s.t_end - s.t_start for s in refined)
    LOGGER.info(
        "silero-vad: %d whisper segments → %d refined; "
        "total speech time %.1fs → %.1fs",
        len(whisper_segments),
        len(refined),
        total_orig,
        total_refined,
    )
    return refined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_cached_alignment(
    aligned_json: Path,
    *,
    song_hash: str,
    lyrics_sha: str,
    allow_manual_override: bool = True,
) -> dict[str, Any] | None:
    """Return the cached payload when it matches the song + canonical lyrics.

    Two escape hatches from the usual ``lyrics_sha256`` check:

    * ``manually_edited: true`` — the visual editor saved this file after a
      user edit. We never silently regenerate those; only ``force=True``
      (the "Re-align from scratch" button) overwrites them. The lyrics text
      may also have drifted slightly (the user can edit both); the editor
      records the text snapshot it was saved against under
      ``manual_lyrics_sha256`` so we can still detect *material* lyric
      changes vs. cosmetic whitespace.
    """
    if not aligned_json.is_file():
        return None
    try:
        with aligned_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read cached alignment %s: %s", aligned_json, exc)
        return None
    if not isinstance(data, dict):
        return None
    if data.get("song_hash") != song_hash:
        return None
    if data.get("schema_version") != LYRICS_ALIGNED_SCHEMA_VERSION:
        return None

    is_manual = bool(data.get("manually_edited"))
    if is_manual and allow_manual_override:
        # Respect manual edits even if the cache-key lyrics have drifted —
        # provided the user hasn't materially rewritten the words. We use
        # the manual_lyrics_sha256 the editor stored at save time if
        # present; otherwise fall back to the regular sha.
        manual_sha = data.get("manual_lyrics_sha256") or data.get("lyrics_sha256")
        if manual_sha == lyrics_sha:
            return data
        LOGGER.warning(
            "Cached alignment for %s is manually edited but lyrics hash "
            "changed (%s → %s); keeping the manual file and returning it. "
            "Click 'Re-align from scratch' if you want WhisperX to rebuild.",
            aligned_json,
            manual_sha,
            lyrics_sha,
        )
        return data

    if data.get("lyrics_sha256") != lyrics_sha:
        return None
    return data


def align_lyrics(
    cache_dir: Path | str,
    lyrics_text: str,
    *,
    force: bool = False,
    device: str | None = None,
    model_name: str = DEFAULT_WHISPER_MODEL,
    compute_type: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: str = DEFAULT_LANGUAGE,
    progress: ProgressFn | None = None,
    use_silero_vad: bool = False,
    use_onset_snap: bool = False,
    use_transcription_anchors: bool = True,
) -> AlignmentResult:
    """
    Align pasted ``lyrics_text`` to the demucs ``vocals.wav`` in ``cache_dir``
    using WhisperX **forced alignment against the user's lyrics**, and persist
    the result as ``lyrics.aligned.json``.

    The user's original word spelling / punctuation / case is preserved; every
    returned timestamp comes from wav2vec2 CTC forced alignment of the user's
    exact text against the vocal stem (whisper is only used to pre-segment
    the audio). Cache is reused when the song hash + canonical lyric tokens
    match, unless ``force=True``.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir does not exist: {cache}")

    vocals_wav = cache / VOCALS_WAV_NAME
    if not vocals_wav.is_file():
        raise FileNotFoundError(
            f"Missing {VOCALS_WAV_NAME} in {cache}; run Analyze with demucs "
            "available first so the aligner has a vocal stem to work on"
        )

    if not lyrics_text or not lyrics_text.strip():
        raise ValueError("lyrics_text is empty; nothing to align")

    song_hash = cache.name
    aligned_json_path = cache / LYRICS_ALIGNED_JSON_NAME
    lines, user_tokens, section_starts, line_anchors = _split_user_lyrics(lyrics_text)
    if not user_tokens:
        raise ValueError("lyrics_text has no word tokens after tokenisation")

    lyrics_sha = _lyrics_cache_key(lyrics_text)

    def _report(p: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, p)), msg)

    if not force:
        cached = _load_cached_alignment(
            aligned_json_path, song_hash=song_hash, lyrics_sha=lyrics_sha
        )
        if cached is not None:
            manual = bool(cached.get("manually_edited"))
            # Log a fingerprint of the first few word timings we're loading
            # so "did the renderer actually pick up my manual edits?" is a
            # one-grep answer in the Gradio log / console output instead of
            # a 2-hour re-render detective story.
            _probe: list[str] = []
            for _i, _w in enumerate((cached.get("words") or [])[:3]):
                if not isinstance(_w, dict):
                    continue
                _probe.append(
                    f"[{_i}]{_w.get('word', '?')!r}@{float(_w.get('t_start', 0.0)):.3f}s"
                )
            LOGGER.info(
                "Loaded cached alignment from %s (manually_edited=%s, %d words); "
                "first: %s",
                aligned_json_path,
                manual,
                len(cached.get("words") or []),
                ", ".join(_probe) if _probe else "(none)",
            )
            msg = (
                "Using manually-edited lyrics.aligned.json "
                "(click Re-align from scratch to discard your edits)"
                if manual
                else "Using cached lyrics.aligned.json"
            )
            _report(1.0, msg)
            words = [
                AlignedWord(
                    word=str(w.get("word", "")),
                    line_idx=int(w.get("line_idx", 0)),
                    t_start=float(w.get("t_start", 0.0)),
                    t_end=float(w.get("t_end", 0.0)),
                    score=(
                        float(w["score"])
                        if isinstance(w.get("score"), (int, float))
                        else None
                    ),
                )
                for w in cached.get("words", [])
                if isinstance(w, dict)
            ]
            return AlignmentResult(
                song_hash=song_hash,
                cache_dir=cache,
                aligned_json=aligned_json_path,
                language=str(cached.get("language", "en")),
                model=str(cached.get("model", model_name)),
                lines=list(cached.get("lines", lines)),
                words=words,
            )

    device_resolved = _pick_device(device)
    compute_type_resolved = compute_type or _default_compute_type(device_resolved)

    _report(0.05, "Preparing lyrics / vocals for alignment…")

    whisper_words, resolved_language, transcription_words = _run_whisperx_forced(
        vocals_wav,
        user_tokens,
        model_name=model_name,
        language=language,
        device=device_resolved,
        compute_type=compute_type_resolved,
        batch_size=batch_size,
        progress=progress,
        user_section_starts=section_starts,
        line_anchors=line_anchors,
        use_silero_vad=use_silero_vad,
        use_transcription_anchors=use_transcription_anchors,
    )

    _report(0.92, "Reconciling forced-alignment output with pasted lyrics…")
    # With forced alignment, ``whisper_words`` is already (almost) a 1-to-1
    # match to ``user_tokens`` — NW is here as a safety net that handles
    # tokeniser edge cases (e.g. the aligner splitting "don't" differently
    # than :func:`_split_user_lyrics`).
    timings, scores = _timings_and_scores_for_user_tokens(user_tokens, whisper_words)
    timings = _enforce_monotonic_per_line(user_tokens, timings)
    timings = _polish_timings(user_tokens, timings)

    # Pull user-anchored line starts into the final timings, then snap to
    # vocal onsets for sub-frame precision where we have them. Order matters:
    # anchors first (hard pin), then onsets (soft pull toward a consonant),
    # then a final monotonic pass to repair any crossings the snaps caused.
    if line_anchors:
        timings = _snap_to_line_anchors(user_tokens, timings, line_anchors)

    # Vocal-onset snap is disabled by default: on sung material, librosa
    # onsets include percussion bleed, vibrato crests, and syllable-internal
    # consonants, and a naive pull-toward-nearest-onset biases timings off
    # real word starts. The code path is preserved for future tuning /
    # speech-style content (``use_onset_snap=True``). The precise CTC
    # frames from wav2vec2 are usually the better answer.
    if use_onset_snap:
        onset_times: list[float] = []
        try:
            from pipeline.vocal_onsets import compute_vocal_onsets

            onset_times = list(compute_vocal_onsets(vocals_wav))
        except Exception as exc:  # noqa: BLE001
            LOGGER.info(
                "Vocal-onset snap skipped (non-fatal): %s. Word starts will "
                "remain at wav2vec2's exact CTC frames.",
                exc,
            )
        if onset_times:
            timings = _snap_to_vocal_onsets(user_tokens, timings, onset_times)

    timings = _enforce_monotonic_per_line(user_tokens, timings)

    words = [
        AlignedWord(
            word=raw_word,
            line_idx=line_idx,
            t_start=float(t0),
            t_end=float(t1),
            score=sc,
        )
        for (raw_word, line_idx), (t0, t1), sc in zip(user_tokens, timings, scores)
    ]

    payload: dict[str, Any] = {
        "schema_version": LYRICS_ALIGNED_SCHEMA_VERSION,
        "song_hash": song_hash,
        "model": model_name,
        "language": resolved_language,
        "vocals_wav": VOCALS_WAV_NAME,
        "lyrics_sha256": lyrics_sha,
        "manually_edited": False,
        "lines": lines,
        "words": [w.to_dict() for w in words],
        # Whisper's own transcribed text with CTC timings, preserved so
        # the visual editor can show "what whisper heard" as ghost labels
        # above the user words. Not used by the rendering pipeline.
        "whisper_words": [
            {
                "word": ww.word,
                "t_start": float(ww.t_start),
                "t_end": float(ww.t_end),
                "score": (
                    float(ww.score) if ww.score is not None else None
                ),
            }
            for ww in transcription_words
        ],
    }

    _report(0.97, "Writing lyrics.aligned.json…")
    tmp_path = aligned_json_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    tmp_path.replace(aligned_json_path)

    _report(1.0, "Lyrics alignment complete")
    return AlignmentResult(
        song_hash=song_hash,
        cache_dir=cache,
        aligned_json=aligned_json_path,
        language=resolved_language,
        model=model_name,
        lines=lines,
        words=words,
    )
