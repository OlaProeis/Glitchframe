# Lyrics aligner

Feature: align user-pasted lyrics to the demucs `vocals.wav` using WhisperX
word-level timestamps, and persist `cache/<hash>/lyrics.aligned.json` for the
kinetic typography layer to consume.

## Dependencies (platform)

On **Windows** with **Python 3.11 or 3.12**, the `lyrics` / `all` / `analysis` extras pin **WhisperX 3.3.0**, **faster-whisper 1.1.0**, **ctranslate2 4.4.0**, and **PyTorch 2.2.2+cu121** (CUDA 12.1) so faster-whisper / CTranslate2 and PyTorch agree on cuDNN DLL names. **Python 3.13** on Windows and **Linux/macOS** use the looser pins in `pyproject.toml` (typically **cu124** and **ctranslate2≥4.5**). Runtime code still tolerates older WhisperX builds that omit `vad_method` / `asr_options`.

## Flow

The aligner uses **forced alignment against the user's pasted lyrics** — the
user's exact text is what wav2vec2 CTC aligns against the audio, so every
output word is one of their words with a phoneme-level timestamp. Whisper's
own transcription is used only to recover VAD-style segment boundaries; its
text is discarded after the Needleman-Wunsch step that maps user tokens onto
those boundaries.

1. Requires `cache/<hash>/vocals.wav` produced by the audio analyzer's demucs
   stem (the aligner raises if it is missing).
2. Runs `whisperx.load_model("large-v3", device, compute_type=...)` on the
   vocal stem (`float16` on CUDA, `int8` on CPU by default) **only** to
   recover speech-activity segment boundaries and whisper's recognised
   words per segment. The `language` hint defaults to `"en"`
   (`DEFAULT_LANGUAGE`). The load call passes two sung-vocals overrides
   (see [Sung-vocals decoding defaults](#sung-vocals-decoding-defaults)):
   - `asr_options` tuned for singing: `no_speech_threshold=0.15`,
     `compression_ratio_threshold=100.0`, `log_prob_threshold=-2.0`,
     `condition_on_previous_text=False`. We deliberately do **not**
     pass the user's lyrics as `initial_prompt` — see the note on that
     row of the table for why.
   - `vad_options` tuned for a near-clean vocal stem: `vad_onset=0.200`,
     `vad_offset=0.100`.

   Older whisperx builds (< ~3.1) that don't accept `asr_options` /
   `vad_options` at load time fall back to the plain signature; a
   `WARNING` is logged so a missing tuning silently regressing the
   transcribed-word count is noticeable from the run log alone.
3. Tokenises the pasted lyrics (blank lines split `line_idx`; whitespace splits
   words; empty lines are skipped). A line whose trimmed content starts with
   `---` (optionally followed by a human-readable tag like
   `--- instrumental` / `--- chorus`) is recognised as an explicit
   **section marker** and collapsed into `section_starts` — see
   [Section markers](#section-markers-). A line whose trimmed content starts
   with `[m:ss]` / `[m:ss.mmm]` / `[h:mm:ss]` is recognised as an
   **inline anchor** — see [Inline anchors](#inline-anchors).
4. Normalises both user tokens and whisper's recognised words (`NFKD` → strip
   combining marks → case-fold → strip everything that isn't `[0-9a-z]`) and
   runs **Needleman-Wunsch** between them to decide which whisper segment
   each user word belongs in. Unmatched user words inherit the segment of
   the nearest matched neighbour (forward fill, then backward fill for any
   leading unanchored run). When whisper recognised nothing usable the full
   user text is fed into a single fallback segment spanning the stem.
5. **(Optional, `use_silero_vad=True`)** Tightens whisper's VAD using Silero
   VAD on the same `vocals.wav` (`pipeline.vocal_vad.detect_vocal_speech_spans`).
   Silero is a small ONNX voice-detection model — ~1 MB, no GPU required. In
   practice Silero is trained on *spoken* speech and routinely under-detects
   on *sung* material (sustained vowels, melisma, soft vocals read as silence),
   which shrinks whisper segments and crams CTC words into too-narrow windows.
   The refinement therefore defaults **OFF**; enable it only for
   speech-adjacent content (spoken-word tracks, podcasts mixed as songs).
   When on, whisper segments are intersected with Silero speech spans;
   segments that overlap no Silero speech are discarded. Silero results are
   cached beside the stem as `vocal_vad.json` keyed by the stem's mtime.
6. **(Default ON, `use_transcription_anchors=True`)** Runs a second CTC
   alignment pass against **whisper's own transcribed text** to produce
   per-word timings for "what whisper heard". Those whisper words are
   NW-matched against the user's tokens (same normalisation as step 4); for
   each user line whose **first** token has a confident identity match the
   helper (`_derive_line_anchors_from_transcription`) emits an auto line
   anchor at the matched whisper word's start time. These auto anchors are
   merged into `line_anchors` (user-supplied `[m:ss]` pins always win), so
   every downstream segment-window clamp that honours a manual anchor also
   honours this auto one. Net effect: long-distance drift across section
   boundaries — the dominant failure mode on sung vocals — is pinned to the
   language-model-aware timings before CTC runs on the user's exact text.
7. Builds one `{text, start, end}` dict per forced-alignment bucket and
   passes them to the same `whisperx.load_align_model` + `whisperx.align`
   model that ran step 6 (the align model is loaded once, reused for both
   passes, then released). Wav2vec2 CTC forced alignment places each user
   word against the audio frame-by-frame within its segment window.
   Bucketing follows one of two rules depending on whether the user
   included markers:
   * **No markers** (default) — one bucket per non-empty whisper VAD
     segment. Skipped buckets handle instrumental passages transparently.
   * **Markers present** — one bucket per user-authored section, placed
     on the timeline by **section-level fingerprint matching + a
     monotonic DP** (see
     [Section-fingerprint placement](#section-fingerprint-placement---path)).
     Each section contributes 2-3 characteristic 3-5-word phrases; those
     are fuzzy-matched against whisper's CTC-aligned transcript, and a
     DP picks one time per section that maximises total match score
     while keeping the sequence strictly monotonic. Sections without a
     usable match are filled by token-weighted interpolation from
     their placed neighbours. This path is what prevents repeated
     choruses from collapsing onto each other and stops intro-noise
     hallucinations from dragging the first section back to `t=0`.
   * **Inline anchors present** — an `[m:ss]` on a section's first line
     is fed to the DP as an immovable forced start (score 2.0 dominates
     any fingerprint match capped at 1.0), so user pins always win over
     automatic placement and the neighbouring sections are forced to
     respect the pin's ordering. Anchors without an explicit `---`
     marker still synthesise implicit section breaks in the parser, so
     the same pin behaviour applies.
8. Runs the **polish** safety net on the aligner's output:
   * `_timings_for_user_tokens` runs another NW pass to reconcile the rare
     case where the aligner tokenised differently than
     `_split_user_lyrics` (e.g. "don't" → "don" + "t"). In the common case
     this is a trivial 1-to-1 mapping. Wav2vec2 **per-word CTC scores**
     from `whisperx.align` are carried through the match step; unmatched
     tokens (filled via interpolation) keep a `score: None` so the editor
     can distinguish real CTC matches from interpolated stand-ins.
   * Unmatched / dropped user tokens are backfilled with
     **character-weighted** gap interpolation: each unmatched word claims a
     share of the gap duration proportional to its character count, so
     unmatched short words ("the") no longer occupy the same slot as long
     ones ("tremendously"). Falls back to uniform-by-index when anchors
     overlap (negative gap duration).
   * `_enforce_monotonic_per_line` nudges timings forward so `t_start` is
     non-decreasing and `t_end >= t_start` within a `line_idx`.
   * `_polish_timings`:
     * Clips outro bleed — when a word's `t_end` would overlap the next
       same-line word's `t_start`, `t_end` is clamped to
       `next.t_start - ADJACENT_WORD_CLAMP_SEC` (~30 ms) so the typography
       layer's outro fade never fights the incoming word.
     * Caps pathological single-word durations at `MAX_WORD_DURATION_SEC`
       (2 s). Guards against the aligner occasionally emitting multi-second
       `t_end` values on sustained notes.
   * `_snap_to_line_anchors` hard-pins each anchored line's first word to
     the user-supplied time, shifting the rest of that line by the same
     delta. Runs only when the user has inline `[m:ss]` anchors.
   * **(Optional, `use_onset_snap=True`)** `_snap_to_vocal_onsets` pulls
     each word's `t_start` to the nearest vocal-stem onset within
     `ONSET_SNAP_WINDOW_SEC` (80 ms). Onsets are computed from `vocals.wav`
     with a voice-tuned librosa configuration
     (`pipeline.vocal_onsets.compute_vocal_onsets`) and cached beside the
     stem as `vocal_onsets.json`. Defaults **OFF**: on sung material the
     onset envelope picks up cymbal bleed, vibrato crests, and
     syllable-internal consonant transitions, and the pull-toward-nearest-
     onset biases `t_start` off the real word start. The code path is
     kept available for future tuning on speech-style content.
   * A final `_enforce_monotonic_per_line` pass repairs any overlap the
     anchor snaps created.
9. Writes `cache/<hash>/lyrics.aligned.json` atomically via a `.json.tmp`
   rename, matching the analyzer's pattern.

The user's **original** word spelling / punctuation / case is preserved on
output — normalisation is only used for matching.

## Section markers (`---`)

The aligner recognises a line whose trimmed content is exactly `---` as an
**explicit section break**. Markers are optional; lyrics without them align
exactly as they did before.

When to use them: WhisperX's VAD sometimes merges repeated musical sections
(e.g. pre-chorus buildup → chorus 1 → verse 2 → pre-chorus → chorus 2) into
a single large speech segment. Without markers, the Needleman-Wunsch token
mapping has to stuff every user word for every merged section into that one
whisper window, which is exactly when the repeated-chorus bleed happens.
With markers, the aligner keeps each user-authored section in its own
forced-alignment bucket and splits the shared window proportionally to each
section's token count, so CTC places each section's words inside its own
slice of the timeline.

Rules:

- A marker line starts with `---`; anything after that prefix is a
  free-form human tag (documentation only). `---`, `--- chorus`,
  `--- instrumental`, `--- solo`, and `--- bridge` are all equivalent from
  the aligner's perspective — they all collapse into a section break.
- `----` (four hyphens) and `-- -` are **normal lyric lines** — the prefix
  must be exactly three hyphens.
- Leading markers (before any lyric token), trailing markers (after the last
  lyric token), and consecutive markers collapse into a single break so no
  empty section is ever emitted.
- Markers are stripped from `lines` and from the rendered typography — they
  only exist as a forced-alignment hint.
- Adding or moving a marker changes the canonical cache key; the next Align
  lyrics click will re-run WhisperX. Unmarked lyrics continue to hash the
  same as they did before this feature, so existing caches stay valid.

Example (the real-world trigger — a vocal run that gets mistaken for the
chorus start):

```
…
Ready for the night
Everything feels right
Turn it up yeah
Summer selecta ecta ecta ecta ecta e e e e e e
---
Turn it up, turn it up
…
```

Without the `---` line, whisper merges the pre-chorus buildup with the
first chorus bar, and the second chorus inherits the compression. With the
marker, `whisperx.align` gets two separate windows (pre-chorus, chorus),
each sized by token count, and CTC aligns inside each one independently.

## Inline anchors

Start any lyric line with `[m:ss]`, `[m:ss.mmm]`, or `[h:mm:ss]` to pin
that line's first word to the given audio time:

```
[0:12]    Verse line one
Everyday thoughts on my mind
[1:04]    Chorus says let it go
```

The bracket block is stripped from the rendered lyric text and never
appears in the typography layer. Each anchor becomes a **hard bound**
during forced alignment — the window up to the anchored line ends at the
anchor, and the next section starts there — so the user can reliably
correct a misplaced section without touching the visual editor. Anchors
also feed a post-CTC snap pass that pins the anchored word's `t_start`
exactly to the user's time, in case the alignment inside the window
drifted by a few frames.

A bracket-only line (`[0:42]` on its own) is treated as an anchor applied
to the *next* lyric line, which is handy when the anchor visually reads
better on its own line. Brackets that appear in the middle of a line
(e.g. `hello [whispered] world`) are left as normal lyric text.

Anchors and `---` markers compose cleanly: an anchor on a marked
section's first line overrides that section's start; an anchor without
any `---` markers auto-synthesises a section break at the anchored line
so the pin behaviour still applies.

## lyrics.aligned.json schema (v3)

```json
{
  "schema_version": 3,
  "song_hash": "<sha256>",
  "model": "large-v3",
  "language": "en",
  "vocals_wav": "vocals.wav",
  "lyrics_sha256": "<sha of canonical tokens+markers+anchors>",
  "manually_edited": false,
  "lines": ["First line as pasted", "Second line", "..."],
  "words": [
    {"word": "Hello", "line_idx": 0, "t_start": 0.50, "t_end": 0.82, "score": 0.94},
    "..."
  ]
}
```

Per-word fields:

- `word`: the user's original spelling / case / punctuation.
- `line_idx`: index into `lines` (after dropping section markers and
  bracket-only anchor lines).
- `t_start`, `t_end`: seconds into `vocals.wav`. Monotonic within a line;
  may overlap across line boundaries (cross-line fade handles that).
- `score`: wav2vec2 CTC confidence from `whisperx.align` (≈ 0..1). Omitted
  when the word was filled via character-weighted interpolation (i.e. the
  user pasted a word whisper didn't transcribe). The [timeline editor](./lyrics-timeline-editor.md)
  colour-codes bars by this field so low-confidence words are obvious.

Top-level `manually_edited` is set to `true` when the file was written
by the visual editor. The aligner honours that flag on cache load: unless
the user clicks **Re-align from scratch**, subsequent Align clicks return
the manual edits unchanged.

## Transcription anchors (auto line pins)

Forced-alignment CTC is blazingly precise *inside* its window, but it has
no language-model awareness: it accepts whatever text you feed it and
crams the phoneme sequence into the audio window, no matter how wrong the
window is. Whisper, in contrast, is a full speech-to-text model that
*hears* the song and outputs what it thinks was said — with per-word
timings once you align that text back to audio.

The **transcription-anchor** pass exploits that: it runs one additional
CTC pass on **whisper's own transcribed text** to get per-word timings
for "what whisper heard", then Needleman-Wunsch-matches those to the
user's tokens. For each user line, the *first matching* token on that
line produces a **line anchor** — identical in effect to the user typing
`[m:ss]` at the start of that line. When the match is the line's first
token the whisper word's `t_start` is used directly; when the match is
a later token on the line, the anchor is back-estimated by subtracting
`local_pos * per_word_duration` (per-word duration clamped to
`[0.18 s, 0.60 s]` so a long melisma can't produce a wildly negative
anchor).

These auto anchors feed the existing per-section window plumbing, so the
final CTC pass on the user's exact lyrics is bucketed into much tighter
per-line windows than whisper's coarse VAD segments alone would provide.
That's the single biggest lever against long-distance drift on sung
material: you get the language model's layout anchoring CTC's phoneme
precision.

### Fuzzy matching

Token matching is **loosened past strict string equality** because sung
vocals are lossy: whisper writes `were` for `"we're"`, `comin` for
`"coming"`, `till` for `"til"`, and so on. Under strict equality, most
of these near-misses never anchor and the forced-alignment pass drifts
on sung material. `_tokens_are_fuzzy_equal` accepts:

- Exact match after normalisation (the common case).
- Mutual prefix of length ≥ 3 chars — catches chopped endings like
  `comin` / `coming`, `till` / `til`.
- Levenshtein distance 1 for tokens of length ≥ 3 — catches typical
  single-char substitutions / insertions / deletions.

Short tokens (< 3 chars) still require an exact match, so `"a"` and
`"i"` (Levenshtein-1 apart) don't falsely fuse. The predicate is
symmetric and is applied to already-normalised token pairs emerging
from Needleman-Wunsch.

### Whisper-word persistence

`_derive_transcription_anchors_via_align` returns both the anchors and
the CTC-timed whisper-transcribed word list. Both are filtered through
`_filter_confident_whisper_words` (drop `score is None` and `score <
0.25`) before use, because forced alignment always places every token
in its input somewhere in the audio window — even tokens whisper
hallucinated or that CTC couldn't actually align. Without the filter,
those junk placements showed up as ghost text drifting over the wrong
audio in the editor, and occasionally produced bad line anchors when
NW happened to match a hallucinated token to the user's lyrics.

The surviving words are serialised into `lyrics.aligned.json` under a
top-level `whisper_words` key and consumed by the visual editor as a
ghost-text overlay on the waveform, letting the user manually re-align
against "what whisper heard" without listening back to the track. The
editor scales each label's opacity by its CTC score so the user can
tell at a glance which labels to trust. The main rendering pipeline
ignores `whisper_words` entirely — it's reference data for the editor.

### Other rules / edge cases

- User-supplied `[m:ss]` anchors always win. The auto pass uses
  `setdefault` into the merged anchor dict, so manual pins are
  untouchable.
- The pass reuses the same loaded align model as the final CTC pass
  (loaded once, two aligns, released once). Overhead is ~1× extra
  `whisperx.align` call; ~1.5–2× total alignment wall-clock on a first
  cache miss. Subsequent re-runs still hit the `lyrics.aligned.json`
  cache so the extra cost only lands on fresh alignments.
- Failure of the extra align pass is non-fatal: the outer flow logs a
  warning and proceeds with whatever manual anchors exist. The user-text
  CTC pass still runs, and the persisted `whisper_words` field is left
  empty so the editor simply shows no ghost labels.

Flag: `align_lyrics(..., use_transcription_anchors=True)` (default ON).

## Sung-vocals decoding defaults

WhisperX's out-of-the-box decoding gates are tuned for spoken speech and
silently throw away large amounts of real singing on the demucs vocal
stem. The aligner therefore passes a small set of overrides at load
time via `whisperx.load_model(..., asr_options=..., vad_options=...)`:

| Key | whisperx default | Our override | Why it matters on singing |
| --- | --- | --- | --- |
| `no_speech_threshold` | `0.6` | `0.15` | Held vowels / melismatic passages routinely trip whisper's no-speech head even when the vocal is clearly audible; the lower bar keeps them. |
| `compression_ratio_threshold` | `2.4` | `100.0` | A legitimate chorus repeating the same hook 4-6× has a very high zlib compression ratio; the default treats it as a hallucination and drops it. Effectively disabled. |
| `log_prob_threshold` | `-1.0` | `-2.0` | Sung tokens have intrinsically lower per-token logprobs than clean speech; the default blanks slow ballads / soft bridges. |
| `condition_on_previous_text` | `False` (3.x) | `False` | Set explicitly so an upstream bump can't silently re-enable hallucinated drift between chunks (once sync is lost on a chorus, prior-token conditioning pulls the decoder into wrong-chorus variants). |
| `initial_prompt` | `None` | `None` (deliberately) | We tried biasing the LM with the user's deduped lyrics. It does raise the raw transcribed-word count, but on stuttery / glitchy intro vocals it makes whisper hallucinate chorus-shaped phrases onto non-lyrical noise, which then drags forced alignment back to `t=0`. Chorus-repeat disambiguation is handled downstream by the fingerprint DP, which is robust to bad transcription inside a section as long as one characteristic phrase is heard anywhere in it — so we get the repeat-resolution benefit without the intro-hallucination cost. |
| `vad_onset` | `0.500` | `0.200` | Demucs stems are near-all-vocal, so we bias toward accepting quieter / breathy passages. |
| `vad_offset` | `0.363` | `0.100` | Same reason; keeps trailing sustained notes attached to their segment. |

Verify the change from the run log alone — `pipeline.lyrics_aligner`
now emits two INFO lines right after transcription:

```
WhisperX transcribe: 9 segments, 198 whisper words recognised, audio=183.5s
WhisperX first-words preview (15/198): They told me not to go so now I'm really going now
```

The preview is the primary sanity check when tuning: if the first words
read as plausible lyric text at plausible times, the decode is healthy;
if it reads as nonsense or repeats one phrase, dial the ASR options
back (the constants live in `pipeline/lyrics_aligner.py` as
`_SUNG_VOCALS_ASR_OPTIONS` / `_SUNG_VOCALS_VAD_OPTIONS`). Older
whisperx builds that don't accept `asr_options` / `vad_options` at load
time fall back to the plain load + emit a `WARNING` so the regression
is visible in the same log.

## Section-fingerprint placement (`---` path)

When the user authors `---` sections, the aligner replaces the
word-level Needleman-Wunsch bucketing with a **section-level**
fingerprint match + monotonic DP. This is the architectural change
that fixes two long-standing failure modes on sung material:

1. **Intro-noise hallucination**: pre-vocal stutter / glitchy audio
   that whisper's LM fills with coherent-looking text no longer drags
   the user's first section to `t=0`, because section 0 is pinned by
   a real fingerprint hit somewhere later in the song.
2. **Repeated-chorus collapse**: two identical chorus sections are
   now forced to pick *different* fingerprint hits by the DP's strict
   temporal-ordering constraint, instead of both inheriting the same
   merged-VAD window and fighting over the same words.

### Flow

1. **Fingerprint extraction** (`_extract_section_fingerprints`) — for
   each user section, enumerate every 3-, 4-, and 5-gram of normalised
   tokens, count how many *other* sections contain the same n-gram,
   and keep the top 3 ranked by
   `(other_section_count asc, length desc, position asc)`. Unique
   phrases beat shared ones, longer phrases beat shorter ones at equal
   uniqueness. Sections of < 3 tokens contribute no fingerprint and
   are placed by interpolation from their neighbours.
2. **Fuzzy match scan**
   (`_find_fingerprint_matches_in_transcript`) — flatten whisper's
   CTC-aligned transcript to `[(normalised_word, t_start), ...]`
   (prefers the confidence-filtered `transcription_words` when
   available; falls back to segment-uniformly-timed whisper words
   otherwise). For each fingerprint, slide a window of `len(phrase)`
   whisper tokens and record a hit when the fraction of
   `_tokens_are_fuzzy_equal` matches passes 0.6. **Temporal-span
   guard**: the window is rejected if the first and last
   fuzzy-matched whisper words span more than
   `max(_FINGERPRINT_MIN_SPAN_SEC, plen × _FINGERPRINT_SEC_PER_WORD)`
   (defaults 3 s / 2 s per word → 6 s for a 3-gram, 10 s for a
   5-gram). This stops CTC-split phrases — where whisper's text is
   right but one word got pinned to intro noise at `t≈0` and the
   rest anchored 20-30 s later on the real vocal — from registering
   as a "match" and anchoring a section at `t=0`. The regression
   this fixes is documented in
   `TestFindFingerprintMatchesInTranscript.test_rejects_match_spanning_too_much_audio_time`
   and
   `TestSectionFingerprintEndToEnd.test_ctc_split_phrase_does_not_drag_section_zero_to_intro`.
   Overlapping matches within the same section (within 0.5 s)
   collapse to the strongest score — two fingerprints hitting the
   same chorus repeat shouldn't both get to compete for one DP
   pick.
3. **Temporal DP assignment** (`_assign_sections_via_temporal_dp`) —
   per section, candidates are the surviving fuzzy hits (score ≥ 0.5,
   under `_FINGERPRINT_DP_MIN_ACCEPT`) plus an optional synthetic
   forced-anchor candidate (score 2.0) when the user pinned the
   section via `[m:ss]`. Standard layer-by-layer DP picks at most one
   candidate per section, enforcing strictly-increasing times and
   maximising total score. Sections that can't satisfy the
   monotonicity constraint return `None` and are left un-placed.
4. **Window construction**
   (`_build_section_windows_from_fingerprints`) — chosen times become
   each section's `start`; un-placed sections are filled by
   token-weighted linear interpolation between their nearest placed
   neighbours. **Token-budget cap on leading/trailing gaps**: un-placed
   sections before the first known anchor (or after the last known
   anchor) are pulled to within `_SEC_PER_TOKEN_BUDGET × tokens` of
   that anchor rather than stretched back to `t=0` / forward to
   `audio_duration`. At 0.6 s/token this is a generous upper bound
   on sung phrasing (≈1.7 words/sec). Without this cap, pinning
   section 1 at `t=22 s` and leaving section 0 un-fingerprinted
   produced a `[0, 22]` window for section 0 even when its content
   realistically only sings in the last 2-3 seconds of that range,
   which forced CTC to cram the user's words across 20 s of
   intro silence (see
   `TestBuildSectionWindowsFromFingerprints.test_leading_unplaced_section_respects_token_budget`).
   A final monotonic sweep with a 0.1 s minimum gap guarantees
   `whisperx.align` always gets well-formed windows. Each section's
   `end` is the next section's `start` (or the audio end).

### Fallback

If the `---` path produces no windows (e.g. every section was too
short to fingerprint and no line anchors were supplied), the aligner
falls back to the historical NW-bucket flow (`_build_forced_alignment_segments`
→ `_build_forced_alignment_segments_by_user_sections`). This is a
safety net only; it should never trigger on a real lyric.

### CTC window widening (per-word timestamp accuracy)

`whisperx.align` runs wav2vec2 CTC **per whisper segment**, bound to
each segment's `[start, end]` window. Faster-whisper's `end`
timestamps truncate aggressively — right after the last decoded word
even when the vocal clearly continues — so when a single segment
contains a chorus repeated three times ("Roll with life / Roll with
life / Roll with life"), CTC is forced to compress all nine words
into whatever audio fits in that narrow window. The result on the
editor waveform is clustered ghost labels: every repeat of the
chorus stacks on top of itself near the segment start instead of
spreading across the seconds it actually sings over.

`_widen_segments_for_ctc` stretches each segment's `t_end` to just
before the next segment's `t_start` (minus `_SEGMENT_WIDEN_GAP_SEC
= 0.05` s), and extends the final segment to the full
`audio_duration`. Segment starts are never touched — pyannote VAD's
onset detection is far more reliable than its offset detection on
sung vocals. The widened segments then feed both the transcription
CTC step (for editor ghost text / line anchors) and the fingerprint
flattening step, so both downstream consumers benefit.

Diagnostic: the `WhisperX transcribe:` log line reports the total
end-slack added (`widened +12.3s of end-slack for CTC`). If that
number is near zero on a song with obvious chorus-clustering in the
editor, the segments are already generously-ended and the clustering
has a different root cause (e.g. whisper genuinely mis-transcribed
repeated text that doesn't exist in the audio). Tests:
`TestWidenSegmentsForCtc`.

### Vocal-activity floor (transcription-as-source-of-truth)

Regardless of which path (fingerprint, NW-bucket, or inline-anchor)
built `forced_segments`, a final guard runs before
`whisperx.align` is invoked:

1. `_first_dense_vocal_activity_time(transcription_words)` finds the
   earliest time where at least `_VOCAL_FLOOR_MIN_WORDS=3` confident
   whisper words appear within a `_VOCAL_FLOOR_WINDOW_SEC=5` s sliding
   window. A single stray hallucinated word near `t=0` (e.g. the
   vocal-chop intro that repeats with effects and fools whisper once)
   fails this density check and is skipped past.
2. `_apply_vocal_activity_floor` shifts `forced_segments[0]["start"]`
   forward to `floor - _VOCAL_FLOOR_BUFFER_SEC` when **all** of:
   * The current first-segment start is at least
     `_VOCAL_FLOOR_MIN_SHIFT_SEC=2` s earlier than the floor.
   * No user line anchor (`[m:ss]`) is earlier than the floor — user
     pins always win over the heuristic.
   * The segment has enough span to accept the shift without
     inverting and without leaving a >1 s residual gap to the floor
     (partial shifts are refused because they leave lyrics still
     mis-placed, just in a tighter box).

This is the explicit "treat the transcription as source of truth"
behaviour: if whisper clearly shows real singing doesn't start for
45 s, the aligner refuses to place user lyrics in that dead zone
even if the upstream placement logic picked a bad start time. Tests:
`TestFirstDenseVocalActivityTime` and `TestApplyVocalActivityFloor`.

### Diagnostic logging

`_run_whisperx_forced` logs one INFO line per alignment run when the
fingerprint path is taken:

```
Section-fingerprint placement: n_sections=5, fingerprints=13 (flat),
matches=8, forced=0; chosen=[15.63, 42.11, 70.04, 98.8, 130.2]
```

`n_sections` is the user's section count, `fingerprints` is the flat
total across all sections, `matches` is the number of surviving fuzzy
hits after dedup, `forced` is the count of sections pinned via user
`[m:ss]` anchors, and `chosen` is the DP's output (with `null` for
un-placed sections). Eyeballing this line tells you whether the
placement is sensible before the CTC pass even runs — if every
section got a time and they're monotonically spaced, the downstream
forced alignment will have tight, correct windows to work within.

The vocal-activity floor logs one of two follow-up lines, *always*,
regardless of whether the fingerprint path ran:

```
Vocal-activity floor: first dense whisper run at ~50.00s;
  shifted forced-alignment segment[0] start forward by 49.50s (new=49.50s)
```

or (no-op case):

```
Vocal-activity floor: first dense whisper run at ~15.30s;
  no shift applied (user anchor earlier, gap below threshold, or
  segment boxed in)
```

If you see user lyrics placed on a silent intro, grep the log for
`Vocal-activity floor` first — if it says "no shift applied", either
the user pinned an early line (intentional), the gap was too small to
matter, or the upstream placement already boxed in segment 0 via a
user pin on segment 1.

## Optional polish passes (`use_silero_vad`, `use_onset_snap`)

Two polishing steps are wired in but **default OFF** because they
consistently regressed quality on sung vocals during real-world testing:

- **`use_silero_vad=True`** — run Silero VAD on `vocals.wav` and
  intersect its speech spans with whisper's VAD segments. Silero is
  trained on spoken speech and tends to reject sustained vowels,
  melisma, and soft vocals as non-speech, which shrinks whisper's
  segments and crams CTC words into too-narrow windows. Useful for
  speech-adjacent content (spoken-word tracks, podcasts mixed as songs);
  actively harmful on most pop / rock / sung material.
- **`use_onset_snap=True`** — after CTC, pull each word's `t_start` to
  the nearest vocal-stem onset within 80 ms. Useful on clean, staccato
  consonants; on sustained vocals the onset envelope also picks up
  cymbal bleed through the stem, vibrato crests, and syllable-internal
  transitions, and the snap biases `t_start` off the real word start.

Both are kept in-tree (with their caches and unit tests) so they can be
re-enabled per-song if a track benefits. The default path is:
*transcription anchors on + VAD/onset snap off*.

## Caching

- Keyed by the per-song cache dir plus a `lyrics_sha256` of the canonical
  tokenised lyrics. Editing the pasted lyrics text invalidates the cache
  automatically.
- Section markers and inline anchors are folded into the canonical only
  when at least one is present, so plain lyrics hash identically to
  pre-feature versions — existing caches stay valid after upgrading.
- `manually_edited: true` bypasses the cache-key check: the visual editor's
  saved timings are always returned as-is, even if the user tweaked the
  lyrics text afterwards (we record the canonical sha at save time under
  `manual_lyrics_sha256` so genuinely new lyrics can still be detected in
  the logs).
- `align_lyrics(..., force=True)` always re-runs WhisperX and overwrites
  any manual edits (the **Re-align from scratch** button in the Lyrics
  timeline tab does exactly this).
- Silero VAD and vocal onsets have their own separate caches
  (`vocal_vad.json`, `vocal_onsets.json`) keyed by the stem's mtime so
  the aligner doesn't redo either pass when only the pasted lyrics
  changed.

## Code

| Piece | Location |
|-------|----------|
| `align_lyrics`, `AlignedWord`, `AlignmentResult` | `pipeline/lyrics_aligner.py` |
| Needleman-Wunsch, normalisation, gap interpolation | `pipeline/lyrics_aligner.py` |
| Inline `[m:ss]` anchor + typed `---` parsing | `_split_user_lyrics`, `_parse_anchor_prefix`, `_is_section_marker` |
| Section-fingerprint placement (marker path) | `_extract_section_fingerprints`, `_find_fingerprint_matches_in_transcript`, `_assign_sections_via_temporal_dp`, `_build_section_windows_from_fingerprints` |
| Vocal-activity floor (transcription-as-source-of-truth) | `_first_dense_vocal_activity_time`, `_apply_vocal_activity_floor` |
| CTC window widening | `_widen_segments_for_ctc` |
| Hard-bound section windows from anchors (fallback) | `_build_forced_alignment_segments_by_user_sections` |
| Onset + anchor snap polish passes (opt-in) | `_snap_to_vocal_onsets`, `_snap_to_line_anchors` |
| Silero-VAD refinement shim (opt-in) | `_refine_segments_with_vocal_vad` |
| Silero VAD wrapper (optional dep) | `pipeline/vocal_vad.py` |
| Vocal-onset detector | `pipeline/vocal_onsets.py` |
| Auto line anchors from whisper transcription | `_derive_transcription_anchors_via_align`, `_derive_line_anchors_from_transcription` |
| Align-lyrics button + handler (`_align_lyrics`) | `app.py` (Lyrics tab) |
| Lyrics timeline editor tab | `app.py` (Lyrics timeline tab), `pipeline/lyrics_editor.py` |
| Vocal stem constant reused | `pipeline/audio_analyzer.VOCALS_WAV_NAME` |

## UI wiring

- **Lyrics tab** exposes the lyrics `gr.Textbox` plus an **Align lyrics**
  button. The handler takes `(song_hash, lyrics_text, run_log)` and reports
  the model / language / word / json-path summary back to the run log.
- Alignment is deliberately **not** chained to Analyze: it depends on the
  user's pasted text, so running it only when the user explicitly clicks
  keeps the WhisperX download / GPU cost out of the fast analyze path.

## Dependencies

Added to the existing `analysis` optional-dependency group:

```
whisperx>=3.1.0
silero-vad>=5.1     # optional: tightens whisper's VAD (small ONNX, no GPU)
```

WhisperX pulls in `faster-whisper`, `pyannote-audio`, and `torchaudio`. Use
the same CUDA 12.x PyTorch wheel as the rest of the project. The aligner
raises a clear `RuntimeError` when `whisperx` is missing, with the path to
`sys.executable` so you can run `… -m pip install -e ".[lyrics]"` into the **same**
environment that runs the app (the Windows launcher `py` often does not use
`.venv`; prefer `.venv\Scripts\python.exe -m pip …` or `python -m pip` after
`Activate.ps1`).

`silero-vad` is optional: the aligner tightens whisper's VAD when it's
importable and silently falls back to whisper-only VAD bounds when it
isn't, so existing installs keep running. The wheel is ~1 MB and needs
no GPU. Install with `python -m pip install silero-vad` into the same
venv.

## Fallback behaviour (no silent failures)

- Missing `vocals.wav` → `FileNotFoundError` with guidance to rerun Analyze
  with demucs available.
- Missing / blank lyrics → `ValueError("lyrics_text is empty…")`.
- `whisperx` not installed → `RuntimeError` with install instructions.
- Forced alignment returns no words → `RuntimeError` with a diagnostic
  mentioning input counts and language (usually indicates a bad vocal stem
  or lyrics that don't match the track).
- UI handler catches the exception and logs a single-line error.

## Schema migration

- v1 (whisper transcription + NW-mapped user text) and v2 (forced
  alignment without scores / manual edits) are both considered stale:
  `_load_cached_alignment` ignores anything whose `schema_version` is not
  the current value, so the first alignment after upgrading always
  regenerates with real forced alignment + per-word CTC scores. v2
  payloads can be re-read by running Align once; the file is overwritten
  in place.

## GPU memory lifecycle

WhisperX loads two GPU-resident models — faster-whisper large-v3 (≈3 GB
FP16) and the wav2vec2 align model (≈500 MB) — and the forced-alignment
flow makes sure **neither outlives its stage** so the next pipeline step
(SDXL background / AnimateDiff) starts with a clean VRAM budget:

- After `model.transcribe(...)` returns, the transcribe model is moved
  to CPU, dereferenced, and `pipeline.gpu_memory.release_cuda_memory()`
  is called. The align model is loaded only afterwards.
- After `whisperx.align(...)` returns (or raises), the align model is
  moved to CPU, dereferenced, and VRAM is released again via the same
  helper.

Both cleanups are in `try/finally` blocks in `_run_whisperx_forced`, so
a CTC backtrack failure or OOM mid-align still drops the weights.

## Related

- Vocal stem + analyser cache: `docs/technical/audio-analyzer.md`
- Visual editor that writes `manually_edited: true`:
  `docs/technical/lyrics-timeline-editor.md`
- UI wiring and progress panel: `docs/technical/gradio-ui.md`
- CUDA cleanup helper: `pipeline/gpu_memory.py`
