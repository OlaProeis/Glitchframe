# Handover — transcription quality on sung vocals is the root problem

> **Audience**: a fresh chat that is picking up this investigation.
> Please read **this whole file before editing anything**. A lot has
> already been tried; we need a different angle, not more of the same.

---

## 1. One-sentence framing

**The ghost-text transcription displayed on the editor waveform is wrong
and sparse** — WhisperX is only recognising ~60% of the sung words in
the user's current test song, and many of the ones it does recognise are
misplaced or nonsensical. Every downstream alignment problem
(misplaced user lyrics, choruses landing on the wrong repeat, stuttering
sections being "filled in") is a **symptom** of this, not the cause.

**Fix the transcription first. Do not touch Needleman–Wunsch, anchor
derivation, section-splitting, or the editor until you've verified the
transcription itself is trustworthy.**

---

## 2. Hard evidence (from `terminals/1.txt`)

Test song: 183.5s, ~200+ sung words in the user-pasted lyrics.

```
WhisperX transcribe: 7 segments, 121 whisper words recognised, audio=183.5s
Whisper-transcript CTC: dropped 8/121 low-confidence or interpolated words (score < 0.25 or None)
Transcription anchors: derived 28 auto line pins (user already pinned 0);
  113 whisper-transcribed words persisted for editor overlay
```

- **121 raw words ÷ 183 s ≈ 40 wpm.** A sung pop song is usually
  80–140 wpm. We're missing roughly half the vocal content.
- After dropping the 8 worst-scoring ones we show 113 ghost labels on
  the waveform. That is what the user sees. The user is correct that
  they are wrong / sparse — the pipeline really did only hear 121
  words, most with low confidence.
- VAD in use: **Pyannote** (whisperx default — see log lines
  `whisperx.vads.pyannote - Performing voice activity detection using
  Pyannote...`). Silero-VAD is OFF by design.
- Transcription input: **`vocals.wav`** (Demucs `htdemucs_ft` stem).
  That's correct, not the raw mix. `load_audio(str(vocals_wav))` is
  confirmed in `pipeline/lyrics_aligner.py:1331`.
- Whisper model: `large-v3`, language forced to user-selected
  (`language=language`), `batch_size=batch_size`. **No** tuned
  decoding params (`initial_prompt`, `no_speech_threshold`,
  `compression_ratio_threshold`, `condition_on_previous_text`,
  `temperature`, `suppress_tokens`). All defaults.

---

## 3. The song and pasted lyrics (for reproducibility)

Cache hash from terminal: `81052fa61ba7a35639b00aa03578552a32c6e2fc88e4fd5c1daae064bf7a7379`
(`G:\DEV\MusicVids\cache\81052fa6…\lyrics.aligned.json` — safe to inspect).

User-pasted lyrics (verbatim, `---` are deliberate section markers;
brackets like `[Chorus]` are **not** present in the pasted text):

```
They told me not to go
So now I'm really going
Now I'm really going
They told me not to go
So now I'm really going
Now I'm really going
---
My blood was never cold
But now it's frozen over
Frozen over
My blood was never cold
But now it's frozen over
Frozen over
---
And you don't have to know
But now you really know me
Now you really know me
You don't have to know
But now you really know me
Now you really know me
[Verse 2]        <-- user has since removed this; do NOT blame it
Now you really know me
Now you really know me
You don't have to know
But now you really know me
Now you really know me
---
My blood was never cold
But now it's frozen over
Frozen over
My blood was never cold
But now it's frozen over
Frozen over
---
And you don't have to know
But now you really know me
Now you really know me
You don't have to know
But now you really know me
Now you really know me
You don't have to know
But now you really know me
Now you really know me
You don't have to know
But now you really know me
Now you really know me
```

Observable pathology in the editor:

- In the intro and first pre-chorus the ghost text roughly lines up.
- Stretches of real vocals have **no** ghost labels (silence in the
  transcription even though vocals are clearly in the stem).
- Around the first chorus the ghost text shows **words from
  elsewhere in the song** (e.g. chorus-region shows phrases that
  actually belong later) and is timed nonsensically.
- Heavily chopped / stuttered vocal sections ("e-e-e-e-e-e") still
  produce "a lot of words there" — whisper is hallucinating text to
  fit the default language-model prior.

---

## 4. What has **already** been tried (do not redo)

See `pipeline/lyrics_aligner.py`, the git log, and
`docs/technical/lyrics-aligner.md`. The previous chat iterated
extensively on *downstream* logic:

| Attempt | Where | Outcome |
| --- | --- | --- |
| NW-match user tokens against whisper words | `_run_whisperx_forced` → `_assign_user_tokens_to_segments` | Works when whisper text is correct; useless when whisper heard nothing / the wrong thing. |
| Transcription anchors (`_derive_line_anchors_from_transcription`) with Levenshtein-1 / prefix fuzzy match (`_tokens_are_fuzzy_equal`) and back-estimation of line-start times | `pipeline/lyrics_aligner.py` | Helpful when whisper text is correct; can **not** rescue anchors where whisper emitted the wrong words. |
| `---` user section markers + proportional time-splitting + anchor-aware overlap resolution | `_build_forced_alignment_segments_by_user_sections`, `_split_overlapping_section_windows` | Works per-section once bounds are right. Can't fix bad bounds caused by a bad transcription. |
| Drop whisper words with CTC `score < 0.25` or `score is None` before anchor derivation / editor persistence (`_filter_confident_whisper_words`) | `pipeline/lyrics_aligner.py` | Cleaned up the worst hallucinations from the editor but kept plenty of wrong-text-at-roughly-plausible-times ones, because CTC `score` rates how cleanly each phoneme string was placed, **not** whether the text itself is correct. |
| Silero-VAD toggle, vocal-onset refinement | `_refine_segments_with_vocal_vad` | Defaulted OFF — was eating sustained vowels and making things worse. |
| Editor ghost-text overlay, opacity by score, drag guide line, multi-select, persistence-after-save | `pipeline/lyrics_editor.py`, `app.py` | Editor itself is solid and round-trips correctly. Not the problem. |

Unit tests covering the above: `tests/test_lyrics_aligner.py`,
`tests/test_lyrics_editor.py` — **all pass**. Don't spend time
re-running them unless you change those modules.

---

## 5. Where the next chat should start looking

Root-causes to investigate **in this order**. Do not go further down
the list until the earlier items are ruled out.

### 5.1 WhisperX decoding / VAD params (cheapest, highest leverage)

File: `pipeline/lyrics_aligner.py`, function `_run_whisperx_forced`
around line 1315.

Current `model.transcribe(audio, batch_size=batch_size, language=language)`
uses whisperx defaults everywhere. For **sung** vocals the defaults are
known to be poor. Try, one at a time, and log raw-word counts:

1. `no_speech_threshold=0.0` (or very low, e.g. `0.1`) — stops whisper
   from marking sustained vowels / soft vocals as silence.
2. `compression_ratio_threshold=2.4` → e.g. `100.0` — disables the
   "too repetitive → drop segment" gate, which is **exactly** what
   fires on a chorus that repeats "Now you really know me" 6 times.
3. `condition_on_previous_text=False` — prevents the LM from drifting
   into hallucinated repeats once it's lost sync.
4. `temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (fallback cascade) —
   whisperx's default is already a cascade but worth verifying the
   exact values being passed.
5. `initial_prompt=<the user's pasted lyrics>` (as a biasing prompt).
   This is the single biggest quality lever for sung vocals: it nudges
   whisper's LM towards the right vocabulary so chorus repeats stop
   getting hallucinated into other chorus variants. Truncate to
   whisper's prompt-token budget (~224 tokens).
6. Bypass Pyannote VAD entirely on vocal stems — the stem is already
   mostly-vocal, and Pyannote is trained on speech, not singing.
   Options: `vad_filter=False` (if supported by the installed
   whisperx build) or pre-compute a single giant speech region so
   VAD doesn't gate anything.

Any of 1 / 2 / 5 alone will probably raise the raw-word count by
30–80%. Log raw-word count and the first ~15 transcribed words after
each change so you can compare.

### 5.2 Verify the vocal stem itself

- Play `cache/81052fa6…/audio/vocals.wav` (or whatever the cached path
  is for this song) and sanity-check the audio whisperx is actually
  seeing. If the vocal stem is clean, this step is done. If the stem
  has lots of bleed / artefacts, that's a separate conversation about
  Demucs model / two-pass separation — but **the current data in the
  transcript suggests the stem is fine**, because the handful of
  whisper words we do get are correct words at correct times. The
  problem is breadth, not accuracy.

### 5.3 If 5.1 isn't enough: try a different ASR backend

Only after 5.1 has been exhausted:

- `whisper-timestamped` (sits on top of openai-whisper, different
  timestamp strategy, often better on music).
- `faster-whisper` direct (whisperx's defaults may be hurting us more
  than faster-whisper's defaults would).
- Separate transcription (for *text*) from forced alignment (for
  *timings*). Use a music-tuned transcriber for the text, then feed
  the user-pasted text to wav2vec2 CTC via `whisperx.align()` with a
  single-segment input covering the whole song.

### 5.4 Only then reconsider the alignment architecture

If and only if the ghost-text on the waveform is visibly
correct/dense and the automatic placement **still** misaligns: the
old investigation's "Option B — direct transcription placement" from
the transcript is the next step. It's already spec'd there. But
there's no point implementing it on top of a bad transcription —
you'd just be matching user lyrics against wrong whisper text.

---

## 6. Explicit non-goals for the next chat

- **Do not** edit `pipeline/lyrics_editor.py`. The editor works; the
  user confirmed so. The only reason the ghost text looks bad in the
  editor is that the backing data (`whisper_words`) is bad.
- **Do not** edit the NW matcher, `_tokens_are_fuzzy_equal`, anchor
  back-estimation, or section-splitting. These all behave correctly
  when given a good transcription.
- **Do not** add more user-facing UI. The user is frustrated; they
  want the automatic pass to be good enough that they don't live in
  the editor.
- **Do not** re-run long full-render tests to "verify" changes. Verify
  at the transcription-count log line. A successful change shows
  `whisper words recognised` climbing from ~121 toward ~200+ on this
  song, with the first ~15 logged words being plausible lyric text.

---

## 7. Key files & line anchors

- `pipeline/lyrics_aligner.py`
  - `_run_whisperx_forced` (~L1274–1500) — **this is where to edit
    transcription params**.
  - `_flatten_transcribe_segments` (~L469) — shape of what whisper
    returned.
  - `_filter_confident_whisper_words` / `_MIN_WHISPER_WORD_SCORE`
    (~L600–640) — low-confidence drop.
  - `_derive_line_anchors_from_transcription` /
    `_tokens_are_fuzzy_equal` — anchor derivation (do not touch for
    now).
  - `align_lyrics` (~L2080) — top-level entry, caching, payload.
- `pipeline/lyrics_editor.py` — editor state, ghost-text rendering,
  drag guide. **Read-only for this task.**
- `app.py` — Gradio wiring. Log lines added during previous chat for
  save / load / render diagnostics.
- `cache/<hash>/lyrics.aligned.json` — canonical artefact. Inspect
  the `whisper_words` array to see exactly what the editor will
  render.
- `terminals/1.txt` — raw log from the session that produced the
  screenshots. Shows the `121 whisper words recognised` line.

---

## 8. Success criteria for this task

A change is "good enough to keep" when, on the cached test song
(hash `81052fa6…`):

1. `WhisperX transcribe: ... N whisper words recognised` log line
   shows **N ≥ 180** (from current 121).
2. `whisper_words` in `lyrics.aligned.json` (after score-filtering)
   contains **≥ 160** entries spread throughout the song (not all
   packed into the first 60 seconds).
3. Loaded into the editor, the ghost text on the waveform reads as
   real song lyrics in roughly the right places — specifically the
   user can verify "Frozen over" appears at the two pre-chorus
   blocks and "Now you really know me" appears at the three chorus
   blocks, not smeared across the verses.

Only after (1)+(2)+(3) pass do we re-evaluate whether automatic user-
lyric placement is good enough or whether we still need the "Option
B" architectural change.
