# Session Handover

## Environment

- **Project:** MusicVids
- **Tech Stack:** Python 3.11, CUDA/PyTorch, Gradio, librosa/demucs/whisperx, diffusers (SDXL / AnimateDiff), moderngl, skia-python, ffmpeg NVENC
- **Context file:** Always read `ai-context.md` first — it contains project rules, architecture, and model selection.
- **Python interpreter:** `.\.venv\Scripts\python.exe` (the Windows `py` launcher does NOT point at the project venv — use the full path).
- **Run tests:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` (pytest is not wired).
- **Branch:** main

## Core Handover Rules

- **NO HISTORY:** Do not include a project history document or past task details unless they directly impact this specific task.
- **SCOPE:** Focus ONLY on the current task detailed below.
- **DO NOT REVERT** the forced-alignment direction — feeding the user's pasted lyrics to `whisperx.align` as ground truth is the correct architectural choice and must stay. The bug is downstream of that decision.

## Current Task

**Fix: `whisperx.align` returns no word-level timings even though transcribe + segment distribution now work.**

### Symptom (most recent run)

User clicks **Align lyrics** in the UI. Log shows:

```
[22:11:39] Align lyrics failed: WhisperX forced alignment returned no word-level timings. Input: 336 user words across 1 segments; language=en. ...
```

After an earlier fix, the run at 22:17:09 got further and logged:

```
WhisperX transcribe: 10 segments, 303 whisper words recognised, audio=236.5s
Forced-alignment input: 9 segments; first=15.96s end=45.27s last=223.61s end=234.85s
```

...but `whisperx.align(...)` still came back with no word-level data, so the error re-fired. That means:

- Transcribe is fine (10 segments, 303 words over 236.5 s — realistic).
- The NW segment-distribution step is fine (9 alignment segments spanning 15.96 s–234.85 s — realistic).
- **The failure is now in either:**
  - (a) `whisperx.align(...)` itself producing an empty / unexpected return shape, or
  - (b) `_extract_whisper_words(...)` not finding the words in whatever shape WhisperX returned.

### What has already been ruled out

- `vocals.wav` is valid (demucs produced it; earlier runs with the old aligner path worked).
- User's pasted lyrics match the track (336 words for a ~4 min song — plausible).
- WhisperX is installed and loading: pyannote VAD and the Lightning checkpoint upgrade ran cleanly.
- Segment distribution is not the bug anymore — previous handover fixed that; see the INFO log above.
- User only renders **English**; `language="en"` is correct. WhisperX logs "No language specified" at transcribe time (the `language=` kwarg may be ignored by the wrapper), but we override with `"en"` for `load_align_model`, so that's not the bug.

### What the next session must do

**Do not guess. Add instrumentation first, then look at the actual shape of the `aligned` return value.**

1. **Read these files in full before changing anything:**
   - `pipeline/lyrics_aligner.py` — especially `_run_whisperx_forced`, `_extract_whisper_words`, `_interp_fill`, `_WhisperWord` dataclass.
   - `docs/technical/lyrics-aligner.md` — architectural intent.
   - `tests/test_lyrics_aligner.py` — the 33 existing unit tests must keep passing (they cover pure helpers hermetically, no WhisperX needed).

2. **Add temporary debug logging** right after the `whisperx.align(...)` call in `_run_whisperx_forced` to dump:
   - `type(aligned)` and `list(aligned.keys())` if dict.
   - `len(aligned.get("word_segments", []))` and a sample of the first 3 entries.
   - `len(aligned.get("segments", []))` and the keys of `aligned["segments"][0]` and a sample of its `words` field.
   - What `_extract_whisper_words(aligned)` returned (length, first 5 words).

3. **Re-run Align lyrics** (user will need to do this — they have a 4 GB SDXL + WhisperX setup on a real GPU; you cannot reproduce locally without the vocal stem).

4. **Based on the shape**, fix `_extract_whisper_words` to parse it correctly. Possible causes to investigate, in order of likelihood:
   - WhisperX ≥ 3.4 may have changed the `align()` return shape; `word_segments` might be gone or nested differently.
   - The `word` field per entry may now be `"text"` or have a leading space not stripped.
   - CTC may be dropping all of the user's words because the alignment language doesn't match (unlikely since `en` is correct, but worth confirming `align_model.config` or equivalent).
   - `return_char_alignments=False` may have shifted the output shape in a newer WhisperX.

5. **Remove the debug logging** once the fix is confirmed; leave the existing structured INFO logs (`WhisperX transcribe:` and `Forced-alignment input:`) in place — they proved essential.

6. **Add a unit test** that feeds a known-good WhisperX-shaped dict into `_extract_whisper_words` so this regresses loudly if WhisperX changes shape again. Keep tests hermetic — do NOT invoke WhisperX in the test.

7. **Do not regenerate `lyrics.aligned.json` caches yourself** — `LYRICS_ALIGNED_SCHEMA_VERSION = 2` already invalidates v1 caches, and the user prefers to run it themselves to see progress in the UI.

### Files recently changed (as of this handover)

These have the latest architecture; study them before proposing changes:

- `pipeline/lyrics_aligner.py` — new forced-alignment flow:
  - `_run_whisperx_forced(vocals_wav, user_tokens, ...)` — two-stage: transcribe for segmentation, then `whisperx.align` with user text per segment.
  - `_flatten_transcribe_segments` — falls back to whitespace-tokenising `seg["text"]` (WhisperX transcribe does NOT emit per-word detail — a previous session discovered this the hard way).
  - `_assign_user_tokens_to_segments` — NW match user → whisper words to decide which segment each user token belongs in; falls back to `_proportional_segment_assignment` if NW finds no matches.
  - `_proportional_segment_assignment` — distributes by whisper-word-count or, if zero, by segment duration. Preserves temporal order.
  - `_build_forced_alignment_segments` — builds `[{text, start, end}]` for `whisperx.align`. Widens sub-100 ms windows and falls back to whole-audio segment if every bucket is empty.
  - `_extract_whisper_words` — **SUSPECT.** Parses `aligned["word_segments"]` first, then `aligned["segments"][*].words`. If WhisperX changed shape, this returns `[]` silently.
  - `LYRICS_ALIGNED_SCHEMA_VERSION = 2` — invalidates old v1 caches.
- `tests/test_lyrics_aligner.py` — 33 tests covering flatten, assignment, proportional fallback, segment building, gap fill, polish, monotonicity. All pass locally.
- `docs/technical/lyrics-aligner.md` — describes the v2 flow. Update if shape changes.

### Known-good architectural constraints (do not break)

- Every returned `AlignedWord.word` must be a **user** word (case/punctuation preserved); we don't display whisper's transcription.
- `_timings_for_user_tokens` + `_enforce_monotonic_per_line` + `_polish_timings` are the post-processing safety net. Keep them — the tests guard their behaviour.
- The cache is keyed by `(song_hash, lyrics_sha256, schema_version)`. When you ship a fix that changes word timings semantically, bump to schema version 3.
- UI handler (`_align_lyrics` in `app.py`) surfaces exceptions as single-line log entries. The error message should stay actionable — include vocal-stem hint and input counts.

## Verification

- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` — must stay at 99 passing (or more if you add tests).
- Manually: delete `cache/<hash>/lyrics.aligned.json`, click **Align lyrics** in the Gradio Lyrics tab on the user's 4-min song, confirm `lyrics.aligned.json` is written and the kinetic typography layer shows the user's exact words on-beat in a Preview 10 s render.
- If align still comes back empty after debug logging, the next step is to check WhisperX version (`.\.venv\Scripts\python.exe -c "import whisperx; print(whisperx.__version__)"`) and consult the installed package's `alignment.py` source to see the actual return shape.

## Key References

- `ai-context.md` — architecture + conventions + "Where Things Live" table.
- `docs/index.md` — documentation index.
- `docs/technical/lyrics-aligner.md` — forced-alignment flow (v2) and cache schema.
- `pipeline/lyrics_aligner.py` — everything lyrics-sync.
- `pipeline/audio_analyzer.py` — `VOCALS_WAV_NAME` constant for the demucs stem.
- `app.py` — `_align_lyrics` handler in the Lyrics tab.
- `.venv\Lib\site-packages\whisperx\` — installed WhisperX source; `alignment.py` is the ground truth for the return shape when shape detection is needed.

## Deferred (do NOT tackle in this session)

- Task Master #20 — Bass-Driven Logo Beat Pulse. Blocked by the alignment bug above; lyrics sync must work end-to-end first.
- Task 19 — scoped preview backgrounds.
- UI-side global lyrics offset slider (±500 ms).
- Bumping typography intro from 0.18 s to 0.25 s.
