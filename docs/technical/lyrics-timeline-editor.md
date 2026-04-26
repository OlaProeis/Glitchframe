# Lyrics timeline editor

Visual per-word timeline editor for `cache/<hash>/lyrics.aligned.json`.
Lives in its own Gradio tab — **Lyrics timeline** — next to the Lyrics
tab. Purpose: let the user inspect and correct WhisperX output before
the kinetic typography layer bakes it into frames, without leaving the
app.

## UX at a glance

1. User ingests audio, runs **Analyze** (which produces `vocals.wav`),
   pastes lyrics, and clicks **Align lyrics** to generate the initial
   `lyrics.aligned.json`.
2. Switches to **Lyrics timeline** and clicks **Load timeline**. The
   vocal-stem waveform renders on a canvas; each word appears as a
   coloured bar (green = strong CTC confidence, yellow = weak, red = very
   weak, grey = no score — i.e. word was filled by gap interpolation).
3. Interacts:
   * **Click a word bar** → audio seeks to `t_start - 0.25 s` and plays.
   * **Drag the middle of a bar** → shifts the word in time, duration
     preserved. A thin yellow vertical guide line appears at the primary
     word's `t_start` for the duration of the drag, so you can line it
     up with a waveform transient by eye. For handle resizes the guide
     tracks the edge you're dragging.
   * **Drag the left / right handle** → adjusts `t_start` / `t_end`
     individually.
   * **Shift- / Ctrl-click a word** → add / remove from the selection
     (bars get a yellow outline). Plain click on an *unselected* word
     replaces the selection with just that word; plain click on a
     *selected* word keeps the current selection so you can group-drag.
   * **Click-and-drag on empty timeline** → rubber-band select every
     word whose centre falls inside the rectangle (Shift/Ctrl held =
     additive, so you can union multiple bands).
   * **Drag any selected word** with multiple selected → moves the
     whole group by one shared delta (clamped so nothing escapes
     `[0, duration]`, relative spacing preserved). The guide line only
     appears for the primary (first) word of the group to keep the
     overlay readable.
   * **Click the waveform background** (no drag) → seeks audio.
   * **Space** → play / pause (uses the `<audio>` element in the same
     tab). **+** / **−** → zoom in / out. **Fit** → zoom to fit.
     **Esc** → clear the selection. **Ctrl/⌘+A** → select all words.
     **Del** / **Backspace** → remove selected word bar(s) from the timeline
     (save to persist); ignored when focus is in an `input`, `textarea`, or `select`.

**Browser text selection** — The scroll area, stage, word track, and each
word bar set CSS `user-select: none` (with `-webkit-user-select` where
needed) so rubber-band and drag operations do not show the native blue
text-selection highlight over the word labels. The help line’s `<kbd>`
key caps use a light `color` on a dark key background so shortcut names
stay readable in both light and dark Gradio themes.

### Whisper ghost-text overlay

Faint italic labels floating above the waveform show **what whisper
actually heard** at each point in the track, with its own CTC word
timings. Labels are staggered across three rows so adjacent words
don't collide at high zoom, and they're non-interactive — they never
steal drags or clicks from the user word bars.

This is the critical aid for songs where the automatic alignment is
poor (heavily chopped vocals, ad-libs, fast rap). The user can read
whisper's transcription directly off the waveform, find the point
where, say, `"walked out"` was heard, and drag the corresponding user
word straight there — no need to listen back to the whole track.

**Per-label opacity** encodes whisper's CTC confidence for that word
(stored as `score` in the payload). Labels with high CTC score render
bright white; low-confidence placements fade toward transparent so
the user can tell at a glance which labels are trustworthy evidence
and which are just the forced aligner guessing. The aligner already
pre-filters anything below `score < 0.25` and all interpolated
(scoreless) entries before persisting — those are reliably junk and
would only clutter the waveform — but the opacity scale still matters
for everything in between.

The data comes from `lyrics.aligned.json`'s `whisper_words` array,
which the aligner populates during the transcription-anchor CTC pass.
Caches produced before this feature shipped simply show no ghost text
(the info bar surfaces `whisper: none (re-align to populate)`), and a
click on **Re-align from scratch** regenerates them alongside a fresh
alignment.
4. Clicks **Save edited timings**. The browser serialises its in-memory
   state (`window._musicvids_editor_state`), Gradio writes it into a
   hidden textbox, and the Python handler validates + persists as the
   new `lyrics.aligned.json` with `manually_edited: true`.
5. Subsequent Align clicks return the manual edits unchanged (the
   aligner's cache-load path respects the flag). **Re-align from
   scratch** clears the flag so the next Align rebuilds from WhisperX.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Gradio tab: "Lyrics timeline"                              │
│                                                            │
│  [Load timeline]   ──► _load_editor(song_hash, log)        │
│                            ├── pipeline.lyrics_editor       │
│                            │     .load_editor_state(cache) │
│                            │     .build_editor_html(state, │
│                            │        audio_url=/file=<abs>) │
│                            └── gr.HTML ←── JS takes over   │
│                                                            │
│  <audio id="mv_editor_audio"                               │
│         src="/file=<abs_path_to_vocals.wav>"               │
│         controls preload="auto">   (lives INSIDE the       │
│                                    editor HTML blob;       │
│                                    Gradio's /file= proxy   │
│                                    serves it because       │
│                                    CACHE_DIR is in         │
│                                    allowed_paths)          │
│                                                            │
│  [Save edited timings]  ── js=() => JSON.stringify(         │
│                                 window._musicvids_editor_state)│
│                          ──► Hidden gr.Textbox             │
│                          ──► _save_editor(song_hash, state,│
│                                           lyrics_text, log)│
│                                └── .save_edited_alignment  │
│                                                            │
│  [Re-align from scratch] ──► _revert_editor(song_hash, log)│
│                                └── .revert_manual_edits    │
└────────────────────────────────────────────────────────────┘
```

All UI state lives in `window._musicvids_editor_state` on the browser
side. There is no custom Gradio component — the editor is a plain
`gr.HTML` with an inline vanilla-JS implementation, which means zero
extra build steps, no JS bundler, and full offline support.

## Data shapes

- **Input** (what `load_editor_state` returns):
  * `song_hash`, `duration_sec`, `sample_rate`
  * `peaks`: list of `(min_norm, max_norm)` pairs (~1600 pairs by
    default) computed from `vocals.wav` by `compute_peaks`.
  * `words`: list of dicts with `word`, `line_idx`, `t_start`, `t_end`,
    `score` (may be `None`).
  * `lines`: list of pasted lyric lines (matches `lyrics.aligned.json`).
  * `manually_edited`: boolean, forwarded to the info bar.
  * `lyrics_sha256`, `vocals_rel_path`: round-tripped for the save
    handler.
  * `whisper_words`: list of `{"word", "t_start", "t_end"}` dicts for
    whisper's own transcription of what it heard. Empty for older
    caches. Rendered as the ghost-text overlay; `save_edited_alignment`
    pulls this forward from the prior cache so it survives manual edits
    (the editor never mutates it — it's reference data, not user data).

- **Output** (what the browser sends back to `save_edited_alignment`):

  ```json
  {
    "song_hash": "<sha>",
    "lines": ["..."],
    "words": [
      {"word": "Hello", "line_idx": 0, "t_start": 0.50, "t_end": 0.82, "score": 0.94},
      "..."
    ]
  }
  ```

  Any fields present in the payload other than `words` / `lines` are
  ignored. The handler always re-derives `song_hash` from the cache
  directory name, never from the payload.

The saved JSON keeps the aligner's schema v3 exactly, plus:

```json
{
  "manually_edited": true,
  "manual_lyrics_sha256": "<sha of current lyrics textbox>"
}
```

## Confidence colour coding

| Score range | Colour   | Meaning                                    |
|-------------|----------|--------------------------------------------|
| `>= 0.60`   | Green    | Strong wav2vec2 CTC match — trust it.      |
| `[0.30, 0.60)` | Yellow | Weak match, usually OK but inspect.        |
| `< 0.30`    | Red      | Very weak — often a mishear, review.       |
| `None`      | Grey     | No CTC match, filled by interpolation.     |

Colours are computed server-side by `_confidence_color_for_score` so
they stay consistent with any server-side visualisation we add later.

## Safety rails

- **Never trusts client state for the song_hash** — the save handler
  always uses the cache-directory name, so a malicious page can't write
  to some other song's cache.
- **Atomic write** via `<path>.json.tmp` + `replace()` so a partial
  write can't corrupt a working alignment.
- **Strict validation** on every word entry (`word`, `line_idx`,
  `t_start`, `t_end` required); bad entries raise `ValueError` and the
  UI surfaces it in the run log.
- **Inverted timings** (`t_end < t_start`) are silently clamped to
  `(t_start, t_start)` so the compositor never sees a negative duration.
- **Empty / missing** vocals / aligned JSON raise `FileNotFoundError`
  with remediation hints; the tab shows the exception in the run log and
  leaves the existing HTML intact.

## Code

| Piece | Location |
|-------|----------|
| Gradio tab + button wiring | `app.py` ("Lyrics timeline" tab) |
| Handlers `_load_editor` / `_save_editor` / `_revert_editor` | `app.py` |
| `EditorState`, `load_editor_state`, `save_edited_alignment`, `revert_manual_edits` | `pipeline/lyrics_editor.py` |
| `compute_peaks` (WAV → downsampled min/max buckets) | `pipeline/lyrics_editor.py` |
| `build_editor_html` (CSS + markup + inline JS) | `pipeline/lyrics_editor.py` |
| Cache-load path that honours `manually_edited` | `pipeline/lyrics_aligner._load_cached_alignment` |

## Dependencies

Nothing new — `numpy` and `soundfile` are already in the core
dependency set; the JS is self-contained and requires no CDN.

## Related

- Alignment pipeline the editor plugs into:
  `docs/technical/lyrics-aligner.md`
- Typography layer that consumes the edited JSON:
  `docs/technical/kinetic-typography.md`
