"""
Backend for the visual lyrics-timeline editor.

The editor lives in a single ``gr.HTML`` component that ships its own
vanilla-JS waveform + word-bar UI (see :func:`build_editor_html`). This
module is responsible for:

* Downsampling ``vocals.wav`` into a peaks array that the browser canvas
  can render cheaply (``~1600`` sample pairs for a typical 4 min song).
* Loading / validating the current ``lyrics.aligned.json`` so the editor
  has per-word ``t_start`` / ``t_end`` / ``score`` to draw.
* Validating and persisting the edited JSON that comes back from the JS
  side, tagging it with ``manually_edited: true`` so the aligner won't
  silently overwrite the user's corrections on the next Align click.

No UI code lives here; the HTML + JS string lives in
:func:`build_editor_html` but is intentionally small so it's easy to read
next to the Python that feeds it.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.lyrics_aligner import (
    LYRICS_ALIGNED_JSON_NAME,
    LYRICS_ALIGNED_SCHEMA_VERSION,
    _lyrics_cache_key,
)
from pipeline._waveform_peaks import DEFAULT_PEAK_WIDTH, compute_peaks

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditorState:
    """All data the editor needs to render one song's timeline."""

    song_hash: str
    duration_sec: float
    sample_rate: int
    peaks: list[tuple[float, float]]  # (min_norm, max_norm) per column
    words: list[dict[str, Any]]
    lines: list[str]
    manually_edited: bool
    lyrics_sha256: str
    vocals_rel_path: str
    # Whisper's own transcription with CTC timings. Shown as faint ghost
    # text above / below the user words so the user can see "what whisper
    # heard at this moment" and manually drag the corresponding user word
    # to match, without having to listen through the whole song. Empty
    # list for older caches that predate whisper-words persistence.
    whisper_words: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# State load / save
# ---------------------------------------------------------------------------


def load_editor_state(
    cache_dir: Path | str,
    *,
    target_peak_width: int = DEFAULT_PEAK_WIDTH,
) -> EditorState:
    """Collect everything the editor needs from ``cache/<song_hash>/``.

    Fails loudly (no silent fallbacks) so the user sees a concrete error:
    missing vocals, missing aligned JSON, etc.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")

    aligned = cache / LYRICS_ALIGNED_JSON_NAME
    if not aligned.is_file():
        raise FileNotFoundError(
            f"{aligned} missing — click 'Align lyrics' first so the editor "
            "has per-word timings to show."
        )
    try:
        with aligned.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"lyrics.aligned.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "words" not in data:
        raise ValueError("lyrics.aligned.json is malformed (no 'words' key).")

    vocals_ref = str(data.get("vocals_wav", "vocals.wav"))
    vocals_path = cache / vocals_ref
    if not vocals_path.is_file():
        raise FileNotFoundError(
            f"Vocal stem missing at {vocals_path}; rerun Analyze with demucs available."
        )

    peaks, sample_rate, duration = compute_peaks(vocals_path, target_peak_width)

    words = [
        {
            "word": str(w.get("word", "")),
            "line_idx": int(w.get("line_idx", 0)),
            "t_start": float(w.get("t_start", 0.0)),
            "t_end": float(w.get("t_end", 0.0)),
            "score": (
                float(w["score"])
                if isinstance(w.get("score"), (int, float))
                else None
            ),
        }
        for w in data.get("words", [])
        if isinstance(w, dict)
    ]
    lines = [str(line) for line in data.get("lines", []) if isinstance(line, str)]

    whisper_words_raw = data.get("whisper_words") or []
    whisper_words: list[dict[str, Any]] = []
    if isinstance(whisper_words_raw, list):
        for w in whisper_words_raw:
            if not isinstance(w, dict):
                continue
            try:
                ww: dict[str, Any] = {
                    "word": str(w.get("word", "")),
                    "t_start": float(w.get("t_start", 0.0)),
                    "t_end": float(w.get("t_end", 0.0)),
                }
            except (TypeError, ValueError):
                continue
            if not ww["word"]:
                continue
            # Preserve CTC score so the editor can fade out lower-
            # confidence labels. Missing scores are kept as None and
            # rendered at the floor opacity.
            sc = w.get("score")
            ww["score"] = float(sc) if isinstance(sc, (int, float)) else None
            whisper_words.append(ww)

    return EditorState(
        song_hash=str(data.get("song_hash", cache.name)),
        duration_sec=float(duration),
        sample_rate=int(sample_rate),
        peaks=peaks,
        words=words,
        lines=lines,
        manually_edited=bool(data.get("manually_edited")),
        lyrics_sha256=str(data.get("lyrics_sha256", "")),
        vocals_rel_path=vocals_ref,
        whisper_words=whisper_words,
    )


def save_edited_alignment(
    cache_dir: Path | str,
    edited_json: str,
    *,
    lyrics_text_snapshot: str | None = None,
) -> Path:
    """Validate and persist edited word timings as the new ``lyrics.aligned.json``.

    Expected input: the JSON string the browser produces, shaped as::

        {
            "song_hash": "<sha>",
            "lines": [...],
            "words": [{"word": str, "line_idx": int, "t_start": float,
                        "t_end": float, "score": float | null}, ...],
        }

    We never trust ``song_hash`` from the payload — the handler passes the
    cache_dir, and the song_hash is taken from the folder name. Any other
    fields are dropped on the floor. The saved file always has
    ``manually_edited: true``; the aligner's cache-load path honours it.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")

    try:
        data = json.loads(edited_json)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Editor payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Editor payload must be a JSON object.")
    if not isinstance(data.get("words"), list):
        raise ValueError("Editor payload is missing a 'words' array.")

    song_hash = cache.name

    clean_words: list[dict[str, Any]] = []
    for w in data["words"]:
        if not isinstance(w, dict):
            continue
        try:
            word = str(w["word"])
            line_idx = int(w["line_idx"])
            t_start = float(w["t_start"])
            t_end = float(w["t_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid word entry from editor {w!r}: {exc}"
            ) from exc
        if t_end < t_start:
            t_end = t_start
        score_raw = w.get("score")
        score: float | None
        if isinstance(score_raw, (int, float)):
            score = float(score_raw)
        else:
            score = None
        entry: dict[str, Any] = {
            "word": word,
            "line_idx": line_idx,
            "t_start": t_start,
            "t_end": t_end,
        }
        if score is not None:
            entry["score"] = score
        clean_words.append(entry)

    lines_raw = data.get("lines") or []
    lines = [str(line) for line in lines_raw if isinstance(line, str)]

    # Record the canonical sha of the user's current lyrics if we can
    # compute it (so a future "same text, minor edit" run still recognises
    # the edits as valid). Falls back to the existing cached sha when no
    # snapshot is supplied. We also pull forward ``whisper_words`` so the
    # editor's ghost-text overlay survives repeated saves (the editor
    # never mutates whisper's transcription — it's purely display data).
    aligned_path = cache / LYRICS_ALIGNED_JSON_NAME
    existing_sha = ""
    preserved_whisper_words: list[dict[str, Any]] = []
    if aligned_path.is_file():
        try:
            with aligned_path.open("r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                existing_sha = str(prev.get("lyrics_sha256") or "")
                prev_ww = prev.get("whisper_words")
                if isinstance(prev_ww, list):
                    preserved_whisper_words = [
                        w for w in prev_ww if isinstance(w, dict)
                    ]
        except Exception:  # noqa: BLE001
            existing_sha = ""

    if lyrics_text_snapshot and lyrics_text_snapshot.strip():
        manual_sha = _lyrics_cache_key(lyrics_text_snapshot)
    elif lines:
        manual_sha = hashlib.sha256(
            ("\n".join(lines) + "\n::editor::").encode("utf-8")
        ).hexdigest()
    else:
        manual_sha = existing_sha

    payload: dict[str, Any] = {
        "schema_version": LYRICS_ALIGNED_SCHEMA_VERSION,
        "song_hash": song_hash,
        "model": str(data.get("model") or "manual-edit"),
        "language": str(data.get("language") or "en"),
        "vocals_wav": str(data.get("vocals_wav") or "vocals.wav"),
        "lyrics_sha256": existing_sha or manual_sha,
        "manually_edited": True,
        "manual_lyrics_sha256": manual_sha,
        "lines": lines,
        "words": clean_words,
        "whisper_words": preserved_whisper_words,
    }

    tmp = aligned_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    tmp.replace(aligned_path)
    LOGGER.info(
        "Saved %d manually-edited word timings to %s",
        len(clean_words),
        aligned_path,
    )
    return aligned_path


def revert_manual_edits(cache_dir: Path | str) -> Path | None:
    """Drop the ``manually_edited`` flag so the next Align re-runs WhisperX.

    Returns the path to the aligned JSON on success, or ``None`` when no
    cached alignment exists (nothing to revert).
    """
    cache = Path(cache_dir)
    aligned_path = cache / LYRICS_ALIGNED_JSON_NAME
    if not aligned_path.is_file():
        return None
    try:
        with aligned_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Can't revert {aligned_path}: not valid JSON ({exc})"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Can't revert {aligned_path}: malformed payload.")
    # Nothing to do if the file is already in auto-aligned state.
    if not data.get("manually_edited"):
        return aligned_path
    data["manually_edited"] = False
    data.pop("manual_lyrics_sha256", None)
    tmp = aligned_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(aligned_path)
    return aligned_path


# ---------------------------------------------------------------------------
# HTML + JS build
# ---------------------------------------------------------------------------


_EDITOR_CSS = """
<style>
  .mv-editor { font-family: system-ui, sans-serif; color: #e5e7eb; }
  .mv-editor .mv-toolbar { display: flex; gap: 8px; align-items: center;
    margin-bottom: 6px; font-size: 13px; }
  .mv-editor .mv-toolbar button { background: #1f2937; color: #f3f4f6;
    border: 1px solid #374151; border-radius: 4px; padding: 4px 10px;
    cursor: pointer; }
  .mv-editor .mv-toolbar button:hover { background: #374151; }
  .mv-editor .mv-toolbar .mv-info { color: #9ca3af; margin-left: auto; }
  .mv-editor .mv-timeline { position: relative; width: 100%;
    background: #0b1220; border: 1px solid #1f2937; border-radius: 6px;
    overflow: hidden; user-select: none; }
  .mv-editor .mv-waveform { display: block; width: 100%; height: 120px; }
  .mv-editor .mv-scroller, .mv-editor .mv-stage, .mv-editor .mv-words,
  .mv-editor .mv-word { -webkit-user-select: none; user-select: none; }
  .mv-editor .mv-words { position: relative; width: 100%; height: 140px;
    background: #111827; border-top: 1px solid #1f2937; overflow: hidden; }
  .mv-editor .mv-word { position: absolute; top: 20px; height: 40px;
    border-radius: 3px; padding: 2px 4px; font-size: 11px;
    line-height: 14px; color: #0b1220; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; cursor: grab; box-sizing: border-box;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.35) inset; }
  .mv-editor .mv-word.dragging { cursor: grabbing; opacity: 0.8; }
  .mv-editor .mv-word .mv-handle { position: absolute; top: 0;
    width: 6px; height: 100%; cursor: ew-resize; background: rgba(0,0,0,0.25); }
  .mv-editor .mv-word .mv-handle.left { left: 0; }
  .mv-editor .mv-word .mv-handle.right { right: 0; }
  .mv-editor .mv-word.selected { outline: 2px solid #f59e0b; }
  .mv-editor .mv-playhead { position: absolute; top: 0; bottom: 0;
    width: 2px; background: #f43f5e; pointer-events: none; z-index: 5; }
  .mv-editor .mv-line-sep { position: absolute; bottom: 6px; height: 8px;
    width: 2px; background: rgba(156, 163, 175, 0.5); }
  .mv-editor .mv-word-label { position: absolute; bottom: 2px; left: 4px;
    font-size: 10px; color: rgba(255,255,255,0.8); pointer-events: none; }
  /* Ghost layer showing whisper's own transcription on top of the
     waveform. Labels are positioned by their t_start and deliberately
     non-interactive so they never steal drags / clicks from user words. */
  .mv-editor .mv-whisper { position: absolute; left: 0; top: 0;
    width: 100%; height: 120px; pointer-events: none; overflow: hidden;
    z-index: 2; }
  /* Base text colour is white; per-label opacity is set inline from the
     CTC score so low-confidence labels fade away. The text-shadow stays
     solid (independent of opacity) to keep the label legible against
     the waveform even at low opacity. */
  .mv-editor .mv-whisper-word { position: absolute; font-size: 11px;
    color: #f8fafc; font-style: italic;
    text-shadow: 0 0 3px rgba(0, 0, 0, 0.95), 0 0 3px rgba(0, 0, 0, 0.95);
    white-space: nowrap; padding: 1px 3px; border-left: 1px solid
    rgba(248, 250, 252, 0.35); line-height: 1; }
  .mv-editor .mv-whisper-word.row-0 { top: 4px; }
  .mv-editor .mv-whisper-word.row-1 { top: 22px; }
  .mv-editor .mv-whisper-word.row-2 { top: 40px; }
  /* Vertical guide shown at the start edge of the primary dragged word
     so the user can line it up to a waveform transient by eye. */
  .mv-editor .mv-drag-guide { position: absolute; top: 0; bottom: 0;
    width: 1px; background: #fde68a; box-shadow: 0 0 4px #fde68a;
    pointer-events: none; z-index: 6; display: none; }
  .mv-editor .mv-scroller { overflow-x: auto; }
  .mv-editor .mv-stage { position: relative; }
  .mv-editor .mv-audio { display: block; width: 100%; margin-top: 6px; }
  .mv-editor .mv-help { color: #9ca3af; font-size: 11px; margin-top: 6px;
    line-height: 1.4; }
  /* Gradio themes often set global kbd/code colors; scope + !important so help
     text stays readable (fixes black-on-black key labels in the UI). */
  .mv-editor .mv-help code {
    background: #e5e7eb !important;
    color: #111827 !important;
    border-radius: 3px;
    padding: 1px 4px;
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  .mv-editor .mv-help kbd {
    background: #374151 !important;
    color: #f9fafb !important;
    border: 1px solid #6b7280 !important;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 11px;
    font-weight: 500;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
</style>
""".strip()


def _confidence_color_for_score(score: float | None) -> str:
    """Return a hex RGB string used to colour a word bar by CTC score."""
    if score is None:
        return "#94a3b8"
    s = max(0.0, min(1.0, float(score)))
    if s < 0.3:
        return "#ef4444"
    if s < 0.6:
        return "#f59e0b"
    return "#22c55e"


def build_editor_html(
    state: EditorState,
    *,
    audio_url: str,
    audio_element_id: str = "mv_editor_real_audio",
    container_id: str = "mv_editor_root",
    state_js_var: str = "_glitchframe_editor_state",
    pixels_per_second: float = 40.0,
) -> str:
    """Return the full HTML blob (CSS + markup + vanilla-JS editor) for ``state``.

    The editor reads its own state from ``window[state_js_var]``, which is
    populated inline from the supplied :class:`EditorState`. All user edits
    flow through JS into that same object; the Save button in the Gradio
    app then reads ``JSON.stringify(window[state_js_var])`` and passes it
    to :func:`save_edited_alignment`.

    ``audio_url`` is a URL the browser can GET to receive the vocal stem —
    usually Gradio's ``/file=<abs_path>`` proxy for a file inside
    ``allowed_paths``. We emit a self-contained ``<audio controls>`` tag
    with that src directly inside the editor so the play/pause/scrub UI
    lives right next to the waveform and the JS has a well-known DOM node
    to drive (no more guessing at Gradio's WaveSurfer internals).
    """
    # Prepare JSON-safe payload for the JS side. ``json.dumps`` gives us
    # valid JS object literal syntax in this case.
    payload = {
        "song_hash": state.song_hash,
        "duration": state.duration_sec,
        "sample_rate": state.sample_rate,
        "peaks": [[float(a), float(b)] for a, b in state.peaks],
        "words": [
            {
                **w,
                "color": _confidence_color_for_score(w.get("score")),
            }
            for w in state.words
        ],
        "lines": state.lines,
        "manually_edited": state.manually_edited,
        "lyrics_sha256": state.lyrics_sha256,
        "vocals_wav": state.vocals_rel_path,
        # Whisper's own transcription (what it heard), surfaced as ghost
        # text on top of the waveform so the user can re-align manually.
        # The score is preserved because the editor scales the label's
        # opacity by it — low-confidence CTC placements are faint so the
        # user can tell at a glance which labels to trust.
        "whisper_words": [
            {
                "word": str(ww.get("word", "")),
                "t_start": float(ww.get("t_start", 0.0)),
                "t_end": float(ww.get("t_end", 0.0)),
                "score": (
                    float(ww["score"])
                    if isinstance(ww.get("score"), (int, float))
                    else None
                ),
            }
            for ww in state.whisper_words
        ],
    }
    payload_json = json.dumps(payload)
    whisper_info = (
        f" · whisper: {len(state.whisper_words)} words"
        if state.whisper_words
        else " · whisper: none (re-align to populate)"
    )
    info = (
        f"{state.song_hash[:8]} · "
        f"{state.duration_sec:.1f}s · "
        f"{len(state.words)} words"
        + whisper_info
        + (" · EDITED" if state.manually_edited else "")
    )
    info_html = html_lib.escape(info)

    # The JS blob contains literal ``%`` characters (e.g. ``line_idx % 3``);
    # to avoid Python's printf-style escaping quirks we inject our handful
    # of substitution variables via plain ``str.replace``. The placeholders
    # below are surrounded with double-underscores so they can't clash with
    # anything the JS might legitimately want to write.
    script = _EDITOR_JS
    for key, val in {
        "__MV_CONTAINER_ID__": container_id,
        "__MV_STATE_JS_VAR__": state_js_var,
        "__MV_AUDIO_ELEMENT_ID__": audio_element_id,
        "__MV_PAYLOAD_JSON__": payload_json,
        "__MV_PIXELS_PER_SECOND__": str(pixels_per_second),
    }.items():
        script = script.replace(key, val)

    # Gradio's ``gr.HTML`` updates its value by setting ``innerHTML`` on the
    # wrapper element. Browsers **intentionally do not execute** ``<script>``
    # tags inserted that way, so a plain ``<script>...</script>`` only ran
    # on the initial page load and silently died on every subsequent
    # "Load timeline" click. The fix is a classic pattern:
    #
    #   1. Park the editor code in a ``<script type="text/plain">``. That
    #      tag is inert (browsers only execute ``text/javascript``), so the
    #      body survives an innerHTML injection verbatim.
    #   2. Add an ``<img src="x" onerror="...">``. The ``onerror`` handler
    #      DOES fire when the element is inserted via innerHTML, and from
    #      there we grab the parked code via ``textContent`` and eval it
    #      with ``new Function(...)``.
    #
    # We also defuse stray ``</script>`` sequences (mostly in the JSON
    # payload — a user-supplied lyric like "</script>" would otherwise end
    # the tag early) by swapping ``</`` for ``<\/`` — safe inside JS strings
    # and invisible to the engine at parse time.
    code_blob = script.replace("</", "<\\/")
    code_tag_id = f"mv_editor_code_{container_id}"
    return (
        f"{_EDITOR_CSS}"
        f"<div class=\"mv-editor\" id=\"{container_id}\">"
        f"  <div class=\"mv-toolbar\">"
        f"    <button type=\"button\" data-mv-action=\"play\">▶ Play / Pause (Space)</button>"
        f"    <button type=\"button\" data-mv-action=\"zoom-in\">+</button>"
        f"    <button type=\"button\" data-mv-action=\"zoom-out\">−</button>"
        f"    <button type=\"button\" data-mv-action=\"zoom-fit\">Fit</button>"
        f"    <span class=\"mv-info\">{info_html}</span>"
        f"  </div>"
        f"  <div class=\"mv-scroller\" data-mv-scroller>"
        f"    <div class=\"mv-stage\" data-mv-stage>"
        f"      <canvas class=\"mv-waveform\" data-mv-waveform></canvas>"
        f"      <div class=\"mv-whisper\" data-mv-whisper></div>"
        f"      <div class=\"mv-words\" data-mv-words></div>"
        f"      <div class=\"mv-playhead\" data-mv-playhead></div>"
        f"      <div class=\"mv-drag-guide\" data-mv-drag-guide></div>"
        f"    </div>"
        f"  </div>"
        # Native HTML5 <audio> lives directly inside the editor so the
        # playback UI is flush with the waveform the user is editing. The
        # JS toolbar drives this exact element (known id), so there's no
        # dependency on Gradio's WaveSurfer DOM.
        f"  <audio class=\"mv-audio\" id=\"{audio_element_id}\" "
        f"src=\"{html_lib.escape(audio_url, quote=True)}\" "
        f"controls preload=\"auto\"></audio>"
        f"  <div class=\"mv-help\">"
        f"    Drag a word bar to shift it. Drag its left / right edge to change "
        f"<code>t_start</code> / <code>t_end</code>. Click a word to seek the audio. "
        f"<kbd>Shift</kbd>/<kbd>Ctrl</kbd>-click to add words to the selection; "
        f"click-and-drag on empty timeline to rubber-band-select. With multiple "
        f"words selected, dragging any of them moves them all together. "
        f"<kbd>Esc</kbd> deselect, <kbd>Ctrl</kbd>+<kbd>A</kbd> select all. "
        f"<kbd>Del</kbd>/<kbd>Backspace</kbd> remove selected word(s). "
        f"<kbd>Space</kbd> play/pause. <kbd>+</kbd>/<kbd>−</kbd> zoom. "
        f"Colour: green = confident, yellow = low confidence, red = very low, grey = no score."
        f"  </div>"
        f"</div>"
        f"<script type=\"text/plain\" id=\"{code_tag_id}\">{code_blob}</script>"
        f"<img src=\"x\" alt=\"\" style=\"display:none\" "
        f"onerror=\"this.remove();"
        f"var _c=document.getElementById('{code_tag_id}');"
        f"if(_c){{try{{(new Function(_c.textContent))();}}"
        f"catch(_e){{console.error('mv-editor init failed',_e);}}}}\">"
    )


# The big JS blob. Uses ``__MV_...__`` placeholder names that
# :func:`build_editor_html` rewrites with ``str.replace`` so we don't have
# to escape the literal ``%`` operators used inside the JS (e.g. for
# ``line_idx % 3`` row striping).
_EDITOR_JS = r"""
(function() {
  const container = document.getElementById("__MV_CONTAINER_ID__");
  if (!container) { return; }
  const state = __MV_PAYLOAD_JSON__;
  // ``state`` is the sole source of truth; every drag mutates it in place,
  // and the Gradio Save button reads ``JSON.stringify(window.__MV_STATE_JS_VAR__)``.
  window.__MV_STATE_JS_VAR__ = state;

  const scroller = container.querySelector("[data-mv-scroller]");
  const stage = container.querySelector("[data-mv-stage]");
  const wave = container.querySelector("[data-mv-waveform]");
  const whisperLayer = container.querySelector("[data-mv-whisper]");
  const words = container.querySelector("[data-mv-words]");
  const playhead = container.querySelector("[data-mv-playhead]");
  const dragGuide = container.querySelector("[data-mv-drag-guide]");
  const toolbar = container.querySelector(".mv-toolbar");

  let pxPerSec = __MV_PIXELS_PER_SECOND__;

  // The editor now ships its OWN ``<audio>`` tag with a known id, so this
  // is a simple ``getElementById``. Scoped to the editor's container so
  // multiple editors on the same page (unlikely but free) don't collide.
  function audio() {
    return container.querySelector("#__MV_AUDIO_ELEMENT_ID__")
        || document.getElementById("__MV_AUDIO_ELEMENT_ID__");
  }

  function setStageWidth() {
    const w = Math.max(600, Math.round(state.duration * pxPerSec));
    stage.style.width = w + "px";
    wave.width = w;
    wave.height = 120;
    words.style.width = w + "px";
    if (whisperLayer) whisperLayer.style.width = w + "px";
    drawWaveform();
    renderWhisperWords();
  }

  function drawWaveform() {
    const ctx = wave.getContext("2d");
    const W = wave.width;
    const H = wave.height;
    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();
    if (!state.peaks || state.peaks.length === 0) { return; }
    ctx.strokeStyle = "#38bdf8";
    ctx.beginPath();
    for (let x = 0; x < W; x++) {
      const idx = Math.min(state.peaks.length - 1,
                           Math.floor((x / W) * state.peaks.length));
      const [mn, mx] = state.peaks[idx];
      const y0 = H / 2 - (mx * H / 2);
      const y1 = H / 2 - (mn * H / 2);
      ctx.moveTo(x + 0.5, y0);
      ctx.lineTo(x + 0.5, y1);
    }
    ctx.stroke();
  }

  function secondsToPx(t) { return t * pxPerSec; }
  function pxToSeconds(x) { return x / pxPerSec; }

  // Render "what whisper heard" as faint italic labels overlaid on the
  // waveform. Labels are staggered across three rows by modulo-index so
  // adjacent words don't visually collide at high zoom. No pointer
  // events — the user drags their OWN words, the ghost layer is purely
  // a reference so they can manually align without listening.
  function renderWhisperWords() {
    if (!whisperLayer) return;
    whisperLayer.innerHTML = "";
    const ww = Array.isArray(state.whisper_words) ? state.whisper_words : [];
    if (ww.length === 0) return;
    ww.forEach((w, i) => {
      const el = document.createElement("span");
      el.className = "mv-whisper-word row-" + (i % 3);
      el.style.left = secondsToPx(w.t_start) + "px";
      // Scale opacity by CTC score so the user can tell at a glance
      // which whisper labels to trust. Interpolated / scoreless entries
      // were already filtered in Python, but fall back to a visible
      // floor just in case an older cache slips one through.
      const sc = (typeof w.score === "number") ? w.score : 0.4;
      const clamped = Math.max(0.25, Math.min(1.0, sc));
      const opacity = 0.30 + 0.55 * clamped;  // 0.44 .. 0.85
      el.style.opacity = opacity.toFixed(2);
      el.textContent = w.word;
      whisperLayer.appendChild(el);
    });
  }

  // Drag guide: thin vertical line shown at the primary dragged word's
  // ``t_start`` so the user can eyeball the start edge against a
  // waveform transient. When multiple words are selected only the
  // first / primary one gets a guide — multiple lines would be noisy
  // and we care most about aligning the *front* of the group anyway.
  function showDragGuide(t) {
    if (!dragGuide) return;
    dragGuide.style.left = secondsToPx(Math.max(0, t)) + "px";
    dragGuide.style.display = "block";
  }
  function hideDragGuide() {
    if (dragGuide) dragGuide.style.display = "none";
  }

  function renderWords() {
    words.innerHTML = "";
    const prevLineById = new Map();
    state.words.forEach((w, i) => {
      const el = document.createElement("div");
      el.className = "mv-word";
      el.dataset.index = String(i);
      el.style.background = w.color || "#64748b";
      const left = secondsToPx(w.t_start);
      const width = Math.max(6, secondsToPx(Math.max(0, w.t_end - w.t_start)));
      el.style.left = left + "px";
      el.style.width = width + "px";
      // Alternate rows per line to reduce overlap noise on dense sections.
      el.style.top = (12 + (w.line_idx % 3) * 44) + "px";

      const label = document.createElement("span");
      label.className = "mv-word-label";
      const sc = (typeof w.score === "number")
        ? " · " + w.score.toFixed(2)
        : "";
      label.textContent = w.word + sc;
      el.appendChild(label);

      const lh = document.createElement("div");
      lh.className = "mv-handle left";
      const rh = document.createElement("div");
      rh.className = "mv-handle right";
      el.appendChild(lh);
      el.appendChild(rh);

      words.appendChild(el);
    });
    attachWordHandlers();
    // Re-render wipes the DOM; re-apply selection classes so zooming or
    // reflow doesn't silently lose the yellow outline on selected words.
    updateSelectionStyling();
  }

  function attachWordHandlers() {
    const nodes = words.querySelectorAll(".mv-word");
    nodes.forEach(node => {
      node.addEventListener("pointerdown", onWordPointerDown);
    });
  }

  // ─── Selection model ──────────────────────────────────────────────────
  // ``selected`` is a Set of word indices. Multi-word drag moves every
  // selected word by the same delta (clamped so none escape [0, duration]).
  // Rubber-band selection (click-drag on empty timeline background) adds
  // every word whose centre falls inside the rectangle.
  const selected = new Set();
  // Set by the rubber-band pointerup so the subsequent synthetic ``click``
  // on the stage doesn't also seek the playhead — we only want one or the
  // other per gesture, never both.
  let suppressNextStageClick = false;

  function updateSelectionStyling() {
    const nodes = words.querySelectorAll(".mv-word");
    nodes.forEach(n => {
      const i = parseInt(n.dataset.index, 10);
      if (selected.has(i)) n.classList.add("selected");
      else n.classList.remove("selected");
    });
    updateSelectionInfo();
  }

  function clearSelection() {
    if (selected.size === 0) return;
    selected.clear();
    updateSelectionStyling();
  }

  function deleteSelected() {
    if (selected.size === 0) return;
    state.words = state.words.filter((_, i) => !selected.has(i));
    selected.clear();
    renderWords();
  }

  function updateSelectionInfo() {
    const info = container.querySelector(".mv-info");
    if (!info) return;
    const baseLabel = info.dataset.baseLabel || info.textContent;
    info.dataset.baseLabel = baseLabel;
    info.textContent = selected.size > 1
      ? baseLabel + "  ·  " + selected.size + " selected"
      : baseLabel;
  }

  let drag = null;
  function onWordPointerDown(ev) {
    ev.preventDefault();
    // Don't let the stage's rubber-band handler also fire for this event.
    ev.stopPropagation();
    const node = ev.currentTarget;
    const idx = parseInt(node.dataset.index, 10);
    const additive = ev.shiftKey || ev.ctrlKey || ev.metaKey;

    const mode = ev.target.classList.contains("mv-handle")
      ? (ev.target.classList.contains("left") ? "left" : "right")
      : "move";

    // Selection semantics (matches typical DAW / finder conventions):
    //   modifier + click unselected → add word to selection
    //   modifier + click selected   → remove from selection; no drag
    //   plain click unselected      → selection becomes {idx}
    //   plain click selected        → keep selection (for group drag)
    if (additive) {
      if (selected.has(idx)) {
        selected.delete(idx);
        updateSelectionStyling();
        return;
      }
      selected.add(idx);
    } else if (!selected.has(idx)) {
      selected.clear();
      selected.add(idx);
    }
    updateSelectionStyling();

    // Resizing via edge handle only makes sense on a single word; the
    // group-drag path is strictly for ``move``. Handles on multi-selection
    // fall back to resizing just the word the user grabbed.
    const dragIndices = (mode === "move")
      ? Array.from(selected)
      : [idx];
    const origs = new Map();
    dragIndices.forEach(i => {
      const ww = state.words[i];
      origs.set(i, { s: ww.t_start, e: ww.t_end });
    });

    drag = {
      primary: idx,
      mode,
      node,
      startX: ev.clientX,
      indices: dragIndices,
      origs,
    };
    node.setPointerCapture(ev.pointerId);
    node.classList.add("dragging");
    if (mode === "move") {
      showDragGuide(state.words[idx].t_start);
    } else if (mode === "left") {
      showDragGuide(state.words[idx].t_start);
    } else if (mode === "right") {
      showDragGuide(state.words[idx].t_end);
    }
    document.addEventListener("pointermove", onWordPointerMove);
    document.addEventListener("pointerup", onWordPointerUp, { once: true });
  }

  function onWordPointerMove(ev) {
    if (!drag) { return; }
    const dx = ev.clientX - drag.startX;
    const dt = pxToSeconds(dx);
    const dur = Math.max(0, state.duration);
    if (drag.mode === "move") {
      // Clamp the delta so no word in the group escapes [0, duration].
      // Using one shared delta preserves the relative spacing between
      // words exactly — that's the whole point of group-move.
      let dtClamped = dt;
      drag.indices.forEach(i => {
        const o = drag.origs.get(i);
        if (o.s + dtClamped < 0) dtClamped = -o.s;
        if (o.e + dtClamped > dur) dtClamped = dur - o.e;
      });
      drag.indices.forEach(i => {
        const o = drag.origs.get(i);
        const w = state.words[i];
        w.t_start = o.s + dtClamped;
        w.t_end = o.e + dtClamped;
        updateWordDom(i);
      });
      showDragGuide(state.words[drag.primary].t_start);
    } else {
      const i = drag.primary;
      const w = state.words[i];
      const o = drag.origs.get(i);
      if (drag.mode === "left") {
        w.t_start = Math.max(0, Math.min(o.e - 0.02, o.s + dt));
        showDragGuide(w.t_start);
      } else if (drag.mode === "right") {
        w.t_end = Math.min(dur, Math.max(o.s + 0.02, o.e + dt));
        showDragGuide(w.t_end);
      }
      updateWordDom(i);
    }
  }

  function onWordPointerUp(ev) {
    if (!drag) { return; }
    hideDragGuide();
    drag.node.classList.remove("dragging");
    const movedPx = Math.abs(ev.clientX - drag.startX);
    // Only the single-word click-without-drag case auto-plays. A click on
    // a word in a multi-selection shouldn't suddenly collapse the
    // selection by seeking off somewhere — we just keep the group selected.
    if (movedPx < 3 && drag.mode === "move" && drag.indices.length === 1) {
      const w = state.words[drag.primary];
      const a = audio();
      if (a) {
        try { a.currentTime = Math.max(0, w.t_start - 0.25); a.play(); }
        catch (e) { /* swallow; user can press Space */ }
      }
    }
    drag = null;
    document.removeEventListener("pointermove", onWordPointerMove);
  }

  function updateWordDom(idx) {
    const node = words.querySelector(".mv-word[data-index=\"" + idx + "\"]");
    if (!node) { return; }
    const w = state.words[idx];
    const left = secondsToPx(w.t_start);
    const width = Math.max(6, secondsToPx(Math.max(0, w.t_end - w.t_start)));
    node.style.left = left + "px";
    node.style.width = width + "px";
  }

  // ─── Rubber-band selection ────────────────────────────────────────────
  let band = null;
  let bandEl = null;

  function ensureBandEl() {
    if (bandEl) return bandEl;
    bandEl = document.createElement("div");
    bandEl.className = "mv-band";
    bandEl.style.position = "absolute";
    bandEl.style.top = "0";
    bandEl.style.bottom = "0";
    bandEl.style.border = "1px dashed #f59e0b";
    bandEl.style.background = "rgba(245, 158, 11, 0.12)";
    bandEl.style.pointerEvents = "none";
    bandEl.style.display = "none";
    bandEl.style.zIndex = "4";
    stage.appendChild(bandEl);
    return bandEl;
  }

  // Word rows in the lyrics editor are laid out via ``top = 12 + (line_idx
  // % 3) * 44`` inside the ``.mv-words`` container (3 alternating rows).
  // When a rubber-band drag starts inside that container we scope the
  // selection + visual band to whichever row the pointer went down over,
  // so users can marquee-select just the words on a single row. Starting
  // the drag over the waveform keeps the legacy all-row behaviour.
  const MV_ROW_HEIGHT_PX = 44;

  function _rowFromPointer(ev) {
    if (!words) return null;
    const wr = words.getBoundingClientRect();
    const y = ev.clientY - wr.top;
    if (y < 0 || y >= wr.height) return null;
    const row = Math.floor(y / MV_ROW_HEIGHT_PX);
    if (row < 0) return 0;
    if (row > 2) return 2;
    return row;
  }

  stage.addEventListener("pointerdown", (ev) => {
    // Words and their handles call stopPropagation in onWordPointerDown,
    // so getting here means the click began on empty timeline / waveform.
    // Still belt-and-suspenders guard in case a new child element forgets.
    if (ev.target.classList) {
      if (ev.target.classList.contains("mv-word") ||
          ev.target.classList.contains("mv-handle") ||
          ev.target.classList.contains("mv-word-label")) {
        return;
      }
    }
    const rect = stage.getBoundingClientRect();
    const additive = ev.shiftKey || ev.ctrlKey || ev.metaKey;
    if (!additive) { clearSelection(); }
    const row = _rowFromPointer(ev);
    band = {
      startX: ev.clientX - rect.left,
      startClientX: ev.clientX,
      origSelected: new Set(selected),
      moved: false,
      row: row,
    };
    const el = ensureBandEl();
    el.style.display = "block";
    el.style.left = band.startX + "px";
    el.style.width = "0px";
    // Row-scoped bands visually clip to that row only so the user sees
    // that the drag won't pick up words on neighbouring rows.
    if (row !== null && words) {
      const wr = words.getBoundingClientRect();
      const top = (wr.top - rect.top) + row * MV_ROW_HEIGHT_PX;
      el.style.top = top + "px";
      el.style.bottom = "auto";
      el.style.height = MV_ROW_HEIGHT_PX + "px";
    } else {
      el.style.top = "0";
      el.style.bottom = "0";
      el.style.height = "";
    }
    document.addEventListener("pointermove", onBandMove);
    document.addEventListener("pointerup", onBandUp, { once: true });
  });

  function onBandMove(ev) {
    if (!band) return;
    const rect = stage.getBoundingClientRect();
    const curX = ev.clientX - rect.left;
    const left = Math.min(band.startX, curX);
    const right = Math.max(band.startX, curX);
    if (bandEl) {
      bandEl.style.left = left + "px";
      bandEl.style.width = (right - left) + "px";
    }
    if (Math.abs(ev.clientX - band.startClientX) > 2) band.moved = true;

    // Rebuild selection from scratch each move so shrinking the band
    // actually *unselects* words the user dragged away from.
    const newSel = new Set(band.origSelected);
    state.words.forEach((w, i) => {
      if (band.row !== null && ((w.line_idx || 0) % 3) !== band.row) return;
      const cx = secondsToPx((w.t_start + w.t_end) / 2);
      if (cx >= left && cx <= right) newSel.add(i);
    });
    selected.clear();
    newSel.forEach(i => selected.add(i));
    updateSelectionStyling();
  }

  function onBandUp(_ev) {
    if (!band) return;
    if (bandEl) {
      bandEl.style.display = "none";
      // Reset inline row-scoping so the next waveform-scoped drag renders
      // full-height again.
      bandEl.style.top = "0";
      bandEl.style.bottom = "0";
      bandEl.style.height = "";
    }
    if (band.moved) { suppressNextStageClick = true; }
    band = null;
    document.removeEventListener("pointermove", onBandMove);
  }

  function tickPlayhead() {
    const a = audio();
    if (a && !a.paused && !a.ended) {
      playhead.style.left = secondsToPx(a.currentTime) + "px";
      // Keep playhead visible while scrolling through a long song.
      const sl = scroller.scrollLeft;
      const sw = scroller.clientWidth;
      const x = secondsToPx(a.currentTime);
      if (x < sl + 40) { scroller.scrollLeft = Math.max(0, x - 40); }
      else if (x > sl + sw - 40) { scroller.scrollLeft = x - sw + 40; }
    }
    requestAnimationFrame(tickPlayhead);
  }

  // Click on waveform / timeline background → seek audio. Suppressed when
  // the click was actually the terminating event of a rubber-band drag
  // (handled in ``onBandUp``) so selecting and seeking don't collide.
  stage.addEventListener("click", (ev) => {
    if (suppressNextStageClick) { suppressNextStageClick = false; return; }
    if (ev.target.classList && ev.target.classList.contains("mv-word")) { return; }
    if (ev.target.classList && ev.target.classList.contains("mv-handle")) { return; }
    const rect = stage.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const t = pxToSeconds(x);
    const a = audio();
    if (a) { try { a.currentTime = Math.max(0, t); } catch (e) {} }
  });

  toolbar.addEventListener("click", (ev) => {
    const action = ev.target && ev.target.getAttribute("data-mv-action");
    if (!action) { return; }
    if (action === "play") {
      const a = audio();
      if (a) { a.paused ? a.play() : a.pause(); }
    } else if (action === "zoom-in") { setZoom(pxPerSec * 1.25); }
    else if (action === "zoom-out") { setZoom(pxPerSec / 1.25); }
    else if (action === "zoom-fit") { setZoom(Math.max(20, (scroller.clientWidth - 8) / Math.max(1, state.duration))); }
  });

  function setZoom(v) {
    pxPerSec = Math.max(4, Math.min(600, v));
    setStageWidth();
    renderWords();
  }

  document.addEventListener("keydown", (ev) => {
    if (ev.target && /INPUT|TEXTAREA|SELECT/.test(ev.target.tagName)) { return; }
    if (!container.offsetParent) { return; }  // hidden tab
    if (ev.code === "Space") {
      ev.preventDefault();
      const a = audio();
      if (a) { a.paused ? a.play() : a.pause(); }
    } else if (ev.key === "+" || ev.key === "=") { setZoom(pxPerSec * 1.25); }
    else if (ev.key === "-" || ev.key === "_") { setZoom(pxPerSec / 1.25); }
    else if (ev.key === "Escape") { clearSelection(); }
    else if (ev.key === "Delete" || ev.key === "Backspace") {
      if (selected.size > 0) {
        ev.preventDefault();
        deleteSelected();
      }
    } else if ((ev.ctrlKey || ev.metaKey) && (ev.key === "a" || ev.key === "A")) {
      ev.preventDefault();
      selected.clear();
      state.words.forEach((_, i) => selected.add(i));
      updateSelectionStyling();
    }
  });

  setStageWidth();
  renderWords();
  requestAnimationFrame(tickPlayhead);
})();
"""


__all__ = [
    "EditorState",
    "build_editor_html",
    "compute_peaks",
    "load_editor_state",
    "revert_manual_edits",
    "save_edited_alignment",
]
