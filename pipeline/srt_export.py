"""Export ``lyrics.aligned.json`` to SubRip (.srt) for external subtitle tools."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pipeline.kinetic_typography import AlignedWord, load_aligned_words

SRT_OUTPUT_NAME = "lyrics.aligned.srt"


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS,mmm`` (SubRip)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    total_ms = max(0, total_ms)
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = min(total_min // 60, 99)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _cue_text(word: str) -> str | None:
    t = (
        (word or "")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )
    return t if t else None


def build_srt_document(words: Sequence[AlignedWord]) -> str:
    """One SubRip cue per aligned word (matches kinetic typography timing)."""
    chunks: list[str] = []
    idx = 1
    for w in words:
        text = _cue_text(w.word)
        if text is None:
            continue
        chunks.append(str(idx))
        chunks.append(
            f"{format_srt_timestamp(w.t_start)} --> {format_srt_timestamp(w.t_end)}"
        )
        chunks.append(text)
        chunks.append("")
        idx += 1
    return "\n".join(chunks)


def export_aligned_json_to_srt(
    aligned_json: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """
    Read aligned lyrics JSON and write ``lyrics.aligned.srt`` beside it unless
    ``output_path`` is given.

    Returns the output path.
    """
    aligned_json = Path(aligned_json)
    _, words = load_aligned_words(aligned_json)
    if not words:
        raise ValueError("No aligned words to export (words list is empty).")
    body = build_srt_document(words)
    if not body.strip():
        raise ValueError("No subtitle cues to export (all words empty).")
    out = output_path or (aligned_json.parent / SRT_OUTPUT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n", encoding="utf-8")
    return out
