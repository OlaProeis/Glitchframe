"""
YouTube-oriented ``metadata.txt`` for a render run: title, description (song
info, optional lyrics, chapter timestamps), and tags derived from song metadata
and the active visual preset.

File format is plain UTF-8 text with ``## SECTION`` headers so it is easy to
read, copy-paste into YouTube, and parse programmatically (see
:func:`parse_metadata_txt`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

METADATA_TXT_NAME = "metadata.txt"
METADATA_FORMAT_VERSION = 1

# Structural segments are unlabeled clusters; use readable placeholders.
_SECTION_NAMES = (
    "Intro",
    "Verse 1",
    "Chorus",
    "Verse 2",
    "Bridge",
    "Verse 3",
    "Outro",
    "Section 8",
)


def format_chapter_timestamp(t_sec: float) -> str:
    """Format seconds for YouTube-style chapter stamps (``H:MM:SS`` or ``M:SS``)."""
    t = max(0.0, float(t_sec))
    total = int(round(t))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _segment_label(index: int) -> str:
    if 0 <= index < len(_SECTION_NAMES):
        return _SECTION_NAMES[index]
    return f"Part {index + 1}"


def chapter_lines_from_analysis(
    analysis: Mapping[str, Any],
) -> list[str]:
    """
    Build ``["M:SS Label", ...]`` from ``analysis.json`` ``segments`` (start
    times and order match the analyzer output).
    """
    raw = analysis.get("segments")
    if not isinstance(raw, list) or not raw:
        return []
    lines: list[str] = []
    for i, seg in enumerate(raw):
        if not isinstance(seg, Mapping):
            continue
        try:
            t0 = float(seg["t_start"])
        except (KeyError, TypeError, ValueError):
            continue
        label = _segment_label(i)
        lines.append(f"{format_chapter_timestamp(t0)} {label}")
    return lines


def build_youtube_title(
    artist: str | None,
    title: str | None,
    *,
    suffix: str = "[Official Visualizer]",
) -> str:
    artist = (artist or "").strip()
    title = (title or "").strip()
    if artist and title:
        return f"{artist} - {title} {suffix}".strip()
    if title:
        return f"{title} {suffix}".strip()
    if artist:
        return f"{artist} {suffix}".strip()
    return f"Music visualizer {suffix}".strip()


def _unique_tags(tags: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        s = str(t).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def build_tags(
    song_metadata: Mapping[str, Any],
    *,
    preset_id: str | None = None,
    preset: Mapping[str, Any] | None = None,
) -> list[str]:
    """Tags: genre, artist, title tokens, preset id / prompt tokens, and defaults."""
    tags: list[str] = []
    genre = song_metadata.get("genre")
    if genre and str(genre).strip():
        tags.append(str(genre).strip())
    artist = song_metadata.get("artist")
    if artist and str(artist).strip():
        tags.append(str(artist).strip())
    title = song_metadata.get("title")
    if title and str(title).strip():
        tags.append(str(title).strip())

    tags.append("music visualizer")

    if preset_id and str(preset_id).strip():
        pid = str(preset_id).strip()
        tags.append(pid)
        for part in pid.split("-"):
            p = part.strip()
            if p and len(p) > 1:
                tags.append(p)

    if preset:
        for key in ("shader", "typo_style"):
            val = preset.get(key)
            if val and str(val).strip():
                tags.append(str(val).strip())

    return _unique_tags(tags)


def _info_lines(song_metadata: Mapping[str, Any], bpm: float | None) -> list[str]:
    lines: list[str] = []
    artist = song_metadata.get("artist")
    title = song_metadata.get("title")
    album = song_metadata.get("album")
    year = song_metadata.get("year")
    genre = song_metadata.get("genre")
    if artist:
        lines.append(f"Artist: {artist}")
    if title:
        lines.append(f"Title: {title}")
    if album:
        lines.append(f"Album: {album}")
    if year:
        lines.append(f"Year: {year}")
    if genre:
        lines.append(f"Genre: {genre}")
    if bpm is not None and bpm > 0:
        lines.append(f"BPM: {bpm:.2f}")
    return lines


def build_description_body(
    song_metadata: Mapping[str, Any],
    *,
    chapter_lines: Sequence[str],
    lyrics_lines: Sequence[str] | None = None,
    bpm: float | None = None,
    footer: str = "Generated with Glitchframe (local).",
) -> str:
    """Multi-line description: credits, optional lyrics, chapters, footer."""
    blocks: list[str] = []
    info = _info_lines(song_metadata, bpm)
    if info:
        blocks.append("\n".join(info))

    if lyrics_lines:
        lyric_text = "\n".join(str(L).rstrip() for L in lyrics_lines if str(L).strip())
        if lyric_text.strip():
            blocks.append("Lyrics:\n" + lyric_text)

    if chapter_lines:
        blocks.append("Chapters:\n" + "\n".join(str(c) for c in chapter_lines))

    blocks.append(footer)
    return "\n\n".join(blocks)


def compose_metadata_txt(
    *,
    youtube_title: str,
    description: str,
    chapter_lines: Sequence[str],
    tags: Sequence[str],
) -> str:
    """Full file content with ``##`` sections."""
    tag_line = ", ".join(str(t) for t in tags)
    chapter_block = "\n".join(str(c) for c in chapter_lines) if chapter_lines else "(none)"
    return (
        f"glitchframe_metadata_version: {METADATA_FORMAT_VERSION}\n"
        f"\n"
        f"## TITLE\n"
        f"{youtube_title}\n"
        f"\n"
        f"## DESCRIPTION\n"
        f"{description}\n"
        f"\n"
        f"## CHAPTERS\n"
        f"{chapter_block}\n"
        f"\n"
        f"## TAGS\n"
        f"{tag_line}\n"
    )


def write_metadata_txt(
    output_dir: str | Path,
    *,
    song_metadata: Mapping[str, Any],
    analysis: Mapping[str, Any],
    preset_id: str | None = None,
    preset: Mapping[str, Any] | None = None,
    lyrics_lines: Sequence[str] | None = None,
    bpm: float | None = None,
    footer: str = "Generated with Glitchframe (local).",
) -> Path:
    """
    Write ``metadata.txt`` under ``output_dir`` (created if missing).

    ``song_metadata`` uses optional keys: ``artist``, ``title``, ``album``,
    ``year``, ``genre`` (same as :class:`orchestrator.OrchestratorInputs.metadata`).
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    if bpm is None and analysis:
        tempo = analysis.get("tempo")
        if isinstance(tempo, dict):
            try:
                b = float(tempo.get("bpm") or 0.0)
                if b > 0:
                    bpm = b
            except (TypeError, ValueError):
                pass

    chapter_lines = chapter_lines_from_analysis(analysis)
    title = build_youtube_title(
        song_metadata.get("artist") if song_metadata else None,
        song_metadata.get("title") if song_metadata else None,
    )
    tags = build_tags(song_metadata or {}, preset_id=preset_id, preset=preset)
    desc = build_description_body(
        song_metadata or {},
        chapter_lines=chapter_lines,
        lyrics_lines=lyrics_lines,
        bpm=bpm,
        footer=footer,
    )
    text = compose_metadata_txt(
        youtube_title=title,
        description=desc,
        chapter_lines=chapter_lines,
        tags=tags,
    )

    out = root / METADATA_TXT_NAME
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(out)
    return out


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_metadata_txt(text: str) -> dict[str, Any]:
    """
    Parse a ``metadata.txt`` produced by :func:`compose_metadata_txt`.

    Returns keys: ``version`` (int|None), ``title``, ``description``, ``chapters``
    (list[str]), ``tags`` (list[str]).
    """
    version: int | None = None
    first_line = text.splitlines()[0] if text else ""
    m_ver = re.match(
        r"(?:glitchframe|musicvids)_metadata_version:\s*(\d+)\s*$",
        first_line.strip(),
    )
    if m_ver:
        version = int(m_ver.group(1))

    sections: dict[str, str] = {}
    for match in _SECTION_RE.finditer(text):
        name = match.group(1).strip().upper()
        start = match.end()
        nxt = _SECTION_RE.search(text, pos=start)
        end = nxt.start() if nxt else len(text)
        sections[name] = text[start:end].strip("\n")

    title = sections.get("TITLE", "").strip()
    description = sections.get("DESCRIPTION", "").strip()
    chapters_raw = sections.get("CHAPTERS", "").strip()
    if chapters_raw == "(none)" or not chapters_raw:
        chapters_list: list[str] = []
    else:
        chapters_list = [ln.strip() for ln in chapters_raw.splitlines() if ln.strip()]

    tags_raw = sections.get("TAGS", "").strip()
    tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]

    return {
        "version": version,
        "title": title,
        "description": description,
        "chapters": chapters_list,
        "tags": tags_list,
    }
