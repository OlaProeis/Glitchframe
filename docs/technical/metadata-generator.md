# Metadata generator (`metadata.txt`)

Feature: build a **YouTube-oriented** `metadata.txt` beside `output.mp4` under `outputs/<run_id>/`: suggested title, description (credits, optional lyrics, chapter timestamps), and tags from song fields and the active visual preset.

## Inputs

- **Song fields** — `OrchestratorInputs.metadata`: optional `artist`, `title`, `album`, `year`, `genre` (cosmetic; not part of `song_hash`).
- **`analysis.json`** — `segments` for chapter start times; `tempo.bpm` for the description when not overridden.
- **Lyrics** — prefer `AlignmentResult.lines` when alignment ran; otherwise non-empty `OrchestratorInputs.lyrics_text` split by line.
- **Preset** — `preset_id` / full preset dict for tags (`shader`, `typo_style`, preset id and hyphen segments). Resolved from `OrchestratorInputs.presets` (`id`, `preset_id`, or `preset` dict); unknown ids skip loaded preset data.

## Output file format (v1)

UTF-8 text, parsing-friendly:

1. First line: `glitchframe_metadata_version: 1` (legacy files may use `musicvids_metadata_version:`)
2. Sections with headers `## TITLE`, `## DESCRIPTION`, `## CHAPTERS`, `## TAGS`

**Title** follows `{Artist} — {Title} [Official Visualizer]` with fallbacks if fields are missing.

**Chapters** are one line per structural segment from the analyzer, ordered by `t_start`, formatted as `M:SS` or `H:MM:SS` plus a placeholder label (`Intro`, `Verse 1`, `Chorus`, …). Timestamps match `analysis.json` segment starts (rounded to whole seconds for stamps).

**Tags** are deduplicated case-insensitively; always includes `music visualizer`.

## Code

| Piece | Location |
|-------|----------|
| Build text, write atomically, `parse_metadata_txt` | `pipeline/metadata.py` |
| `write_run_metadata(run_output_dir, inputs=..., analysis_doc=..., alignment=...)` | `orchestrator.py` |
| Segment schema | `docs/technical/audio-analyzer.md` |

Call **`write_run_metadata`** after a full render when the run directory exists (same stage as `thumbnail.png`).

## Tests

`tests/test_metadata.py` — chapter lines vs `analysis.json` segments, round-trip parse, orchestrator wiring.

## Related

- `docs/technical/pipeline-orchestrator.md` — `OrchestratorInputs` and cache vs `run_id`.
- `docs/technical/lyrics-aligner.md` — aligned lines for descriptions.
