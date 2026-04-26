# Effects timeline — unittest coverage

Maps the **effects-timeline stack** test modules to behaviour so regressions are easy to trace. Run the full project suite with:

`.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

## Module map

| Test module | Focus |
|-------------|--------|
| `tests/test_effects_timeline.py` | `EffectsTimeline` / `EffectClip` JSON load/save, `validate_*`, unknown settings, atomic tmp files, schema version, `auto_reactivity_master` bounds, multi-kind round-trip. |
| `tests/test_effects_editor.py` | `load_editor_state`, `save_edited_timeline` (JSON + `song_hash` rules), `bake_auto_schedule` (dedupe, zoom/glitch bake), `build_editor_html` smoke, `build_ghost_events`. |
| `tests/test_screen_shake.py` | `shake_offset`: inside/outside clip, determinism, amplitude, overlap sum, non-shake clips ignored. |
| `tests/test_color_invert.py` | `invert_mix`: defaults, clamping, overlap cap, non-invert clips ignored, non-finite `t`. |
| `tests/test_zoom_punch.py` | `zoom_scale`: ease/width, overlap max, identity when peak ≤ 1, defaults. |
| `tests/test_chromatic_aberration.py` | `apply_chromatic_aberration`: inactive window, `shift_px=0`, G fixed, determinism, overlap sum, non-chromatic ignored. |
| `tests/test_scanline_tear.py` | `apply_scanline_tear`: inactive window, `intensity=0`, determinism, wrap/black modes, RGB lockstep on flat colour, clip order, shape guard. |
| `tests/test_compositor_effects_timeline.py` | Compositor helpers: `auto_reactivity_master`, scaled pulses, user BEAM / glitch merge, `_build_frame_effects_context`, `_apply_frame_effects` (zoom/shake/chromatic/scanline/invert, post-pass order). |

## Related docs

- Model: `docs/technical/effects-timeline.md`
- Compositor wiring: `docs/technical/effects-timeline-compositor.md`
- Renderers overview: `docs/technical/effects-timeline-renderers.md`
