# Effects timeline — unittest coverage

Maps the **effects-timeline stack** test modules to behaviour so regressions are easy to trace. Run the full project suite with:

`.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

## Module map

| Test module | Focus |
|-------------|--------|
| `tests/test_effects_timeline.py` | `EffectsTimeline` / `EffectClip` JSON load/save, `validate_*`, unknown settings, atomic tmp files, schema version, `auto_reactivity_master` bounds, multi-kind round-trip, **legacy `ZOOM_PUNCH` clip dropped on load**. |
| `tests/test_effects_editor.py` | `load_editor_state`, `save_edited_timeline` (JSON + `song_hash` rules), `bake_auto_schedule` (shake dedupe, RMS-impact glitch bake), `build_editor_html` smoke (all nine rows), `build_ghost_events`. |
| `tests/test_screen_shake.py` | `shake_offset`: inside/outside clip, determinism, amplitude, overlap sum, non-shake clips ignored. |
| `tests/test_color_invert.py` | `invert_mix`: defaults, clamping, overlap cap, non-invert clips ignored, non-finite `t`. |
| `tests/test_fade.py` | `fade_alpha` / `apply_fade`: in/out direction, `smoothstep` vs `linear` ease, `peak_alpha` clamp, overlap (max), full-black short-circuit, non-fade ignored, non-finite `t`. |
| `tests/test_pixel_smear.py` | `apply_pixel_smear`: inactive window, zero-`intensity`/`density` short-circuits, deterministic streaks per `song_hash`, non-smear ignored, shape guard. |
| `tests/test_block_glitch.py` | `apply_block_glitch`: inactive window, zero-{intensity, displace} short-circuits, deterministic per seed, oversized `block_size_px` clamped, non-block ignored, shape guard. |
| `tests/test_chromatic_aberration.py` | `apply_chromatic_aberration`: inactive window, `shift_px=0`, G fixed, determinism, overlap sum, non-chromatic ignored. |
| `tests/test_scanline_tear.py` | `apply_scanline_tear`: inactive window, `intensity=0`, determinism, wrap/black modes, RGB lockstep on flat colour, clip order, shape guard. |
| `tests/test_compositor_effects_timeline.py` | Compositor helpers: `auto_reactivity_master`, scaled pulses, user BEAM / glitch merge, `_build_frame_effects_context`, `_apply_frame_effects` (shake/smear/block/chromatic/scanline/invert/fade, fade runs last). |

## Related docs

- Model: `docs/technical/effects-timeline.md`
- Compositor wiring: `docs/technical/effects-timeline-compositor.md`
- Renderers overview: `docs/technical/effects-timeline-renderers.md`
