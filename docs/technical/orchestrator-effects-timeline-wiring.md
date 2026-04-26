# Orchestrator → effects timeline wiring

End-to-end preview and full render load the per-song effects schedule from the cache and pass it (plus an effective reactivity master) into the compositor. The Gradio **Effects timeline** tab still owns editing `cache/<hash>/effects_timeline.json`; this path is the read side for `orchestrate_preview_10s` and `orchestrate_full_render`.

## `OrchestratorInputs`

- **`effects_timeline_enabled`** (default `true`) — when `false`, the compositor receives `effects_timeline=None` and `auto_reactivity_master=1.0` (legacy behaviour, no user JSON on that path).
- **`auto_reactivity_master`** (default `1.0`) — per-run trim, combined with the value stored in the loaded timeline file (see below).

## Loading and `CompositorConfig`

After `orchestrate_analysis` sets `state.cache_dir`, `_effects_compositor_config` calls `pipeline.effects_timeline.load`. Missing JSON yields an empty default `EffectsTimeline` (master `1.0` on the object). The function returns a tuple for `CompositorConfig`:

- **`effects_timeline`** — the loaded model (or `None` if disabled).
- **`auto_reactivity_master`** — **effective** scalar: `max(0, OrchestratorInputs.auto_reactivity_master) * EffectsTimeline.auto_reactivity_master` from disk. The file’s field is what the editor saves; the orchestrator field scales that for the current render. The compositor’s `_auto_reactivity_master` reads only `CompositorConfig.auto_reactivity_master`, not a second copy from the timeline object.

## Tests and related docs

- `tests/test_orchestrator_effects_timeline.py` — helper behaviour, disabled path, on-disk master × input, non-positive input trim.
- `docs/technical/pipeline-orchestrator.md` — public API and progress notes including this section.
- `docs/technical/effects-timeline-compositor.md` — what the compositor does with `CompositorConfig` once the orchestrator has filled it.
