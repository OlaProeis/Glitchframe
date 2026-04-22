# Logo rim lights — traveling-wave field (tasks 25–28 overview)

Builds on `compute_logo_rim_prep` (see [`logo-rim-lights-prep.md`](./logo-rim-lights-prep.md)) to produce a
per-frame **premultiplied RGBA patch** that drops into the existing logo stack
via `pipeline.logo_composite._blend_premult_rgba_patch` (same contract as the
neon glow patch). This page summarizes **behaviour, controls, and contracts**;
deep dives live in the linked rim docs.

## Behaviour at a glance

| Concern | Behaviour |
|--------|------------|
| **Mask / stroke fallback** | `prep.use_line_features` gates `prep.line_mask`. When confidence is low (e.g. solid bright fills), `line_mask` is cleared and only halo + inward bleed drive the field — no silent stroke lighting. `line_boost` is a no-op in that mode. |
| **Multi-colour** | `rim_color_layers` 2–3 adds HSV-spread tints, per-layer phase offsets, optional `hue_drift_per_sec`, and `song_hash`-seeded hue nudge. Halo-only logos cap at **two** tints even if three layers are requested. See [`logo-rim-lights-color.md`](./logo-rim-lights-color.md). |
| **Beat / envelope modulation** | Optional `RimAudioModulation` scales glow gain, adds angular phase offset, and scales effective inward bleed (bass). Compositor stepper: [`logo-rim-audio-modulation.md`](./logo-rim-audio-modulation.md). |
| **Determinism** | Same `(prep, t, config, audio_mod)` → same bytes; no RNG. `song_hash` stabilizes palette nudges across runs. |
| **A/B vs classic neon** | Rim is a separate premult patch blended in compositor order; use `LogoGlowMode` / `glow_amount` to compare rim-only, classic-only, or layered looks. See [`logo-rim-compositing.md`](./logo-rim-compositing.md). Gradio → orchestrator mapping: [`logo-rim-branding-ui.md`](./logo-rim-branding-ui.md). |

## Module & API

- **Module:** `pipeline/logo_rim_lights.py`
- **`RimLightConfig`** — frozen dataclass of geometry / phase / colour params:
  - `pad_px` (default `24`) — border around the logo so halo + blur fit.
  - `rim_rgb` `(R, G, B)` — emissive colour (0--255).
  - `intensity`, `opacity_pct` — overall gain / final opacity.
  - `halo_spread_px`, `halo_boost` — outer radial falloff off the alpha edge.
  - `inward_mix` in `[0, 1]`, `inward_depth_px` — how strongly and how far the
    light bleeds into the shape from the edge (reads as light moving into white
    strokes when enabled).
  - `line_boost` — weight of `prep.line_mask` (zero automatically when
    `prep.use_line_features` is False).
  - `blur_px` — Gaussian blur on the radial field, applied before angular
    modulation so the rotating wave stays crisp.
  - `waves`, `wave_sharpness`, `phase_hz`, `phase_offset`, `wave_floor` —
    traveling angular wave `cos(waves*theta + 2*pi*phase_hz*t + phase_offset)`
    around `prep.centroid_xy`; `wave_sharpness >= 1` pinches the lobes, and
    `wave_floor` keeps a baseline glow between peaks.
  - Task **26** — `rim_color_layers` (1--3), `color_spread_rad`, `layer_phase_offsets`,
    `hue_drift_per_sec`, `song_hash`, `flicker_amount`; see
    [logo-rim-lights-color.md](./logo-rim-lights-color.md).
- **`compute_logo_rim_light_patch(prep, *, t, config=RimLightConfig())`** →
  `(patch, pad)` where `patch` is `uint8 (H + 2*pad, W + 2*pad, 4)` premultiplied
  RGBA. Blend with `_blend_premult_rgba_patch(dst, patch, x0 - pad, y0 - pad)`.

## Algorithm

1. Zero-pad `prep.alpha_f` and `prep.line_mask` by `pad_px`.
2. Binarise the alpha (`> 0.5`) and compute two distance transforms:
   - `dt_out` — distance from each outside pixel to the alpha edge (outer halo
     falloff).
   - `dt_in` — distance from each inside pixel to the alpha edge (inward bleed
     depth).
3. Build three contributions, each in `[0, 1]`:
   - `halo = smoothstep(1 - dt_out / halo_spread_px) * halo_boost`.
   - `inward = smoothstep(1 - dt_in / inward_depth_px) * alpha_pad * inward_mix`.
   - `line = line_mask_pad * line_boost`.
4. Sum to a **radial base**, then Gaussian-blur by `blur_px` (single L-channel
   PIL pass, rescaled to preserve dynamic range).
5. Multiply by the angular lobe
   `(0.5 + 0.5*cos(waves*theta + phi(t)))^wave_sharpness`, rescaled by
   `wave_floor` to keep an optional baseline rim.
6. Final emissive is clamped to `[0, 1]` and premultiplied:
   `a = emissive * intensity * (opacity_pct/100)`; `rgb = rim_rgb * a`.
7. Fully transparent logos / zero intensity return an all-zero patch of the
   correct `(H + 2p, W + 2p, 4)` shape so downstream blends are a no-op.

## Contract & edge cases

- **Premultiplied output:** for each pixel, `R, G, B <= alpha` (tested).
- **Halo-only fallback:** when `prep.use_line_features` is False (e.g. solid
  fills), the line term is zero automatically; `halo` + `inward` still drive a
  visible rim.
- **Stateless / deterministic:** identical `(prep, t, config)` → identical
  bytes.
- **No NaNs:** all paths go through `np.clip` before the uint8 cast.

## Tests

`tests/test_logo_rim_lights.py::TestRimLightPatch`:

- Shape + dtype + premultiplied invariant.
- Phase travels: patches at `t=0` vs `t=1` differ when `phase_hz > 0`; identical
  when `phase_hz == 0`.
- `phase_offset` shifts the static angular pattern when `phase_hz == 0`.
- `line_boost` is ignored when `use_line_features` is False; with stroke prep
  and halo/inward disabled, raising `line_boost` turns a zero patch into visible
  line-driven emission.
- `inward_mix = 0` produces zero interior alpha (with halo/line/angular off);
  `inward_mix = 1` raises interior alpha measurably.
- Empty alpha returns a zero patch of the right shape.
- Opacity / intensity scale alpha; `opacity_pct = 0` → fully transparent.
- Determinism across repeated calls; same patch with identical `audio_mod`.
- Extreme `RimAudioModulation` scalers clamp inside the patch path (finite output).
- Solid-fill logos (no line features) still produce a visible halo.

Task **26** multicolour, **27** audio, and **28** compositor wiring are also
covered in `tests/test_logo_rim_lights.py`, `tests/test_logo_composite_rim.py`,
and `tests/test_orchestrator_logo_rim_inputs.py`.

## Integration

`compute_logo_rim_light_patch` is invoked from
`pipeline.logo_composite.build_rim_light_premult_patch` / `composite_logo_onto_frame`
when the compositor enables rim lighting; see [`logo-rim-compositing.md`](logo-rim-compositing.md).
Multi-colour and audio-driven modulation build on the field produced here.
