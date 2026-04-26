# Waveform peaks downsampling

Shared helper that turns a WAV file into a compact list of min/max columns for browser canvas waveforms (lyrics timeline editor today; effects timeline editor later).

## Module

- **`pipeline/_waveform_peaks.py`** — internal pipeline module (leading underscore).

## API

- **`compute_peaks(wav_path, target_width=DEFAULT_PEAK_WIDTH)`** → **`(peaks, sample_rate, duration_sec)`**
  - **`peaks`:** `list[tuple[float, float]]` — each pair is `(min, max)` normalized to **[-1, 1]** (per-signal peak normalization so quiet tracks still read).
  - Stereo files are mono-mixed before bucketing.
  - Bucket count is capped so each bucket spans at least **`_MIN_SAMPLES_PER_BUCKET`** samples (short audio does not explode to thousands of empty buckets).

## Consumers

- **`pipeline/lyrics_editor.py`** — imports and re-exports **`compute_peaks`** (and uses **`DEFAULT_PEAK_WIDTH`** for **`load_editor_state`**). The HTML/JS payload shape for peaks must stay stable.
- **`pipeline/effects_editor.py`** — uses the same helper for the effects-timeline editor canvas; the JS draws a filled min→max band plus top/bottom outline so individual transients read at default zoom.

## Default peak width

**`DEFAULT_PEAK_WIDTH = 6000`** — picks a bucket size around **30–40 ms** for a typical 3–4 min song so kicks and snares land on distinct columns in the editor canvas. The JSON payload (each pair ≈ 2 floats) stays under **~140 KB**, well inside normal HTML page budgets.

## Tests

- **`tests/test_lyrics_editor.py`** — peak behaviour, a guard that **`pipeline._waveform_peaks.compute_peaks`** matches the public re-export, and a regression lock on **`DEFAULT_PEAK_WIDTH`** staying at high resolution.

## Related

- `docs/technical/lyrics-timeline-editor.md` — editor that embeds peaks in the page payload.
