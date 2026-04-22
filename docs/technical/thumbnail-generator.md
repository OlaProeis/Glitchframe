# Thumbnail generator

Run-scoped **1920×1080** PNG for uploads, produced beside `output.mp4` under `outputs/<run_id>/thumbnail.png`.

## Frame time

`pipeline.thumbnail.pick_thumbnail_time(analysis)` reads `analysis.json`:

1. If **at least two** `segments` exist: start from **`segments[1]["t_start"]`** (second structural block as a chorus proxy). When **`downbeats`** are present, use the **first downbeat ≥ that start**; otherwise use **`t_start`** alone.
2. Otherwise: **1 s box-smoothed RMS** over `rms.values` at `rms.fps` (fallback: top-level `fps`), then **argmax**; sample time is **`(index + 0.5) / fps`** to match compositor frame centering.

The result is clipped to **`duration_sec`**.

## Compositor integration

- **`pipeline.compositor.render_single_frame`**: renders one **RGB** frame (background → reactive shader → optional kinetic words → optional logo). Thumbnails call it with **`aligned_words=None`** and **`title_text` cleared** so the burned-in artist/title card is not sampled (the Skia overlay below is the only title on the PNG); background + shader + logo still match the video.
- **`pipeline.compositor.render_full_video`**: optional **`thumbnail_line`** (non-empty string, e.g. `Artist — Title`) and **`thumbnail_palette`** (preset `colors` list). After a successful ffmpeg encode, **`pipeline.thumbnail.save_thumbnail_png`** runs and **`CompositorResult.thumbnail_png`** points at the file when written.

## Title overlay

`save_thumbnail_png` draws a single centered line with **Skia** (font from **`CompositorConfig.font_path`**, size scaled down if wider than ~92% of frame width). Fill and shadow use **`thumbnail_palette[0]`** / **`[1]`** when provided, else **`CompositorConfig.base_color`** and **`shadow_color`**. The image is resized to **1920×1080** with Pillow when the compositor resolution differs, then saved as PNG.

## Related

- Kinetic typography (per-word video layer): `docs/technical/kinetic-typography.md`
- Full compositor pipeline: `docs/technical/frame-compositor.md`
