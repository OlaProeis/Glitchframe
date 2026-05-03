"""
Keyframes timeline editor (Gradio ``gr.HTML`` + vanilla JS) and upload staging.

Reuses the waveform / zoom / drag patterns from :mod:`pipeline.effects_editor`
but with a **single row** of segment clips tied to SDXL/upload anchors.
"""

from __future__ import annotations

import html as html_lib
import json
import math
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from pipeline._waveform_peaks import DEFAULT_PEAK_WIDTH, compute_peaks
from pipeline.audio_analyzer import ANALYSIS_JSON_NAME
from pipeline.background_stills import (
    DEFAULT_GEN_HEIGHT,
    DEFAULT_GEN_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_MODEL_ID,
    DEFAULT_WIDTH,
    _atomic_replace_path,
    _background_dir,
    _keyframe_path,
    _load_manifest,
    plan_keyframes,
)
from pipeline.effects_editor import _resolve_wav_for_peaks
from pipeline.keyframes_timeline import (
    KEYFRAMES_TIMELINE_FILENAME,
    KEYFRAMES_TIMELINE_SCHEMA_VERSION,
    UPLOAD_STAGING_DIRNAME,
    KeyframeTimelineEntry,
    KeyframesTimeline,
    load_keyframes_timeline,
    persist_timeline_and_manifest,
    save_keyframes_timeline,
    timeline_identity,
)

_MIN_SEG = 0.05


def gradio_file_url(abs_path: Path | str) -> str:
    """Gradio ``/file=...`` URL for paths under ``allowed_paths`` (matches ``app._editor_audio_url``)."""
    as_posix = str(Path(abs_path).resolve()).replace("\\", "/")
    return "/file=" + urllib.parse.quote(as_posix, safe="/:")


def _load_analysis_mapping(cache: Path) -> dict[str, Any]:
    path = cache / ANALYSIS_JSON_NAME
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _duration_from_analysis(analysis: Mapping[str, Any]) -> float:
    raw = analysis.get("duration_sec")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def load_keyframes_editor_state(
    cache_dir: str | Path,
    *,
    preset_prompt: str,
    preset_id: str,
    target_width: int = DEFAULT_WIDTH,
    target_height: int = DEFAULT_HEIGHT,
    gen_width: int = DEFAULT_GEN_WIDTH,
    gen_height: int = DEFAULT_GEN_HEIGHT,
    model_id: str = DEFAULT_MODEL_ID,
    target_peak_width: int = DEFAULT_PEAK_WIDTH,
) -> dict[str, Any]:
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")
    song_hash = cache.name
    analysis = _load_analysis_mapping(cache)
    duration = _duration_from_analysis(analysis)
    if duration <= 0.0:
        raise FileNotFoundError(
            f"{ANALYSIS_JSON_NAME} missing or invalid in {cache}; analyze audio first."
        )

    wav = _resolve_wav_for_peaks(cache)
    peaks, sample_rate, _wav_dur = compute_peaks(wav, target_peak_width)
    peaks_list = [[float(a), float(b)] for a, b in peaks]

    timeline = load_keyframes_timeline(cache)
    if timeline is not None and timeline.entries:
        ordered = sorted(timeline.entries, key=lambda e: float(e.t_sec))
    else:
        plans = plan_keyframes(analysis, preset_prompt)
        ordered = [
            KeyframeTimelineEntry(
                id=f"kf-{i}",
                t_sec=float(p.t_sec),
                prompt=str(p.prompt),
                source="sdxl",
            )
            for i, p in enumerate(plans)
        ]

    keyframes: list[dict[str, Any]] = []
    for i, e in enumerate(ordered):
        t0 = float(e.t_sec)
        if i + 1 < len(ordered):
            t1 = float(ordered[i + 1].t_sec)
        else:
            t1 = float(duration)
        keyframes.append(
            {
                "id": e.id,
                "t_start": t0,
                "duration_s": max(_MIN_SEG, t1 - t0),
                "prompt": str(e.prompt),
                "source": str(e.source),
            }
        )

    keyframe_previews: list[dict[str, str]] = []
    staging_root = _background_dir(cache) / UPLOAD_STAGING_DIRNAME
    for i, e in enumerate(ordered):
        staged = staging_root / f"{e.id}.png"
        if staged.is_file():
            url = gradio_file_url(staged)
        else:
            png = _keyframe_path(cache, i)
            url = gradio_file_url(png) if png.is_file() else ""
        keyframe_previews.append({"id": str(e.id), "url": url})

    return {
        "schema_version": KEYFRAMES_TIMELINE_SCHEMA_VERSION,
        "song_hash": song_hash,
        "duration": float(duration),
        "sample_rate": int(sample_rate),
        "peaks": peaks_list,
        "keyframes": keyframes,
        "keyframe_previews": keyframe_previews,
        "preset_id": str(preset_id),
        "preset_prompt": str(preset_prompt),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "gen_width": int(gen_width),
        "gen_height": int(gen_height),
        "model_id": str(model_id),
        "selected_target_id": None,
    }


def persist_keyframes_timeline_from_loaded_state(
    cache_dir: str | Path,
    state: Mapping[str, Any],
) -> Path:
    """
    Write ``keyframes_timeline.json`` from a dict returned by
    :func:`load_keyframes_editor_state` so disk exists before **Save timeline**
    (needed for operations that patch the timeline by id, e.g. single-slot SDXL).
    Does not update ``manifest.json``.
    """
    cache = Path(cache_dir)
    entries: list[KeyframeTimelineEntry] = []
    for item in state.get("keyframes") or []:
        if not isinstance(item, dict):
            continue
        entries.append(
            KeyframeTimelineEntry(
                id=str(item["id"]),
                t_sec=float(item["t_start"]),
                prompt=str(item.get("prompt", "")),
                source=(
                    "upload"
                    if str(item.get("source", "sdxl")).lower() == "upload"
                    else "sdxl"
                ),
            )
        )
    if not entries:
        raise ValueError("No keyframes in editor state")
    tl = KeyframesTimeline(
        schema_version=KEYFRAMES_TIMELINE_SCHEMA_VERSION,
        manually_edited=True,
        entries=tuple(entries),
        target_width=int(state.get("target_width", DEFAULT_WIDTH)),
        target_height=int(state.get("target_height", DEFAULT_HEIGHT)),
    )
    save_keyframes_timeline(cache, tl)
    return cache / KEYFRAMES_TIMELINE_FILENAME


def save_keyframes_editor_payload(
    cache_dir: str | Path,
    json_payload: str | bytes,
    *,
    song_hash_from_dir: str | None = None,
    preset_id: str,
    preset_prompt: str,
    model_id: str,
    gen_width: int,
    gen_height: int,
) -> Path:
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")
    canonical = cache.name
    if song_hash_from_dir is not None and song_hash_from_dir != canonical:
        raise ValueError(
            f"song_hash_from_dir {song_hash_from_dir!r} != cache {canonical!r}"
        )
    raw = (
        json_payload.decode("utf-8")
        if isinstance(json_payload, (bytes, bytearray))
        else str(json_payload)
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Keyframes payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Keyframes payload must be a JSON object.")
    sh = data.get("song_hash")
    if sh is not None and str(sh) != canonical:
        raise ValueError(
            f"Payload song_hash {sh!r} does not match cache directory {canonical!r}."
        )
    kfs = data.get("keyframes")
    if not isinstance(kfs, list):
        raise ValueError("keyframes array required")
    entries: list[KeyframeTimelineEntry] = []
    for item in kfs:
        if not isinstance(item, dict):
            continue
        entries.append(
            KeyframeTimelineEntry(
                id=str(item["id"]),
                t_sec=float(item["t_start"]),
                prompt=str(item.get("prompt", "")),
                source=("upload" if str(item.get("source", "sdxl")).lower() == "upload" else "sdxl"),
            )
        )
    if not entries:
        raise ValueError("At least one keyframe is required")
    tw = int(data.get("target_width", DEFAULT_WIDTH))
    th = int(data.get("target_height", DEFAULT_HEIGHT))
    timeline = KeyframesTimeline(
        schema_version=KEYFRAMES_TIMELINE_SCHEMA_VERSION,
        manually_edited=True,
        entries=tuple(entries),
        target_width=tw,
        target_height=th,
    )
    analysis = _load_analysis_mapping(cache)
    if not analysis:
        raise FileNotFoundError(f"{ANALYSIS_JSON_NAME} missing in {cache}")

    prev_tl = load_keyframes_timeline(cache)
    if prev_tl is not None:
        prev_ids: tuple[str, ...] | None = timeline_identity(prev_tl.entries)
    else:
        man = _load_manifest(cache)
        prev_ids = (
            tuple(f"kf-{i}" for i in range(man.num_keyframes)) if man is not None else None
        )

    persist_timeline_and_manifest(
        cache,
        timeline,
        preset_id=preset_id,
        preset_prompt=preset_prompt,
        model_id=model_id,
        gen_width=int(gen_width),
        gen_height=int(gen_height),
        analysis=analysis,
        previous_ids_ordered=prev_ids,
    )
    return cache / "keyframes_timeline.json"


def apply_upload_crop(
    cache_dir: str | Path,
    *,
    entry_id: str,
    image_path: str | Path,
    nx: float,
    ny: float,
    nw: float,
    nh: float,
    out_width: int,
    out_height: int,
) -> Path:
    """Crop normalized rect ``(nx,ny,nw,nh)`` in ``[0,1]``, resize, save to upload staging."""
    from pipeline.keyframes_timeline import UPLOAD_STAGING_DIRNAME
    from pipeline.background_stills import _background_dir

    cache = Path(cache_dir)
    bg = _background_dir(cache)
    staging = bg / UPLOAD_STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    dst = staging / f"{entry_id.strip()}.png"

    nx = float(max(0.0, min(1.0, nx)))
    ny = float(max(0.0, min(1.0, ny)))
    nw = float(max(0.01, min(1.0, nw)))
    nh = float(max(0.01, min(1.0, nh)))
    if nx + nw > 1.0:
        nw = 1.0 - nx
    if ny + nh > 1.0:
        nh = 1.0 - ny

    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    with Image.open(src) as im:
        im_rgb = im.convert("RGB")
        w, h = im_rgb.size
        x0 = int(math.floor(nx * w))
        y0 = int(math.floor(ny * h))
        x1 = int(math.ceil((nx + nw) * w))
        y1 = int(math.ceil((ny + nh) * h))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        cropped = im_rgb.crop((x0, y0, x1, y1)).resize(
            (int(out_width), int(out_height)), Image.LANCZOS
        )
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    cropped.save(str(tmp), format="PNG", optimize=False)
    cropped.close()
    _atomic_replace_path(tmp, dst)
    return dst


def build_crop_canvas_html(
    *,
    image_url: str,
    target_width: int | None = None,
    target_height: int | None = None,
) -> str:
    """Crop UI with drag rectangle; sets ``window._glitchframe_crop_rect`` (normalized).

    Selection is **locked** to ``target_width:target_height`` so the resized output
    is not stretched non-uniformly.
    """
    import uuid

    tw = int(target_width) if target_width and int(target_width) > 0 else 1920
    th = int(target_height) if target_height and int(target_height) > 0 else 1080
    uid = uuid.uuid4().hex[:10]
    safe = html_lib.escape(image_url, quote=True)
    js = rf"""
(function() {{
  window._glitchframe_crop_rect = {{ nx: 0, ny: 0, nw: 1, nh: 1 }};
  const OUT_W = {tw};
  const OUT_H = {th};
  const AR = OUT_W / Math.max(1, OUT_H);
  const img = document.getElementById("kf_img_{uid}");
  const cv = document.getElementById("kf_cv_{uid}");
  if (!img || !cv) return;
  const ctx = cv.getContext("2d");
  let drawing = false;
  let ax = 0, ay = 0, rx = 0, ry = 0;
  let anchorX = 0, anchorY = 0;

  function norm() {{
    const W = cv.width, H = cv.height;
    if (W <= 0 || H <= 0) return;
    let x = Math.min(ax, rx);
    let y = Math.min(ay, ry);
    let rw = Math.max(4, Math.abs(rx - ax));
    let rh = rw / AR;
    rw = Math.min(rw, W);
    rh = rw / AR;
    if (rh > H) {{
      rh = H;
      rw = rh * AR;
    }}
    x = clamp(x, 0, Math.max(0, W - rw));
    y = clamp(y, 0, Math.max(0, H - rh));
    window._glitchframe_crop_rect = {{
      nx: x / W, ny: y / H, nw: rw / W, nh: rh / H
    }};
  }}

  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

  /** Rectangle from fixed anchor to drag point, aspect locked to AR; fitted in canvas. */
  function lockAspectFromDrag(ax0, ay0, px, py, cw, ch) {{
    const dx = px - ax0;
    const dy = py - ay0;
    let adx = Math.max(4, Math.abs(dx));
    let ady = Math.max(4, Math.abs(dy));
    let w, h;
    if (adx / ady > AR) {{
      w = adx;
      h = w / AR;
    }} else {{
      h = ady;
      w = h * AR;
    }}
    let x0 = dx >= 0 ? ax0 : ax0 - w;
    let y0 = dy >= 0 ? ay0 : ay0 - h;
    w = Math.min(w, cw);
    h = w / AR;
    if (h > ch) {{
      h = ch;
      w = h * AR;
    }}
    x0 = clamp(x0, 0, Math.max(0, cw - w));
    y0 = clamp(y0, 0, Math.max(0, ch - h));
    if (x0 + w > cw) x0 = Math.max(0, cw - w);
    if (y0 + h > ch) y0 = Math.max(0, ch - h);
    return {{ x0, y0, w, h }};
  }}

  function centeredAspectRect(cw, ch) {{
    const cwAR = cw / Math.max(1, ch);
    let w, h;
    if (AR >= cwAR) {{
      h = ch;
      w = h * AR;
      if (w > cw) {{ w = cw; h = w / AR; }}
    }} else {{
      w = cw;
      h = w / AR;
      if (h > ch) {{ h = ch; w = h * AR; }}
    }}
    const x0 = (cw - w) / 2;
    const y0 = (ch - h) / 2;
    return {{ x0, y0, w, h }};
  }}

  function paint() {{
    const iw = img.naturalWidth, ih = img.naturalHeight;
    if (!iw || !ih) return;
    let cw = cv.parentElement ? Math.max(280, cv.parentElement.clientWidth - 16) : 800;
    cw = Math.min(cw, 920);
    const scale = Math.min(cw / iw, 520 / ih);
    cv.width = Math.round(iw * scale);
    cv.height = Math.round(ih * scale);
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      Math.min(ax, rx), Math.min(ay, ry),
      Math.abs(rx - ax), Math.abs(ry - ay)
    );
    norm();
  }}

  img.addEventListener("load", function() {{
    paint();
    const r = centeredAspectRect(cv.width, cv.height);
    ax = r.x0; ay = r.y0; rx = r.x0 + r.w; ry = r.y0 + r.h;
    anchorX = ax; anchorY = ay;
    paint();
  }});

  cv.addEventListener("pointerdown", function(ev) {{
    const r = cv.getBoundingClientRect();
    drawing = true;
    const sx = (ev.clientX - r.left) * (cv.width / r.width);
    const sy = (ev.clientY - r.top) * (cv.height / r.height);
    anchorX = clamp(sx, 0, cv.width);
    anchorY = clamp(sy, 0, cv.height);
    ax = anchorX; ay = anchorY;
    rx = anchorX; ry = anchorY;
    try {{ cv.setPointerCapture(ev.pointerId); }} catch (_e) {{}}
    paint();
  }});

  cv.addEventListener("pointermove", function(ev) {{
    if (!drawing) return;
    const r = cv.getBoundingClientRect();
    const sx = (ev.clientX - r.left) * (cv.width / r.width);
    const sy = (ev.clientY - r.top) * (cv.height / r.height);
    const box = lockAspectFromDrag(anchorX, anchorY, sx, sy, cv.width, cv.height);
    ax = box.x0; ay = box.y0; rx = box.x0 + box.w; ry = box.y0 + box.h;
    paint();
  }});

  cv.addEventListener("pointerup", function() {{ drawing = false; norm(); }});
}})();
""".strip()
    code_blob = js.replace("</", "<\\/")
    boot_id = f"kf_boot_{uid}"
    return f"""
<div class="mv-kf-crop-host" style="margin-top:8px;">
  <img id="kf_img_{uid}" crossorigin="anonymous" src="{safe}" alt=""
    style="max-width:100%;display:block"/>
  <div style="font-size:12px;color:#94a3b8;margin:8px 0 6px;">
    Drag on the canvas to resize the crop. The red frame stays locked to
    <b>{tw}×{th}</b> (your video output aspect) so the result is not stretched.
  </div>
  <canvas id="kf_cv_{uid}" class="mv-kf-crop-canvas"></canvas>
</div>
<script type="text/plain" id="{boot_id}">{code_blob}</script>
<img src="x" alt="" style="display:none"
  onerror="this.remove();var b=document.getElementById('{boot_id}');if(b){{try{{(new Function(b.textContent))();}}catch(e){{console.error('kf-crop',e);}}}}">
""".strip()


def mark_keyframe_entry_as_upload(cache_dir: Path | str, entry_id: str) -> None:
    """Set ``source: upload`` for one timeline entry (after staging PNG exists)."""
    cache = Path(cache_dir)
    kt = load_keyframes_timeline(cache)
    if kt is None:
        return
    new_entries = tuple(
        KeyframeTimelineEntry(
            id=e.id,
            t_sec=e.t_sec,
            prompt=e.prompt,
            source="upload" if e.id == entry_id else e.source,
        )
        for e in kt.entries
    )
    save_keyframes_timeline(
        cache,
        KeyframesTimeline(
            schema_version=kt.schema_version,
            manually_edited=True,
            entries=new_entries,
            target_width=kt.target_width,
            target_height=kt.target_height,
        ),
    )


def keyframe_id_choices(cache_dir: Path | str) -> list[str]:
    cache = Path(cache_dir)
    kt = load_keyframes_timeline(cache)
    if kt is not None and kt.entries:
        return [e.id for e in sorted(kt.entries, key=lambda x: float(x.t_sec))]
    man = _load_manifest(cache)
    if man is not None:
        return [f"kf-{i}" for i in range(man.num_keyframes)]
    return []


def keyframe_entry_id_at_index(cache_dir: Path | str, index: int) -> str | None:
    """Gallery / ``keyframe_N.png`` order matches sorted timeline ids."""
    ids = keyframe_id_choices(cache_dir)
    if index < 0 or index >= len(ids):
        return None
    return ids[index]


def keyframe_prompt_for_entry_id(cache_dir: Path | str, entry_id: str) -> str:
    cache = Path(cache_dir)
    kt = load_keyframes_timeline(cache)
    if kt is None:
        return ""
    eid = str(entry_id).strip()
    for e in kt.entries:
        if e.id == eid:
            return str(e.prompt)
    return ""


def list_keyframe_gallery_paths(cache_dir: Path | str) -> list[str]:
    from pipeline.background_stills import _keyframe_path

    cache = Path(cache_dir)
    man = _load_manifest(cache)
    if man is None:
        return []
    out: list[str] = []
    for i in range(man.num_keyframes):
        p = _keyframe_path(cache, i)
        if p.is_file():
            out.append(str(p.resolve()))
    return out


def keyframe_png_abs_path_for_index(cache_dir: Path | str, index: int) -> Path | None:
    """Absolute path to ``keyframe_{index}.png`` if it exists."""
    from pipeline.background_stills import _keyframe_path

    cache = Path(cache_dir)
    man = _load_manifest(cache)
    if man is None or index < 0 or index >= man.num_keyframes:
        return None
    p = _keyframe_path(cache, index)
    return p.resolve() if p.is_file() else None


def generate_sdxl_keyframes_for_cache(
    cache_dir: str | Path,
    *,
    preset_id: str,
    preset_prompt: str,
    width: int,
    height: int,
    force_regenerate_sdxl: bool = False,
    regenerate_indices: set[int] | None = None,
    progress: Callable[..., Any] | None = None,
) -> None:
    """Run SDXL for missing keyframes; optionally regenerate every ``sdxl`` slot or explicit indices."""
    from pipeline import background as bg_mod

    cache = Path(cache_dir)
    regen: set[int] | None = None
    if regenerate_indices is not None:
        regen = set(regenerate_indices)
    elif force_regenerate_sdxl:
        kt = load_keyframes_timeline(cache)
        if kt and kt.entries:
            ordered = sorted(kt.entries, key=lambda e: float(e.t_sec))
            regen = {i for i, e in enumerate(ordered) if e.source == "sdxl"}

    bg = bg_mod.create_background_source(
        bg_mod.MODE_SDXL_STILLS,
        cache,
        preset_id=preset_id,
        preset_prompt=preset_prompt,
        width=int(width),
        height=int(height),
        sdxl_ken_burns=False,
        sdxl_rife_morph=False,
    )
    try:
        bg.ensure_keyframes(
            progress=progress,
            apply_rife=False,
            force_regenerate_indices=regen,
        )
    finally:
        bg.close()


KeyframesEditorCss = """
<style>
  /* Light chrome for Gradio tab; dark “studio” only around waveform + timeline. */
  .mv-kf { font-family: ui-sans-serif, system-ui, sans-serif;
    color: #334155; background: transparent; padding: 0; border-radius: 0; }
  .mv-kf-dark { background: #0b1220; color: #e2e8f0;
    border: 1px solid #1e293b; border-radius: 8px; padding: 10px 12px 12px;
    margin-bottom: 12px; }
  .mv-kf-toolbar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    margin-bottom: 8px; }
  .mv-kf-toolbar button { cursor: pointer; padding: 6px 10px; border-radius: 6px;
    border: 1px solid #334155; background: #1e293b; color: #e2e8f0;}
  .mv-kf-toolbar button:hover { background: #334155; }
  .mv-kf-info { margin-left: auto; color: #94a3b8; font-size: 12px; }
  .mv-kf-body { display: flex; gap: 8px; }
  .mv-kf-labels { width: 110px; flex-shrink: 0; font-size: 12px; color: #94a3b8; }
  .mv-kf-label-head { height: 120px; display: flex; align-items: flex-end;
    padding-bottom: 4px; }
  .mv-kf-label-row { height: 36px; display: flex; align-items: center;
    border-left: 3px solid #a855f7; padding-left: 6px; font-weight: 600;
    color: #e879f9; }
  .mv-kf-scroller { overflow-x: auto; flex: 1; border: 1px solid #1e293b;
    border-radius: 6px; background: #020617; }
  .mv-kf-stage { position: relative; min-height: 156px; }
  .mv-kf-wave { display: block; background: #020617; }
  .mv-kf-rows { position: relative; }
  .mv-kf-row { position: relative; height: 36px; background: rgba(30,41,59,.35);
    border-top: 1px solid #1e293b;}
  .mv-kf-clip { position: absolute; top: 3px; height: 30px; border-radius: 4px;
    box-sizing: border-box; border: 1px solid rgba(0,0,0,.35);
    cursor: grab; display: flex; align-items: center; padding: 0 6px;
    min-width: 24px; overflow: hidden; white-space: nowrap; font-size: 11px;
    color: #f8fafc; font-weight: 600; user-select: none;
    text-shadow: 0 1px 2px rgba(0,0,0,.55); }
  .mv-kf-clip:active { cursor: grabbing; }
  .mv-kf-clip.selected { outline: 2px solid #f8fafc; z-index: 3; }
  .mv-kf-clip.upload { background: linear-gradient(135deg,#34d399,#10b981); }
  .mv-kf-clip.sdxl { background: linear-gradient(135deg,#c084fc,#a855f7); }
  .mv-kf-clip.dragging { opacity: 0.9; z-index: 5; }
  .mv-kf-handle { position: absolute; top: 0; bottom: 0; width: 6px;
    cursor: ew-resize; z-index: 2; }
  .mv-kf-handle.left { left: 0; }
  .mv-kf-handle.right { right: 0; }
  .mv-kf-playhead { position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
    background: #f43f5e; pointer-events: none; z-index: 4; }
  .mv-kf-drag-guide { position: absolute; top: 0; bottom: 0; width: 1px;
    background: #fbbf24; pointer-events: none; z-index: 6; display: none; }
  .mv-kf-audio { width: 100%; margin-top: 10px; }
  .mv-kf-help { font-size: 11px; color: #64748b; margin-top: 10px; line-height: 1.45; }
  .mv-kf-preview { margin-top: 0; padding-top: 0; border-top: none; }
  .mv-kf-preview-head { margin-bottom: 6px; }
  .mv-kf-preview-cap { font-size: 12px; color: #475569; line-height: 1.35; }
  .mv-kf-preview-frame { position: relative; min-height: 120px; max-height: 240px;
    background: #020617; border-radius: 8px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(15,23,42,0.06);
    display: flex; align-items: center; justify-content: center; overflow: hidden; }
  .mv-kf-preview-img { max-width: 100%; max-height: 240px; object-fit: contain;
    display: none; }
  .mv-kf-preview-empty { font-size: 12px; color: #94a3b8; padding: 16px; text-align: center; }
  .mv-kf-clip.playing { box-shadow: 0 0 0 2px #f43f5e inset; z-index: 4; }
  .mv-kf-actions { margin-top: 14px; padding: 0; background: transparent; border-radius: 0;
    border: none; }
  .mv-kf-actions label.mv-kf-act-label {
    display: block; font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 6px; }
  .mv-kf-inline-prompt { width: 100%; box-sizing: border-box; min-height: 52px; border-radius: 8px;
    border: 1px solid #d1d5db; background: #fff; color: #1f2937; font-size: 13px; padding: 8px;
    font-family: ui-sans-serif, system-ui, sans-serif; resize: vertical;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
  .mv-kf-act-btns { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
  .mv-kf-act-btns button {
    cursor: pointer; padding: 8px 12px; border-radius: 8px;
    border: 1px solid #d1d5db; background: #f9fafb; color: #374151; font-size: 12px; }
  .mv-kf-act-btns button:hover { background: #f3f4f6; border-color: #cbd5e1; }
  .mv-kf-target-hint { font-size: 11px; color: #64748b; margin-top: 6px; }
  .mv-kf-crop { margin-top: 14px; padding-top: 12px; border-top: 1px solid #1e293b; }
  .mv-kf-crop-canvas { display: block; max-width: 100%; border: 1px solid #334155;
    cursor: crosshair; background: #020617; }
  .mv-kf-modal-back { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.55);
    z-index: 50; align-items: center; justify-content: center; }
  .mv-kf-modal-back.open { display: flex; }
  .mv-kf-modal { background: #1e293b; border-radius: 8px; padding: 14px;
    min-width: 320px; max-width: 90vw; border: 1px solid #475569; }
  .mv-kf-modal textarea { width: 100%; min-height: 72px; box-sizing: border-box;
    border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0;
    font-size: 13px; padding: 8px; }
  .mv-kf-modal label { font-size: 12px; color: #94a3b8; display: block; margin-top: 8px; }
  .mv-kf-modal select { width: 100%; padding: 6px; margin-top: 4px; border-radius: 6px; }
  .mv-kf-modal-actions { display: flex; gap: 8px; margin-top: 12px; justify-content: flex-end; }
</style>
""".strip()


def build_keyframes_editor_html(
    state: Mapping[str, Any],
    *,
    audio_url: str,
    container_id: str,
    state_js_var: str = "_glitchframe_keyframes_state",
    audio_element_id: str = "mv_kf_audio",
    pixels_per_second: float = 40.0,
) -> str:
    payload = {
        "schema_version": int(state.get("schema_version", KEYFRAMES_TIMELINE_SCHEMA_VERSION)),
        "song_hash": str(state.get("song_hash", "")),
        "duration": float(state.get("duration", 0.0)),
        "sample_rate": int(state.get("sample_rate", 0) or 0),
        "peaks": state.get("peaks") or [],
        "keyframes": list(state.get("keyframes") or []),
        "keyframe_previews": list(state.get("keyframe_previews") or []),
        "target_width": int(state.get("target_width", DEFAULT_WIDTH)),
        "target_height": int(state.get("target_height", DEFAULT_HEIGHT)),
    }
    payload_json = json.dumps(payload)
    song_hash = payload["song_hash"]
    n = len(payload["keyframes"])
    info = html_lib.escape(f"{song_hash[:8] or '—'} · {payload['duration']:.1f}s · {n} stills")

    script = _KF_JS.replace("__MV_CONTAINER_ID__", container_id)
    script = script.replace("__MV_STATE_JS_VAR__", state_js_var)
    script = script.replace("__MV_AUDIO_ELEMENT_ID__", audio_element_id)
    script = script.replace("__MV_PAYLOAD_JSON__", payload_json)
    script = script.replace("__MV_PIXELS_PER_SECOND__", str(pixels_per_second))
    code_blob = script.replace("</", "<\\/")
    code_tag_id = f"mv_kf_code_{container_id}"
    audio_src = html_lib.escape(audio_url, quote=True)

    return f"""{KeyframesEditorCss}
<div class="mv-kf" id="{html_lib.escape(container_id)}">
  <div class="mv-kf-dark">
  <div class="mv-kf-toolbar">
    <button type="button" data-mv-kf-action="play">▶ Play / Pause</button>
    <button type="button" data-mv-kf-action="zoom-in">+</button>
    <button type="button" data-mv-kf-action="zoom-out">−</button>
    <button type="button" data-mv-kf-action="zoom-fit">Fit</button>
    <span class="mv-kf-info">{info}</span>
  </div>
  <div class="mv-kf-body">
    <div class="mv-kf-labels">
      <div class="mv-kf-label-head">waveform</div>
      <div class="mv-kf-label-row">Background stills</div>
    </div>
    <div class="mv-kf-scroller" data-mv-kf-scroller>
      <div class="mv-kf-stage" data-mv-kf-stage>
        <canvas class="mv-kf-wave" data-mv-kf-wave></canvas>
        <div class="mv-kf-rows"><div class="mv-kf-row" data-mv-kf-row></div></div>
        <div class="mv-kf-playhead" data-mv-kf-playhead></div>
        <div class="mv-kf-drag-guide" data-mv-kf-drag-guide></div>
      </div>
    </div>
  </div>
  </div>
  <div class="mv-kf-preview" data-mv-kf-preview>
    <div class="mv-kf-preview-head">
      <span class="mv-kf-preview-cap" data-mv-kf-preview-cap>
        Play or click a clip — preview follows the playhead.
      </span>
    </div>
    <div class="mv-kf-preview-frame">
      <img class="mv-kf-preview-img" data-mv-kf-preview-img alt="" />
      <div class="mv-kf-preview-empty" data-mv-kf-preview-empty>
        Generate SDXL stills (or save uploads) to see keyframe images here.
      </div>
    </div>
  </div>
  <div class="mv-kf-actions">
    <label class="mv-kf-act-label">Prompt for the target clip (SDXL)</label>
    <textarea class="mv-kf-inline-prompt" data-mv-kf-inline-prompt rows="2"
      placeholder="Click a clip on the waveform, edit the prompt, then Regenerate."></textarea>
    <div class="mv-kf-act-btns">
      <button type="button" data-mv-kf-fire="regen">Regenerate (SDXL)</button>
      <button type="button" data-mv-kf-fire="replace">Replace with image…</button>
      <button type="button" data-mv-kf-fire="crop">Crop keyframe</button>
    </div>
    <div class="mv-kf-target-hint" data-mv-kf-target-hint>Target slot: — (click a clip)</div>
  </div>
  <audio class="mv-kf-audio" id="{html_lib.escape(audio_element_id)}"
    src="{audio_src}" controls preload="auto"></audio>
  <div class="mv-kf-help">
    <b>Preview</b> follows the playhead while audio plays; when paused, it shows the <b>selected</b>
    slot (click a clip or use the slot dropdown). <b>Click</b> a clip (no drag) to seek
    and pick the slot for Regenerate / crop. Drag clips to move; drag <b>edges</b> to retime the next boundary.
    <b>Double-click</b> a clip for prompt / source. Use the panel below for image upload + Apply crop.
    <b>Space</b> play/pause; <b>+</b>/<b>−</b> zoom; <b>Esc</b> closes the keyframe dialog.
  </div>
  <div class="mv-kf-modal-back" data-mv-kf-modal-back>
    <div class="mv-kf-modal">
      <strong style="font-size:14px">Keyframe</strong>
      <label>Prompt (SDXL)</label>
      <textarea data-mv-kf-edit-prompt></textarea>
      <label>Source</label>
      <select data-mv-kf-edit-source>
        <option value="sdxl">SDXL generate</option>
        <option value="upload">Upload / crop</option>
      </select>
      <div class="mv-kf-modal-actions">
        <button type="button" data-mv-kf-edit-cancel>Cancel</button>
        <button type="button" data-mv-kf-edit-ok>OK</button>
      </div>
    </div>
  </div>
</div>
<script type="text/plain" id="{code_tag_id}">{code_blob}</script>
<img src="x" alt="" style="display:none" onerror="this.remove();var _c=document.getElementById('{code_tag_id}');if(_c){{try{{(new Function(_c.textContent))();}}catch(_e){{console.error('mv-kf-editor',_e);}}}}">
"""


_KF_JS = r"""
(function () {
  const MIN_GAP = 0.05;
  const container = document.getElementById("__MV_CONTAINER_ID__");
  if (!container) return;
  const state = __MV_PAYLOAD_JSON__;
  window.__MV_STATE_JS_VAR__ = state;
  let pxPerSec = __MV_PIXELS_PER_SECOND__;
  const ROW_H = 36;
  const WAVE_H = 120;

  const scroller = container.querySelector("[data-mv-kf-scroller]");
  const stage = container.querySelector("[data-mv-kf-stage]");
  const wave = container.querySelector("[data-mv-kf-wave]");
  const row = container.querySelector("[data-mv-kf-row]");
  const playhead = container.querySelector("[data-mv-kf-playhead]");
  const guide = container.querySelector("[data-mv-kf-drag-guide]");
  const modalBack = container.querySelector("[data-mv-kf-modal-back]");
  const modalPrompt = container.querySelector("[data-mv-kf-edit-prompt]");
  const modalSource = container.querySelector("[data-mv-kf-edit-source]");
  let editId = null;
  const previewImg = container.querySelector("[data-mv-kf-preview-img]");
  const previewCap = container.querySelector("[data-mv-kf-preview-cap]");
  const previewEmpty = container.querySelector("[data-mv-kf-preview-empty]");
  const inlinePrompt = container.querySelector("[data-mv-kf-inline-prompt]");
  const targetHint = container.querySelector("[data-mv-kf-target-hint]");
  const previewsById = new Map(
    (state.keyframe_previews || []).map((x) => [String(x.id), x.url || ""])
  );

  function audio() {
    return document.getElementById("__MV_AUDIO_ELEMENT_ID__");
  }
  function secondsToPx(t) { return t * pxPerSec; }
  function pxToSeconds(x) { return x / pxPerSec; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function sortKf() {
    state.keyframes.sort((a, b) => a.t_start - b.t_start);
  }
  function recalcDurations() {
    sortKf();
    const dur = state.duration;
    const kf = state.keyframes;
    for (let i = 0; i < kf.length; i++) {
      if (i + 1 < kf.length) {
        kf[i].duration_s = Math.max(MIN_GAP, kf[i + 1].t_start - kf[i].t_start);
      } else {
        kf[i].duration_s = Math.max(MIN_GAP, dur - kf[i].t_start);
      }
    }
  }

  function setStageWidth() {
    const w = Math.max(600, Math.round(state.duration * pxPerSec));
    stage.style.width = w + "px";
    wave.width = w;
    wave.height = WAVE_H;
    row.style.width = w + "px";
    drawWaveform();
    renderKf();
    fillInlinePromptFromSelected();
  }
  function drawWaveform() {
    const ctx = wave.getContext("2d");
    const W = wave.width, H = wave.height, mid = H / 2;
    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, W, H);
    const peaks = state.peaks || [];
    if (!peaks.length) return;
    const top = new Float32Array(W);
    const bot = new Float32Array(W);
    const scale = peaks.length / W;
    for (let x = 0; x < W; x++) {
      const lo = Math.floor(x * scale);
      const hi = Math.max(lo + 1, Math.floor((x + 1) * scale));
      let mn = peaks[lo][0], mx = peaks[lo][1];
      for (let i = lo + 1; i < hi && i < peaks.length; i++) {
        const p = peaks[i];
        if (p[0] < mn) mn = p[0];
        if (p[1] > mx) mx = p[1];
      }
      top[x] = mid - mx * mid;
      bot[x] = mid - mn * mid;
    }
    ctx.fillStyle = "#a855f7";
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, top[0]);
    for (let x = 1; x < W; x++) ctx.lineTo(x, top[x]);
    for (let x = W - 1; x >= 0; x--) ctx.lineTo(x, bot[x]);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  let selected = null;
  function findKf(id) { return state.keyframes.find((k) => k.id === id); }
  function trackSelected(id) {
    selected = id;
    state.selected_target_id = id ? id : null;
  }

  function kfIndexAtTime(t) {
    sortKf();
    const kf = state.keyframes;
    if (!kf.length) return -1;
    let idx = 0;
    const tt = Number(t) || 0;
    for (let i = 0; i < kf.length; i++) {
      if (kf[i].t_start <= tt + 1e-5) idx = i;
      else break;
    }
    return idx;
  }

  function syncSlotDropdown(entryId) {
    const wrap = document.getElementById("mv_kf_slot_dd");
    if (!wrap || !entryId) return;
    const id = String(entryId);
    const sel = wrap.querySelector("select");
    if (sel) {
      const ok = Array.from(sel.options).some((o) => o.value === id);
      if (!ok) return;
      if (sel.value !== id) {
        sel.value = id;
        sel.dispatchEvent(new Event("input", { bubbles: true }));
        sel.dispatchEvent(new Event("change", { bubbles: true }));
      }
      return;
    }
    const inputEl = wrap.querySelector('input:not([type="hidden"])') || wrap.querySelector("input");
    if (!inputEl) return;
    if (inputEl.value !== id) {
      /* Do not focus: off-screen dropdown focus makes the viewport jump. */
      inputEl.value = id;
      inputEl.dispatchEvent(new Event("input", { bubbles: true }));
      inputEl.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function gradioPromptInput() {
    const el = document.getElementById("mv_kf_gr_prompt");
    if (!el) return null;
    if (el.matches && el.matches("textarea, input")) return el;
    return el.querySelector("textarea")
      || el.querySelector("input[type=\"text\"]")
      || el.querySelector("input");
  }
  function syncPromptToGradio() {
    const gr = gradioPromptInput();
    if (!inlinePrompt || !gr) return;
    if (gr.value !== inlinePrompt.value) {
      gr.value = inlinePrompt.value;
      gr.dispatchEvent(new Event("input", { bubbles: true }));
      gr.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }
  function fillInlinePromptFromSelected() {
    if (!inlinePrompt) return;
    const k = selected ? findKf(selected) : null;
    inlinePrompt.value = k ? (k.prompt || "") : "";
    syncPromptToGradio();
  }
  function clickGradioAction(name) {
    syncPromptToGradio();
    const ids = { regen: "mv_kf_btn_regen", replace: "mv_kf_btn_replace", crop: "mv_kf_btn_crop" };
    const el = document.getElementById(ids[name]);
    if (!el) return;
    const target = el.matches && el.matches("button") ? el : el.querySelector("button");
    if (target) target.click();
  }

  function updatePlayheadFollow() {
    const a = audio();
    const t = (a && !isNaN(a.currentTime)) ? a.currentTime : 0;
    const idx = kfIndexAtTime(t);
    sortKf();
    const playing = a && !a.paused && !a.ended;
    // While paused, the playhead often sits at 0s while the first anchor is
    // much later — kfIndexAtTime would stick on index 0 and every slot looked
    // like the first still. Prefer the selected target when not playing so
    // Regenerate / dropdown targeting matches the preview image.
    let k = null;
    if (playing) {
      k = idx >= 0 ? state.keyframes[idx] : null;
    } else if (selected) {
      k = findKf(selected) || null;
    } else {
      k = idx >= 0 ? state.keyframes[idx] : null;
    }
    if (previewCap) {
      if (k) {
        const tag = playing ? "▶ " : "⏸ ";
        previewCap.textContent = tag + (k.source === "upload" ? "▣ " : "◆ ")
          + k.id + " @ " + k.t_start.toFixed(2) + "s — "
          + (k.prompt || "").slice(0, 72) + ((k.prompt && k.prompt.length > 72) ? "…" : "");
      } else {
        previewCap.textContent = "—";
      }
    }
    const url = k ? (previewsById.get(k.id) || "") : "";
    if (previewImg && previewEmpty) {
      if (url) {
        previewImg.style.display = "";
        previewEmpty.style.display = "none";
        const bust = url + (url.indexOf("?") >= 0 ? "&" : "?") + "mvcb=" + Date.now();
        previewImg.src = bust;
      } else {
        previewImg.style.display = "none";
        previewImg.removeAttribute("src");
        previewEmpty.style.display = "";
        previewEmpty.textContent = k
          ? "No PNG on disk for this slot yet — generate stills or apply an upload."
          : "Generate SDXL stills (or save uploads) to see keyframe images here.";
      }
    }
    row.querySelectorAll(".mv-kf-clip").forEach((el) => {
      const id = el.getAttribute("data-kf-id");
      const kk = findKf(id);
      const kidx = kk ? state.keyframes.indexOf(kk) : -1;
      el.classList.toggle("playing", playing && kidx === idx);
    });
    if (targetHint) {
      targetHint.textContent = selected
        ? ("Target slot: " + selected + " — used for Regenerate / Replace / Crop.")
        : "Target slot: — click a clip, or use the dropdown under Upload.";
    }
  }

  function renderKf() {
    recalcDurations();
    row.querySelectorAll(".mv-kf-clip").forEach((n) => n.remove());
    (state.keyframes || []).forEach((k) => {
      const el = document.createElement("div");
      el.className = "mv-kf-clip " + (k.source === "upload" ? "upload" : "sdxl");
      el.setAttribute("data-kf-id", k.id);
      if (selected === k.id) el.classList.add("selected");
      el.style.left = secondsToPx(Math.max(0, k.t_start)) + "px";
      el.style.width = Math.max(8, secondsToPx(k.duration_s)) + "px";
      el.textContent = (k.source === "upload" ? "▣ " : "◆ ") +
        (k.prompt || "").slice(0, 28) + (k.prompt && k.prompt.length > 28 ? "…" : "");
      const lh = document.createElement("div");
      lh.className = "mv-kf-handle left";
      const rh = document.createElement("div");
      rh.className = "mv-kf-handle right";
      el.appendChild(lh);
      el.appendChild(rh);
      el.addEventListener("pointerdown", onKfDown);
      el.addEventListener("dblclick", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        openEdit(k.id);
      });
      row.appendChild(el);
    });
    updatePlayheadFollow();
  }

  /** Toggle .selected without destroying clip nodes (keeps drag + pointer capture valid). */
  function applyKfSelection() {
    row.querySelectorAll(".mv-kf-clip").forEach((node) => {
      const clipId = node.getAttribute("data-kf-id");
      node.classList.toggle("selected", clipId === selected);
    });
  }

  let drag = null;
  function onKfDown(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    const el = ev.currentTarget;
    const id = el.getAttribute("data-kf-id");
    const k = findKf(id);
    if (!k) return;
    trackSelected(id);
    sortKf();
    const idx = state.keyframes.indexOf(k);
    syncSlotDropdown(id);
    applyKfSelection();
    fillInlinePromptFromSelected();
    const handle = ev.target.closest(".mv-kf-handle");
    const mode = handle
      ? (handle.classList.contains("left") ? "left" : "right")
      : "move";
    drag = {
      id, mode, el,
      startX: ev.clientX,
      startY: ev.clientY,
      moved: false,
      clickSeek: !handle && mode === "move",
      origStart: k.t_start,
      origDur: k.duration_s,
      origNextStart: (idx + 1 < state.keyframes.length)
        ? state.keyframes[idx + 1].t_start : null,
    };
    try { el.setPointerCapture(ev.pointerId); } catch (_e) {}
    el.classList.add("dragging");
    document.addEventListener("pointermove", onKfMove);
    document.addEventListener("pointerup", onKfUp, { once: true });
  }
  function onKfMove(ev) {
    if (!drag) return;
    if (Math.abs(ev.clientX - drag.startX) > 4 || Math.abs(ev.clientY - drag.startY) > 4) {
      drag.moved = true;
    }
    const dt = pxToSeconds(ev.clientX - drag.startX);
    const k = findKf(drag.id);
    if (!k) return;
    const dur = state.duration;
    const idx = state.keyframes.indexOf(k);
    if (drag.mode === "move") {
      let newT = drag.origStart + dt;
      newT = clamp(newT, 0, dur - MIN_GAP);
      if (idx > 0) {
        newT = Math.max(newT, state.keyframes[idx - 1].t_start + MIN_GAP);
      }
      if (idx + 1 < state.keyframes.length) {
        const nextT = state.keyframes[idx + 1].t_start;
        newT = Math.min(newT, nextT - MIN_GAP);
      }
      k.t_start = newT;
      recalcDurations();
    } else if (drag.mode === "left") {
      let newT = clamp(drag.origStart + dt, 0, k.t_start + k.duration_s - MIN_GAP);
      if (idx > 0) newT = Math.max(newT, state.keyframes[idx - 1].t_start + MIN_GAP);
      k.t_start = newT;
      recalcDurations();
    } else {
      if (idx + 1 < state.keyframes.length && drag.origNextStart != null) {
        const next = state.keyframes[idx + 1];
        let newNextT = clamp(
          drag.origNextStart + dt,
          k.t_start + MIN_GAP,
          dur,
        );
        if (idx + 2 < state.keyframes.length) {
          newNextT = Math.min(newNextT, state.keyframes[idx + 2].t_start - MIN_GAP);
        }
        next.t_start = newNextT;
      }
      recalcDurations();
    }
    drag.el.style.left = secondsToPx(k.t_start) + "px";
    drag.el.style.width = Math.max(8, secondsToPx(k.duration_s)) + "px";
    if (drag.mode === "right" && idx + 1 < state.keyframes.length) {
      const next = state.keyframes[idx + 1];
      const nextEl = Array.from(row.querySelectorAll(".mv-kf-clip")).find(
        (el) => el.getAttribute("data-kf-id") === next.id
      );
      if (nextEl) {
        nextEl.style.left = secondsToPx(next.t_start) + "px";
        nextEl.style.width = Math.max(8, secondsToPx(next.duration_s)) + "px";
      }
    }
  }
  function onKfUp() {
    const d = drag;
    if (d) {
      d.el.classList.remove("dragging");
      const k = findKf(d.id);
      if (k && d.clickSeek && !d.moved) {
        const a = audio();
        if (a) try { a.currentTime = k.t_start; } catch (_e) {}
        trackSelected(d.id);
      }
      if (k) syncSlotDropdown(k.id);
    }
    drag = null;
    recalcDurations();
    renderKf();
    fillInlinePromptFromSelected();
    document.removeEventListener("pointermove", onKfMove);
  }

  function openEdit(id) {
    const k = findKf(id);
    if (!k) return;
    editId = id;
    modalPrompt.value = k.prompt || "";
    modalSource.value = k.source === "upload" ? "upload" : "sdxl";
    modalBack.classList.add("open");
  }
  function closeEdit() {
    modalBack.classList.remove("open");
    editId = null;
  }
  modalBack.querySelector("[data-mv-kf-edit-cancel]").addEventListener("click", closeEdit);
  modalBack.querySelector("[data-mv-kf-edit-ok]").addEventListener("click", () => {
    const k = findKf(editId);
    if (k) {
      k.prompt = modalPrompt.value.trim();
      k.source = modalSource.value === "upload" ? "upload" : "sdxl";
    }
    if (k && editId === selected && inlinePrompt) {
      inlinePrompt.value = k.prompt || "";
      syncPromptToGradio();
    }
    closeEdit();
    renderKf();
  });

  function tickPlayhead() {
    const a = audio();
    if (a && !isNaN(a.currentTime)) {
      const x = secondsToPx(a.currentTime);
      playhead.style.left = x + "px";
      if (!a.paused && !a.ended) {
        const sl = scroller.scrollLeft, sw = scroller.clientWidth;
        if (x < sl + 40) scroller.scrollLeft = Math.max(0, x - 40);
        else if (x > sl + sw - 40) scroller.scrollLeft = x - sw + 40;
      }
    }
    updatePlayheadFollow();
    requestAnimationFrame(tickPlayhead);
  }

  stage.addEventListener("click", (ev) => {
    if (ev.target.closest(".mv-kf-clip")) return;
    const rect = stage.getBoundingClientRect();
    const t = clamp(pxToSeconds(ev.clientX - rect.left), 0, state.duration);
    const a = audio();
    if (a) try { a.currentTime = t; } catch (_e) {}
    updatePlayheadFollow();
  });

  container.querySelectorAll("[data-mv-kf-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const act = btn.getAttribute("data-mv-kf-action");
      const a = audio();
      if (act === "play" && a) {
        if (a.paused) a.play(); else a.pause();
      } else if (act === "zoom-in") { pxPerSec *= 1.25; setStageWidth(); }
      else if (act === "zoom-out") { pxPerSec /= 1.25; setStageWidth(); }
      else if (act === "zoom-fit") { pxPerSec = __MV_PIXELS_PER_SECOND__; setStageWidth(); }
    });
  });

  if (inlinePrompt) {
    inlinePrompt.addEventListener("input", () => {
      const kk = selected ? findKf(selected) : null;
      if (kk) kk.prompt = inlinePrompt.value;
      syncPromptToGradio();
    });
  }
  container.querySelectorAll("[data-mv-kf-fire]").forEach((b) => {
    b.addEventListener("click", () => {
      clickGradioAction(b.getAttribute("data-mv-kf-fire"));
    });
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.target && /INPUT|TEXTAREA|SELECT/.test(ev.target.tagName)) return;
    if (!container.offsetParent) return;
    if (ev.code === "Space") {
      ev.preventDefault();
      const a = audio();
      if (a) {
        try {
          if (a.paused) void a.play();
          else a.pause();
        } catch (_e) {}
      }
      return;
    }
    if (ev.key === "Escape") {
      if (modalBack && modalBack.classList.contains("open")) {
        ev.preventDefault();
        closeEdit();
      }
      return;
    }
    if (ev.key === "+" || ev.key === "=") {
      pxPerSec *= 1.25;
      setStageWidth();
      return;
    }
    if (ev.key === "-" || ev.key === "_") {
      pxPerSec /= 1.25;
      setStageWidth();
      return;
    }
    if (ev.key === "Delete" || ev.key === "Backspace") {
      if (!selected || document.activeElement === modalPrompt) return;
      if (state.keyframes.length <= 1) return;
      ev.preventDefault();
      state.keyframes = state.keyframes.filter((k) => k.id !== selected);
      trackSelected(null);
      recalcDurations();
      renderKf();
      fillInlinePromptFromSelected();
    }
  });

  (function bindSlotDropdownFollow() {
    const wrap = document.getElementById("mv_kf_slot_dd");
    if (!wrap) return;
    const onPick = () => {
      const sel = wrap.querySelector("select");
      const inputEl = wrap.querySelector('input:not([type="hidden"])') || wrap.querySelector("input");
      const v = sel ? String(sel.value || "") : inputEl ? String(inputEl.value || "") : "";
      if (!v) return;
      trackSelected(v);
      renderKf();
      fillInlinePromptFromSelected();
    };
    wrap.addEventListener("change", onPick, true);
    wrap.addEventListener("input", onPick, true);
  })();

  setStageWidth();
  requestAnimationFrame(tickPlayhead);
})();
"""

__all__ = [
    "apply_upload_crop",
    "build_crop_canvas_html",
    "build_keyframes_editor_html",
    "generate_sdxl_keyframes_for_cache",
    "gradio_file_url",
    "keyframe_entry_id_at_index",
    "keyframe_id_choices",
    "keyframe_png_abs_path_for_index",
    "keyframe_prompt_for_entry_id",
    "list_keyframe_gallery_paths",
    "load_keyframes_editor_state",
    "mark_keyframe_entry_as_upload",
    "persist_keyframes_timeline_from_loaded_state",
    "save_keyframes_editor_payload",
]
