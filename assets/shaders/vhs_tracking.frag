#version 330

// VHS-style overlay: horizontal scanlines, per-line tracking jitter, simulated
// RGB channel separation, and occasional noise bands. ``onset_pulse`` spikes
// the glitch amount so kicks chew the image; the palette is used for tint,
// the noise-band highlight color, and the onset flash.
//
// Scanlines, chromatic aberration and hash-noise tracking bands are
// well-known public-domain VHS simulation techniques.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_lo   → extra per-line track-slip on kicks.
//   transient_mid  → chromatic-aberration punch on snares (R/B split).
//   transient_hi   → tracking-burst density: noise bands flare on hats.
//   build_tension  → mutes chroma + tightens scanline contrast so the
//                    build reads as "signal starved", then snaps loose.
//   drop_hold      → chroma bleed + extra noise-band intensity tinted
//                    toward palette[4] that decays with the afterglow.
//   bar_phase      → slow vertical roll of the palette ramp across the
//                    4-beat bar so the tape never feels frozen to time.

in vec2 v_uv;
out vec4 out_color;

uniform vec2 resolution;
uniform float time;
uniform float beat_phase;
uniform float bar_phase;       // 0..1 across the current 4-beat bar
uniform float rms;
uniform float onset_pulse;
uniform float onset_env;       // continuous normalised onset-strength envelope
uniform float bass_hit;
uniform float transient_lo;    // low-band transient (kick / sub)
uniform float transient_mid;   // mid-band transient (snare / body)
uniform float transient_hi;    // high-band transient (hats / air)
uniform float build_tension;   // 0..1 pre-drop smoothstep ramp
uniform float drop_hold;       // post-drop exponential afterglow
uniform float intensity;
uniform float band_energies[8];
uniform vec3 u_palette[5];
uniform int u_palette_size;
uniform sampler2D u_background;
uniform float u_comp_background;

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 palette_ramp(float t) {
    int n = max(1, u_palette_size);
    float f = clamp(t, 0.0, 0.9999) * float(n - 1);
    int i0 = int(floor(f));
    int i1 = int(min(float(n - 1), floor(f) + 1.0));
    return mix(palette_pick(i0), palette_pick(i1), fract(f));
}

const float TAU = 6.28318530717958647692;

void main() {
    // All reactive signals arrive in [0, 1] by contract; clamp defensively.
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    vec2 uv = v_uv;

    // Per-scanline jitter: quantise y to scanlines, pick a random horizontal
    // offset per (line, tick) pair, and only apply it on rare "track-slipped"
    // lines (high hash values). transient_lo widens the slip on kicks.
    float line = floor(v_uv.y * resolution.y);
    float tick = floor(time * 24.0);
    float line_seed = hash21(vec2(line, tick));
    float glitch = 0.012 + 0.035 * onset_pulse + 0.028 * bass_hit
                 + 0.028 * t_lo + 0.020 * hold;
    // Pre-drop: scanline jitter tightens (fewer slipped lines, smaller
    // offsets) so the tape reads "locked" into the build.
    glitch *= mix(1.0, 0.55, tension);
    float x_shift = (line_seed - 0.5) * glitch * step(0.92, line_seed);
    uv.x = clamp(uv.x + x_shift, 0.0, 1.0);

    // Base tone: vertical palette ramp with a slow horizontal wobble so the
    // frame isn't perfectly striped. bar_phase adds a slow vertical roll so
    // the tape never feels frozen.
    float ramp_t = uv.y
                 + 0.08 * sin(uv.x * 14.0 + time * 1.6)
                 + 0.12 * sin(bar_phase * TAU);
    vec3 col = palette_ramp(ramp_t);

    // Chromatic-aberration mimic: offset the R/B channels by sampling the
    // palette ramp at slightly shifted y values (no background required in
    // overlay mode). transient_mid + drop_hold drive the split.
    float split = 0.004 + 0.012 * onset_pulse + 0.010 * bass_hit
                + 0.014 * t_mid + 0.012 * hold;
    vec3 col_r = palette_ramp(ramp_t + split);
    vec3 col_b = palette_ramp(ramp_t - split);
    col.r = mix(col.r, col_r.r, 0.6);
    col.b = mix(col.b, col_b.b, 0.6);

    // Scanline darkening: fine vertical stripes at the pixel pitch.
    float scan = 0.68 + 0.32 * (0.5 + 0.5 * sin(v_uv.y * resolution.y * 3.14159));
    col *= scan;

    // Occasional bright noise bands (tracking errors). Hats push density;
    // drop_hold makes them more likely for a couple of seconds post-drop.
    float band_thr = mix(0.82, 0.72, 0.5 * t_hi + 0.5 * hold);
    float band = step(band_thr, hash21(vec2(0.0, floor(v_uv.y * 64.0 + time * 9.0))));
    col += palette_pick(1) * band
           * (0.15 + 0.32 * onset_pulse + 0.28 * bass_hit + 0.26 * t_hi);

    // Post-drop chroma bleed tinted toward palette[4].
    col += palette_pick(4) * band * (0.32 * hold);

    // Beat flash accents a palette slot on every beat.
    col += palette_pick(2) * (1.0 - beat_phase) * 0.10;

    // Pre-drop chroma mute: pull toward luma so the signal looks starved.
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.50 * tension);

    // Grain keeps flat areas from looking CG-clean.
    float grain = hash21(v_uv * resolution + time * 73.0) - 0.5;
    col += vec3(grain) * 0.06;

    float rms_g = rms * rms;
    // Content-driven alpha: opacity tracks the rendered ``col`` luminance
    // so dark scanlines / blanking regions let the SDXL/AnimateDiff
    // background read through. The previous flat 0.55 base painted the
    // frame at >50 % opacity regardless of content. Audio lift keeps the
    // VHS layer punchy on hits and during drops.
    // See ``docs/technical/reactive-shader-layer.md``.
    float content = clamp(dot(col, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    float audio_lift = 0.22 * rms_g + 0.14 * bass_hit + 0.20 * hold;
    float alpha = clamp((content + audio_lift) * intensity, 0.0, 1.0);
    vec3 rgb_pre = col * alpha;
    vec4 ov = vec4(rgb_pre, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
