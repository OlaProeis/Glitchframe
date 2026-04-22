#version 330

// VHS-style overlay: horizontal scanlines, per-line tracking jitter, simulated
// RGB channel separation, and occasional noise bands. ``onset_pulse`` spikes
// the glitch amount so kicks chew the image; the palette is used for tint,
// the noise-band highlight color, and the onset flash.
//
// Scanlines, chromatic aberration and hash-noise tracking bands are
// well-known public-domain VHS simulation techniques.

in vec2 v_uv;
out vec4 out_color;

uniform vec2 resolution;
uniform float time;
uniform float beat_phase;
uniform float rms;
uniform float onset_pulse;
uniform float bass_hit;
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

void main() {
    vec2 uv = v_uv;

    // Per-scanline jitter: quantise y to scanlines, pick a random horizontal
    // offset per (line, tick) pair, and only apply it on rare "track-slipped"
    // lines (high hash values).
    float line = floor(v_uv.y * resolution.y);
    float tick = floor(time * 24.0);
    float line_seed = hash21(vec2(line, tick));
    float glitch = 0.012 + 0.035 * onset_pulse + 0.028 * bass_hit;
    float x_shift = (line_seed - 0.5) * glitch * step(0.92, line_seed);
    uv.x = clamp(uv.x + x_shift, 0.0, 1.0);

    // Base tone: vertical palette ramp with a slow horizontal wobble so the
    // frame isn't perfectly striped.
    float ramp_t = uv.y + 0.08 * sin(uv.x * 14.0 + time * 1.6);
    vec3 col = palette_ramp(ramp_t);

    // Chromatic-aberration mimic: offset the R/B channels by sampling the
    // palette ramp at slightly shifted y values (no background required in
    // overlay mode).
    float split = 0.004 + 0.012 * onset_pulse + 0.010 * bass_hit;
    vec3 col_r = palette_ramp(ramp_t + split);
    vec3 col_b = palette_ramp(ramp_t - split);
    col.r = mix(col.r, col_r.r, 0.6);
    col.b = mix(col.b, col_b.b, 0.6);

    // Scanline darkening: fine vertical stripes at the pixel pitch.
    float scan = 0.68 + 0.32 * (0.5 + 0.5 * sin(v_uv.y * resolution.y * 3.14159));
    col *= scan;

    // Occasional bright noise bands (tracking errors). We quantise y into
    // chunky bands that shift with time so the bands feel like rolling dropouts.
    float band = step(0.82, hash21(vec2(0.0, floor(v_uv.y * 64.0 + time * 9.0))));
    col += palette_pick(1) * band * (0.15 + 0.32 * onset_pulse + 0.28 * bass_hit);

    // Beat flash accents a palette slot on every beat.
    col += palette_pick(2) * (1.0 - beat_phase) * 0.10;

    // Grain keeps flat areas from looking CG-clean.
    float grain = hash21(v_uv * resolution + time * 73.0) - 0.5;
    col += vec3(grain) * 0.06;

    float rms_g = rms * rms;
    float alpha = clamp(intensity * (0.55 + 0.18 * rms_g + 0.10 * bass_hit), 0.0, 1.0);
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
