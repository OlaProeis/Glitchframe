#version 330

// Soft, tasteful lo-fi wash: palette-driven vertical gradient, sparse
// soft-edged bokeh circles, animated film grain, and a warm vignette. The
// audio reactivity is deliberately gentle so this preset still feels like
// background art rather than a visualiser — ``rms`` nudges warmth, beats
// give a tiny breath, and ``onset_pulse`` softly blooms from the accent.
//
// Animated hash-grain, Gaussian-ish bokeh and radial vignettes are
// public-domain image-processing techniques.

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
    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 uv = v_uv;
    vec2 centered = v_uv - 0.5;
    centered.x *= aspect;

    // Base wash: gentle diagonal palette ramp that drifts slowly with time
    // so the "paper" doesn't feel static even when nothing loud is playing.
    float ramp_t = 0.65 * (1.0 - uv.y)
                 + 0.35 * uv.x
                 + 0.04 * sin(time * 0.15 + uv.x * 3.2);
    vec3 base = palette_ramp(ramp_t);

    // Soft bokeh dots on a coarse grid: each cell gets a random center,
    // radius, and brightness, and only cells above a threshold render one.
    vec2 grid = vec2(4.0, 2.4);
    vec2 cell = floor(uv * grid);
    vec2 cell_uv = fract(uv * grid);
    float seed = hash21(cell + 17.0);
    vec2 dot_c = vec2(0.30 + 0.40 * hash21(cell + 3.1),
                      0.30 + 0.40 * hash21(cell + 7.9));
    float dot_r = 0.18 + 0.22 * hash21(cell + 11.3);
    float d = length((cell_uv - dot_c) * vec2(1.0, grid.x / grid.y));
    float bokeh = smoothstep(dot_r, dot_r * 0.35, d) * step(0.58, seed);
    base += palette_pick(0) * bokeh * 0.18;
    base += palette_pick(3) * bokeh * 0.08;

    // Animated film grain: per-pixel hash with a time seed so the grain
    // "moves" every frame like photographic grain.
    float grain = hash21(v_uv * resolution + vec2(time * 91.3, time * 57.7));
    base += vec3(grain - 0.5) * 0.10;

    float rms_g = rms * rms;
    // Warmth follows sustained level gently; kicks add a hair more.
    base += palette_pick(1) * (0.07 * rms_g + 0.05 * bass_hit);

    // Very soft beat breath. Intentionally tiny — lo-fi shouldn't punch.
    base *= 1.0 + (1.0 - beat_phase) * 0.035 + bass_hit * 0.02;

    // Accent: prefer shaped kicks over every onset.
    base += palette_pick(4) * (0.04 * onset_pulse + 0.05 * bass_hit);

    // Warm vignette: darkens corners, leaves center rich.
    float r = length(centered);
    float vig = smoothstep(0.95, 0.28, r);
    base *= 0.55 + 0.55 * vig;

    float alpha = clamp(intensity * (0.55 + 0.14 * rms_g + 0.08 * bass_hit), 0.0, 1.0);
    vec3 rgb_pre = base * alpha;
    vec4 ov = vec4(rgb_pre, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
