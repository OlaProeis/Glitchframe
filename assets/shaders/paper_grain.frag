#version 330

// Soft, tasteful lo-fi wash: palette-driven vertical gradient, sparse
// soft-edged bokeh circles, animated film grain, and a warm vignette. The
// audio reactivity is deliberately gentle so this preset still feels like
// background art rather than a visualiser — ``rms`` nudges warmth, beats
// give a tiny breath, and ``onset_pulse`` softly blooms from the accent.
//
// Animated hash-grain, Gaussian-ish bokeh and radial vignettes are
// public-domain image-processing techniques.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``). Lo-fi preset → everything
// stays **tiny**; we never punch, we just gently lean:
//   transient_hi   → +10% grain amplitude on hats (barely perceptible).
//   transient_mid  → soft bokeh lift on snares via palette[2].
//   build_tension  → desaturates + tightens the vignette a touch.
//   drop_hold      → warm palette[4] wash that fades with the afterglow.
//   bar_phase      → slow drift of the bokeh centers across each bar so
//                    the "paper" doesn't feel frozen even without onsets.

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
    // Lo-fi aesthetic → we deliberately use *small* weights below.
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

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
    // bar_phase drifts every center so the dots wander across each bar.
    vec2 bar_drift = 0.04 * vec2(sin(bar_phase * TAU), cos(bar_phase * TAU));
    vec2 grid = vec2(4.0, 2.4);
    vec2 cell = floor(uv * grid);
    vec2 cell_uv = fract(uv * grid);
    float seed = hash21(cell + 17.0);
    vec2 dot_c = vec2(0.30 + 0.40 * hash21(cell + 3.1),
                      0.30 + 0.40 * hash21(cell + 7.9))
               + bar_drift;
    float dot_r = 0.18 + 0.22 * hash21(cell + 11.3);
    float d = length((cell_uv - dot_c) * vec2(1.0, grid.x / grid.y));
    float bokeh = smoothstep(dot_r, dot_r * 0.35, d) * step(0.58, seed);
    base += palette_pick(0) * bokeh * 0.18;
    base += palette_pick(3) * bokeh * 0.08;
    // Tiny snare lift on the mid-slot through the bokeh.
    base += palette_pick(2) * bokeh * (0.05 * t_mid);

    // Animated film grain: per-pixel hash with a time seed so the grain
    // "moves" every frame like photographic grain. Hats nudge amplitude.
    float grain = hash21(v_uv * resolution + vec2(time * 91.3, time * 57.7));
    base += vec3(grain - 0.5) * (0.10 + 0.03 * t_hi);

    float rms_g = rms * rms;
    // Warmth follows sustained level gently; kicks add a hair more.
    base += palette_pick(1) * (0.07 * rms_g + 0.05 * bass_hit);

    // Very soft beat breath. Intentionally tiny — lo-fi shouldn't punch.
    base *= 1.0 + (1.0 - beat_phase) * 0.035 + bass_hit * 0.02;

    // Accent: prefer shaped kicks over every onset, plus a small post-drop
    // warm wash tinted toward palette[4].
    base += palette_pick(4) * (0.04 * onset_pulse + 0.05 * bass_hit + 0.08 * hold);

    // Pre-drop: desaturate slightly so the paper reads "faded".
    float luma = dot(base, vec3(0.2126, 0.7152, 0.0722));
    base = mix(base, vec3(luma), 0.22 * tension);

    // Warm vignette: darkens corners, leaves center rich. Tightens a
    // touch during the build so focus pulls inward.
    float r = length(centered);
    float vig_outer = mix(0.95, 0.82, tension);
    float vig = smoothstep(vig_outer, 0.28, r);
    base *= 0.55 + 0.55 * vig;

    float alpha = clamp(intensity
                        * (0.55 + 0.14 * rms_g + 0.08 * bass_hit + 0.08 * hold),
                        0.0, 1.0);
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
