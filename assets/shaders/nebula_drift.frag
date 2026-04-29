#version 330

// Layered value-noise nebula that drifts slowly across the frame. Bass
// energies inflate cloud density so low-frequency hits make the nebula
// breathe; ``onset_pulse`` blooms a bright highlight from the last palette
// slot. A sparse high-frequency star layer twinkles with ``time``.
//
// All techniques (value noise, FBM, smoothstep-interpolated hash) are
// textbook public-domain GLSL patterns — no third-party shader code.

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

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
        v += amp * vnoise(p);
        p = p * 2.02 + vec2(7.3, 11.7);
        amp *= 0.5;
    }
    return v;
}

void main() {
    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 uv = v_uv;
    uv.x *= aspect;

    // Drift: mostly time + gentle RMS curve; kicks add a separate breath.
    float rms_g = rms * rms;
    vec2 drift = vec2(time * (0.014 + 0.020 * rms_g + 0.018 * bass_hit),
                     -time * (0.016 + 0.018 * rms_g + 0.016 * bass_hit));

    float cloud = fbm(uv * 2.4 + drift);
    cloud += 0.35 * fbm(uv * 5.7 - drift * 1.7 + vec2(13.0, -9.0));
    cloud = clamp(cloud, 0.0, 1.4);

    // Low-end: prefer shaped kick envelope over raw mel bins (less shimmer).
    float bass = bass_hit * 0.65
               + (band_energies[0] + band_energies[1]) * 0.12;
    cloud += 0.16 * bass;

    // Sparse stars: threshold on a high-frequency hash, modulated by a
    // per-star twinkle.
    float star_seed = hash21(floor(uv * 620.0 + 3.1));
    float star = step(0.9965, star_seed);
    float twinkle = 0.4 + 0.6 * sin(time * 3.0 + star_seed * 31.0);
    float stars = star * twinkle;

    // Ramp through four palette picks with smooth thresholds so tone
    // transitions are gradual rather than striped.
    float t = clamp(cloud * 0.9, 0.0, 1.0);
    vec3 c0 = palette_pick(0);
    vec3 c1 = palette_pick(1);
    vec3 c2 = palette_pick(2);
    vec3 c3 = palette_pick(3);
    vec3 low = mix(c0, c1, smoothstep(0.0, 0.55, t));
    vec3 high = mix(c2, c3, smoothstep(0.45, 1.0, t));
    vec3 neb = mix(low, high, smoothstep(0.35, 0.75, t));

    // Accent bloom: kicks more than raw onsets (fewer sparkles on hats).
    neb += palette_pick(4) * (0.14 * onset_pulse + 0.18 * bass_hit);

    // Stars are near-white highlights on top of the nebula.
    neb = mix(neb, vec3(1.0), stars * 0.85);

    // Radial vignette so the nebula feels focal, not wallpaper.
    vec2 centered = v_uv - 0.5;
    centered.x *= aspect;
    float r = length(centered);
    float vignette = smoothstep(1.05, 0.18, r);

    // Content-driven alpha: lean on ``cloud`` so dark voids of space let
    // the SDXL/AnimateDiff background read through. (Previously
    // ``0.50 + 0.55 * cloud`` painted the whole frame at >=50 % opacity.)
    // See ``docs/technical/reactive-shader-layer.md``.
    float alpha = clamp(cloud * vignette * intensity, 0.0, 1.0);
    vec3 col = neb * alpha;
    vec4 ov = vec4(col, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
