#version 330

// Minimal void background for the voidcat-ASCII compositor: dark wash, light
// grain, palette-tinted vignette, optional edge shimmer from transients. The
// cat lives in the CPU ASCII layer, not here.

in vec2 v_uv;
out vec4 out_color;

uniform vec2 resolution;
uniform float time;
uniform float beat_phase;
uniform float bar_phase;
uniform float rms;
uniform float onset_pulse;
uniform float onset_env;
uniform float bass_hit;
uniform float transient_lo;
uniform float transient_mid;
uniform float transient_hi;
uniform float build_tension;
uniform float drop_hold;
uniform float intensity;
uniform float band_energies[8];
uniform vec3 u_palette[5];
uniform int u_palette_size;
uniform sampler2D u_background;
uniform float u_comp_background;

const float TAU = 6.28318530717958647692;

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

void main() {
    // This pass must stay visible on near-black video backgrounds. Older
    // coefficients with alpha capped at 0.45 and palette * 0.1 made premul
    // ``ov.rgb`` almost zero over black (result ≈ (1-a)*bg).
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);
    float rms_g = rms * rms;
    float i_eff = max(intensity, 0.2);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * vec2(2.0 * aspect, 2.0);
    float d = length(p);
    float vig = 1.0 - smoothstep(0.35, 1.1, d);

    float n = hash21(floor(v_uv * vec2(200.0, 200.0 * resolution.y / max(resolution.x, 1.0))));
    vec3 p0 = palette_pick(0);
    vec3 p1 = palette_pick(1);
    vec3 p3 = palette_pick(3);
    vec3 p4 = palette_pick(4);
    // Brighter base wash so ``col * a`` is non-negligible when bg is (0,0,0)
    vec3 c0 = p0 * (0.38 + 0.35 * rms_g + 0.22 * hold);
    vec3 c1 = mix(p1, p3, 0.35 * t_lo + 0.25 * t_hi);
    vec3 col = c0;
    col += c1 * (0.12 + 0.28 * t_hi) * vig;
    col += (n - 0.5) * 0.06 * (0.5 + onset_env);
    col += p4 * 0.10 * t_hi * (1.0 - vig);
    col = mix(col, col * 0.62, 0.5 * tension);
    // Ambient colour lift: keeps frame from reading as raw black
    col += 0.04 * p1 + 0.03 * p4;

    float a = (0.32 + 0.38 * rms + 0.14 * t_lo + 0.18 * hold) * i_eff;
    a = clamp(a, 0.28, 0.78);
    vec4 ov = vec4(col * a, a);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        out_color = vec4(ov.rgb + bg * (1.0 - ov.a), 1.0);
    } else {
        out_color = ov;
    }
}
