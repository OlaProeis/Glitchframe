#version 330

// Voidcat: dark void, pixel-quantized overlay — a small cat silhouette
// “chases” a bright laser dot along a smooth parametric path. The cat
// follows the same path with a time lag so it trails the target. Kicks and
// transients brighten the laser and add jitter; build/drop signals tint the void.
//
// All motion is analytic in ``time`` / ``beat_phase`` (no feedback buffer).

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
const float PI = 3.14159265359;

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

// --- Path: laser in [0,1]^2; cat uses same function with lagged phase -------
vec2 laser_uv_at(float phase) {
    float j = 0.11 * transient_hi + 0.07 * onset_pulse + 0.05 * onset_env;
    float s = phase;
    vec2 u = vec2(
        0.5 + 0.44 * sin(s * 1.07) * cos(s * 0.31)
            + j * sin(s * 11.0 + bar_phase * TAU),
        0.5 + 0.40 * cos(s * 0.92) * sin(s * 0.27)
            + j * cos(s * 10.0)
    );
    return clamp(u, vec2(0.03), vec2(0.97));
}

// Convert [0,1] UV to aspect-correct centered plane (x wider on landscape)
vec2 uv01_to_plane(vec2 u1) {
    float aspect = resolution.x / max(resolution.y, 1.0);
    return vec2((u1.x - 0.5) * 2.0 * aspect, (u1.y - 0.5) * 2.0);
}

float sd_disk(vec2 p, vec2 c, float r) {
    return length(p - c) - r;
}

float sd_box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Rounded cat silhouette in local space (y is up toward laser in unrotated frame)
float cat_sdf(vec2 pr) {
    // Ears (two small circles)
    float e1 = length(pr - vec2(-0.055, 0.10)) - 0.038;
    float e2 = length(pr - vec2(0.055, 0.10)) - 0.038;
    float ears = min(e1, e2);
    // Head
    float head = length(pr - vec2(0.0, 0.04)) - 0.085;
    // Body
    float body = sd_box(pr - vec2(0.0, -0.07), vec2(0.075, 0.095));
    // Tail hook
    float tail = length(pr - vec2(-0.11, -0.12)) - 0.028;
    tail = min(tail, length(pr - vec2(-0.09, -0.05)) - 0.022);
    float d = min(min(min(ears, head), body), tail);
    // Whisker pad / snout nub
    d = min(d, length(pr - vec2(0.0, -0.02)) - 0.04);
    return d;
}

void main() {
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float rms_g = rms * rms;
    float aspect = resolution.x / max(resolution.y, 1.0);

    // Motion phase: speed reacts to loudness and kicks
    float w = 0.95 + 0.55 * rms_g + 0.45 * bass_hit + 0.25 * t_lo;
    float phase = time * w + beat_phase * 0.35 + 0.18 * sin(time * 0.4);

    vec2 laser01 = laser_uv_at(phase);
    float chase_lag = 0.52 + 0.12 * sin(time * 0.33) + 0.08 * t_mid;
    vec2 cat01 = laser_uv_at(phase - chase_lag);
    // Nudge cat toward current laser on big hits (shortens gap visually)
    cat01 = mix(cat01, laser01, 0.12 * bass_hit + 0.08 * t_lo);

    vec2 L = uv01_to_plane(laser01);
    vec2 C = uv01_to_plane(cat01);
    vec2 p_full = vec2((v_uv.x - 0.5) * 2.0 * aspect, (v_uv.y - 0.5) * 2.0);

    // Pixel quantize (retro grid)
    float g = 56.0 + 16.0 * t_hi;
    vec2 p = floor(p_full * g) / g;

    vec2 to_laser = L - C;
    float dist_lc = length(to_laser);
    float ang = atan(to_laser.y, to_laser.x);
    float ca = cos(-ang + PI * 0.5);
    float sa = sin(-ang + PI * 0.5);
    float stretch = 1.0 + 0.12 * bass_hit + 0.08 * t_mid;
    vec2 pr = p - C;
    pr.x /= stretch;
    pr = vec2(ca * pr.x - sa * pr.y, sa * pr.x + ca * pr.y);
    // Face scale: slightly larger on snare
    pr /= 0.78 + 0.06 * rms;
    float d_cat = cat_sdf(pr);
    float cat_m = smoothstep(0.018, 0.0, d_cat);

    // Eyes (two pixels)
    float eye_l = length(pr - vec2(-0.03, 0.045)) - 0.012;
    float eye_r = length(pr - vec2(0.03, 0.045)) - 0.012;
    float eyes = smoothstep(0.006, 0.0, min(eye_l, eye_r));
    // Blink on some beats
    float blink = smoothstep(0.08, 0.0, abs(beat_phase - 0.5) - 0.42);
    eyes *= 1.0 - blink * 0.85;

    // Laser dot + corona
    float dl = length(p - L);
    float r_dot = 0.028 + 0.014 * t_hi + 0.01 * hold;
    float near_l = 1.0 - smoothstep(0.0, r_dot * 1.25, dl);
    float laser_glow = exp(-dl * 14.0) * (0.55 + 0.45 * t_hi);
    // Cheap beam line from cat toward laser (very soft)
    vec2 e = normalize(L - C + 1e-5);
    vec2 wv = p - C;
    float along = dot(wv, e);
    float perp = abs(dot(wv, vec2(-e.y, e.x)));
    float line_ok = step(0.0, along) * step(along, dist_lc);
    float beam = exp(-perp * 80.0) * line_ok * 0.22 * (0.3 + 0.7 * t_hi);
    beam *= (1.0 - tension * 0.4);

    float n = hash21(floor(v_uv * vec2(220.0, 220.0 * resolution.y / max(resolution.x, 1.0))));

    vec3 c_cat = mix(palette_pick(0), palette_pick(1), 0.55);
    vec3 c_eye = mix(palette_pick(2), vec3(1.0), 0.4);

    vec3 laser_col = mix(palette_pick(3), vec3(1.0, 0.2, 0.35), 0.55);
    laser_col = mix(laser_col, vec3(1.0, 0.9, 0.95), 0.2 + 0.3 * hold);
    float flick = 0.7 + 0.3 * sin(time * 50.0) * t_hi;
    vec3 l_core = laser_col * (0.9 + 0.5 * onset_pulse) * flick;

    // Laser strength (0..1): core + corona + beam
    float u_las = clamp(near_l * 0.95 + laser_glow * 0.48 + beam, 0.0, 1.0);
    vec3 c_body = mix(c_cat, c_eye, eyes);
    c_body += (n - 0.5) * 0.04 * cat_m;
    float u_cat = cat_m;
    // No cat: laser only. With cat: body under laser sheen.
    vec3 c_pix = mix(
        l_core,
        mix(c_body, l_core, u_las),
        step(0.001, u_cat));
    float cover = max(u_cat, u_las);
    float a = cover * (0.28 + 0.32 * max(rms, u_las) + 0.08 * (1.0 - tension));
    a = clamp(a * intensity, 0.0, 1.0);
    vec3 premul = c_pix * a;
    vec4 ov = vec4(premul, a);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 out_rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(out_rgb, 1.0);
    } else {
        out_color = ov;
    }
}
