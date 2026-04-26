#version 330

// Milkdrop-inspired spectral feedback: zoom-rotate-smear of the previous
// frame plus a fresh interference "plasma" layer driven by the mel bands,
// beat/bar phase, RMS, onset envelopes, bass/transients, build tension, and
// drop hold. Optional RGB split on ``transient_hi`` for a futuristic streak.

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
uniform sampler2D u_prev_frame;
uniform float u_has_prev;

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
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);
    float rms_g = rms * rms;
    float i_eff = clamp(intensity, 0.15, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 q = v_uv - 0.5;
    q.x *= aspect;

    // --- Feedback warp: zoom (bass / RMS), twist (mid / bar), spectral curl ---
    float zoom = 0.955
               + 0.035 * rms_g
               + 0.042 * bass_hit
               + 0.030 * t_lo
               - 0.018 * tension
               + 0.012 * hold
               + 0.008 * onset_env;
    float spin = 0.0014 * (1.0 + 2.2 * rms_g)
               + 0.0055 * t_mid
               + 0.0028 * sin(bar_phase * TAU)
               + 0.012 * t_hi * sin(time * 14.0 + length(q) * 18.0);
    float cr = cos(spin);
    float sr = sin(spin);
    q = vec2(cr * q.x - sr * q.y, sr * q.x + cr * q.y) * zoom;

    float spec_w =
          band_energies[0] * 0.010 + band_energies[1] * 0.009
        + band_energies[2] * 0.008 + band_energies[3] * 0.007
        + band_energies[4] * 0.006 + band_energies[5] * 0.005
        + band_energies[6] * 0.004 + band_energies[7] * 0.004;
    spec_w *= 1.0 + 0.6 * beat_phase + 0.35 * bass_hit;
    q += vec2(
        spec_w * sin(time * 3.3 + q.y * 9.0 + beat_phase * TAU * 2.0),
        spec_w * cos(time * 2.9 - q.x * 8.0 + bar_phase * TAU)
    );

    q.x /= aspect;
    vec2 uv_s = q + 0.5;
    uv_s = clamp(uv_s, 0.002, 0.998);

    float split = (0.0018 + 0.004 * t_hi + 0.002 * onset_pulse) / max(aspect, 0.5);
    vec2 off = vec2(split, 0.0);
    vec4 prev_c = texture(u_prev_frame, uv_s);
    vec3 prev_rgb = vec3(
        texture(u_prev_frame, clamp(uv_s + off, 0.002, 0.998)).r,
        prev_c.g,
        texture(u_prev_frame, clamp(uv_s - off, 0.002, 0.998)).b
    );

    float decay = 0.895 + 0.045 * hold - 0.055 * tension + 0.025 * rms_g;
    decay += 0.02 * (1.0 - tension) * t_hi;
    decay = clamp(decay, 0.82, 0.97);
    vec3 trail = prev_rgb * decay * u_has_prev;

    // --- Fresh plasma layer (polar interference + per-bin phase) ---
    vec2 p = (v_uv - 0.5) * vec2(aspect, 1.0);
    float rad = length(p);
    float ang = atan(p.y, p.x);

    float spec_phase =
          band_energies[0] * 5.0 + band_energies[1] * 4.2
        + band_energies[2] * 3.6 + band_energies[3] * 3.0
        + band_energies[4] * 2.4 + band_energies[5] * 2.0
        + band_energies[6] * 1.6 + band_energies[7] * 1.3;

    float wob = sin(beat_phase * TAU * 2.0) * 0.15 + cos(bar_phase * TAU) * 0.1;
    float plasma =
          sin(rad * (22.0 + 8.0 * rms) - time * 2.8 + spec_phase * 2.1 + wob)
        * cos(ang * (6.0 + 2.0 * hold) + time * 1.15 + bar_phase * TAU * 0.5)
        + 0.35 * sin(rad * 44.0 + bass_hit * 5.0 - time * 4.0);
    plasma = plasma * 0.5 + 0.5;

    vec3 c_lo = mix(palette_pick(0), palette_pick(1), plasma);
    vec3 c_hi = mix(palette_pick(2), palette_pick(3), 1.0 - plasma);
    vec3 fresh = mix(c_lo, c_hi, 0.5 + 0.45 * sin(time * 0.37 + rad * 6.0));

    float grain = hash21(floor(v_uv * resolution / 4.0)) - 0.5;
    fresh += vec3(grain * 0.04) * (0.4 + onset_env + 0.5 * t_hi);
    fresh = mix(fresh, palette_pick(4), 0.12 + 0.35 * onset_pulse + 0.2 * onset_env);
    fresh += palette_pick(3) * (0.28 * t_mid + 0.42 * t_hi);
    fresh += vec3(0.55, 0.92, 1.0) * t_hi * 0.22 * (1.0 - 0.6 * tension);
    fresh = mix(fresh, fresh * vec3(0.72, 0.78, 0.95), 0.55 * tension);

    float emit = (0.20 + 0.62 * rms_g + 0.48 * bass_hit + 0.22 * t_lo)
               * (1.0 - 0.38 * tension)
               * (1.0 + 0.55 * hold + 0.35 * onset_pulse);
    emit *= i_eff;

    vec3 acc = trail + fresh * emit;
    acc = acc / (1.0 + acc * 0.32);

    float vign = 1.0 - smoothstep(0.35, 1.15, length(v_uv - 0.5) * 1.85);
    acc *= mix(0.72, 1.0, vign) * (0.88 + 0.12 * (1.0 - tension));

    float a_base = 0.32 + 0.38 * sqrt(clamp(rms, 0.0, 1.0))
                 + 0.28 * bass_hit + 0.18 * hold + 0.12 * t_mid;
    float alpha = clamp(a_base * i_eff, 0.20, 0.90);
    vec3 premul = acc * alpha;

    vec4 ov = vec4(premul, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
