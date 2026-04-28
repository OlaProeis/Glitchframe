#version 330

// Milkdrop-inspired spectral feedback: zoom-rotate-smear of the previous
// frame plus a fresh interference layer (soft plasma + sharp rings/rays).
// Crisp detail + brighter chill sections: higher floors on emit/alpha/fine
// structure, beat/onset gates so rings/beams/grid pulse more often, and a
// second scaled grid + polar moiré layer driven by a shared ``gate``.
// Polish: continuous grain (no screen-pixel blocks), fwidth-softened grid,
// peak whites on hits, gentler global tone-map so highlights can approach 1.0.

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
    float rms_soft = sqrt(clamp(rms, 0.0, 1.0));
    float i_eff = clamp(intensity, 0.15, 1.0);
    // Peaks twice per beat — extra chances for structure to pop without waiting for a drop
    float pulse_beat = pow(sin(beat_phase * TAU), 2.0);
    // Shared 0..1 gate: scales secondary pattern + boosts highlights between drops
    float spec_sum = band_energies[0] + band_energies[1] + band_energies[2]
                   + band_energies[3] + band_energies[4] + band_energies[5]
                   + band_energies[6] + band_energies[7];
    spec_sum = clamp(spec_sum * 0.11, 0.0, 1.0);

    float gate = clamp(
          0.18
        + 0.52 * onset_pulse
        + 0.42 * bass_hit
        + 0.38 * t_mid
        + 0.32 * pulse_beat
        + 0.22 * onset_env
        + 0.32 * t_hi
        + 0.35 * hold
        + 0.14 * rms_soft
        + 0.12 * spec_sum,
        0.0,
        1.0
    );

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 q = v_uv - 0.5;
    q.x *= aspect;

    // --- Feedback warp (slightly milder = less sampled blur per frame) ---
    float zoom = 0.965
               + 0.028 * rms_g
               + 0.034 * bass_hit
               + 0.024 * t_lo
               - 0.015 * tension
               + 0.010 * hold
               + 0.006 * onset_env;
    float spin = 0.0011 * (1.0 + 2.2 * rms_g)
               + 0.0042 * t_mid
               + 0.0024 * sin(bar_phase * TAU)
               + 0.009 * t_hi * sin(time * 14.0 + length(q) * 18.0);
    float cr = cos(spin);
    float sr = sin(spin);
    q = vec2(cr * q.x - sr * q.y, sr * q.x + cr * q.y) * zoom;

    float spec_w =
          band_energies[0] * 0.010 + band_energies[1] * 0.009
        + band_energies[2] * 0.008 + band_energies[3] * 0.007
        + band_energies[4] * 0.006 + band_energies[5] * 0.005
        + band_energies[6] * 0.004 + band_energies[7] * 0.004;
    spec_w *= 0.62 * (1.0 + 0.6 * beat_phase + 0.35 * bass_hit);
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

    float decay = 0.875 + 0.040 * hold - 0.050 * tension + 0.022 * rms_g;
    decay += 0.018 * (1.0 - tension) * t_hi;
    decay = clamp(decay, 0.76, 0.93);
    vec3 trail = prev_rgb * decay * u_has_prev;

    // --- Fresh layer: soft plasma + sharp rings/rays (readable fine structure) ---
    vec2 p = (v_uv - 0.5) * vec2(aspect, 1.0);
    float rad = length(p);
    float ang = atan(p.y, p.x);

    float spec_phase =
          band_energies[0] * 5.0 + band_energies[1] * 4.2
        + band_energies[2] * 3.6 + band_energies[3] * 3.0
        + band_energies[4] * 2.4 + band_energies[5] * 2.0
        + band_energies[6] * 1.6 + band_energies[7] * 1.3;

    float wob = sin(beat_phase * TAU * 2.0) * 0.15 + cos(bar_phase * TAU) * 0.1;
    float plasma_raw =
          sin(rad * (26.0 + 10.0 * rms) - time * 2.9 + spec_phase * 2.1 + wob)
        * cos(ang * (6.0 + 2.0 * hold) + time * 1.15 + bar_phase * TAU * 0.5)
        + 0.32 * sin(rad * 52.0 + bass_hit * 5.0 - time * 4.2)
        + 0.18 * sin(rad * 88.0 - time * 5.5 + spec_phase * 3.0);
    float plasma = plasma_raw * 0.5 + 0.5;

    // Thin bright concentric ridges (high exponent = sharp peaks)
    float ring_wave = sin(rad * (40.0 + 14.0 * rms + 6.0 * bass_hit) - time * 3.4
                    + spec_phase * 1.7 + 0.4 * sin(ang * 3.0));
    float rings = pow(clamp(abs(ring_wave), 0.0, 1.0), 8.0);
    rings += 0.55 * pow(clamp(abs(sin(rad * 78.0 - time * 6.0 + spec_phase * 2.0)), 0.0, 1.0), 12.0);

    // Narrow radial beams (audio-scalloped count)
    float beam_k = 10.0 + 7.0 * beat_phase + 3.0 * rms + 4.0 * t_mid;
    float beams = pow(clamp(abs(cos(ang * beam_k + time * 2.1 + bar_phase * TAU)), 0.0, 1.0), 16.0);
    beams += 0.4 * pow(clamp(abs(cos(ang * (beam_k * 1.37) - time * 1.6)), 0.0, 1.0), 20.0);

    // Brighter floors so structure reads in quiet sections; gate widens peaks
    float fine = rings * (0.30 + 0.50 * rms_g + 0.32 * onset_env + 0.22 * pulse_beat + 0.18 * rms_soft)
                     * (0.65 + 0.35 * gate)
               + beams * (0.24 + 0.42 * t_hi + 0.28 * onset_pulse + 0.20 * pulse_beat + 0.14 * beat_phase)
                     * (0.60 + 0.40 * gate);

    // --- Second pattern: scaled/rotating grid + polar moiré (``gate`` drives scale & strength) ---
    float dens = mix(14.0, 46.0, gate) + 10.0 * rms_soft + 6.0 * pulse_beat;
    float br = bar_phase * TAU * 0.18 + time * 0.07 + 0.4 * onset_pulse;
    vec2 guv = v_uv - 0.5;
    guv = vec2(cos(br) * guv.x - sin(br) * guv.y, sin(br) * guv.x + cos(br) * guv.y) + 0.5;
    vec2 gf = fract(guv * dens) - 0.5;
    float lw = 0.042 - 0.018 * gate + 0.012 * (1.0 - rms_soft);
    float ax = fwidth(gf.x) * 1.25;
    float ay = fwidth(gf.y) * 1.25;
    float gx = 1.0 - smoothstep(0.0, lw + ax, abs(gf.x));
    float gy = 1.0 - smoothstep(0.0, lw + ay, abs(gf.y));
    float grid = pow(max(gx, gy), 4.0);

    float moire_scl = mix(18.0, 52.0, gate) + 8.0 * bass_hit + 6.0 * pulse_beat;
    float moire = pow(
        clamp(abs(sin(rad * moire_scl - time * 2.7 + spec_phase + ang * 2.0)), 0.0, 1.0),
        7.0
    );

    float pattern_b = clamp(grid * (0.35 + 0.65 * gate) + moire * (0.25 + 0.55 * gate), 0.0, 1.0);

    vec3 c_lo = mix(palette_pick(0), palette_pick(1), plasma);
    vec3 c_hi = mix(palette_pick(2), palette_pick(3), 1.0 - plasma);
    vec3 fresh = mix(c_lo, c_hi, 0.5 + 0.45 * sin(time * 0.37 + rad * 8.0));
    // Ambient wash so the frame is never empty black between hits
    fresh += mix(palette_pick(2), palette_pick(1), 0.5) * (0.10 + 0.14 * rms_soft + 0.06 * pulse_beat);

    vec3 edge_col = mix(palette_pick(1), palette_pick(4), 0.5 + 0.5 * sin(time * 0.9 + spec_phase));
    fresh = mix(fresh, fresh + edge_col * 1.15, clamp(fine, 0.0, 1.0));

    vec3 pb_col = mix(palette_pick(3), palette_pick(4), 0.4 + 0.6 * gate);
    fresh += pb_col * pattern_b * (0.22 + 0.62 * gate + 0.15 * onset_env);

    // High-frequency hash in continuous UV — avoids large square blocks from
    // ``floor(v_uv * resolution / …)`` (read as chunky “low res” in encodes).
    vec2 g_uv = v_uv * vec2(1447.0, 1021.0) + vec2(time * 13.7, -time * 9.2);
    float grain = hash21(g_uv) + 0.5 * hash21(g_uv * 1.91 + 17.0) - 0.75;
    fresh += vec3(grain * 0.028) * (0.35 + onset_env + 0.55 * t_hi);
    fresh = mix(fresh, palette_pick(4), 0.12 + 0.35 * onset_pulse + 0.2 * onset_env);
    fresh += palette_pick(3) * (0.28 * t_mid + 0.42 * t_hi);
    fresh += vec3(0.55, 0.92, 1.0) * t_hi * 0.22 * (1.0 - 0.6 * tension);
    fresh = mix(fresh, fresh * vec3(0.72, 0.78, 0.95), 0.32 * tension);

    float emit = (0.36 + 0.48 * rms_g + 0.42 * rms_soft + 0.44 * bass_hit + 0.24 * t_lo)
               * (1.0 - 0.22 * tension)
               * (1.0 + 0.50 * hold + 0.32 * onset_pulse)
               + 0.10 * onset_env
               + 0.07 * pulse_beat
               + 0.12 * pattern_b;
    emit *= i_eff;

    vec3 acc = trail + fresh * emit;
    // Partial tone-map only — leaves more headroom for peak mix toward white
    acc = acc / (1.0 + acc * 0.07);

    float vign = 1.0 - smoothstep(0.35, 1.15, length(v_uv - 0.5) * 1.85);
    acc *= mix(0.84, 1.0, vign) * (0.90 + 0.10 * (1.0 - tension));

    // Near-white cores on strong hits (premultiplied path: push rgb toward alpha)
    float peak = clamp(
          onset_pulse * 0.95
        + bass_hit * 0.85
        + hold * 0.45
        + gate * 0.35
        + t_hi * 0.25
        + fine * 0.2,
        0.0,
        1.0
    );
    float peak2 = peak * peak;
    acc = mix(acc, vec3(1.0), peak2 * 0.62);
    acc += vec3(1.0) * 0.12 * peak * (0.35 + 0.65 * clamp(fine, 0.0, 1.0));

    float a_base = 0.40 + 0.36 * rms_soft + 0.26 * rms_g
                 + 0.26 * bass_hit + 0.16 * hold + 0.11 * t_mid
                 + 0.10 * onset_env + 0.08 * pulse_beat + 0.07 * gate
                 + 0.12 * peak;
    float alpha = clamp(a_base * i_eff, 0.30, 0.94);
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
