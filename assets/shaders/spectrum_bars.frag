#version 330

// Reactive 8-band spectrum bars. Each column of the screen maps to one
// ``band_energies`` slot; bar height grows with the band energy, pulses on
// ``onset_pulse``, and its color shifts between two tones across ``beat_phase``.
// Output is premultiplied RGBA so it can be alpha-blended over a background.

in vec2 v_uv;
out vec4 out_color;

uniform vec2 resolution;
uniform float time;
uniform float beat_phase;    // 0..1 between consecutive beats
uniform float rms;           // global loudness envelope
uniform float onset_pulse;   // 0..1 decaying after each onset peak
uniform float bass_hit;      // smoothed kick envelope (matches compositor)
uniform float intensity;     // user intensity slider (0..1)
uniform float band_energies[8];
uniform vec3 u_palette[5];
uniform int u_palette_size;
uniform sampler2D u_background;
uniform float u_comp_background; // 0 = premultiplied RGBA overlay; 1 = blend over u_background

const float BAR_COUNT = 8.0;
const float BAR_GAP = 0.10;  // fraction of each column used as gutter

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

void main() {
    float col_f = v_uv.x * BAR_COUNT;
    int band = int(clamp(floor(col_f), 0.0, BAR_COUNT - 1.0));
    float f = fract(col_f);

    float energy = band_energies[band];
    float rms_g = rms * rms;
    float h = clamp(
        energy * 0.58 + bass_hit * 0.22 + rms_g * 0.12 + onset_pulse * 0.05,
        0.0,
        1.0
    );
    float base_y = 0.08;
    float top_y = base_y + h * 0.78;

    float in_bar_y = step(base_y, v_uv.y) * step(v_uv.y, top_y);
    float in_bar_x = step(BAR_GAP * 0.5, f) * step(f, 1.0 - BAR_GAP * 0.5);
    float bar = in_bar_y * in_bar_x;

    float glow_y = smoothstep(top_y + 0.03, top_y, v_uv.y) *
                   step(base_y, v_uv.y);
    float glow = glow_y * in_bar_x * (0.35 + 0.35 * bass_hit + 0.25 * onset_pulse);

    float pulse = 1.0 - beat_phase;
    // Walk the palette across the 8 bars so each bar gets its own hue, then
    // shift toward the "beat" color on every beat.
    vec3 cool = palette_pick(band);
    vec3 hot  = palette_pick(band + 1);
    vec3 col = mix(cool, hot, pulse) *
               (0.85 + 0.25 * sin(time * 1.7 + float(band)));

    float alpha = clamp(bar + glow * 0.6, 0.0, 1.0) * intensity;
    vec4 ov = vec4(col * alpha, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
