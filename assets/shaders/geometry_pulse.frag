#version 330

// Concentric rings emanating from the center. Ring sharpness pops on each
// ``onset_pulse``, spacing modulates with ``time`` and ``rms``, and ring
// brightness is modulated by the spectrum band at each ring's radius.

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

const float PI = 3.14159265359;
const float RING_COUNT = 7.0;

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= resolution.x / max(resolution.y, 1.0);
    float r = length(uv);

    float band_idx = clamp(r * 8.0, 0.0, 7.0);
    int bi = int(floor(band_idx));
    float e = band_energies[bi];

    float rms_g = rms * rms;
    float ring = abs(sin(r * RING_COUNT * PI - time * (1.05 + rms_g * 0.85)));
    float thickness = 0.020 + 0.030 * bass_hit + 0.022 * onset_pulse + 0.018 * e;
    float mask = smoothstep(thickness + 0.015, thickness * 0.4, ring);

    float radial_fade = smoothstep(1.15, 0.10, r);
    float beat_flash = (1.0 - beat_phase) * 0.45;

    // Each ring picks its base hue from the palette, and overall tone shifts
    // with ``rms`` so quiet sections sit on early palette colors and loud
    // sections push toward later ones.
    vec3 ring_hue = palette_pick(bi);
    vec3 loud_hue = palette_pick(bi + 2);
    vec3 base = mix(ring_hue, loud_hue, rms_g);
    vec3 col = base * (0.55 + 0.45 * e + 0.35 * bass_hit + beat_flash);

    float alpha = mask * radial_fade * (0.32 + 0.45 * bass_hit + 0.28 * onset_pulse)
                  * intensity;
    alpha = clamp(alpha, 0.0, 1.0);
    vec4 ov = vec4(col * alpha, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
