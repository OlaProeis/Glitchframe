#version 330

// Cellular particle field. Each grid cell hosts a single point whose radius
// is driven by one of the eight spectrum bands and by ``onset_pulse``. The
// whole field drifts upward proportional to ``rms`` so louder sections feel
// more alive. Colors interpolate between two palette tones via ``beat_phase``.

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

const float CELL_SIZE = 0.045;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

void main() {
    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 uv = v_uv;
    uv.x *= aspect;
    float rms_g = rms * rms;
    uv.y += time * (0.032 + 0.055 * rms_g + 0.045 * bass_hit);

    vec2 grid = uv / CELL_SIZE;
    vec2 cell = floor(grid);
    vec2 frac = fract(grid);

    float h0 = hash(cell);
    float h1 = hash(cell + vec2(17.0, 3.0));
    float h2 = hash(cell + vec2(91.0, 57.0));

    vec2 center = vec2(0.25 + 0.5 * h1, 0.25 + 0.5 * h2);
    float d = distance(frac, center);

    int band = int(mod(h0 * 8.0, 8.0));
    float e = band_energies[band];

    float radius = 0.12 + e * 0.24 + bass_hit * 0.20 + onset_pulse * 0.10 * h0;
    float core = smoothstep(radius, radius * 0.25, d);
    float halo = smoothstep(radius * 1.8, radius, d) * 0.35;
    float brightness = (core + halo) * (0.38 + 0.55 * e + 0.25 * bass_hit);

    // Each cell picks its pair of palette colors deterministically from its
    // hash so nearby cells aren't identical, and beat_phase smoothly swaps
    // between them.
    int pick_a = int(floor(h0 * float(max(1, u_palette_size))));
    vec3 cold = palette_pick(pick_a);
    vec3 warm = palette_pick(pick_a + 1);
    vec3 col = mix(cold, warm, beat_phase);
    col *= 0.8 + 0.3 * sin(time * 0.7 + h0 * 6.2831);

    float alpha = clamp(brightness, 0.0, 1.0) * intensity;
    vec4 ov = vec4(col * alpha, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
