#version 330

// Two-step domain-warped FBM producing iridescent liquid-metal patterns
// remapped onto the preset palette. ``onset_pulse`` emits a radial ripple
// from the center, and low-frequency bands bias the overall brightness so
// the "liquid" feels louder on bass hits.
//
// Domain-warping (f(p + f(p + f(p)))) is a public-domain technique
// popularised by Inigo Quilez's noise articles; the GLSL here is a fresh
// implementation, not a port.

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
        p = p * 2.03 + vec2(5.7, 9.1);
        amp *= 0.5;
    }
    return v;
}

// Palette-position lookup with linear interpolation between adjacent picks
// so the warped-noise scalar produces a continuous gradient.
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
    uv.x *= aspect;

    // Flow direction changes slowly so loops don't feel mechanical.
    float rms_g = rms * rms;
    float flow = time * (0.032 + 0.028 * rms_g + 0.024 * bass_hit);

    vec2 q = vec2(
        fbm(uv + vec2(0.0, flow)),
        fbm(uv + vec2(5.2, flow * 1.3 + 1.3))
    );
    vec2 r = vec2(
        fbm(uv + 3.8 * q + vec2(1.7, 9.2 + flow * 0.6)),
        fbm(uv + 3.8 * q + vec2(8.3, 2.8 - flow * 0.5))
    );
    float v = fbm(uv + 4.0 * r);

    vec3 col = palette_ramp(v * 1.15);

    // Onset ripple: concentric wave that fades off-center.
    vec2 c = v_uv - 0.5;
    c.x *= aspect;
    float rad = length(c);
    float ripple = 0.5 + 0.5 * sin(rad * 32.0 - time * 4.2);
    float ripple_mask = (0.55 * onset_pulse + 0.45 * bass_hit)
                        * smoothstep(0.52, 0.08, rad);
    col += palette_pick(3) * ripple * ripple_mask * 0.32;

    float bass = bass_hit * 0.55 + (band_energies[0] + band_energies[1]) * 0.2;
    col *= 0.78 + 0.38 * bass + 0.10 * rms_g;
    col += palette_pick(2) * (1.0 - beat_phase) * 0.08;

    float alpha = clamp(0.55 + 0.40 * v, 0.0, 1.0) * intensity;
    vec3 rgb_pre = col * alpha;
    vec4 ov = vec4(rgb_pre, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
