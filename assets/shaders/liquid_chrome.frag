#version 330

// Two-step domain-warped FBM producing iridescent liquid-metal patterns
// remapped onto the preset palette. ``onset_pulse`` emits a radial ripple
// from the center, and low-frequency bands bias the overall brightness so
// the "liquid" feels louder on bass hits.
//
// Domain-warping (f(p + f(p + f(p)))) is a public-domain technique
// popularised by Inigo Quilez's noise articles; the GLSL here is a fresh
// implementation, not a port.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_lo   → kick warp: briefly boosts the domain-warp flow so
//                    the sheet ripples harder on every low-band hit.
//   transient_mid  → snare highlight: short mid-slot tint on peaks.
//   transient_hi   → hat shimmer: high-band specular punch on the warped
//                    noise tips.
//   build_tension  → cools the palette (pulls toward early slots) and
//                    slows the flow so the build feels viscous.
//   drop_hold      → post-drop palette[4] bloom + brightness lift that
//                    decays with the afterglow.
//   bar_phase      → rotates the flow basis once per 4-beat bar so the
//                    liquid doesn't track raw ``time``.

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
    // All reactive signals arrive in [0, 1] by contract; clamp defensively
    // so a dense transient stack never exceeds the scale budget below.
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 uv = v_uv;
    uv.x *= aspect;

    // Flow direction changes slowly so loops don't feel mechanical.
    // build_tension slows the flow pre-drop; transient_lo punches it on
    // kicks; drop_hold adds a small release boost.
    float rms_g = rms * rms;
    float flow_speed = (0.032 + 0.028 * rms_g + 0.024 * bass_hit + 0.038 * t_lo)
                     * mix(1.0, 0.55, tension)
                     + 0.020 * hold;
    float flow = time * flow_speed;

    // Bar-rotated flow basis so the liquid never tracks raw time.
    float bar_angle = bar_phase * TAU;
    vec2 bar_drift = vec2(cos(bar_angle), sin(bar_angle)) * 0.45;

    vec2 q = vec2(
        fbm(uv + vec2(0.0, flow) + bar_drift),
        fbm(uv + vec2(5.2, flow * 1.3 + 1.3) - bar_drift)
    );
    vec2 r = vec2(
        fbm(uv + 3.8 * q + vec2(1.7, 9.2 + flow * 0.6)),
        fbm(uv + 3.8 * q + vec2(8.3, 2.8 - flow * 0.5))
    );
    float v = fbm(uv + 4.0 * r);

    vec3 col = palette_ramp(v * 1.15);

    // Cold-slot pull during pre-drop builds: blend toward an early-slot
    // ramp so the palette narrows and cools.
    vec3 cold = palette_ramp(v * 0.55);
    col = mix(col, cold, 0.45 * tension);

    // Onset ripple: concentric wave that fades off-center.
    vec2 c = v_uv - 0.5;
    c.x *= aspect;
    float rad = length(c);
    float ripple = 0.5 + 0.5 * sin(rad * 32.0 - time * 4.2);
    float ripple_mask = (0.55 * onset_pulse + 0.45 * bass_hit + 0.35 * t_lo)
                        * smoothstep(0.52, 0.08, rad);
    col += palette_pick(3) * ripple * ripple_mask * 0.32;

    // Hat-driven specular punch on noise tips (where v is bright).
    col += palette_pick(3) * (0.22 * t_hi) * smoothstep(0.55, 0.95, v);

    // Snare mid-slot highlight — short, layered on top of the ramp.
    col += palette_pick(2) * (0.12 * t_mid);

    float bass = bass_hit * 0.55 + (band_energies[0] + band_energies[1]) * 0.2;
    col *= 0.78 + 0.38 * bass + 0.10 * rms_g;
    col += palette_pick(2) * (1.0 - beat_phase) * 0.08;

    // Post-drop bloom via palette[4]; weighted by the noise intensity so
    // the bloom sits on the highlights rather than washing the frame.
    col += palette_pick(4) * (0.32 * hold) * smoothstep(0.25, 0.85, v);

    // Pre-drop desaturation complements the cold-slot pull above.
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.30 * tension);

    float alpha = clamp(0.55 + 0.40 * v, 0.0, 1.0)
                * intensity
                * (1.0 + 0.20 * hold);
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
