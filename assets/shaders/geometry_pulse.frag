#version 330

// Concentric rings emanating from the center. Ring sharpness pops on each
// ``onset_pulse``, spacing modulates with ``time`` and ``rms``, and ring
// brightness is modulated by the spectrum band at each ring's radius.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_lo   → additional ring-thickness punch on kicks.
//   transient_mid  → snare snap: short ring sharpness spike on every snare.
//   transient_hi   → fine inter-ring shimmer multiplier on hats.
//   build_tension  → tightens ring spacing (compresses the pattern) and
//                    desaturates so the build feels held in.
//   drop_hold      → outer-ring bloom via palette[4] + saturation surge.
//   bar_phase      → rotates a faint angular modulation across each bar
//                    so the rings never look perfectly radial / looped.

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

const float PI = 3.14159265359;
const float TAU = 6.28318530717958647692;
const float RING_COUNT = 7.0;

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

void main() {
    // All reactive signals arrive in [0, 1] by contract; clamp defensively
    // so a dense transient stack never exceeds the mix()/scale budget.
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= resolution.x / max(resolution.y, 1.0);
    float r = length(uv);

    float band_idx = clamp(r * 8.0, 0.0, 7.0);
    int bi = int(floor(band_idx));
    float e = band_energies[bi];

    float rms_g = rms * rms;
    // Ring spacing tightens pre-drop so the pattern compresses; relaxes
    // back as drop_hold releases.
    float ring_count_eff = RING_COUNT * mix(1.0, 1.28, tension)
                         * mix(1.0, 0.92, hold);
    // Faint angular modulation rotated by bar_phase — rings stay
    // concentric overall but never look perfectly radial.
    float angle = atan(uv.y, uv.x);
    float ring_arg = r * ring_count_eff * PI
                   - time * (1.05 + rms_g * 0.85)
                   + 0.25 * sin(angle + bar_phase * TAU);
    float ring = abs(sin(ring_arg));
    float thickness = 0.020 + 0.030 * bass_hit + 0.022 * onset_pulse
                    + 0.018 * e + 0.028 * t_mid + 0.018 * t_lo;
    float mask = smoothstep(thickness + 0.015, thickness * 0.4, ring);

    float radial_fade = smoothstep(1.15, 0.10, r);
    float beat_flash = (1.0 - beat_phase) * 0.45;

    // Each ring picks its base hue from the palette, and overall tone shifts
    // with ``rms`` so quiet sections sit on early palette colors and loud
    // sections push toward later ones.
    vec3 ring_hue = palette_pick(bi);
    vec3 loud_hue = palette_pick(bi + 2);
    vec3 base = mix(ring_hue, loud_hue, rms_g);
    vec3 col = base * (0.55 + 0.45 * e + 0.35 * bass_hit + beat_flash
                       + 0.20 * t_hi);

    // Outer-ring bloom on drops: weight toward the outer radius so the
    // shockwave feels like it's expanding, using palette[4] as the accent.
    float outer_mask = smoothstep(0.35, 0.90, r) * radial_fade;
    col += palette_pick(4) * hold * outer_mask * 0.55;

    // Pre-drop desaturation — tightens the palette into the build.
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.40 * tension);

    float alpha = mask * radial_fade
                * (0.32 + 0.45 * bass_hit + 0.28 * onset_pulse + 0.30 * hold)
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
