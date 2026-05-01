#version 330

// Reactive 8-band spectrum bars. Each column of the screen maps to one
// ``band_energies`` slot; bar height grows with the band energy, pulses on
// ``onset_pulse``, and its color shifts between two tones across ``beat_phase``.
// Output is premultiplied RGBA so it can be alpha-blended over a background.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_hi   → sparkle density on the bar tips (hat / air hits).
//   transient_mid  → short mid-slot flash washed across all bars on snares.
//   build_tension  → desaturates colour and compresses bar height range
//                    pre-drop so the build reads as a "held breath".
//   drop_hold      → post-drop palette[4] bloom + overall intensity punch.
//   bar_phase      → slow cool↔hot palette bias across the 4-beat bar so
//                    the gradient doesn't look locked to raw ``time``.

in vec2 v_uv;
out vec4 out_color;

uniform vec2 resolution;
uniform float time;
uniform float beat_phase;    // 0..1 between consecutive beats
uniform float bar_phase;     // 0..1 across the current 4-beat bar
uniform float rms;           // global loudness envelope
uniform float onset_pulse;   // 0..1 decaying after each onset peak
uniform float onset_env;     // continuous normalised onset-strength envelope
uniform float bass_hit;      // smoothed kick envelope (matches compositor)
uniform float transient_lo;  // low-band transient (kick / sub)
uniform float transient_mid; // mid-band transient (snare / body)
uniform float transient_hi;  // high-band transient (hats / air)
uniform float build_tension; // 0..1 pre-drop smoothstep ramp
uniform float drop_hold;     // post-drop exponential afterglow
uniform float intensity;     // user intensity slider (0..1)
uniform float band_energies[8];
uniform vec3 u_palette[5];
uniform int u_palette_size;
uniform sampler2D u_background;
uniform float u_comp_background; // 0 = premultiplied RGBA overlay; 1 = blend over u_background

const float BAR_COUNT = 8.0;
const float BAR_GAP = 0.10;  // fraction of each column used as gutter
const float TAU = 6.28318530717958647692;

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

void main() {
    // All reactive signals arrive in [0, 1] by contract; clamp defensively
    // so a dense transient stack (snare + hat + drop_hold on the same
    // frame) can never exceed the mix()/scale budget below.
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float col_f = v_uv.x * BAR_COUNT;
    int band = int(clamp(floor(col_f), 0.0, BAR_COUNT - 1.0));
    float f = fract(col_f);

    float energy = band_energies[band];
    float rms_g = rms * rms;
    float h_raw = energy * 0.58 + bass_hit * 0.22 + rms_g * 0.12 + onset_pulse * 0.05;
    // Pre-drop: compress the dynamic range toward a mid-height so the build
    // reads as a flat "held breath"; drop_hold briefly lifts the ceiling.
    float h = clamp(mix(h_raw, h_raw * 0.65 + 0.12, tension) + 0.10 * hold,
                    0.0, 1.0);
    float base_y = 0.08;
    float top_y = base_y + h * 0.78;

    float in_bar_y = step(base_y, v_uv.y) * step(v_uv.y, top_y);
    float in_bar_x = step(BAR_GAP * 0.5, f) * step(f, 1.0 - BAR_GAP * 0.5);
    float bar = in_bar_y * in_bar_x;

    // Bar-tip glow. transient_hi adds a fine-grain sparkle on hats so the
    // upper band pops without affecting the bar body below.
    float glow_y = smoothstep(top_y + 0.03, top_y, v_uv.y) *
                   step(base_y, v_uv.y);
    float glow = glow_y * in_bar_x *
                 (0.30 + 0.28 * bass_hit + 0.18 * onset_pulse + 0.35 * t_hi);

    // Cool↔hot slot pair walks across the 8 bars as before, but bar_phase
    // adds a slow per-bar bias so the gradient never tracks raw time.
    float pulse = 1.0 - beat_phase;
    float bar_shift = 0.5 + 0.5 * sin(bar_phase * TAU + float(band) * 0.30);
    float mix_w = clamp(mix(pulse, bar_shift, 0.40), 0.0, 1.0);
    vec3 cool = palette_pick(band);
    vec3 hot  = palette_pick(band + 1);
    vec3 col = mix(cool, hot, mix_w) *
               (0.85 + 0.25 * sin(time * 1.7 + float(band)));

    // Mid-slot flash washes subtly across every bar on snares.
    col += palette_pick(2) * (0.12 * t_mid);

    // Post-drop bloom via palette[4]; rides alongside onset_pulse so the
    // drop lands as a compound flash.
    col += palette_pick(4) * (0.14 * hold + 0.05 * onset_pulse);

    // Pre-drop desaturation: blend toward luma so the build feels tighter
    // and colder. Snaps back when drop_hold kicks in.
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.45 * tension);

    float alpha = clamp(bar + glow * 0.5, 0.0, 1.0)
                  * intensity
                  * (1.0 + 0.12 * hold);
    vec4 ov = vec4(col * alpha, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
