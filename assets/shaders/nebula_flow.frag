#version 330

// Nebula flow — pilot shader for the Phase-2 reactive-signal vocabulary.
// Shares the value-noise / FBM scaffolding of ``nebula_drift`` but rewires
// the main body to consume ``transient_lo/mid/hi``, ``build_tension``,
// ``drop_hold`` and ``bar_phase`` so an A/B render against the ``cosmic``
// preset exposes the new dynamics without changing the base aesthetic or
// the shipped 5-colour palette.
//
// Signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_lo   → additive cloud breathing on every kick (fast attack).
//   transient_mid  → mid-slot colour flash on snares / body hits.
//   transient_hi   → sparkle density + twinkle brightness on hats / air.
//   build_tension  → dampens drift, pulls mix toward the cold palette
//                    slots (u_palette[0/1]) and desaturates.
//   drop_hold      → post-drop palette[4] bloom + briefly higher FBM
//                    persistence so finer structure surfaces for ~2 s.
//   bar_phase      → rotates the drift basis per bar so the look never
//                    feels locked to raw ``time``.
//   onset_env      → sustained shimmer layered on top of onset_pulse.
//
// All techniques (value noise, FBM, smoothstep-interpolated hash) are
// textbook public-domain GLSL patterns — no third-party shader code.

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

// Detail-weighted FBM: ``persistence`` rises with drop_hold so higher
// octaves retain more amplitude right after a drop, then relaxes back.
float fbm(vec2 p, float detail) {
    float v = 0.0;
    float amp = 0.5;
    float persistence = mix(0.5, 0.72, clamp(detail, 0.0, 1.0));
    for (int i = 0; i < 4; i++) {
        v += amp * vnoise(p);
        p = p * 2.02 + vec2(7.3, 11.7);
        amp *= persistence;
    }
    return v;
}

void main() {
    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 uv = v_uv;
    uv.x *= aspect;

    // All reactive signals arrive in [0, 1] by contract; clamp defensively
    // so a dense transient stack (kick + snare + hat + drop_hold landing
    // on the same frame) can never exceed the mix()/scale budget below.
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);
    float env = clamp(onset_env, 0.0, 1.0);
    float pulse = clamp(onset_pulse, 0.0, 1.0);

    // Drift basis rotates once per 4-beat bar so the nebula never tracks
    // raw time. build_tension dampens motion into the build; drop_hold
    // briefly re-energises it on the way out.
    float bar_angle = bar_phase * TAU;
    vec2 drift_basis = vec2(cos(bar_angle), sin(bar_angle));
    float drift_speed = (0.016 + 0.022 * rms * rms)
                      * mix(1.0, 0.35, tension)
                      + 0.014 * hold;
    vec2 drift = drift_basis * (time * drift_speed)
               + vec2(0.0, -time * 0.004);

    // Low-end breathing: transient_lo is the primary "inflate" knob,
    // bass_hit adds a slightly slower sustain so the two complement.
    float breath = 0.35 * t_lo + 0.12 * bass_hit;

    float cloud = (0.5 + 0.5 * breath) * fbm(uv * 2.4 + drift, hold);
    cloud += 0.35 * fbm(uv * 5.7 - drift * 1.7 + vec2(13.0, -9.0), hold * 0.6);
    cloud += 0.18 * env;
    cloud = clamp(cloud, 0.0, 1.4);

    // Sparse star field: transient_hi lowers the threshold (more stars
    // visible on hats) and amps the twinkle so the upper band pops.
    float star_thr = mix(0.9970, 0.9935, t_hi);
    float star_seed = hash21(floor(uv * 620.0 + 3.1));
    float star = step(star_thr, star_seed);
    float twinkle = 0.4 + 0.6 * sin(time * 3.0 + star_seed * 31.0);
    float stars = star * twinkle * (1.0 + 0.8 * t_hi);

    // Palette ramp identical in shape to nebula_drift so the A/B reads as
    // a dynamics change, not a colour change.
    float grad = clamp(cloud * 0.9, 0.0, 1.0);
    vec3 c0 = palette_pick(0);
    vec3 c1 = palette_pick(1);
    vec3 c2 = palette_pick(2);
    vec3 c3 = palette_pick(3);
    vec3 low = mix(c0, c1, smoothstep(0.0, 0.55, grad));
    vec3 high = mix(c2, c3, smoothstep(0.45, 1.0, grad));
    vec3 neb = mix(low, high, smoothstep(0.35, 0.75, grad));

    // Cold-slot pull during pre-drop builds: lerp toward the c0/c1 blend
    // then partially desaturate so the build feels tighter and colder.
    vec3 cold = mix(c0, c1, smoothstep(0.2, 0.8, grad));
    neb = mix(neb, cold, 0.65 * tension);
    float luma = dot(neb, vec3(0.2126, 0.7152, 0.0722));
    neb = mix(neb, vec3(luma), 0.35 * tension);

    // Snare / body flash on the mid palette slot. Kept small so it layers
    // over the bass breath rather than overwriting it.
    neb += palette_pick(2) * (0.10 * t_mid);

    // Post-drop bloom via palette[4]; onset_pulse / bass_hit ride alongside
    // at half weight so the drop lands as a compound flash.
    neb += palette_pick(4) * (0.28 * hold + 0.14 * pulse + 0.14 * bass_hit);

    // Stars are near-white highlights on top of the nebula.
    neb = mix(neb, vec3(1.0), clamp(stars * 0.85, 0.0, 1.0));

    // Radial vignette — tightens during the build so focus pulls inward.
    vec2 centered = v_uv - 0.5;
    centered.x *= aspect;
    float r = length(centered);
    float vig_outer = mix(1.05, 0.85, tension);
    float vignette = smoothstep(vig_outer, 0.18, r);

    // Content-driven alpha: the nebula is the rendered ``cloud`` field plus
    // ``stars`` highlights. Use ``cloud`` as the primary opacity term so
    // dark voids of space let the SDXL/AnimateDiff background read
    // through, with a small ``hold`` lift so post-drop blooms still pump.
    // The previous ``0.50 + 0.55 * cloud`` baked a 50 % floor that painted
    // the entire frame even where ``cloud`` was zero.
    // See ``docs/technical/reactive-shader-layer.md``.
    float alpha = clamp(cloud
                        * vignette
                        * intensity
                        * (1.0 + 0.30 * hold),
                        0.0, 1.0);
    vec3 col = neb * alpha;
    vec4 ov = vec4(col, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
