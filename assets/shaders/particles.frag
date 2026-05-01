#version 330

// Cellular particle field. Each grid cell hosts a single point whose radius
// is driven by one of the eight spectrum bands and by ``onset_pulse``. The
// whole field drifts upward proportional to ``rms`` so louder sections feel
// more alive. Colors interpolate between two palette tones via ``beat_phase``.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_lo   → small kick-driven radius bounce on every cell.
//   transient_mid  → mid-slot colour bias on snares (brief flash).
//   transient_hi   → per-cell twinkle amplitude + a fraction of cells
//                    briefly flare on hats (sparkle density).
//   build_tension  → dampens drift speed and compresses cell radius so
//                    the field "tightens" pre-drop.
//   drop_hold      → post-drop radius + brightness lift tinted toward
//                    palette[4]; snaps back as the afterglow decays.
//   bar_phase      → rotates the drift basis once per 4-beat bar so the
//                    field never feels locked to raw ``time``.

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

const float CELL_SIZE = 0.045;
const float TAU = 6.28318530717958647692;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

void main() {
    // All reactive signals arrive in [0, 1] by contract; clamp defensively
    // so a dense transient stack (kick + snare + hat + drop_hold) can
    // never exceed the mix()/scale budget below.
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 uv = v_uv;
    uv.x *= aspect;
    float rms_g = rms * rms;

    // Drift basis rotates once per 4-beat bar so the field never tracks
    // raw time; build_tension dampens speed into the build, drop_hold
    // briefly re-energises it on the way out.
    float bar_angle = bar_phase * TAU;
    vec2 drift_basis = vec2(sin(bar_angle), cos(bar_angle));
    float drift_speed = (0.032 + 0.055 * rms_g + 0.045 * bass_hit)
                      * mix(1.0, 0.45, tension)
                      + 0.030 * hold;
    uv += drift_basis * (time * drift_speed);

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

    // Radius: kick-driven bounce via transient_lo, hat-driven sparkle on
    // a fraction of cells via transient_hi, plus a post-drop lift.
    float radius = 0.12 + e * 0.24 + bass_hit * 0.20 + onset_pulse * 0.10 * h0
                 + 0.08 * t_lo + 0.12 * t_hi * step(0.65, h1)
                 + 0.10 * hold;
    // Pre-drop compression: cells shrink slightly during the build so the
    // field tightens; drop_hold above undoes this on the release.
    radius *= mix(1.0, 0.78, tension);

    float core = smoothstep(radius, radius * 0.25, d);
    float halo = smoothstep(radius * 1.8, radius, d) * 0.20;
    float brightness = (core + halo) * (0.34 + 0.48 * e + 0.20 * bass_hit
                                        + 0.14 * hold);

    // Each cell picks its pair of palette colors deterministically from its
    // hash so nearby cells aren't identical, and beat_phase smoothly swaps
    // between them.
    int pick_a = int(floor(h0 * float(max(1, u_palette_size))));
    vec3 cold = palette_pick(pick_a);
    vec3 warm = palette_pick(pick_a + 1);
    vec3 col = mix(cold, warm, beat_phase);
    // Twinkle: base LFO plus per-cell hat sparkle.
    col *= 0.8 + 0.3 * sin(time * 0.7 + h0 * TAU) + 0.25 * t_hi * h1;

    // Snare flash via the mid palette slot (small, so it layers rather
    // than overwrites the cell's own hue).
    col += palette_pick(2) * (0.14 * t_mid);

    // Post-drop bloom via palette[4]; only lights cells that already carry
    // a core so empty space stays dark.
    col += palette_pick(4) * (0.16 * hold) * core;

    // Pre-drop desaturation — tightens into the build.
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.28 * tension);

    float alpha = clamp(brightness, 0.0, 1.0) * intensity * (1.0 + 0.09 * hold);
    vec4 ov = vec4(col * alpha, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
