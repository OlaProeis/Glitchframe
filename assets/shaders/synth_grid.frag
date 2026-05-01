#version 330

// Retrowave composition: perspective ground grid scrolling toward the viewer,
// horizon glow, and a sliced neon sun above the horizon. Grid line thickness
// kicks on each beat via ``beat_phase``; sun-slice phase advances with
// ``time``; ground scroll speed includes ``rms`` so louder sections drive
// faster parallax. Palette picks drive horizon, sun top, sun bottom, grid,
// and accent.
//
// 1/z perspective grids and horizon-sun slicing are standard public-domain
// demoscene techniques.
//
// Phase-2 signal mapping (mirrors the authoring guide in
// ``docs/technical/reactive-shader-layer.md``):
//   transient_lo   → kick-driven scroll-speed punch on the ground plane.
//   transient_mid  → grid-line snap on snares (briefly thicker lines).
//   transient_hi   → extra sparkle in the upper sky region on hats.
//   build_tension  → cools + desaturates the sky and tightens perspective
//                    toward the horizon so the build feels compressed.
//   drop_hold      → post-drop horizon bloom via palette[4] + sun halo
//                    boost that decays with the afterglow.
//   bar_phase      → sun drifts slightly horizontally across each bar so
//                    the horizon composition never looks locked.

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

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

const float TAU = 6.28318530717958647692;

void main() {
    // All reactive signals arrive in [0, 1] by contract; clamp defensively.
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    // NDC-ish space: x in [-aspect, +aspect], y in [-1, +1].
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= aspect;

    float rms_g = rms * rms;
    float horizon = 0.05;

    vec3 sky_lo = palette_pick(0);
    vec3 sky_hi = palette_pick(1);
    vec3 sun_lo = palette_pick(2);
    vec3 sun_hi = palette_pick(3);
    vec3 grid_col = palette_pick(4);

    vec3 col = vec3(0.0);
    // Ground is the focal element — we want it opaque regardless of the
    // underlying background so the grid reads cleanly. The sky stays
    // slightly transparent so SDXL/AnimateDiff backgrounds can still peek
    // through around the sun.
    float local_alpha = 1.0;

    if (uv.y < horizon) {
        // Ground plane with 1/depth perspective; tighter tiles near horizon.
        // transient_lo punches scroll speed on kicks; build_tension slows
        // the parallax pre-drop; drop_hold briefly re-energises it.
        float depth = max(horizon - uv.y, 0.001);
        float u = uv.x / depth;
        float scroll = (0.50 + 0.42 * rms_g + 0.28 * bass_hit + 0.38 * t_lo)
                     * mix(1.0, 0.55, tension)
                     + 0.28 * hold;
        float v = 1.0 / depth + time * scroll;

        // Anti-aliased grid lines. Using ``fwidth`` keeps line width roughly
        // constant in screen space, which is essential for 1/z perspective
        // grids — without it, far-field pixels change so fast per pixel that
        // every pixel looks like a line and the ground collapses to a wash.
        // transient_mid snaps lines thicker on snares.
        float beat_boost = 0.5 * (1.0 - beat_phase) + 0.35 * onset_pulse
                         + 0.28 * bass_hit + 0.35 * t_mid + 0.20 * hold;
        float line_u = 1.0 - smoothstep(fwidth(u) * (0.5 + beat_boost),
                                        fwidth(u) * (1.5 + beat_boost),
                                        abs(fract(u) - 0.5));
        float line_v = 1.0 - smoothstep(fwidth(v) * (0.5 + beat_boost),
                                        fwidth(v) * (1.5 + beat_boost),
                                        abs(fract(v) - 0.5));
        // Fade out the grid once lines get tighter than a couple of pixels;
        // that's the horizon haze and stops the pile-up at the vanishing point.
        float u_density = clamp(1.0 / (fwidth(u) * 6.0), 0.0, 1.0);
        float v_density = clamp(1.0 / (fwidth(v) * 6.0), 0.0, 1.0);
        float line = max(line_u * u_density, line_v * v_density);

        // Ground darkens far from the camera so distant lines don't wash out.
        float ground_fade = smoothstep(-1.0, horizon, uv.y);
        vec3 ground = sky_lo * (0.15 + 0.45 * ground_fade);
        col = ground + grid_col * line * (0.55 + 0.65 * ground_fade);
        local_alpha = 1.0;
    } else {
        // Sky: vertical ramp from horizon to zenith.
        float sky_t = clamp((uv.y - horizon) / (1.0 - horizon), 0.0, 1.0);
        col = mix(sky_lo, sky_hi, sky_t);

        // Sun disc centered above the horizon. bar_phase drifts it slightly
        // so the horizon composition doesn't look locked.
        vec2 sun_center = vec2(0.12 * sin(bar_phase * TAU), horizon + 0.38);
        float sun_r = length(uv - sun_center);
        float disc = smoothstep(0.34, 0.30, sun_r);

        // Vertical slices advance with time so the sun appears to roll.
        float slice = step(0.5, fract((uv.y - horizon) * 14.0 - time * 0.55));
        vec3 sun_col = mix(sun_hi, sun_lo,
                           clamp((uv.y - horizon) / 0.42, 0.0, 1.0));
        col = mix(col, sun_col, disc * slice);

        // Horizon halo: exponential falloff in both directions. drop_hold
        // blooms it hard via palette[4] for a couple of seconds post-drop.
        float halo = exp(-abs(uv.y - horizon) * 7.5)
                   * (0.38 + 0.16 * rms_g + 0.14 * bass_hit + 0.30 * hold);
        col += sun_hi * halo * 0.30;
        col += grid_col * halo * (0.38 * hold);

        // High-band sparkle at the top of the frame (cymbals / hats area);
        // transient_hi amps it directly.
        float highs = (band_energies[6] + band_energies[7]) * 0.5 + 0.35 * t_hi;
        col += grid_col * highs * smoothstep(0.35, 1.0, sky_t) * 0.2;

        // Let higher sky fade toward transparent so the backdrop bleeds in;
        // stay mostly opaque near the horizon and sun to preserve contrast.
        local_alpha = mix(0.90, 0.52, smoothstep(horizon, 1.0, uv.y));
        // Sun disc is always fully opaque over the backdrop.
        local_alpha = max(local_alpha, disc);
    }

    // Overall beat pop — keep subtle so the grid doesn't strobes.
    col += grid_col * (1.0 - beat_phase) * 0.07;

    // Pre-drop: cool + desaturate the sky-and-horizon composition so the
    // build reads as compressed. Snaps back as drop_hold rises.
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.35 * tension);
    col = mix(col, sky_lo, 0.20 * tension);

    float alpha = clamp(intensity * local_alpha * (1.0 + 0.08 * hold),
                        0.0, 1.0);
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
