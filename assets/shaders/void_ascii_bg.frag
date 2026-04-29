#version 330

// Minimal void background pass for the voidcat-ASCII compositor. Historically
// this layer painted a palette-tinted noise wash on top of the SDXL/AnimateDiff
// still, which read as drifting "clouds" over the cleanly-rendered background
// and washed out the CPU ASCII grid that lives on top. The reactive shader's
// job here is intentionally close to a no-op: gently darken the background so
// the bright ASCII glyphs pop, and pulse the darken on bass/drop hits to add
// a hint of motion. The cat + grid (the actual subject) live in the
// :mod:`pipeline.voidcat_ascii` CPU layer, not here.

in vec2 v_uv;
out vec4 out_color;

uniform vec2 resolution;
uniform float time;
uniform float beat_phase;
uniform float bar_phase;
uniform float rms;
uniform float onset_pulse;
uniform float onset_env;
uniform float bass_hit;
uniform float transient_lo;
uniform float transient_mid;
uniform float transient_hi;
uniform float build_tension;
uniform float drop_hold;
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

void main() {
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * vec2(2.0 * aspect, 2.0);
    float r = length(p);

    // Soft radial mask: corners darken slightly, centre stays clean. Stays
    // small so SDXL detail in the middle of the frame reads through.
    float corner_mask = smoothstep(0.55, 1.20, r);

    // Background-only path: the layer is otherwise transparent so the SDXL
    // / AnimateDiff still renders untouched at the centre. ``corner_darken``
    // pulses gently with bass + post-drop afterglow so quiet sections sit
    // calm and loud sections dip a hair.
    float audio_pulse = 0.55 * bass_hit + 0.30 * hold + 0.18 * t_lo;
    float corner_darken = (0.10 + 0.10 * audio_pulse) * corner_mask;

    // Tiny accent tint at the corners only, so the palette informs the
    // darkening without painting a visible cloud over the whole frame.
    vec3 tint = mix(palette_pick(0), palette_pick(1), 0.5);

    // Premultiplied output: the shader contributes a near-black tint at the
    // corners and zero at the centre, so the composite formula
    // ``ov.rgb + bg * (1 - ov.a)`` falls through to ``bg`` everywhere except
    // the soft outer ring. Alpha is driven entirely by ``corner_darken`` so
    // silent uniforms still leave the SDXL still visible (regression test
    // ``test_silent_uniforms_let_bg_dominate``).
    float a = clamp(corner_darken * intensity, 0.0, 0.45);
    vec3 col = tint * 0.10;  // very dark accent toward the rim
    vec3 premul = col * a;
    vec4 ov = vec4(premul, a);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        out_color = vec4(ov.rgb + bg * (1.0 - ov.a), 1.0);
    } else {
        out_color = ov;
    }
}
