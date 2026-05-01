#version 330

// First-person "flight" through a dark cylindrical lattice tunnel: perspective
// uses 1/|uv| (vanishing point at center), spiral twist on the angular grid,
// emissive grid lines + node spheres, and cheap bloom (core + wide halo). Scroll
// is monotonic forward flight (time-driven) with audio surges; slow camera roll
// and cumulative corkscrew twist avoid bar-synced back-and-forth sway.
// Kicks pulse line width and emissive mix; transients and ``drop_hold`` add sparks.
//
// Cylindrical perspective + fwidth anti-aliased lines follow common demoscene
// patterns; no third-party shader code.

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

void main() {
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= aspect;

    // Camera roll: slow continuous spin, surges on kicks (no bar-length reversal)
    float rms_g = rms * rms;
    float roll_spd = 0.042 + 0.028 * rms_g + 0.20 * bass_hit + 0.10 * t_lo
                   - 0.020 * tension + 0.012 * hold;
    float roll = time * roll_spd + 0.06 * sin(time * 0.29);
    float cr = cos(roll);
    float sr = sin(roll);
    p = vec2(cr * p.x - sr * p.y, sr * p.x + cr * p.y);

    float r = length(p);
    float r_safe = max(r, 0.0012);
    float ang = atan(p.y, p.x);

    // 1/r perspective: small r = down the tunnel (far), large r = rim (near)
    float z = 1.0 / r_safe;
    // Forward flight: steady cruise + audio surges (depth scroll dominates gv)
    float scroll_mult = 0.55 + 0.60 * rms_g + 0.55 * bass_hit
                      + 0.42 * t_lo
                      - 0.32 * tension
                      + 0.30 * hold;
    float scroll = time * (1.38 + 1.28 * scroll_mult) * (1.0 + 0.25 * t_mid);

    // Corkscrew: cumulative time twist + audio (avoid sin(bar_phase) sway)
    float twist = 0.36 + 0.10 * rms_g + 0.12 * bass_hit + 0.07 * t_lo
                - 0.12 * tension + 0.10 * hold
                + time * 0.031;
    float spiral = z * twist * (1.0 + 0.15 * t_hi);
    float rings = 18.0 * mix(1.0, 1.12, hold);
    float gu = (ang * 6.0 + spiral + scroll * 0.4) * rings / TAU
             + 0.25 * sin(z * 0.35 + time * 0.5);
    float gv = z * 9.0 - scroll * 1.92 + time * 0.09;
    vec2 g = vec2(gu, gv);

    vec2 f = fract(g) - 0.5;
    // Anti-aliased line mask (fwidth tames moiré at the vanishing point)
    float w_line = 0.018 + 0.022 * bass_hit + 0.012 * t_mid + 0.06 * t_lo
                 + 0.008 * (band_energies[0] + band_energies[1]);
    float ax = fwidth(f.x) * 1.1;
    float ay = fwidth(f.y) * 1.1;
    float lx = 1.0 - smoothstep(w_line, w_line + ax, abs(f.x));
    float ly = 1.0 - smoothstep(w_line, w_line + ay, abs(f.y));
    float line = max(lx, ly);

    // Node sparks at cell corners
    vec2 fr = fract(g);
    float d_c = min(
        min(length(fr), length(fr - vec2(1.0, 0.0))),
        min(length(fr - vec2(0.0, 1.0)), length(fr - vec2(1.0, 1.0)))
    );
    float node_w = 0.04 + 0.05 * bass_hit + 0.03 * hold;
    float node = 1.0 - smoothstep(0.0, node_w, d_c);
    // Bloom on nodes (wide soft falloff)
    float node_halo = 1.0 - smoothstep(0.0, node_w * 3.2, d_c);
    // Extra pulse on onsets
    node += 0.45 * node_halo * (0.2 * onset_pulse + 0.12 * onset_env + 0.2 * t_mid);

    // Color: red/orange in foreground (high r), cool blue/purple in the core
    float depth_01 = clamp(1.0 - r * 1.4, 0.0, 1.0);
    vec3 hot = mix(palette_pick(0), palette_pick(1), 0.5 + 0.5 * rms_g);
    vec3 cool = mix(palette_pick(2), palette_pick(3), 0.4 + 0.4 * z * 0.01);
    vec3 accent = palette_pick(4);
    vec3 line_col = mix(hot, cool, smoothstep(0.1, 0.85, depth_01));
    float luma_t = dot(line_col, vec3(0.2126, 0.7152, 0.0722));
    line_col = mix(line_col, vec3(luma_t), 0.45 * tension);

    // Per-band twinkle on high bins (distant "electric" feel)
    float e_hi = band_energies[4] * 0.12 + band_energies[5] * 0.10
               + band_energies[6] * 0.10 + band_energies[7] * 0.12;
    line_col += cool * (0.15 * t_hi + 0.12 * e_hi) * (0.2 + 0.8 * depth_01);
    // Drop: accent surges
    line_col = mix(
        line_col,
        line_col + accent * 0.7,
        hold * (0.25 + 0.55 * smoothstep(0.2, 0.9, r))
    );

    // Wireframe emissive (lines + corner nodes) with line bloom. Halo and
    // beat-flash terms bumped (2026-04) so the lattice reads cleanly against
    // SDXL/AnimateDiff cyberpunk backdrops; the structural ``core`` stays
    // tight so the tunnel still has clear lines instead of a wash.
    float line_glow = 1.0 - smoothstep(w_line, w_line * 6.0, abs(f.x));
    line_glow = max(line_glow, 1.0 - smoothstep(w_line, w_line * 6.0, abs(f.y)));
    float struct_mask = max(line, node * 1.3);
    float core = struct_mask;
    float line_luma = dot(line_col, vec3(0.299, 0.587, 0.114));
    float halo = (line_glow * 0.32 + node_halo * 0.52) * (0.55 + 0.45 * line_luma);

    float bright = core * 1.20
                 + halo * 0.78
                 + 0.32 * (1.0 - beat_phase) * line
                 + 0.26 * onset_pulse * (line + node);

    // Floating sparks (streaks toward center) — high-band / hat-driven
    vec2 sp = p * 38.0 + vec2(time * 7.0, -time * 5.0);
    float spk = 0.0;
    for (int i = 0; i < 3; i++) {
        vec2 c = sp + float(i) * vec2(19.0, 13.0);
        float h = hash21(floor(c));
        vec2 o = fract(c) - 0.5;
        float s = 1.0 - smoothstep(0.0, 0.2 + 0.15 * t_hi, length(o));
        float pick = step(0.92, h);
        spk += pick * s * h * (0.35 + 0.65 * t_hi) * (0.4 + 0.6 * (1.0 - depth_01));
    }
    line_col = mix(hot, vec3(1.0, 0.85, 0.5), 0.25 * t_hi) * 0.15 * spk + line_col * (1.0 + 0.5 * spk);

    // Radial falloff: rim a bit dimmer, center glows; edge vignette for speed
    float edge_vig = smoothstep(1.35, 0.28, r) * (0.4 + 0.6 * smoothstep(0.0, 0.45, r));
    // Audio multiplier with a cleaner floor + steeper bass slope so the
    // lattice pumps harder on every kick instead of sitting at a flat
    // ~0.55 baseline that SDXL detail otherwise wins out over. The legacy
    // ``base_dark`` term that contributed a flat dark wash everywhere on
    // the frame has been removed — it was the "fog" that softened the
    // tunnel against busy backdrops without adding any visible structure.
    float audio_gain = 0.62 + 0.30 * (band_energies[0] + band_energies[1])
                     + 0.42 * bass_hit + 0.18 * t_lo;
    float alpha_geo = (bright * 1.05 * edge_vig + spk * 0.35) * audio_gain;

    // Occasional "thruster" flash from the deep tunnel on bass
    float tunnel_flash = (1.0 - smoothstep(0.0, 0.12, r))
                       * (0.28 * bass_hit + 0.16 * t_lo) * (1.0 - tension);
    vec3 emissive = line_col * bright * (0.85 + 0.40 * e_hi) + hot * tunnel_flash
                  + accent * (0.20 * node * (1.0 + 2.0 * hold) + 0.16 * t_mid);

    // Saturate line cores toward white on peaks. Without this, thin coloured
    // lines disappear against busy SDXL stills (e.g. orange flames behind a
    // teal/violet tunnel). The pow gate keeps faint mids coloured but pushes
    // bright structural pixels toward an almost-white rim that reads on any
    // background.
    float core_peak = pow(clamp(bright * 0.65, 0.0, 1.0), 3.0);
    emissive = mix(emissive, vec3(1.55), core_peak * 0.40);

    // The structural mask (lines + nodes) gets its own dedicated alpha
    // channel so the lattice geometry punches through any background,
    // independent of the soft emit / glow envelope. ``struct_mask`` is
    // already a sharp 0..1 line-or-node indicator, so this only lifts
    // alpha where there *is* line — empty void pixels stay fully
    // transparent and the background reads through them cleanly.
    float struct_alpha = clamp(
        struct_mask * (0.85 + 0.40 * bass_hit + 0.25 * t_lo) * edge_vig,
        0.0, 1.0
    );

    float alpha = clamp(
        max(struct_alpha,
            length(emissive) * 0.26 + alpha_geo * 1.05 + tunnel_flash * 0.42
        ) * intensity,
        0.0,
        1.0
    );
    vec3 col = emissive * alpha;
    // Premul path
    vec4 ov = vec4(col, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
