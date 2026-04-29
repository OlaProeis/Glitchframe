#version 330

// Future-plasma Milkdrop-style preset:
//   * Curl-noise *translation* feedback (no inward radial zoom — kills the
//     tunnel composition the previous revision had drifted into).
//   * Polar wedge kaleidoscope (6 ↔ 8 fold per quarter-bar) on the fresh
//     layer — the iconic Milkdrop mirror symmetry.
//   * Domain-warped FBM smoke filaments (``f(p + f(p + f(p)))``) replace
//     concentric rings / radial beams with organic flowing threads.
//   * Audio-driven Lissajous "waveform" trace mirrored through the
//     kaleidoscope — the signature Milkdrop element, summed over all 8
//     ``band_energies`` so each frequency bin warps the curve differently.
//   * Slow 5-slot hue-cycling palette ramp so the look never sits on a
//     single colour pair.
//
// All techniques (value-noise FBM, curl from FBM partial derivatives,
// polar wedge fold, domain warp) are textbook public-domain GLSL — no
// third-party shader code.

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
uniform sampler2D u_prev_frame;
uniform float u_has_prev;

const float TAU = 6.28318530717958647692;

vec3 palette_pick(int idx) {
    int n = max(1, u_palette_size);
    int i = ((idx % n) + n) % n;
    return u_palette[i];
}

// Cycling palette ramp — the input ``t`` is mod-tiled across the palette
// so a continuous monotonic input produces a continuous hue cycle through
// all 5 slots and back to slot 0 (rather than clamping at the last slot).
vec3 palette_ramp(float t) {
    int n = max(1, u_palette_size);
    float fn = float(n);
    float f = mod(t * fn, fn);
    int i0 = int(floor(f));
    int i1 = int(mod(float(i0) + 1.0, fn));
    return mix(palette_pick(i0), palette_pick(i1), fract(f));
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
        p = p * 2.05 + vec2(7.3, 11.7);
        amp *= 0.55;
    }
    return v;
}

// Divergence-free 2D curl from FBM partial derivatives — gives an
// organic flow field that translates the feedback texture instead of
// pulling everything toward the centre.
vec2 curl(vec2 p) {
    const float e = 0.06;
    float a = fbm(p + vec2(0.0, e));
    float b = fbm(p - vec2(0.0, e));
    float c = fbm(p + vec2(e, 0.0));
    float d = fbm(p - vec2(e, 0.0));
    return vec2(a - b, -(c - d));
}

// Polar wedge kaleidoscope: fold the angle into ``[0, TAU/n]`` then
// reflect across the bisector so adjacent wedges mirror each other.
// ``n`` must be an integer ≥ 2 for the wedges to tile the circle.
vec2 kaleido(vec2 p, float n) {
    float r = length(p);
    float a = atan(p.y, p.x);
    float wedge = TAU / max(n, 2.0);
    a = mod(a, wedge);
    a = abs(a - wedge * 0.5);
    return r * vec2(cos(a), sin(a));
}

void main() {
    float t_lo = clamp(transient_lo, 0.0, 1.0);
    float t_mid = clamp(transient_mid, 0.0, 1.0);
    float t_hi = clamp(transient_hi, 0.0, 1.0);
    float tension = clamp(build_tension, 0.0, 1.0);
    float hold = clamp(drop_hold, 0.0, 1.0);
    float pulse = clamp(onset_pulse, 0.0, 1.0);
    float env = clamp(onset_env, 0.0, 1.0);
    float rms_g = rms * rms;
    float rms_soft = sqrt(clamp(rms, 0.0, 1.0));
    float i_eff = clamp(intensity, 0.15, 1.0);

    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 p = v_uv - 0.5;
    p.x *= aspect;

    // -----------------------------------------------------------------
    //  FEEDBACK WARP — translate through curl-noise + tiny bar swirl.
    //  Critical: no inward radial zoom (that's the tunnel look we are
    //  walking away from). A small bass-driven out-breathe stops the
    //  feedback texture from pooling at the centre on a long hold.
    // -----------------------------------------------------------------
    vec2 cp = p * 1.20 + vec2(time * 0.030, time * 0.018);
    vec2 flow = curl(cp);
    float flow_amp = (0.0030 + 0.0042 * rms_g + 0.0050 * t_lo
                    + 0.0034 * bass_hit + 0.0022 * env)
                   * mix(1.0, 0.55, tension)
                   * (1.0 + 0.55 * hold);
    vec2 warp = flow * flow_amp;

    // Tiny bar-synced swirl — a few mrad per frame, audio-jittered.
    float swirl = 0.00080
                + 0.00140 * sin(bar_phase * TAU)
                + 0.00280 * t_mid
                + 0.00170 * t_hi
                + 0.00060 * sin(time * 0.21);
    float cs = cos(swirl);
    float sn = sin(swirl);
    vec2 q = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
    // Subtle out-breathing pulse — opposite of the original inward zoom.
    float breathe = 1.0 + 0.014 * bass_hit + 0.010 * t_lo - 0.008 * tension;
    q *= breathe;
    q.x /= aspect;

    vec2 uv_s = q + 0.5 + warp;
    uv_s = clamp(uv_s, 0.001, 0.999);

    // Hi-band chromatic split on the trail sample.
    float split = (0.0014 + 0.0042 * t_hi + 0.0022 * pulse) / max(aspect, 0.5);
    vec2 ox = vec2(split, 0.0);
    vec4 prev_c = texture(u_prev_frame, uv_s);
    vec3 prev_rgb = vec3(
        texture(u_prev_frame, clamp(uv_s + ox, 0.001, 0.999)).r,
        prev_c.g,
        texture(u_prev_frame, clamp(uv_s - ox, 0.001, 0.999)).b
    );
    float decay = 0.870 + 0.055 * hold - 0.060 * tension + 0.022 * rms_g;
    decay = clamp(decay, 0.74, 0.95);
    vec3 trail = prev_rgb * decay * u_has_prev;

    // -----------------------------------------------------------------
    //  KALEIDOSCOPE — alternate 6 ↔ 8 fold every quarter-bar so the
    //  symmetry shifts on the downbeat without sliding through invalid
    //  non-integer wedge counts (which would tear at angle = ±π).
    // -----------------------------------------------------------------
    float n_step = floor(bar_phase * 4.0);
    float n_fold = 6.0 + 2.0 * mod(n_step, 2.0);
    vec2 kp = kaleido(p, n_fold);

    // -----------------------------------------------------------------
    //  FRESH LAYER — domain-warped FBM smoke filaments inside the wedge.
    //  Spectrum-weighted offsets so each band paints into the warp.
    // -----------------------------------------------------------------
    float spec_w =
          band_energies[0] * 0.020 + band_energies[1] * 0.018
        + band_energies[2] * 0.014 + band_energies[3] * 0.012
        + band_energies[4] * 0.010 + band_energies[5] * 0.008
        + band_energies[6] * 0.007 + band_energies[7] * 0.006;
    float spec_sum = band_energies[0] + band_energies[1] + band_energies[2]
                   + band_energies[3] + band_energies[4] + band_energies[5]
                   + band_energies[6] + band_energies[7];
    spec_sum = clamp(spec_sum * 0.125, 0.0, 1.0);

    vec2 d_off = vec2(time * 0.045, -time * 0.030)
               + 0.42 * vec2(sin(bar_phase * TAU), cos(bar_phase * TAU));
    vec2 q1 = vec2(
        fbm(kp * 1.6 + d_off),
        fbm(kp * 1.6 + d_off + vec2(5.2, 1.3))
    );
    vec2 q2 = vec2(
        fbm(kp * 2.2 + 3.8 * q1 + vec2(1.7, 9.2 + spec_w * 6.0)),
        fbm(kp * 2.2 + 3.8 * q1 + vec2(8.3, 2.8 - spec_w * 6.0))
    );
    float fbm_v = fbm(kp * 2.6 + 4.0 * q2);

    // Bright filament ridges (Milkdrop "smoke threads").
    float ridge = 1.0 - abs(fbm_v - 0.5) * 2.0;
    ridge = pow(clamp(ridge, 0.0, 1.0), 6.0);
    // Secondary fine ridge for crunch on hats / drops.
    float fine_v = fbm(kp * 6.5 + 2.0 * q2 + vec2(time * 0.09, -time * 0.07));
    float fine = pow(clamp(1.0 - abs(fine_v - 0.5) * 2.0, 0.0, 1.0), 14.0);

    // -----------------------------------------------------------------
    //  WAVEFORM TRACE — glowing horizontal+vertical sinusoids whose
    //  amplitude is summed from all 8 band_energies. Drawn in folded
    //  ``kp`` so the kaleidoscope mirror multiplies it across the screen.
    // -----------------------------------------------------------------
    float wphase = time * (0.85 + 0.55 * rms) + beat_phase * TAU * 0.5;
    float wy = 0.0;
    float wx = 0.0;
    for (int i = 0; i < 8; i++) {
        float k = float(i + 1);
        float fk = 2.4 + k * 1.45;
        wy += band_energies[i] * sin(kp.x * fk + wphase + k * 0.71);
        wx += band_energies[i] * cos(kp.y * fk + wphase * 1.3 - k * 0.71);
    }
    float wave_gain = 0.058 + 0.048 * pulse + 0.038 * bass_hit + 0.026 * t_mid;
    wy *= wave_gain;
    wx *= wave_gain;
    float trace_w = 0.0040 + 0.0070 * t_hi + 0.0040 * env;
    float trace_y = 1.0 - smoothstep(0.0, trace_w * 1.6, abs(kp.y - wy));
    float trace_x = 1.0 - smoothstep(0.0, trace_w * 1.6, abs(kp.x - wx));
    // Soft halo around each line (wider falloff, lower amplitude).
    float halo_y = 1.0 - smoothstep(0.0, trace_w * 6.0, abs(kp.y - wy));
    float halo_x = 1.0 - smoothstep(0.0, trace_w * 6.0, abs(kp.x - wx));
    float trace = clamp(trace_y + trace_x + 0.35 * (halo_y + halo_x), 0.0, 1.6);

    // -----------------------------------------------------------------
    //  PALETTE — slow continuous hue cycle across all 5 slots, with
    //  bar / FBM / beat modulation so the colour never freezes.
    // -----------------------------------------------------------------
    float hue = time * 0.038
              + bar_phase * 0.45
              + 0.20 * fbm_v
              + 0.18 * sin(beat_phase * TAU * 0.5)
              + 0.10 * spec_w;
    vec3 base = palette_ramp(hue);
    vec3 hot = palette_ramp(hue + 0.18 + 0.18 * t_mid);
    vec3 spark = palette_ramp(hue + 0.55 + 0.30 * t_hi);
    vec3 trace_col = palette_ramp(hue + 0.78);

    vec3 fresh = mix(base * 0.55, base * 1.05, smoothstep(0.25, 0.75, fbm_v));
    fresh = mix(fresh, hot * 1.30, ridge * (0.55 + 0.45 * env));
    fresh = mix(fresh, spark, fine * (0.45 + 0.55 * t_hi + 0.25 * pulse));
    fresh += trace_col * trace * (0.32 + 0.85 * pulse + 0.45 * t_mid + 0.32 * env);

    // Drop afterglow — palette[4] bloom riding the bright FBM tips only.
    fresh += palette_pick(4) * (0.30 * hold) * smoothstep(0.30, 0.85, fbm_v);

    // Build-up dampening: pull toward cool slots and desaturate.
    vec3 cold = mix(palette_pick(0), palette_pick(1), smoothstep(0.20, 0.80, fbm_v));
    fresh = mix(fresh, cold, 0.55 * tension);
    float luma = dot(fresh, vec3(0.2126, 0.7152, 0.0722));
    fresh = mix(fresh, vec3(luma), 0.30 * tension);

    // Audio-gated ambient wash. The reactive layer is composited as a
    // *premultiplied overlay* (see ``docs/technical/reactive-shader-layer.md``)
    // so anything we lay down here also lifts the alpha in :func:`alpha`
    // below — a constant ambient term would paint the whole frame opaque
    // even during silent passages and hide the SDXL/AnimateDiff background.
    // Gate on ``rms_soft + spec_sum`` so the wash ramps in with energy
    // rather than sitting under every frame.
    float wash_amp = 0.10 * rms_soft + 0.06 * spec_sum;
    fresh += mix(palette_pick(2), palette_pick(1), 0.5) * wash_amp;

    // Continuous-UV hash grain (no chunky pixel blocks in encodes).
    vec2 g_uv = v_uv * vec2(1447.0, 1021.0) + vec2(time * 13.7, -time * 9.2);
    float grain = hash21(g_uv) + 0.5 * hash21(g_uv * 1.91 + 17.0) - 0.75;
    fresh += vec3(grain) * (0.022 + 0.038 * t_hi + 0.022 * env);

    // -----------------------------------------------------------------
    //  EMIT / TRAIL COMBINE
    // -----------------------------------------------------------------
    float emit = (0.40 + 0.42 * rms_g + 0.36 * bass_hit + 0.30 * t_lo
                  + 0.20 * env + 0.18 * pulse + 0.20 * hold
                  + 0.14 * ridge + 0.20 * trace)
               * (1.0 - 0.22 * tension)
               * i_eff;

    vec3 acc = trail + fresh * emit;
    // Soft Reinhard tone-map with headroom for peak whites.
    acc = acc / (1.0 + acc * 0.085);

    // Peak whites on hits — push small bright cores so drops feel snappy.
    float peak = clamp(
          pulse * 0.85
        + bass_hit * 0.70
        + hold * 0.40
        + t_hi * 0.30
        + ridge * 0.25
        + trace * 0.55,
        0.0, 1.0);
    acc = mix(acc, vec3(1.0), peak * peak * 0.55);

    // Edge-soft vignette so the corners don't out-shine the centre.
    float vign = 1.0 - smoothstep(0.45, 1.10, length(v_uv - 0.5) * 1.8);
    acc *= mix(0.84, 1.0, vign);

    // Content-driven alpha. Every other shader in ``assets/shaders/`` derives
    // alpha from the rendered ``acc`` (see ``particles.frag``,
    // ``tunnel_flight.frag``, ``geometry_pulse.frag``) so dark / empty regions
    // of the shader output let the background show through. The previous
    // formula here had a hard floor of 0.30 and a constant 0.42 base, which
    // painted at minimum 30 % opacity over every pixel of every frame and
    // completely hid the SDXL stills the compositor uploads as ``u_background``.
    //
    // ``content`` is the per-pixel luminance of the toned-mapped accumulator
    // (already includes trails + peak whites + vignette), and ``audio_lift``
    // is a small additive bonus on hits so the overlay punches harder during
    // loud sections without painting the quiet ones. No floor: silent +
    // empty pixels read as alpha = 0 and the SDXL still passes through
    // untouched.
    float content = clamp(dot(acc, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    float audio_lift = 0.22 * bass_hit + 0.18 * peak + 0.14 * hold
                     + 0.10 * env + 0.08 * trace;
    float alpha = clamp((content + audio_lift) * i_eff, 0.0, 0.95);
    vec3 premul = acc * alpha;

    vec4 ov = vec4(premul, alpha);
    if (u_comp_background > 0.5) {
        vec3 bg = texture(u_background, v_uv).rgb;
        vec3 rgb = ov.rgb + bg * (1.0 - ov.a);
        out_color = vec4(rgb, 1.0);
    } else {
        out_color = ov;
    }
}
