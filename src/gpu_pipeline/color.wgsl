// -------------------------------
// Uniforms
// -------------------------------
struct Uniforms {
    center_x:           f32,
    center_x_exp:       i32,
    center_y:           f32,
    center_y_exp:       i32,
    scale:              f32,
    scale_exp:          i32,
    max_iter:           u32,
    ref_orb_count:      u32,
    perturb_err_thresh: f32,
    view_width:         f32,
    view_height:        f32,
    render_width:       u32,
    render_height:      u32,
    render_tex_width:   f32,
    render_tex_height:  f32,
    grid_size:          u32,
    grid_width:         u32,
    sample_count:       u32,
    jitter_strength:    f32,
    sample_avg_bias:    f32,
    render_flags:       u32,
    stripe_trap_arg1:   f32,
    stripe_trap_arg2:   f32,
    stripe_trap_arg3:   f32,
    stripe_trap_arg4:   f32,
    trap_shape:          u32,
    trap_palette_cycles: f32,
    trap_iter_skip_frac: f32,
    // Unused by color; declared to keep the uniform layout aligned with the
    // calc shaders (fractal-function fields live mid-struct in SceneUniform).
    formula_power:       u32,
    julia_c_re:          f32,
    julia_c_im:          f32,
    rot_cos:            f32,
    rot_sin:            f32,
    stateful_kind:      u32,
    color_scalar_mapping_mode:      u32,
    color_scaler_mapping_strength:  f32,
    palette_tex_width:  u32,
    palette_len:        u32,
    palette_cycles:     f32,
    palette_offset:     f32,
    palette_gamma:      f32,
    distance_multiplier:        f32,
    glow_intensity:             f32,
    neighbor_scale_multiplier:  f32,
    ambient_intensity:          f32,
    key_light_intensity:        f32,
    key_light_azimuth:          f32,
    key_light_elevation:        f32,
    fill_light_intensity:       f32,
    fill_light_azimuth:         f32,
    fill_light_elevation:       f32,
    specular_intensity:         f32,
    specular_power:             f32,
    ao_darkness:                f32,
    rim_intensity:              f32,
    rim_power:                  f32,
    // --- Histogram (adaptive) coloring ---
    hist_eq_amount:             f32,
    hist_black_pct:             f32,
    hist_white_pct:             f32,
    hist_temporal_alpha:        f32,
    // --- Tier 3 ---
    hist_bin_count:             u32,
    hist_blur_radius:           u32,
    hist_log_binning:           u32,
    hist_include_interior:      u32,
    palette_interp_mode:        u32,
    _pad_palette0:              u32,
    _pad_palette1:              u32,
    _pad_palette2:              u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

// Render flags from scene uniforms
const DEBUG_COLORING: u32   = 1u << 0;
const SHOW_GLITCH: u32      = 1u << 1;
const SMOOTH_COLORING: u32  = 1u << 2;
const USE_DE: u32           = 1u << 3;
const USE_STRIPES: u32      = 1u << 4;
const USE_TRAPS: u32          = 1u << 13;
const USE_TRAP_INTERIOR: u32  = 1u << 14;
const USE_HISTOGRAM: u32      = 1u << 16;
const ENABLE_GLOW: u32        = 1u << 5;
const ENABLE_KEY_LIGHT: u32 = 1u << 6;
const ENABLE_FILL_LIGHT: u32= 1u << 7;
const ENABLE_SPEC: u32      = 1u << 8;
const ENABLE_AO: u32        = 1u << 9;
const ENABLE_RIM: u32       = 1u << 10;
const RECALC_FRACTAL: u32   = 1u << 11;

// From mandelbrot calculation shader (per-pixel calculations)
@group(0) @binding(1)
var calc_tex: texture_2d<f32>;

@group(0) @binding(2)
var palette_tex: texture_2d<f32>;

@group(0) @binding(3)
var render_tex : texture_storage_2d<rgba8unorm, write>;

// Histogram-equalization inputs (built by histogram.wgsl). Read-only here.
// HIST_BINS is the allocation/max; the active count is uni.hist_bin_count.
const HIST_BINS: u32 = 1024u;
@group(0) @binding(4) var<storage, read> hist_cdf: array<f32, HIST_BINS>;
@group(0) @binding(5) var<storage, read> hist_range: array<u32, 2>;

// Position of fi within [lo, hi], in [0,1]. MUST match bin_frac() in histogram.wgsl.
fn hist_bin_frac(fi: f32, lo: f32, hi: f32) -> f32 {
    if (uni.hist_log_binning != 0u) {
        let a   = log(1.0 + fi);
        let alo = log(1.0 + lo);
        let ahi = log(1.0 + hi);
        return clamp((a - alo) / max(ahi - alo, 1e-20), 0.0, 1.0);
    }
    return clamp((fi - lo) / max(hi - lo, 1e-20), 0.0, 1.0);
}

// Map a raw fractional iteration count to its equalized rank in [0,1] via the
// on-screen escape-time CDF, interpolating between adjacent bins. Falls back to
// the plain linear normalization if the range is degenerate (no spread).
fn hist_equalize(fi: f32) -> f32 {
    let lo = bitcast<f32>(hist_range[0]);
    let hi = bitcast<f32>(hist_range[1]);
    if (hi <= lo) {
        return fi / f32(uni.max_iter);
    }
    let n = clamp(uni.hist_bin_count, 1u, HIST_BINS);
    let pos = hist_bin_frac(fi, lo, hi) * f32(n - 1u);
    let i0 = u32(floor(pos));
    let i1 = min(i0 + 1u, n - 1u);
    return mix(hist_cdf[i0], hist_cdf[i1], fract(pos));
}

fn map_color_scalar(t: f32) -> f32 {
    let k = uni.color_scaler_mapping_strength;

    switch (uni.color_scalar_mapping_mode) {
        case 0u: { // Linear
            return t;
        }
        case 1u: { // Power
            return pow(t, k);
        }
        case 2u: { // Log
            return log(1.0 + k * t) / log(1.0 + k);
        }
        case 3u: { // Atan
            return atan(k * t) / atan(k);
        }
        default: {
            return t;
        }
    }
}

// sRGB (gamma ~2.2 approx) <-> linear light helpers for perceptual palette blends.
fn srgb_to_linear(c: vec3f) -> vec3f { return pow(c, vec3f(2.2)); }
fn linear_to_srgb(c: vec3f) -> vec3f {
    // Clamp: Oklab blends can land slightly out of gamma; pow(neg, frac) = NaN.
    return pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
}

// Bjorn Ottosson's Oklab (expects linear sRGB in, linear sRGB out).
fn linear_srgb_to_oklab(c: vec3f) -> vec3f {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = pow(l, 1.0 / 3.0);
    let m_ = pow(m, 1.0 / 3.0);
    let s_ = pow(s, 1.0 / 3.0);
    return vec3f(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}
fn oklab_to_linear_srgb(c: vec3f) -> vec3f {
    let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3f(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

// Blend two palette texels (sRGB, 0..1) per uni.palette_interp_mode.
fn interp_palette(c0: vec3f, c1: vec3f, frac: f32) -> vec3f {
    switch (uni.palette_interp_mode) {
        case 0u: { // Linear (sRGB)
            return mix(c0, c1, frac);
        }
        case 1u: { // Smoothstep
            let f = frac * frac * (3.0 - 2.0 * frac);
            return mix(c0, c1, f);
        }
        case 2u: { // Gamma-correct (blend in linear light)
            return linear_to_srgb(mix(srgb_to_linear(c0), srgb_to_linear(c1), frac));
        }
        case 3u: { // Cosine
            let f = 0.5 - 0.5 * cos(frac * 3.14159265359);
            return mix(c0, c1, f);
        }
        case 4u: { // OkLab (perceptual)
            let a = linear_srgb_to_oklab(srgb_to_linear(c0));
            let b = linear_srgb_to_oklab(srgb_to_linear(c1));
            return linear_to_srgb(oklab_to_linear_srgb(mix(a, b, frac)));
        }
        default: {
            return mix(c0, c1, frac);
        }
    }
}

fn palette_lookup(t_in: f32) -> vec3f {
    let tex_width = f32(uni.palette_tex_width);
    let palette_len = f32(uni.palette_len);

    var t = t_in * palette_len * uni.palette_cycles + (uni.palette_offset * palette_len);
    t = t % tex_width;

    let i0 = i32(floor(t));
    let i1 = i32(min(f32(i0 + 1), tex_width - 1.0));

    let frac = fract(t);

    let c0 = textureLoad(palette_tex, vec2<i32>(i0, 0), 0).rgb;
    let c1 = textureLoad(palette_tex, vec2<i32>(i1, 0), 0).rgb;

    return interp_palette(c0, c1, frac);
}

fn calculate_surface_normals(pix: vec2i) -> vec3f {
    var eps: i32 = i32(uni.neighbor_scale_multiplier);
    if (eps <= 0) {
        eps = 1;
    }

    let dx_r = textureLoad(calc_tex, pix + vec2i(eps, 0), 0).y;
    let dx_l = textureLoad(calc_tex, pix - vec2i(eps, 0), 0).y;
    let dy_t = textureLoad(calc_tex, pix + vec2i(0, eps), 0).y;
    let dy_b = textureLoad(calc_tex, pix - vec2i(0, eps), 0).y;

    let dx = dx_r - dx_l;
    let dy = dy_t - dy_b;

    let scale = ldexp(uni.scale, uni.scale_exp);
    eps *= 2;
    let grad = vec3f(dx, dy, scale * f32(eps));
    return normalize(grad);
}


// Calculate light direction from azimuth and elevation, in degres
fn light_dir(az_deg: f32, el_deg: f32) -> vec3f {
    let az = radians(az_deg);
    let el = radians(el_deg);

    let x = cos(el) * cos(az);
    let y = cos(el) * sin(az);
    let z = sin(el);

    return normalize(vec3f(x,y,z));
}

fn calculate_diffuse(d: f32, N: vec3f) -> f32 {
    let key_light  = light_dir(uni.key_light_azimuth, uni.key_light_elevation);
    let fill_light = light_dir(uni.fill_light_azimuth, uni.fill_light_elevation);
    var diffuse = uni.ambient_intensity;

    if ((uni.render_flags & ENABLE_KEY_LIGHT) != 0) {
        diffuse += uni.key_light_intensity * max(dot(N, key_light),0.0);
    }
    if ((uni.render_flags & ENABLE_FILL_LIGHT) != 0) {
        diffuse += uni.fill_light_intensity * max(dot(N, fill_light),0.0);
    }

    let view = vec3f(0.0, 0.0, 1.0);

    if ((uni.render_flags & ENABLE_SPEC) != 0) {
        // For specular lighting
        let half_vec = normalize(key_light + view);
        var spec = pow(max(dot(N, half_vec), 0.0), uni.specular_power);
        spec *= exp(-d * pow(2.0, uni.distance_multiplier));
    
        diffuse += uni.specular_intensity * spec;
    }

    if ((uni.render_flags & ENABLE_RIM) != 0) {
        let rim = pow(1.0 - max(dot(N, view),0.0), uni.rim_power * uni.rim_power);
        diffuse += rim * uni.rim_intensity;
    }

    return clamp(diffuse, 0.0, 4.0);
}

const OVERSAMPLE_GUARD = 50;

// ---------------------------------------------
// Compute entry point
// ---------------------------------------------
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));

    // Bounds checking based on real screen size
    if (pix.x >= i32(uni.render_width + OVERSAMPLE_GUARD) || pix.y >= i32(uni.render_height + OVERSAMPLE_GUARD)) {
        return;
    }

    var f_max_iters = f32(uni.max_iter);

    // Load results of the mandelbrot computation, stored in the rgba32float texture.
    let result = textureLoad(calc_tex, pix, 0);
    var it = u32(result.x);
    let fi = result.x;
    var d = result.y;
    let stripe_avg = result.z;

    let N = calculate_surface_normals(pix);

    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        f_max_iters += 1;
    }

    var t = fi / f_max_iters;

    // Adaptive (histogram) coloring: blend the linear normalization toward the
    // equalized CDF rank, with percentile black/white clipping in CDF space.
    if ((uni.render_flags & USE_HISTOGRAM) != 0) {
        let cdf = hist_equalize(fi);
        let denom = max(uni.hist_white_pct - uni.hist_black_pct, 1e-4);
        let clipped = clamp((cdf - uni.hist_black_pct) / denom, 0.0, 1.0);
        t = mix(t, clipped, uni.hist_eq_amount);
    }

    if ((uni.render_flags & USE_STRIPES) != 0) {
        t = mix(t, t + (stripe_avg - 0.5), uni.stripe_trap_arg2);
    }

    // User-costimizable mapping functions on normalized escape-time - i.e color scalar
    t = map_color_scalar(t);
    
    var color = palette_lookup(t);

    if ((uni.render_flags & DEBUG_COLORING) != 0) {
        color = vec3f(t, t*t, pow(t, 0.5));
    }

    color = pow(color, vec3f(1.0 / uni.palette_gamma));

    if ((uni.render_flags & USE_TRAPS) != 0) {
        // stripe_avg holds trap_min (min orbit distance to trap shape) when traps are active.
        let trap_val = stripe_avg;
        let tex_width = f32(uni.palette_tex_width);
        let palette_len = f32(uni.palette_len);
        var tt = trap_val * palette_len * uni.trap_palette_cycles + (uni.palette_offset * palette_len);
        tt = tt % tex_width;
        let ti0 = i32(floor(tt));
        let ti1 = i32(min(f32(ti0 + 1), tex_width - 1.0));
        let trap_color = pow(
            interp_palette(
                textureLoad(palette_tex, vec2i(ti0, 0), 0).rgb,
                textureLoad(palette_tex, vec2i(ti1, 0), 0).rgb,
                fract(tt)),
            vec3f(1.0 / uni.palette_gamma));
        // stripe_trap_arg4 is the trap blend weight [0, 1]
        color = mix(color, trap_color, uni.stripe_trap_arg4);
    }

    if ((uni.render_flags & USE_DE) != 0) {
        var diffuse = calculate_diffuse(d, N);
        color *= diffuse;
    }

    let scale = ldexp(uni.scale, uni.scale_exp);
    d /= scale; // Glow and AO seem to work better with scale as a factor

    if ((uni.render_flags & ENABLE_GLOW) != 0) {
        let glow = 1.0 / (1.0 + d * pow(2.0, uni.distance_multiplier));
        color += glow * uni.glow_intensity;
    }

    if ((uni.render_flags & ENABLE_AO) != 0) {
        // AO lighting
        let ao = exp(-d * pow(2.0, uni.distance_multiplier));
        color *= mix(uni.ao_darkness, 1.0, ao);
    }

    let trap_interior = (uni.render_flags & USE_TRAPS) != 0
                     && (uni.render_flags & USE_TRAP_INTERIOR) != 0;
    if (it >= uni.max_iter && !trap_interior) {
        color = vec3f(0.0, 0.0, 0.0);
    }

    textureStore(render_tex, pix, vec4f(color, 1.0));
}
