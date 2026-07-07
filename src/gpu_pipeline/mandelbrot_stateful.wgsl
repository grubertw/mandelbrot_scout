// Stateful holomorphic formulas (f32) — formulas that carry extra iteration
// state beyond z, so they can't be expressed by the de-baked f_step helpers.
// Currently: Manowar (c-plane), naive + perturbation (Option A: no rebasing).
//
// Manowar:  z_{n+1} = z_n^2 + z_{n-1} + c   (holomorphic, second-order)
//           seed z_0 = c, z_{-1} = c        (c = pixel, parameter plane)
//
// Bindings mirror mandelbrot_f32.wgsl exactly so the same bind-group layout and
// buffers are reused by the pipeline (the orbit buffers go unused on the naive
// path but are declared for layout compatibility).

const DEBUG_COLORING: u32   = 1u << 0;
const GLITCH_FIX: u32       = 1u << 1;
const SMOOTH_COLORING: u32  = 1u << 2;
const USE_DE: u32           = 1u << 3;
const USE_STRIPES: u32      = 1u << 4;
const USE_TRAPS: u32        = 1u << 13;

const ESCAPED_BIT: u32       = 1u << 0u;
const PERTURB_BIT: u32       = 1u << 1u;
const MAX_ITER_BIT: u32      = 1u << 3u;

const BAILOUT: f32 = 128.0;

const PROBE_X_FRAC: f32 = 0.66;
const PROBE_Y_FRAC: f32 = 0.66;

fn c32_mul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Smooth iteration offset (escape rate ~2 for these formulas).
fn smooth_offset(mag2: f32) -> f32 {
    return 1.0 - log(log(mag2) * 0.5) / log(2.0);
}

// -------------------------------
// Uniforms (layout identical to mandelbrot_f32.wgsl's prefix)
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
    formula_power:       u32,
    julia_c_re:          f32,
    julia_c_im:          f32,
    rot_cos:             f32,
    rot_sin:             f32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

@group(0) @binding(1) var noise_tex: texture_2d<f32>;

@group(0) @binding(2)
var calc_out_tex: texture_storage_2d<rgba32float, write>;

struct GpuRefOrbitLocation {
    c_ref_re:               f32,
    c_ref_im:               f32,
    max_ref_iters:          u32,
    center_offset_re:       f32,
    center_offset_re_exp:   i32,
    center_offset_im:       f32,
    center_offset_im_exp:   i32,
};
@group(0) @binding(3)
var<storage, read> orbit_location : array<GpuRefOrbitLocation>;

@group(0) @binding(4)
var<storage, read> rank_one_orbit: array<vec2f>;

@group(0) @binding(5)
var<storage, read> rank_two_orbit: array<vec2f>;

fn wrap(coord: vec2i, size: vec2i) -> vec2i {
    return vec2i(
        ((coord.x % size.x) + size.x) % size.x,
        ((coord.y % size.y) + size.y) % size.y
    );
}

fn hash2(p: vec2i, s: u32) -> vec2i {
    let h = u32(p.x) * 1664525u + u32(p.y) * 1013904223u + s * 374761393u;
    return vec2i(i32((h >> 8u) & 127u), i32((h >> 16u) & 127u));
}

fn get_jitter(pix: vec2i, sample_idx: u32) -> vec2f {
    let tex_size = vec2i(textureDimensions(noise_tex));
    let offset = hash2(pix, sample_idx);
    let coord = wrap(pix + offset, tex_size);
    let noise = textureLoad(noise_tex, coord, 0).xy;
    return noise * 2.0 - 1.0;
}

fn rotate_off(o: vec2f) -> vec2f {
    return vec2f(o.x * uni.rot_cos - o.y * uni.rot_sin,
                 o.x * uni.rot_sin + o.y * uni.rot_cos);
}

fn build_c_from_scene(pix: vec2i, jitter: vec2f) -> vec2f {
    let rw = f32(uni.render_width);
    let rh = f32(uni.render_height);
    let jittered_pix = vec2f(pix) + jitter * uni.jitter_strength;
    let u = f32(jittered_pix.x) / rw;
    let v = f32(jittered_pix.y) / rh;
    let cu = u - 0.5;
    let cv = v - 0.5;
    let scale = ldexp(uni.scale, uni.scale_exp);
    let offset = rotate_off(vec2f(cu * uni.view_width, cv * uni.view_height)) * scale;
    return vec2f(ldexp(uni.center_x, uni.center_x_exp),
                 ldexp(uni.center_y, uni.center_y_exp)) + offset;
}

fn build_delta_c_from_orbit_location(pix: vec2i, orbit_idx: u32, jitter: vec2f) -> vec2f {
    let rw = f32(uni.render_width);
    let rh = f32(uni.render_height);
    let jittered_pix = vec2f(pix) + jitter * uni.jitter_strength;
    let u = f32(jittered_pix.x) / rw;
    let v = f32(jittered_pix.y) / rh;
    let cu = u - 0.5;
    let cv = v - 0.5;
    let scale = ldexp(uni.scale, uni.scale_exp);
    let offset = rotate_off(vec2f(cu * uni.view_width, cv * uni.view_height)) * scale;
    let orbit = orbit_location[orbit_idx];
    let delta_from_center_to_ref_orb = vec2f(
        ldexp(orbit.center_offset_re, orbit.center_offset_re_exp),
        ldexp(orbit.center_offset_im, orbit.center_offset_im_exp)
    );
    return delta_from_center_to_ref_orb + offset;
}

fn load_ref_orbit(orbit_idx: u32, it: u32) -> vec2f {
    var Z_c32 = vec2f(0.0, 0.0);
    if (orbit_idx == 0u) { Z_c32 = rank_one_orbit[it]; }
    else if (orbit_idx == 1u) { Z_c32 = rank_two_orbit[it]; }
    return Z_c32;
}

fn trap_dist(z: vec2f) -> f32 {
    switch (uni.trap_shape) {
        case 1u: { return min(abs(z.x), abs(z.y)); }
        case 2u: { return max(abs(z.x), abs(z.y)); }
        case 3u: { return abs(z.x); }
        case 4u: { return abs(z.y); }
        case 5u: {
            let r = length(z);
            if (r < 1e-10) { return 1e10; }
            let theta = atan2(z.y, z.x);
            let b = max(uni.stripe_trap_arg1, 0.01);
            let arms = max(uni.stripe_trap_arg2, 1.0);
            let phase = (log(r) / b - theta) * arms / 6.283185307;
            let frac = phase - floor(phase);
            return min(frac, 1.0 - frac) * 2.0;
        }
        default: { return abs(length(z) - uni.stripe_trap_arg1); }
    }
}

// -------------------------------
// Manowar (naive): z_{n+1} = z_n^2 + z_{n-1} + c,  seed z_0 = z_{-1} = c.
// -------------------------------
fn manowar(c: vec2f) -> vec4f {
    var z = c;         // z_0 = c (parameter plane)
    var z_prev = c;    // z_{-1} = z_0
    var i: u32 = 0u;
    let max_i = uni.max_iter;
    var mag_z: f32 = 0.0;
    var escape_mag_z: f32 = 0.0;
    var extra: u32 = 0u;
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;
    var trap_min: f32 = 1e30;
    var trap_sum: f32 = 0.0;
    var trap_count: f32 = 0.0;
    let trap_skip: u32 = u32(f32(max_i) * uni.trap_iter_skip_frac);
    let trap_sharpness: f32 = uni.stripe_trap_arg3;
    var flags: u32 = 0u;

    for (i = 0u; i < max_i; i = i + 1u) {
        let z_new = c32_mul(z, z) + z_prev + c;
        z_prev = z;
        z = z_new;

        if ((uni.render_flags & USE_TRAPS) != 0 && i >= trap_skip) {
            let td = trap_dist(z);
            if (trap_sharpness > 0.0) { trap_sum += exp(-td * trap_sharpness); trap_count += 1.0; }
            else { trap_min = min(trap_min, td); }
        } else if ((uni.render_flags & USE_STRIPES) != 0) {
            let angle = atan2(z.y, z.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_trap_arg1);
            stripe = pow(stripe, uni.stripe_trap_arg3);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        mag_z = z.x * z.x + z.y * z.y;
        if (mag_z > BAILOUT) {
            flags |= ESCAPED_BIT;
            if (extra >= 2) { break; }
            extra += 1;
        }
        if ((flags & ESCAPED_BIT) == 0) { escape_mag_z = mag_z; }
    }

    if (i == max_i) { flags |= MAX_ITER_BIT; }

    var fi = f32(i - extra);
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        fi = clamp(fi + smooth_offset(max(escape_mag_z, 1e-30)), 0.0, f32(max_i));
        if (i == max_i) { fi = f32(max_i); }
    }

    var chan_z: f32 = 0.0;
    if ((uni.render_flags & USE_TRAPS) != 0) {
        if (trap_sharpness > 0.0) { chan_z = trap_sum / max(trap_count, 1.0); }
        else { chan_z = trap_min; }
    } else if ((uni.render_flags & USE_STRIPES) != 0) {
        chan_z = stripe_sum / stripe_count;
    }
    return vec4f(fi, 0.0, chan_z, f32(flags)); // distance=0 (DE deferred)
}

// -------------------------------
// Manowar perturbation (Option A: no rebasing). Seed z_0 = c, so
// dz_0 = dz_{-1} = dc = delta_c (like Julia). Advance:
//   dz_{n+1} = 2*Z_n*dz_n + dz_n^2 + dz_{n-1} + dc
// z_{n-1}'s reference term cancels; only the previous DELTA is carried. No glitch
// rebasing yet: relies on a high-iteration reference staying well-matched (dz
// never overtaking z). Glitches at extreme depth are the known limit, like
// Mandelbrot with GLITCH_FIX off.
// -------------------------------
fn manowar_perturb(delta_c: vec2f) -> vec4f {
    var dz = delta_c;       // dz_0    = c_pixel - c_ref
    var dz_prev = delta_c;  // dz_{-1} = dz_0
    var i: u32 = 0u;
    var ref_i: u32 = 0u;
    let max_i = uni.max_iter;
    let max_ref_i = orbit_location[0u].max_ref_iters;
    var mag_z: f32 = 0.0;
    var escape_mag_z: f32 = 0.0;
    var extra: u32 = 0u;
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;
    var trap_min: f32 = 1e30;
    var trap_sum: f32 = 0.0;
    var trap_count: f32 = 0.0;
    let trap_skip: u32 = u32(f32(max_i) * uni.trap_iter_skip_frac);
    let trap_sharpness: f32 = uni.stripe_trap_arg3;
    var flags: u32 = PERTURB_BIT;

    for (i = 0u; i < max_i; i = i + 1u) {
        if (ref_i >= max_ref_i) { break; } // OOB guard (no rebasing)
        let Z = load_ref_orbit(0u, ref_i);
        let z = Z + dz;   // reconstructed pixel value z_n

        if ((uni.render_flags & USE_TRAPS) != 0 && i >= trap_skip) {
            let td = trap_dist(z);
            if (trap_sharpness > 0.0) { trap_sum += exp(-td * trap_sharpness); trap_count += 1.0; }
            else { trap_min = min(trap_min, td); }
        } else if ((uni.render_flags & USE_STRIPES) != 0) {
            let angle = atan2(z.y, z.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_trap_arg1);
            stripe = pow(stripe, uni.stripe_trap_arg3);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        mag_z = z.x * z.x + z.y * z.y;
        if (mag_z > BAILOUT) {
            flags |= ESCAPED_BIT;
            if (extra >= 2) { break; }
            extra += 1;
        }
        if ((flags & ESCAPED_BIT) == 0) { escape_mag_z = mag_z; }

        let dz_new = c32_mul(Z + Z, dz) + c32_mul(dz, dz) + dz_prev + delta_c;
        dz_prev = dz;
        dz = dz_new;
        ref_i += 1u;
    }

    if (i == max_i) { flags |= MAX_ITER_BIT; }

    var fi = f32(i - extra);
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        fi = clamp(fi + smooth_offset(max(escape_mag_z, 1e-30)), 0.0, f32(max_i));
        if (i == max_i) { fi = f32(max_i); }
    }

    var chan_z: f32 = 0.0;
    if ((uni.render_flags & USE_TRAPS) != 0) {
        if (trap_sharpness > 0.0) { chan_z = trap_sum / max(trap_count, 1.0); }
        else { chan_z = trap_min; }
    } else if ((uni.render_flags & USE_STRIPES) != 0) {
        chan_z = stripe_sum / stripe_count;
    }
    return vec4f(fi, 0.0, chan_z, f32(flags));
}

const OVERSAMPLE_GUARD = 50;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));
    var c_for_log = vec2f(0.0, 0.0);

    if (pix.x >= i32(uni.render_width + OVERSAMPLE_GUARD) || pix.y >= i32(uni.render_height + OVERSAMPLE_GUARD)) {
        return;
    }

    var sample_idx: u32 = 0u;
    var max_iters: f32 = 0.0;
    var min_iters: f32 = 1e27;
    var accum_dist = 0.0;
    var accum_stripe = 0.0;
    var accum_flags = 0u;

    for (sample_idx = 0; sample_idx < uni.sample_count; sample_idx++) {
        let jitter = get_jitter(pix, sample_idx);
        var results = vec4f(0.0, 0.0, 0.0, 0.0);
        if (uni.ref_orb_count > 0) {
            let delta_c = build_delta_c_from_orbit_location(pix, 0u, jitter);
            results = manowar_perturb(delta_c);
            c_for_log = delta_c;
        } else {
            let c = build_c_from_scene(pix, jitter);
            results = manowar(c);
            c_for_log = c;
        }

        max_iters = max(results.x, max_iters);
        min_iters = min(results.x, min_iters);
        accum_dist += results.y;
        accum_stripe += results.z;
        accum_flags |= u32(results.w);
    }

    let sc = f32(uni.sample_count);
    let accum_iters = mix(min_iters, max_iters, uni.sample_avg_bias);
    accum_dist /= sc;
    accum_stripe /= sc;

    textureStore(calc_out_tex, pix,
        vec4f(accum_iters, accum_dist, accum_stripe, f32(accum_flags)));

    if (   pix.x == i32(f32(uni.render_width) * PROBE_X_FRAC)
        && pix.y == i32(f32(uni.render_height) * PROBE_Y_FRAC) ) {
        debug_out.center_x     = c_for_log.x;
        debug_out.center_x_exp = 0;
        debug_out.center_y     = c_for_log.y;
        debug_out.center_y_exp = 0;
        debug_out.scale        = uni.scale;
        debug_out.scale_exp    = uni.scale_exp;
        debug_out.max_iters    = uni.max_iter;
        debug_out.fi           = accum_iters;
        debug_out.distance     = accum_dist;
        debug_out.stripe_avg   = accum_stripe;
        debug_out.flags        = u32(accum_flags);
        debug_out.rebase_count = 0u;
        debug_out.bla_max_step      = 0u;
        debug_out.bla_step_count    = 0u;
        debug_out.bla_iters_skipped = 0u;
    }
}

struct DebugOut {
    center_x:           f32,
    center_x_exp:       i32,
    center_y:           f32,
    center_y_exp:       i32,
    scale:              f32,
    scale_exp:          i32,
    max_iters:          u32,
    fi:                 f32,
    distance:           f32,
    stripe_avg:         f32,
    flags:              u32,
    rebase_count:       u32,
    bla_max_step:       u32,
    bla_step_count:     u32,
    bla_iters_skipped:  u32,
};

@group(1) @binding(0)
var<storage, read_write> debug_out: DebugOut;
