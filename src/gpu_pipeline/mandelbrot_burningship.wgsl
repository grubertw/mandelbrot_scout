// Burning Ship (f32) — single reference, exact perturbation via `diffabs`, plus
// ordinary Zhuoran rebasing. Ported from Fraktaler-3 (src/hybrid.h, floatexp.h).
// NO sign-glitch detection and NO multi-reference are needed. BLA deferred (mat2).
//
// z_{n+1} = (|Re z| + i|Im z|)^2 + c
//         = ( x^2 - y^2 + Re c ,  2|x||y| + Im c )
//
// Perturbation is EXACT: fold the delta with diffabs and the reference with abs,
// then apply the usual square perturbation. With W = (|X|,|Y|) and
// dw = (diffabs(X,dx), diffabs(Y,dy)):
//   dz' = (2 W + dw) * dw + dc      (complex multiply)
// diffabs(c,d) = |c+d| - |c| handles every sign case (including axis crossings)
// in closed form, so there is nothing to "detect". Rebasing to the critical
// point 0 is valid because diffabs(0,d) = |d|.
//
// Bindings mirror mandelbrot_f32.wgsl exactly so the same bind-group layout and
// orbit/debug buffers are reused by the pipeline.

const DEBUG_COLORING: u32   = 1u << 0;
const GLITCH_FIX: u32       = 1u << 1;
const SMOOTH_COLORING: u32  = 1u << 2;
const USE_DE: u32           = 1u << 3;
const USE_STRIPES: u32      = 1u << 4;
const USE_TRAPS: u32        = 1u << 13;

const ESCAPED_BIT: u32       = 1u << 0u;
const PERTURB_BIT: u32       = 1u << 1u;
const PERTURB_ERR_BIT: u32   = 1u << 2u;  // a Zhuoran rebase happened
const MAX_ITER_BIT: u32      = 1u << 3u;

const BAILOUT: f32 = 128.0;

// Off-center debug probe (matches the other shaders).
const PROBE_X_FRAC: f32 = 0.66;
const PROBE_Y_FRAC: f32 = 0.66;

fn c32_mul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Exact perturbation of |.|: diffabs(c, d) = |c + d| - |c|, all sign cases.
// Ported verbatim from Fraktaler-3 (src/floatexp.h). This is what makes Burning
// Ship perturbation exact — the axis fold needs no glitch detection.
fn diffabs(c: f32, d: f32) -> f32 {
    let cd = c + d;
    let c2d = 2.0 * c + d;
    if (c >= 0.0) { return select(-c2d, d, cd >= 0.0); }
    return select(-d, c2d, cd > 0.0);
}

// Continuous (smooth) iteration offset; burning ship escapes at rate ~2, so the
// power-2 form is right (formula_power is 2 for BurningShip). `mag` is the FIRST
// magnitude past `bailout` (same |z|^2 convention as the callers), so
// log(mag)/log(bailout) >= 1 and log(log(...)) is finite — no NaN near |z|=1.
// See mandelbrot_f32.wgsl for the full rationale.
fn smooth_offset(mag: f32, bailout: f32) -> f32 {
    return 1.0 - log(log(mag) / log(bailout)) / log(f32(uni.formula_power));
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
    rot_cos:            f32,
    rot_sin:            f32,
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

// Rotate a world-space offset by the viewport rotation (identity when rot_cos=1).
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
// Formula: burning ship  z' = (|x| + i|y|)^2 + c
// -------------------------------

// Absolute step (naive path).
fn bship_step(z: vec2f, c: vec2f) -> vec2f {
    return vec2f(z.x * z.x - z.y * z.y + c.x,
                 2.0 * abs(z.x) * abs(z.y) + c.y);
}

// Exact Burning Ship perturbation advance (Fraktaler-3 method): fold the delta
// via diffabs and the reference via abs, then the ordinary square perturbation
//   dz' = (2 W + dw) * dw + dc,   W = |Z|-folded reference, dw = folded delta.
fn bship_perturb_step(Z: vec2f, dz: vec2f, dc: vec2f) -> vec2f {
    let dw = vec2f(diffabs(Z.x, dz.x), diffabs(Z.y, dz.y));
    let W  = vec2f(abs(Z.x), abs(Z.y));
    return c32_mul(2.0 * W + dw, dw) + dc;
}

// -------------------------------
// Naive (no reference orbit)
// -------------------------------
fn burningship(pix: vec2f) -> vec4f {
    var z = vec2f(0.0, 0.0);
    let c = pix;
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
        z = bship_step(z, c);

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
        if ((flags & ESCAPED_BIT) == 0) {
            if (mag_z > BAILOUT) {
                escape_mag_z = mag_z;   // first magnitude past bailout (finite)
                flags |= ESCAPED_BIT;
            }
        } else {
            // Count/break on the flag (NOT mag_z) so an overflowed |z| (inf/NaN at
            // high power) can't skip the break and stall the loop to max_iter. See
            // mandelbrot_f32.wgsl for the full rationale.
            extra += 1;
            if (extra >= 2) { break; }
        }
    }

    if (i == max_i) { flags |= MAX_ITER_BIT; }

    var fi = f32(i - extra);
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        fi = clamp(fi + smooth_offset(max(escape_mag_z, BAILOUT + 1e-3), BAILOUT), 0.0, f32(max_i));
        if (i == max_i) { fi = f32(max_i); }
    }

    var chan_z: f32 = 0.0;
    if ((uni.render_flags & USE_TRAPS) != 0) {
        if (trap_sharpness > 0.0) { chan_z = trap_sum / max(trap_count, 1.0); }
        else { chan_z = trap_min; }
    } else if ((uni.render_flags & USE_STRIPES) != 0) {
        chan_z = stripe_sum / stripe_count;
    }
    return vec4f(fi, 0.0, chan_z, f32(flags)); // distance=0 (DE deferred: mat2)
}

// -------------------------------
// Perturbation (single reference, sign-tracked)
// -------------------------------
fn burningship_perturb(delta_c: vec2f) -> vec4f {
    var dz = vec2f(0.0, 0.0);   // parameter plane: z0 = 0
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
    let glitch_fix = (uni.render_flags & GLITCH_FIX) != 0;
    var flags: u32 = PERTURB_BIT;

    for (i = 0u; i < max_i; i = i + 1u) {
        if (ref_i >= max_ref_i) { break; } // OOB guard
        var Z = load_ref_orbit(0u, ref_i);

        // Zhuoran rebase (Fraktaler-3, hybrid.h): when the reconstructed value is
        // smaller than the delta (or the reference is exhausted), restart the
        // reference at the critical point 0. Valid for Burning Ship because
        // diffabs(0, d) = |d| — the fold at 0 is exact. Gated on GLITCH_FIX so the
        // raw (un-rebased) perturbation can still be shown.
        if (glitch_fix) {
            let z_full = Z + dz;
            if (dot(z_full, z_full) < dot(dz, dz) || ref_i + 1u >= max_ref_i) {
                dz    = z_full;
                ref_i = 0u;
                Z     = load_ref_orbit(0u, 0u);  // = 0 (critical point)
                flags |= PERTURB_ERR_BIT;
            }
        }

        let z = Z + dz;   // reconstructed pixel value z_n = Z_n + dz_n

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
        if ((flags & ESCAPED_BIT) == 0) {
            if (mag_z > BAILOUT) {
                escape_mag_z = mag_z;   // first magnitude past bailout (finite)
                flags |= ESCAPED_BIT;
            }
        } else {
            // Count/break on the flag (NOT mag_z) so an overflowed |z| (inf/NaN at
            // high power) can't skip the break and stall the loop to max_iter. See
            // mandelbrot_f32.wgsl for the full rationale.
            extra += 1;
            if (extra >= 2) { break; }
        }

        dz = bship_perturb_step(Z, dz, delta_c);
        ref_i += 1u;
    }

    if (i == max_i) { flags |= MAX_ITER_BIT; }

    var fi = f32(i - extra);
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        fi = clamp(fi + smooth_offset(max(escape_mag_z, BAILOUT + 1e-3), BAILOUT), 0.0, f32(max_i));
        if (i == max_i) { fi = f32(max_i); }
    }

    var chan_z: f32 = 0.0;
    if ((uni.render_flags & USE_TRAPS) != 0) {
        if (trap_sharpness > 0.0) { chan_z = trap_sum / max(trap_count, 1.0); }
        else { chan_z = trap_min; }
    } else if ((uni.render_flags & USE_STRIPES) != 0) {
        chan_z = stripe_sum / stripe_count;
    }
    return vec4f(fi, 0.0, chan_z, f32(flags)); // distance=0 (DE deferred: mat2)
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
        var results = vec4f(0.0, 0.0, 0.0, 0.0);
        let jitter = get_jitter(pix, sample_idx);

        if (uni.ref_orb_count > 0) {
            let delta_c = build_delta_c_from_orbit_location(pix, 0u, jitter);
            results = burningship_perturb(delta_c);
            c_for_log = delta_c;
        } else {
            let c = build_c_from_scene(pix, jitter);
            results = burningship(c);
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
