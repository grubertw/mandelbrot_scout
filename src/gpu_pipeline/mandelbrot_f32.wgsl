const DEBUG_COLORING: u32   = 1u << 0;
const GLITCH_FIX: u32       = 1u << 1;
const SMOOTH_COLORING: u32  = 1u << 2;
const USE_DE: u32           = 1u << 3;
const USE_STRIPES: u32      = 1u << 4;
const ENABLE_GLOW: u32      = 1u << 5;
const USE_TRAPS: u32        = 1u << 13; // overrides USE_STRIPES; shares stripe_trap_arg slots
const ENABLE_KEY_LIGHT: u32 = 1u << 6;
const ENABLE_FILL_LIGHT: u32= 1u << 7;
const ENABLE_SPEC: u32      = 1u << 8;
const ENABLE_AO: u32        = 1u << 9;
const ENABLE_RIM: u32       = 1u << 10;

const ESCAPED_BIT: u32              = 1u << 0u;
const PERTURB_BIT: u32              = 1u << 1u;
const PERTURB_ERR_BIT: u32          = 1u << 2u;
const MAX_ITER_BIT: u32             = 1u << 3u;
const ORBIT_SHIFT: u32              = 20u;

const MAX_U32: u32 = 0xFFFFFFFFu;
const MAX_P: u32 = 32u;

// Off-center debug probe (must match mandelbrot_fexp.wgsl so the shared debug
// buffer samples the same screen location regardless of which shader ran).
const PROBE_X_FRAC: f32 = 0.66;
const PROBE_Y_FRAC: f32 = 0.66;

// For multiplying 2 complex numbers,
// which is NOT the same is a matrix multiply (or dot product)
fn c32_mul(a: vec2f, b: vec2f) -> vec2f {
    let r2 = a.x * b.x;
    let i2 = a.y * b.y;
    let ri = a.x * b.y;
    let ir = a.y * b.x;
    return vec2f(r2 - i2, ri + ir);
}

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
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

@group(0) @binding(1) var noise_tex: texture_2d<f32>;

// Mandelbrot computation results are written to these textures
@group(0) @binding(2)
var calc_out_tex: texture_storage_2d<rgba32float, write>;

fn build_c_from_scene(pix: vec2i, jitter: vec2f) -> vec2f {
    let rw = f32(uni.render_width);
    let rh = f32(uni.render_height);

    let vw = uni.view_width;
    let vh = uni.view_height;

    let jittered_pix = vec2f(pix) + jitter * uni.jitter_strength;

    let u = f32(jittered_pix.x) / rw;
    let v = f32(jittered_pix.y) / rh;

    // Center → [-0.5, 0.5]
    let cu = u - 0.5;
    let cv = v - 0.5;

    // Reconstruct actual scale from FExp mantissa + exponent.
    let scale = ldexp(uni.scale, uni.scale_exp);

    // Map into world space using VIEW size
    let offset = vec2f(cu * vw, cv * vh) * scale;

    // Reconstruct center from FExp mantissa + exponent.
    let cx = ldexp(uni.center_x, uni.center_x_exp);
    let cy = ldexp(uni.center_y, uni.center_y_exp);

    return vec2f(cx, cy) + offset;
}

// Trap shapes (trap_shape values):
//   0 = circle ring       : |  |z| - arg1  |   (arg1=0 → point at origin)
//   1 = cross             :  min(|Re(z)|, |Im(z)|)
//   2 = square            :  max(|Re(z)|, |Im(z)|)
//   3 = line Re           :  |Re(z)|
//   4 = line Im           :  |Im(z)|
//   5 = log spiral        :  arg1=tightness (b, growth rate), arg2=arm count
fn trap_dist(z: vec2f) -> f32 {
    switch (uni.trap_shape) {
        case 1u: { return min(abs(z.x), abs(z.y)); }
        case 2u: { return max(abs(z.x), abs(z.y)); }
        case 3u: { return abs(z.x); }
        case 4u: { return abs(z.y); }
        case 5u: { // logarithmic spiral
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

fn mandelbrot(c: vec2f) -> vec4f {
    var z = vec2f(0.0, 0.0);

    var i: u32 = 0u;
    let max_i: u32 = uni.max_iter;
    var mag_z: f32 = 0.0;
    // Extra tracking for DE/surface normals
    var escape_mag_z: f32 = 0.0;
    var dz = vec2f(0.0, 0.0);
    var extra: u32 = 0u; // Iterate a few past bailout for better DE values.
    // For stripe-averaging
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;
    // For orbit traps (only one of stripes/traps is active at a time)
    var trap_min: f32 = 1e30;
    var trap_sum: f32 = 0.0;   // smooth accumulation path (sharpness > 0)
    var trap_count: f32 = 0.0;
    let trap_skip: u32 = u32(f32(max_i) * uni.trap_iter_skip_frac);
    let trap_sharpness: f32 = uni.stripe_trap_arg3;
    var flags: u32 = 0u;

    for (i = 0u; i < max_i; i = i + 1u) {
        if ((uni.render_flags & USE_DE) != 0) {
            // Derivitive tracking for DE
            dz = 2.0 * c32_mul(dz, z) + vec2f(1.0, 0.0);
        }

        // update z
        z = c32_mul(z, z) + c;

        if ((uni.render_flags & USE_TRAPS) != 0 && i >= trap_skip) {
            let td = trap_dist(z);
            if (trap_sharpness > 0.0) {
                trap_sum += exp(-td * trap_sharpness);
                trap_count += 1.0;
            } else {
                trap_min = min(trap_min, td);
            }
        } else if ((uni.render_flags & USE_STRIPES) != 0) {
            let angle = atan2(z.y, z.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_trap_arg1);
            stripe = pow(stripe, uni.stripe_trap_arg3);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        // Bailout
        mag_z = length(z);
        if (mag_z > 32.0) {
            flags |= ESCAPED_BIT;
            // Make extra iterations past escape for better DE approximation.
            if (extra >= 2) {
                break;
            }
            extra += 1;
        }

        if ((flags & ESCAPED_BIT) == 0) {
            escape_mag_z = mag_z;
        }
    }

    if (i == max_i) {
        flags |= MAX_ITER_BIT;
    }

    var fi = f32(i - extra);
    // Replace with smooth iters if enabled
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        let safe_mag_z = max(escape_mag_z, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag_z) * 0.5), 0.0, f32(max_i));
        if (i == max_i) {
            fi = f32(max_i);
        }
    }
    // Finalize DE
    var distance: f32 = 0.0;
    if ((uni.render_flags & USE_DE) != 0) {
        let r = mag_z;
        let dr = max(length(dz), 1e-30);
        distance = 0.5 * r * log(r) / dr;
    }
    // Channel z carries either trap value or stripe average (mutually exclusive).
    var chan_z: f32 = 0.0;
    if ((uni.render_flags & USE_TRAPS) != 0) {
        if (trap_sharpness > 0.0) {
            chan_z = trap_sum / max(trap_count, 1.0);
        } else {
            chan_z = trap_min;
        }
    } else if ((uni.render_flags & USE_STRIPES) != 0) {
        chan_z = stripe_sum / stripe_count;
    }

    return vec4(fi, distance, chan_z, f32(flags));
}

struct GpuRefOrbitLocation {
    c_ref_re:               f32,
    c_ref_im:               f32,
    max_ref_iters:          u32,
    center_offset_re:       f32,
    center_offset_re_exp:   i32,
    center_offset_im:       f32,
    center_offset_im_exp:   i32,
};

// Location information for each Reference Orbit
// For perturbation to succed without precision loss, both the reference orbit's
// location, AND it's delta from screen/viewport center must be sent to the GPU.
// Note that screen center is always tracked by the Scene using high (Rug) precision,
// And it is this value that serves as a basis for intial pixel deltas (taken
// through Df rounding pushed to the uniform). Although seeds are spawned from the
// rounded Df values, precision is 'regained' after the scout calculates a high-
// precision reference orbit (and qualifies it). The Scene then calculates a delta
// from it's (Rug) center, and pushes this as 'center_offset', PER FRAME.
// Now, delta_c can be computed 'safely', without the catastrophic cancelation that
// happens if the GPU were to attempt calculating the delta between ref orbit and
// screen center on it's own.
@group(0) @binding(3)
var<storage, read> orbit_location : array<GpuRefOrbitLocation>;

// Ranked ReferenceOrbits, in ComplexDf format
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
    return vec2i(
        i32((h >> 8u) & 127u),
        i32((h >> 16u) & 127u)
    );
}

fn get_jitter(pix: vec2i, sample_idx: u32) -> vec2f {
    let tex_size = vec2i(textureDimensions(noise_tex));

    let offset = hash2(pix, sample_idx);

    let coord = wrap(pix + offset, tex_size);
    let noise = textureLoad(noise_tex, coord, 0).xy;
    return noise * 2.0 - 1.0; // map [0,1] → [-1,1]
}

fn build_delta_c_from_orbit_location(pix: vec2i, orbit_idx: u32, jitter: vec2f) -> vec2f {
    let rw = f32(uni.render_width);
    let rh = f32(uni.render_height);

    let vw = uni.view_width;
    let vh = uni.view_height;

    let jittered_pix = vec2f(pix) + jitter * uni.jitter_strength;

    let u = f32(jittered_pix.x) / rw;
    let v = f32(jittered_pix.y) / rh;

    let cu = u - 0.5;
    let cv = v - 0.5;

    // Reconstruct actual scale from FExp mantissa + exponent.
    let scale = ldexp(uni.scale, uni.scale_exp);

    let offset = vec2f(cu * vw, cv * vh) * scale;
    let orbit = orbit_location[orbit_idx];

    // Reconstruct actual f32 offsets from FExp mantissa + exponent.
    let delta_from_center_to_ref_orb = vec2f(
        ldexp(orbit.center_offset_re, orbit.center_offset_re_exp),
        ldexp(orbit.center_offset_im, orbit.center_offset_im_exp)
    );

    return delta_from_center_to_ref_orb + offset;
}

fn load_ref_orbit(orbit_idx: u32, it: u32) -> vec2f {
    var Z_c32 = vec2f(0.0, 0.0);

    if (orbit_idx == 0u) {
        Z_c32 = rank_one_orbit[it];
    }
    else if (orbit_idx == 1u) {
        Z_c32 = rank_two_orbit[it];
    }

    return Z_c32;
}

// -------------------------------
// Mandelbrot Perturbance
// -------------------------------
fn mandelbrot_perturb(delta_c: vec2f) -> vec4f {
    var dz = vec2f(0.0, 0.0);
    var i: u32 = 0u;
    var ref_i: u32 = 0u;
    let max_i = uni.max_iter;
    let max_ref_i = orbit_location[0u].max_ref_iters;
    var mag_z: f32 = 0.0;
    // Extra tracking for DE/surface normals
    // Note, this is a derivitive of 'full z', which
    // is taken after Z + dz, but of the 'previous iteration'
    // similar to how it is in the absolute recurrance.
    var dzdc = vec2f(0.0, 0.0);
    var z = vec2f(0.0, 0.0);
    var escape_mag_z: f32 = 0.0;
    var extra: u32 = 0u; // Iterate a few past bailout for better DE values.
    // For stripe-averaging
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;
    // For orbit traps (only one of stripes/traps is active at a time)
    var trap_min: f32 = 1e30;
    var trap_sum: f32 = 0.0;   // smooth accumulation path (sharpness > 0)
    var trap_count: f32 = 0.0;
    let trap_skip: u32 = u32(f32(max_i) * uni.trap_iter_skip_frac);
    let trap_sharpness: f32 = uni.stripe_trap_arg3;
    var flags: u32 = PERTURB_BIT;

    for (i = 0u; i < max_i; i = i + 1u) {
        // Load reference orbit Z_n
        var Z = load_ref_orbit(0u, ref_i);
        ref_i += 1u;

        // λ_n = 2 * Z_n
        let lambda = Z + Z;

        // dz_{n+1} = λ_n * dz_n + dz_n^2 + Δc
        dz = c32_mul(lambda, dz) + c32_mul(dz, dz) + delta_c;

        if ((uni.render_flags & USE_DE) != 0) {
            // Derivitive tracking of 'reconstructed z'm for DE
            dzdc = 2.0 * c32_mul(dzdc, z) + vec2f(1.0, 0.0);
        }

        // Reconstructed z for escape testing
        Z = load_ref_orbit(0u, ref_i);
        z = Z + dz;

        if ((uni.render_flags & USE_TRAPS) != 0 && i >= trap_skip) {
            let td = trap_dist(z);
            if (trap_sharpness > 0.0) {
                trap_sum += exp(-td * trap_sharpness);
                trap_count += 1.0;
            } else {
                trap_min = min(trap_min, td);
            }
        } else if ((uni.render_flags & USE_STRIPES) != 0) {
            let angle = atan2(z.y, z.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_trap_arg1);
            stripe = pow(stripe, uni.stripe_trap_arg3);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        // Standard bailout
        mag_z = z.x * z.x + z.y * z.y;
        if (mag_z > 128.0) {
             flags |= ESCAPED_BIT;
            // Make extra iterations past escape for better DE approximation.
            if (extra >= 2) {
                break;
            }
            extra += 1;
        }

        if ((flags & ESCAPED_BIT) == 0) {
            escape_mag_z = mag_z;
        }

        let mag_dz = dz.x * dz.x + dz.y * dz.y;

        // Detect numerical error from |dz|/|z|
        // i.e. Zhouran's glitch detection, where same RefOrb can be reused.
        //let ratio = mag_dz / (mag_z + 1e-30);
        //max_glitch_ratio = max(max_glitch_ratio, ratio);
        if ( ((uni.render_flags & GLITCH_FIX) != 0) &&
             (mag_z < mag_dz * uni.perturb_err_thresh || ref_i + 1u == max_ref_i)) {
            dz = z;
            ref_i = 0u;
            flags |= PERTURB_ERR_BIT;
        }
    }

    if (i == max_i) {
        flags |= MAX_ITER_BIT;
    }

    var fi = f32(i - extra);
    // For smooth iterations
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        let safe_mag_z = max(escape_mag_z, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag_z) * 0.5), 0.0, f32(max_i));
        if (i == max_i) {
            fi = f32(max_i);
        }
    }
    // Finalize DE calculation
    var distance: f32 = 0.0;
    if ((uni.render_flags & USE_DE) != 0) {
        let r = mag_z;
        var dr = sqrt(length(dzdc));
        distance = 0.5 * r * log(r) / dr;
    }

    // Channel z carries either trap value or stripe average (mutually exclusive).
    var chan_z: f32 = 0.0;
    if ((uni.render_flags & USE_TRAPS) != 0) {
        if (trap_sharpness > 0.0) {
            chan_z = trap_sum / max(trap_count, 1.0);
        } else {
            chan_z = trap_min;
        }
    } else if ((uni.render_flags & USE_STRIPES) != 0) {
        chan_z = stripe_sum / stripe_count;
    }

    return vec4f(fi, distance, chan_z, f32(flags));
}

// The texture sampler in display.wgsl likes to have a few more pixels to interplolate
// especially when oversampling (i.e. res_factor > 1)
const OVERSAMPLE_GUARD = 50;
const OVERSAMPLE_EPSILON = 1e-4;

// ---------------------------------------------
// Compute entry point
// ---------------------------------------------
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));
    var c_for_log = vec2f(0.0, 0.0);

    // Bounds checking based on real screen size
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

        // Check if there is a (qualified) reference orbit available and,
        // use the perturbance path, if found.
        if (uni.ref_orb_count > 0) {
            // Attempt perturbance with the 1st qualified orbit
            let delta_c = build_delta_c_from_orbit_location(pix, 0u, jitter);
            results = mandelbrot_perturb(delta_c);
            c_for_log = delta_c;
        }
        else {
            let c = build_c_from_scene(pix, jitter);
            results = mandelbrot(c);
            c_for_log = c;
        }

        max_iters = max(results.x, max_iters);
        min_iters = min(results.x, min_iters);
     
        accum_dist += results.y;
        accum_stripe += results.z;
        accum_flags |= u32(results.w);
    }

    // Average
    let sc = f32(uni.sample_count);
    let accum_iters = mix(min_iters, max_iters, uni.sample_avg_bias);
    accum_dist /= sc;
    accum_stripe /= sc;

    textureStore(calc_out_tex, pix,
        vec4f(accum_iters, accum_dist, accum_stripe, f32(accum_flags)));

    // -- DEBUG --
    if (   pix.x == i32(f32(uni.render_width) * PROBE_X_FRAC)
        && pix.y == i32(f32(uni.render_height) * PROBE_Y_FRAC) ) {
        // exponents 0: the f32 shader's center is a plain world coordinate.
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
        debug_out.bla_max_step     = 0u;  // BLA is FExp-path only
        debug_out.bla_step_count   = 0u;
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
