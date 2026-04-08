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
    center_y:           f32,
    scale:              f32,
    max_iter:           u32,
    ref_orb_count:      u32,
    view_width:         f32,
    view_height:        f32,
    render_width:       u32,
    render_height:      u32,
    render_tex_width:   f32,
    render_tex_height:  f32,
    grid_size:          u32,
    grid_width:         u32,
    render_flags:       u32,
    stripe_density:     f32,
    stripe_strength:    f32,
    stripe_gamma:       f32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

const DEBUG_COLORING: u32   = 1u << 0;
const GLITCH_FIX: u32       = 1u << 1;
const SMOOTH_COLORING: u32  = 1u << 2;
const USE_DE: u32           = 1u << 3;
const USE_STRIPES: u32      = 1u << 4;
const ENABLE_GLOW: u32      = 1u << 5;
const ENABLE_KEY_LIGHT: u32 = 1u << 6;
const ENABLE_FILL_LIGHT: u32= 1u << 7;
const ENABLE_SPEC: u32      = 1u << 8;
const ENABLE_AO: u32        = 1u << 9;
const ENABLE_RIM: u32       = 1u << 10;

fn build_c_from_scene(pix: vec2i) -> vec2f {
    let rw = f32(uni.render_width);
    let rh = f32(uni.render_height);

    let vw = uni.view_width;
    let vh = uni.view_height;

    // Normalize pixel → [0,1]
    let u = (f32(pix.x) + 0.5) / rw;
    let v = (f32(pix.y) + 0.5) / rh;

    // Center → [-0.5, 0.5]
    let cu = u - 0.5;
    let cv = v - 0.5;

    // Map into world space using VIEW size
    let offset = vec2f(cu * vw, cv * vh) * uni.scale;

    return vec2f(uni.center_x, uni.center_y) + offset;
}

const ESCAPED_BIT: u32              = 1u << 0u;
const PERTURB_BIT: u32              = 1u << 1u;
const PERTURB_ERR_INNER_BIT: u32    = 1u << 2u;
const PERTURB_ERR_OUTER_BIT: u32    = 1u << 3u;
const MAX_ITER_BIT: u32             = 1u << 4u;
const ORBIT_SHIFT: u32              = 20u;

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
    var flags: u32 = 0u;

    for (i = 0u; i < max_i; i = i + 1u) {
        if ((uni.render_flags & USE_DE) != 0) {
            // Derivitive tracking for DE
            dz = 2.0 * c32_mul(dz, z) + vec2f(1.0, 0.0);
        }

        // update z
        z = c32_mul(z, z) + c;

        if ((uni.render_flags & USE_STRIPES) != 0) {
            // for stripe-averaging
            let angle = atan2(z.y, z.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
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
    // final stripe average
    var stripe_avg: f32 = 0.0;
    if ((uni.render_flags & USE_STRIPES) != 0) {
        stripe_avg = stripe_sum / stripe_count;
    }
    return vec4f(fi, distance, stripe_avg, f32(flags));
}

// All mandelbrot computation results are written to this texture
@group(0) @binding(1)
var calc_out_tex: texture_storage_2d<rgba32float, write>;

struct GpuRefOrbitLocation {
    c_ref_re:           f32,
    c_ref_im:           f32,
    r_valid:            f32,
    max_ref_iters:      u32,
    center_offset_re:   f32,
    center_offset_im:   f32,
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
@group(0) @binding(2)
var<storage, read> orbit_location : array<GpuRefOrbitLocation>;

// Ranked ReferenceOrbits, in ComplexDf format
@group(0) @binding(3)
var<storage, read> rank_one_orbit: array<vec2f>;

@group(0) @binding(4)
var<storage, read> rank_two_orbit: array<vec2f>;

fn build_delta_c_from_orbit_location(pix: vec2i, orbit_idx: u32) -> vec2f {
    let rw = f32(uni.render_width);
    let rh = f32(uni.render_height);

    let vw = uni.view_width;
    let vh = uni.view_height;

    let u = (f32(pix.x) + 0.5) / rw;
    let v = (f32(pix.y) + 0.5) / rh;

    let cu = u - 0.5;
    let cv = v - 0.5;

    let offset = vec2f(cu * vw, cv * vh) * uni.scale;
    let orbit = orbit_location[orbit_idx];

    let delta_from_center_to_ref_orb = vec2f(
        orbit.center_offset_re,
        orbit.center_offset_im
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

const BETA = 0.001;

// -------------------------------
// Mandelbrot Perturbance
// -------------------------------
fn mandelbrot_perturb(delta_c: vec2f) -> vec4f {
    var dz = vec2f(0.0, 0.0);
    var i: u32 = 0u;
    let max_i = uni.max_iter;
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
    var flags: u32 = PERTURB_BIT;

    for (i = 0u; i < max_i; i = i + 1u) {
        // Load reference orbit Z_n
        let Z = load_ref_orbit(0u, i);

        // λ_n = 2 * Z_n
        let lambda = Z + Z;

        // dz_{n+1} = λ_n * dz_n + dz_n^2 + Δc
        dz = c32_mul(lambda, dz) + c32_mul(dz, dz) + delta_c;

        if ((uni.render_flags & USE_DE) != 0) {
            // Derivitive tracking of 'reconstructed z'm for DE
            dzdc = 2.0 * c32_mul(dzdc, z) + vec2f(1.0, 0.0);
        }

        // Absolute z for escape testing
        z = Z + dz;

        if ((uni.render_flags & USE_STRIPES) != 0) {
            // for stripe-averaging
            let angle = atan2(z.y, z.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        // Standard bailout
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

        let mag_dz = length(dz);
        let mag_Z = length(Z);

        if (mag_z < BETA * mag_dz) {
            flags |= PERTURB_ERR_INNER_BIT;
        }
        if (mag_dz > BETA * mag_Z) {
            flags |= PERTURB_ERR_OUTER_BIT;
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

    // final stripe average
    var stripe_avg: f32 = 0.0;
    if ((uni.render_flags & USE_STRIPES) != 0) {
        stripe_avg = stripe_sum / stripe_count;
    }
    return vec4f(fi, distance, stripe_avg, f32(flags));
}

// The texture sampler in display.wgsl likes to have a few more pixels to interplolate
// especially when oversampling (i.e. res_factor > 1)
const OVERSAMPLE_GUARD = 50;

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

    var results = vec4f(0.0, 0.0, 0.0, 0.0);

    // Check if there is a (qualified) reference orbit available and,
    // use the perturbance path, if found.
    if (uni.ref_orb_count > 0) {
        // Attempt perturbance with the 1st qualified orbit
        let delta_c = build_delta_c_from_orbit_location(pix, 0u);
        results = mandelbrot_perturb(delta_c);
        c_for_log = delta_c;

    }
    else {
        let c = build_c_from_scene(pix);
        results = mandelbrot(c);
        c_for_log = c;
    }

    textureStore(calc_out_tex, pix, results);

    // -- DEBUG --
    if (   pix.x == i32(f32(uni.render_width) * 0.5)
        && pix.y == i32(f32(uni.render_height) * 0.5) ) {
        debug_out.center_x = c_for_log.x;
        debug_out.center_y = c_for_log.y;
        debug_out.max_iters = uni.max_iter;
        debug_out.fi = results.x;
        debug_out.distance = results.y;
        debug_out.stripe_avg = results.z;
        debug_out.flags = u32(results.w);
    }
}

struct DebugOut {
    center_x:           f32,
    center_y:           f32,
    max_iters:          u32,
    fi:                 f32,
    distance:           f32,
    stripe_avg:         f32,
    flags:              u32,
};

@group(1) @binding(0)
var<storage, read_write> debug_out: DebugOut;
