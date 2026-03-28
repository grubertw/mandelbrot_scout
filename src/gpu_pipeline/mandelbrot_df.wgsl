// -------------------------------
// Double-float structure
// -------------------------------
struct Df {
    hi: f32,
    lo: f32,
};

struct ComplexDf {
    r: Df,
    i: Df,
};

const one: f32 = 1.0;

// -------------------------------
// Double-float arithmatic operations
// -------------------------------
fn split_df(a: f32) -> Df {
    let c = (f32(1u << 12u) + 1.0) * a;
    let a_big = c - a;
    let a_hi = c * one - a_big;
    let a_lo = a * one - a_hi;

    return Df(a_hi, a_lo);
}

// Special sum operation when a > b
fn quickTwoSum(a: f32, b: f32) -> Df {
	let x = (a + b) * one;
	let b_virt = (x - a) * one;
	let y = b - b_virt;
	return Df(x, y);
}

fn twoSum(a: f32, b: f32) -> Df {
	let x = (a + b);
	let b_virt = (x - a) * one;
	let a_virt = (x - b_virt) * one;
	let b_err = b - b_virt;
	let a_err = a - a_virt;
	let y = a_err + b_err;
	return Df(x, y);
}

fn twoSub(a: f32, b: f32) -> Df {
	let s = (a - b);
	let v = (s * one - a) * one;
	let err = (a - (s - v) * one) * one - (b + v);
	return Df(s, err);
}

fn twoProd(a: f32, b: f32) -> Df {
	let x = a * b;
	let a2 = split_df(a);
	let b2 = split_df(b);
	let err1 = x - (a2.hi * b2.hi * one) * one;
	let err2 = err1 - (a2.lo * b2.hi * one) * one;
	let err3 = err2 - (a2.hi * b2.lo * one) * one;
	let y = a2.lo * b2.lo - err3;
	return Df(x, y);
}

fn df_add(a: Df, b: Df) -> Df {
	var s = twoSum(a.hi, b.hi);
	var t = twoSum(a.lo, b.lo);
	s.lo += t.hi;
	s = quickTwoSum(s.hi, s.lo);
	s.lo += t.lo;
	s = quickTwoSum(s.hi, s.lo);
	return s;
}

fn df_sub(a: Df, b: Df) -> Df {
	var s = twoSub(a.hi, b.hi);
	var t = twoSub(a.lo, b.lo);
	s.lo += t.hi;
	s = quickTwoSum(s.hi, s.lo);
	s.lo += t.lo;
	s = quickTwoSum(s.hi, s.lo);
	return Df(s.hi, s.lo);
}

fn df_mul(a: Df, b: Df) -> Df {
	var p = twoProd(a.hi, b.hi);
	p.lo += a.hi * b.lo;
	p.lo += a.lo * b.hi;
	p = quickTwoSum(p.hi, p.lo);
	return p;
}

//fn df_add(a: Df, b: Df) -> Df {
//    let s = a.hi + b.hi;
//    let e = (a.hi - s) + b.hi + a.lo + b.lo;
//    return Df(s, e);
//}
//fn df_sub(a: Df, b: Df) -> Df {
//    let s = a.hi - b.hi;
//    let e = (a.hi - s) - b.hi + a.lo - b.lo;
//    return Df(s, e);
//}
//fn df_mul(a: Df, b: Df) -> Df {
//    let p = a.hi * b.hi;
//    let e = a.hi * b.lo + a.lo * b.hi;
//    return Df(p, e);
//}

// -------------------------------
// Convert a regular f32 into a Df (lo = 0.0)
// -------------------------------
fn df_from_f32(x: f32) -> Df {
    return Df(x, 0.0);
}

fn df_from_i32(i: i32) -> Df {
    return Df(f32(i), 0.0);
}

// multiply Df by scalar f32
fn df_mul_f32(a: Df, b: f32) -> Df {
    return df_mul(a, Df(b, 0.0));
}


fn df_mag2(v: Df) -> f32 {
    let v_abs = abs(v.hi) + abs(v.lo);
    return v_abs * v_abs;
}

fn df_mag2_upper(zx: Df, zy: Df) -> f32 {
    let ax = abs(zx.hi) + abs(zx.lo);
    let ay = abs(zy.hi) + abs(zy.lo);
    return ax * ax + ay * ay;
}

fn cdf_mag2(a: ComplexDf) -> f32 {
    let abs_r = abs(a.r.hi) + abs(a.r.lo);
    let abs_i = abs(a.i.hi) + abs(a.i.lo);
    return abs_r * abs_r + abs_i * abs_i;
}

// -------------------------------
// Complex double-float operations
// z = x + i*y
// -------------------------------
fn cdf_add(a: ComplexDf, b: ComplexDf) -> ComplexDf {
    var r: ComplexDf;
    r.r = df_add(a.r, b.r);
    r.i = df_add(a.i, b.i);
    return r;
}

fn cdf_sub(a: ComplexDf, b: ComplexDf) -> ComplexDf {
    var r: ComplexDf;
    r.r = df_sub(a.r, b.r);
    r.i = df_sub(a.i, b.i);
    return r;
}

// complex multiply: (ar + i ai) * (br + i bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
fn cdf_mul(a: ComplexDf, b: ComplexDf) -> ComplexDf {
    let rr = df_mul(a.r, b.r);
    let ii = df_mul(a.i, b.i);
    var real = df_sub(rr, ii);
    let rbi = df_mul(a.r, b.i);
    let ibr = df_mul(a.i, b.r);
    var imag = df_add(rbi, ibr);
    var out: ComplexDf;
    out.r = real;
    out.i = imag;
    return out;
}

// -------------------------------
// Uniforms
// -------------------------------
struct Uniforms {
    center_x_hi:        f32,
    center_y_hi:        f32,
    scale_hi:           f32,
    max_iter:           u32,
    ref_orb_count:      u32,
    screen_width:       u32,
    screen_height:      u32,
    grid_size:          u32,
    grid_width:         u32,
    palette_size:       u32,
    palette_frequency:  f32,
    palette_offset:     f32,
    palette_gamma:      f32,
    render_flags:       u32,
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
    stripe_density:             f32,
    stripe_strength:            f32,
    stripe_gamma:               f32,
    rim_intensity:              f32,
    rim_power:                  f32,
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

fn build_c_from_scene(pix: vec2<i32>) -> ComplexDf {
    let half_w = f32(uni.screen_width) * 0.5;
    let half_h = f32(uni.screen_height) * 0.5;

    let dx_i = pix.x - i32(half_w);
    let dy_i = i32(half_h) - pix.y; // y-axis increases downward, so must flip!

    let dx_df = df_from_i32(dx_i);
    let dy_df = df_from_i32(dy_i);

    // Scene scale
    let scale = Df(uni.scale_hi, uni.scale_lo);

    // offset = dx*scale + dy*scale
    let off_x = df_mul(dx_df, scale);
    let off_y = df_mul(dy_df, scale);

    // Scene/Viewport center
    let center_x = Df(uni.center_x_hi, uni.center_x_lo);
    let center_y = Df(uni.center_y_hi, uni.center_y_lo);

    let c = ComplexDf(df_add(center_x, off_x), df_add(center_y, off_y)); 
    return c;
}

const ESCAPED_BIT: u32              = 1u << 0u;
const PERTURB_BIT: u32              = 1u << 1u;
const PERTURB_ERR_INNER_BIT: u32    = 1u << 2u;
const PERTURB_ERR_OUTER_BIT: u32    = 1u << 3u;
const MAX_ITER_BIT: u32             = 1u << 4u;
const ORBIT_SHIFT: u32              = 20u;

// -------------------------------
// Mandelbrot iteration using DF arithmetic.
// Returns iteration count (u32).
// -------------------------------
fn mandelbrot(c: ComplexDf) -> vec4<f32> {
    var z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));

    var i: u32 = 0u;
    let max_i: u32 = uni.max_iter;
    var mag2: f32 = 0.0;
    // Extra tracking for DE/surface normals
    var escape_mag2: f32 = 0.0;
    var dz = vec2<f32>(0.0, 0.0);
    var extra: u32 = 0u; // Iterate a few past bailout for better DE values.
    // For stripe-averaging
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;
    var flags: u32 = 0u;

    for (i = 0u; i < max_i; i = i + 1u) {
        let zx2 = df_mul(z.r, z.r);
        let zy2 = df_mul(z.i, z.i);
        let zxy = df_mul(z.r, z.i);

        // real = zx2 - zy2 + c.r
        let real_part = df_add(df_sub(zx2, zy2), c.r);

        // imag = 2*zx*zy + c.i  -> 2*zxy + c.i
        let imag_part = df_add(df_add(zxy, zxy), c.i);

        if ((uni.render_flags & USE_DE) != 0) {
            // Derivitive tracking for DE
            let dz_new = vec2<f32>(
                2.0 * (z.r.hi * dz.x - z.i.hi * dz.y) + 1.0,
                2.0 * (z.r.hi * dz.y + z.i.hi * dz.x)
            );
            dz = dz_new;
        }

        // update z
        z.r = real_part;
        z.i = imag_part;

        if ((uni.render_flags & USE_STRIPES) != 0) {
            // for stripe-averaging
            let angle = atan2(z.i.hi, z.r.hi);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        // Bailout
        mag2 = cdf_mag2(z);
        if (mag2 > 1024.0) {
            flags |= ESCAPED_BIT;
            // Make extra iterations past escape for better DE approximation.
            if (extra >= 2) {
                break;
            }
            extra += 1;
        }

        if ((flags & ESCAPED_BIT) == 0) {
            escape_mag2 = mag2;
        }
    }

    if (i == max_i) {
        flags |= MAX_ITER_BIT;
    }

    var fi = f32(i - extra);
    // Replace with smooth iters if enabled
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        let safe_mag2 = max(escape_mag2, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag2) * 0.5), 0.0, f32(max_i));
        if (i == max_i) {
            fi = f32(max_i);
        }
    }
    // Finalize DE
    var distance: f32 = 0.0;
    if ((uni.render_flags & USE_DE) != 0) {
        let r = sqrt(mag2);
        let dr = max(length(dz), 1e-30);
        distance = 0.5 * r * log(r) / dr;
    }
    // final stripe average
    var stripe_avg: f32 = 0.0;
    if ((uni.render_flags & USE_STRIPES) != 0) {
        stripe_avg = stripe_sum / stripe_count;
    }
    return vec4<f32>(fi, distance, stripe_avg, f32(flags));
}

// All mandelbrot computation results are written to this texture
@group(0) @binding(1)
var calc_out_tex: texture_storage_2d<rgba32float, write>;

struct GpuRefOrbitLocation {
    c_ref_re_hi:            f32,
    c_ref_re_lo:            f32,
    c_ref_im_hi:            f32,
    c_ref_im_lo:            f32,
    max_ref_iters:          u32,
    center_offset_re_hi:    f32,
    center_offset_re_lo:    f32,
    center_offset_im_hi:    f32,
    center_offset_im_lo:    f32,
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
var<storage, read> rank_one_orbit: array<ComplexDf>;

@group(0) @binding(4)
var<storage, read> rank_two_orbit: array<ComplexDf>;

// delta_c - along with Zref-n0 - are needed to start pertubation.
fn build_delta_c_from_orbit_location(pix: vec2<i32>, orbit_idx: u32) -> ComplexDf {
    let half_w = f32(uni.screen_width) * 0.5;
    let half_h = f32(uni.screen_height) * 0.5;
    let scale = Df(uni.scale_hi, uni.scale_lo);

    let orbit = orbit_location[orbit_idx];

    let dx_i = pix.x - i32(half_w);
    let dy_i = i32(half_h) - pix.y;

    let dx_df = df_from_i32(dx_i);
    let dy_df = df_from_i32(dy_i);

    let off_x = df_mul(dx_df, scale);
    let off_y = df_mul(dy_df, scale);

    let delta_from_center_to_ref_orb = ComplexDf(
        Df(orbit.center_offset_re_hi, orbit.center_offset_re_lo),
        Df(orbit.center_offset_im_hi, orbit.center_offset_im_lo)
    );

    let delta_c = ComplexDf(
        df_add(delta_from_center_to_ref_orb.r, off_x),
        df_add(delta_from_center_to_ref_orb.i, off_y)
    );

    return delta_c;
}

fn load_ref_orbit(orbit_idx: u32, it: u32) -> ComplexDf {
    if (orbit_idx == 0u) {
        return rank_one_orbit[it];
    }
    else if (orbit_idx == 1u) {
        return rank_two_orbit[it];
    }
    else {
        return ComplexDf(Df(0.0, 0.0), Df(0.0, 0.0));
    }
}

const BETA = 0.001;

// -------------------------------
// Mandelbrot Perturbance
// -------------------------------
fn mandelbrot_perturb(delta_c: ComplexDf) -> vec4<f32> {
    var dz = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var i: u32 = 0u;
    let max_i = uni.max_iter;
    var mag2: f32 = 0.0;
    // Extra tracking for DE/surface normals
    // Note, this is a derivitive of 'full z', which
    // is taken after Z + dz, but of the 'previous iteration'
    // similar to how it is in the absolute recurrance.
    var dzdc = vec2<f32>(0.0, 0.0);
    var z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var escape_mag2: f32 = 0.0;
    var extra: u32 = 0u; // Iterate a few past bailout for better DE values.
    // For stripe-averaging
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;
    var flags: u32 = PERTURB_BIT;

    for (i = 0u; i < max_i; i = i + 1u) {
        // Load reference orbit Z_n
        let Z = load_ref_orbit(0u, i);

        // λ_n = 2 * Z_n
        let lambda = cdf_add(Z, Z);

        // dz_{n+1} = λ_n * dz_n + dz_n^2 + Δc
        let dz2 = cdf_mul(dz, dz);
        dz = cdf_add(
            cdf_add(cdf_mul(lambda, dz), dz2),
            delta_c
        );

        if ((uni.render_flags & USE_DE) != 0) {
            // Derivitive tracking of 'reconstructed z'm for DE
            let dz_new = vec2<f32>(
                2.0 * (z.r.hi * dzdc.x - z.i.hi * dzdc.y) + 1.0,
                2.0 * (z.r.hi * dzdc.y + z.i.hi * dzdc.x)
            );
            dzdc = dz_new;
        }

        // Absolute z for escape testing
        z = cdf_add(Z, dz);

        if ((uni.render_flags & USE_STRIPES) != 0) {
            // for stripe-averaging
            let angle = atan2(z.i.hi, z.r.hi);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        // Standard bailout
        mag2 = cdf_mag2(z);
        if (mag2 > 1024.0) {
             flags |= ESCAPED_BIT;
            // Make extra iterations past escape for better DE approximation.
            if (extra >= 2) {
                break;
            }
            extra += 1;
        }

        if ((flags & ESCAPED_BIT) == 0) {
            escape_mag2 = mag2;
        }

        let abs_z = sqrt(mag2);
        let abs_dz = sqrt(cdf_mag2(dz));
        let abs_Z = sqrt(cdf_mag2(Z));

        if (abs_z < BETA * abs_dz) {
            flags |= PERTURB_ERR_INNER_BIT;
        }
        if (abs_dz > BETA * abs_Z) {
            flags |= PERTURB_ERR_OUTER_BIT;
        }
    }

    if (i == max_i) {
        flags |= MAX_ITER_BIT;
    }

    var fi = f32(i - extra);
    // For smooth iterations
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        let safe_mag2 = max(escape_mag2, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag2) * 0.5), 0.0, f32(max_i));
        if (i == max_i) {
            fi = f32(max_i);
        }
    }
    // Finalize DE calculation
    var distance: f32 = 0.0;
    if ((uni.render_flags & USE_DE) != 0) {
        let r = sqrt(mag2);
        var dr = sqrt(length(dzdc));
        distance = 0.5 * r * log(r) / dr;
    }

    // final stripe average
    var stripe_avg: f32 = 0.0;
    if ((uni.render_flags & USE_STRIPES) != 0) {
        stripe_avg = stripe_sum / stripe_count;
    }
    return vec4<f32>(fi, distance, stripe_avg, f32(flags));
}

// ---------------------------------------------
// Compute entry point
// ---------------------------------------------
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2<i32>(i32(gid.x), i32(gid.y));

    // Bounds checking based on real screen size
    if (pix.x >= i32(uni.screen_width) || pix.y >= i32(uni.screen_height)) {
        return;
    }

    var results = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // Check if there is a (qualified) reference orbit available and,
    // use the perturbance path, if found.
    if (uni.ref_orb_count > 0) {
        // Attempt perturbance with the 1st qualified orbit
        let delta_c = build_delta_c_from_orbit_location(pix, 0u);
        results = mandelbrot_perturb(delta_c);

    }
    else {
        let c = build_c_from_scene(pix);
        results = mandelbrot(c);
    }

    textureStore(calc_out_tex, pix, results);

    // -- DEBUG --
    if (   pix.x == i32(f32(uni.screen_width) * 0.5)
        && pix.y == i32(f32(uni.screen_height) * 0.5) ) {
        debug_out.max_iters = uni.max_iter;
        debug_out.fi = results.x;
        debug_out.distance = results.y;
        debug_out.stripe_avg = results.z;
        debug_out.flags = u32(results.w);
    }
}

struct DebugOut {
    center_x_hi:        f32,
    center_x_lo:        f32,
    center_y_hi:        f32,
    center_y_lo:        f32,
    max_iters:          u32,
    fi:                 f32,
    distance:           f32,
    stripe_avg:         f32,
    flags:              u32,
};

@group(1) @binding(0)
var<storage, read_write> debug_out: DebugOut;
