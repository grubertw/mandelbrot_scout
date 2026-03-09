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

// -------------------------------
// Double-float arithmatic operations
// -------------------------------
fn df_add(a: Df, b: Df) -> Df {
    let s = a.hi + b.hi;
    let e = (a.hi - s) + b.hi + a.lo + b.lo;
    return Df(s, e);
}

fn df_sub(a: Df, b: Df) -> Df {
    let s = a.hi - b.hi;
    let e = (a.hi - s) - b.hi + a.lo - b.lo;
    return Df(s, e);
}

fn df_mul(a: Df, b: Df) -> Df {
    let p = a.hi * b.hi;
    let e = a.hi * b.lo + a.lo * b.hi;
    return Df(p, e);
}

fn df_div(a: Df, b: Df) -> Df {
    let p = a.hi / b.hi;
    let e = a.hi / b.lo + a.lo / b.hi;
    return Df(p, e);
}

fn df_neg(a: Df) -> Df {
    var out: Df;
    out.hi = -a.hi;
    out.lo = -a.lo;
    return out;
}

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
    center_x_lo:        f32,
    center_y_hi:        f32,
    center_y_lo:        f32,
    scale_hi:           f32,
    scale_lo:           f32,
    max_iter:           u32,
    ref_orb_count:      u32,
    screen_width:       u32,
    screen_height:      u32,
    grid_size:          u32,
    grid_width:         u32,
    palette_frequency:  f32,
    palette_offset:     f32,
    palette_gamma:      f32,
    render_flags:       u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

const DEBUG_COLORING: u32   = 1u << 0;
const GLITCH_FIX: u32       = 1u << 1;

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

// -------------------------------
// Mandelbrot iteration using DF arithmetic.
// Returns iteration count (u32).
// -------------------------------
fn mandelbrot(c: ComplexDf) -> u32 {
    var z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
 
    var i: u32 = 0u;
    let max_i: u32 = uni.max_iter;

    for (i = 0u; i < max_i; i = i + 1u) {
        let zx2 = df_mul(z.r, z.r);
        let zy2 = df_mul(z.i, z.i);
        let zxy = df_mul(z.r, z.i);

        // real = zx2 - zy2 + c.r
        let real_part = df_add(df_sub(zx2, zy2), c.r);

        // imag = 2*zx*zy + c.i  -> 2*zxy + c.i
        let imag_part = df_add(df_add(zxy, zxy), c.i);

        // update z
        z.r = real_part;
        z.i = imag_part;

        // Bailout
        let mag2 = cdf_mag2(z);
        if (mag2 > 16.0) {
            break;
        }
    }

    return i;
}

// Pertubation feedback into the reduce (compute) shader
// Also contains iteration count, and the orbit used (i.e. rank-1, rank-2, ect)
@group(1) @binding(0)
var flags_tex: texture_storage_2d<r32uint, write>;

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
@group(1) @binding(1)
var<storage, read> orbit_location : array<GpuRefOrbitLocation>;

// Ranked ReferenceOrbits, in ComplexDf format
@group(1) @binding(2)
var<storage, read> rank_one_orbit: array<ComplexDf>;

@group(1) @binding(3)
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

const ITER_MASK: u32        = 0x0000FFFFu;
const ESCAPED_BIT: u32      = 1u << 16u;
const PERTURB_BIT: u32      = 1u << 17u;
const PERTURB_ERR_BIT: u32  = 1u << 18u;
const MAX_ITER_BIT: u32     = 1u << 19u;
const ORBIT_SHIFT: u32      = 20u;

fn record_pix_feedback(pix: vec2<i32>, orbit_idx: u32, it: u32, flags_in: u32) {
    var flags = flags_in | (it & ITER_MASK);
    if (it < uni.max_iter) {
        flags = flags | ESCAPED_BIT;
    }
    else {
        flags = flags | MAX_ITER_BIT;
    }
    if (uni.ref_orb_count > 0) {
        flags = flags | PERTURB_BIT;
        flags = flags | (orbit_idx << ORBIT_SHIFT);
    }

    textureStore(flags_tex, pix, vec4<u32>(flags, 0u, 0u, 0u));
}

const BETA = 4.0; // Used for dynamic tracking of perturbation error

// -------------------------------
// Mandelbrot Perturbance 
// Inputs:
// 1) orbit/tile index (into reference orbit texture)
// 2) delta_c (from ref_c to pixel)
// Outputs:
// 1) Iteration count
// -------------------------------
fn mandelbrot_perturb(orbit_idx: u32, delta_c: ComplexDf) -> vec2<u32> {
    var dz = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var i: u32 = 0u;
    var ref_i: u32 = 0u;
    var flags: u32 = 0u;
    let max_i = uni.max_iter;
    let max_ref_iters = orbit_location[orbit_idx].max_ref_iters;

    for (i = 0u; i < max_i; i = i + 1u) {
        // Load reference orbit Z_n
        let Z = load_ref_orbit(orbit_idx, ref_i);
        ref_i += 1;

        // λ_n = 2 * Z_n
        let lambda = cdf_add(Z, Z);

        // dz_{n+1} = λ_n * dz_n + dz_n^2 + Δc
        let dz2 = cdf_mul(dz, dz);
        dz = cdf_add(
            cdf_add(cdf_mul(lambda, dz), dz2),
            delta_c
         );

        // Absolute z for escape testing
        let z = cdf_add(Z, dz);

        // Standard bailout
        let mag2_z = cdf_mag2(z);
        if (mag2_z > 16.0) {
            break;
        }

        if ((uni.render_flags & GLITCH_FIX) != 0) {
            let mag2_dz = cdf_mag2(dz);
            let abs_z = max((abs(z.r.hi) + abs(z.r.lo)), (abs(z.i.hi) + abs(z.i.lo))) * BETA;

            if (abs_z < mag2_dz || (ref_i == max_ref_iters)) {
                // Apply Zhuoran's glitch-fix, which resets where Z taken
                // as we continue iterating on DeltaZn.
                dz = z;
                ref_i = 0;
            }
        }
    }

    return vec2<u32>(i, flags);
}

//
// Color bind groups and logic
//
@group(2) @binding(0)
var palette_tex: texture_2d<f32>;

@group(2) @binding(1)
var palette_sampler: sampler;

fn palette_lookup(t: f32) -> vec3<f32> {
    return textureSample(palette_tex, palette_sampler,
        vec2<f32>(t * uni.palette_frequency + uni.palette_offset, 0.5)).rgb;
}

// -------------------------------
// Fullscreen triangle VS
// -------------------------------
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos: vec2<f32>;
    switch (vid) {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>( 3.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0,  3.0); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }
    return vec4<f32>(pos, 0.0, 1.0);
}

// -------------------------------
// Fragment shader
// -------------------------------
@fragment
fn fs_main(@builtin(position) coords: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(i32(coords.x), i32(coords.y));
    var it: u32 = 0u;
    var flags: u32 = 0u;
    let orbit_idx: u32 = 0;
    var c = build_c_from_scene(pix);

    // Check if there is a (qualified) reference orbit available and,
    // use the perturbance path, if found.
    if (uni.ref_orb_count > 0) {
        // Attempt perturbance with the 1st qualified orbit
        let delta_c = build_delta_c_from_orbit_location(pix, orbit_idx);
        let p_res = mandelbrot_perturb(orbit_idx, delta_c);

        // if error was detected, try to perturb with the next qualified ref orbit.
        //orbit_idx += 1;
        //if (((p_res.y & PERTURB_ERR_BIT) != 0u) && uni.ref_orb_count > orbit_idx) {}
        it = p_res.x;
        flags = p_res.y;
        c = delta_c;
    }
    else {
        it = mandelbrot(c);
    }

    // Color logic
    var t = f32(it) / f32(uni.max_iter);
    t = pow(t, uni.palette_gamma);
    var color = palette_lookup(t);

    if ((uni.render_flags & DEBUG_COLORING) != 0) {
        color = vec3<f32>(t, t*t, pow(t, 0.5));
    }

    if (it == uni.max_iter) {
        // If in the set, color black
        color = vec3(0.0, 0.0, 0.0);
    }

    // Record per-pixel flags for the reduce/compute shader
    record_pix_feedback(pix, orbit_idx, it, flags);

    // -- DEBUG --
    if (   pix.x == i32(f32(uni.screen_width) * 0.5)
        && pix.y == i32(f32(uni.screen_height) * 0.5) ) {
        debug_out.center_x_hi = c.r.hi;
        debug_out.center_x_lo = c.r.lo;
        debug_out.center_y_hi = c.i.hi;
        debug_out.center_y_lo = c.i.lo;
        debug_out.max_iters = uni.max_iter;
        debug_out.iter = it;
        debug_out.t = t;
    }

    return vec4<f32>(color, 1.0);
}

struct DebugOut {
    center_x_hi:        f32,
    center_x_lo:        f32,
    center_y_hi:        f32,
    center_y_lo:        f32,
    max_iters:          u32,
    iter:               u32,
    t:                  f32,
};

@group(3) @binding(0)
var<storage, read_write> debug_out: DebugOut;
