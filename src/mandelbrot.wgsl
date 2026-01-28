struct DebugOut {
    c_ref_re_hi: f32,
    c_ref_re_lo: f32,
    c_ref_im_hi: f32,
    c_ref_im_lo: f32,
    delta_c_re_hi: f32,
    delta_c_re_lo: f32,
    delta_c_im_hi: f32,
    delta_c_im_lo: f32,
    orbit_idx:     u32,
    orbit_meta_ref_len: u32,
    perturb_escape_seq: u32,
    last_valid_i: u32,
    abs_i: u32,
    last_valid_z_re_hi: f32,
    last_valid_z_re_lo: f32,
    last_valid_z_im_hi: f32,
    last_valid_z_im_lo: f32,
};

@group(3) @binding(0)
var<storage, read_write> debug_out: DebugOut;

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
    pix_dx_hi:          f32,
    pix_dx_lo:          f32,
    pix_dy_hi:          f32,
    pix_dy_lo:          f32,
    screen_width:       f32,
    screen_height:      f32,
    screen_tile_size:   f32,
    max_iter:           u32,
    ref_orb_len:        u32,
    ref_orb_count:      u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

struct FrameFeedback {
    max_lambda_re_hi:   f32,
    max_lambda_re_lo:   f32,
    max_lambda_im_hi:   f32,
    max_lambda_im_lo:   f32,
    max_delta_z_re_hi:  f32,
    max_delta_z_re_lo:  f32,
    max_delta_z_im_hi:  f32,
    max_delta_z_im_lo:  f32,
    escape_ratio:       f32,
};
@group(1) @binding(0) var<storage, read_write> frame_fb: FrameFeedback;

// ----------------------------
// Reference orbit from ScoutEngine
// ----------------------------
@group(2) @binding(0)
var ref_orbit_tex : texture_2d<f32>;

fn load_ref_orbit(orbit_idx: u32, iter: u32) -> ComplexDf {
    let base_y = i32(orbit_idx * 4u);
    let x = i32(iter);

    let re_hi = textureLoad(ref_orbit_tex, vec2<i32>(x, base_y + 0), 0).x;
    let re_lo = textureLoad(ref_orbit_tex, vec2<i32>(x, base_y + 1), 0).x;
    let im_hi = textureLoad(ref_orbit_tex, vec2<i32>(x, base_y + 2), 0).x;
    let im_lo = textureLoad(ref_orbit_tex, vec2<i32>(x, base_y + 3), 0).x;

    return ComplexDf(Df(re_hi, re_lo), Df(im_hi, im_lo));
}

struct OrbitMeta {
    ref_len: u32,        // how many Z_n are valid
    escape_index: u32,   // or 0xFFFFFFFF if None
    flags: u32,          // bitmask (future use)
    pad: u32,            // 16-byte alignment
};

@group(2) @binding(1)
var<storage, read> orbit_meta : array<OrbitMeta>;

// Screen-space tile → orbit slot
@group(2) @binding(2)
var tile_orbit_index_tex : texture_2d<u32>;

fn load_tile_orbit_index(pix: vec2<i32>) -> u32 {
    let tile_x = pix.x / i32(uni.screen_tile_size);
    let tile_y = pix.y / i32(uni.screen_tile_size);

    return textureLoad(
        tile_orbit_index_tex,
        vec2<i32>(tile_x, tile_y),
        0
    ).x;
}

@group(2) @binding(3)
var last_valid_i_tex: texture_storage_2d<r32uint, write>;

@group(2) @binding(4)
var orbit_idx_tex:    texture_storage_2d<r32uint, write>;

@group(2) @binding(5)
var flags_tex:        texture_storage_2d<r32uint, write>;

fn record_perturb_feedback(pix: vec2<i32>, 
                last_valid_i: u32, orbit_idx: u32, perturb_flags: u32) {
    textureStore(last_valid_i_tex, pix, vec4<u32>(last_valid_i, 0u, 0u, 0u));
    textureStore(orbit_idx_tex,    pix, vec4<u32>(orbit_idx, 0u, 0u, 0u));
    textureStore(flags_tex,        pix, vec4<u32>(perturb_flags, 0u, 0u, 0u));
}

// ---------- Build c from integer pixel offsets using CPU-provided pix_dx/pix_dy ----------
fn build_c_from_frag(pix: vec2<i32>) -> ComplexDf {
    // integer center (half window). Compute half in f32 then cast to i32 is OK here because width is pixel count.
    let half_w: i32 = i32(uni.screen_width * 0.5);
    let half_h: i32 = i32(uni.screen_height * 0.5);

    let dx_i: i32 = pix.x - half_w;
    let dy_i: i32 = pix.y - half_h;

    let dx_df = df_from_i32(dx_i);
    let dy_df = df_from_i32(dy_i);

    // load pix vectors supplied by CPU (each is a Df)
    let pix_dx = Df(uni.pix_dx_hi, uni.pix_dx_lo);
    let pix_dy = Df(uni.pix_dy_hi, uni.pix_dy_lo);

    // offset = dx*pix_dx + dy*pix_dy
    let off_x = df_mul(dx_df, pix_dx);
    let off_y = df_mul(dy_df, pix_dy);

    let center_x = Df(uni.center_x_hi, uni.center_x_lo);
    let center_y = Df(uni.center_y_hi, uni.center_y_lo);

    let c = ComplexDf(df_add(center_x, off_x), df_add(center_y, off_y)); 
    return c;
}


// -------------------------------
// Mandelbrot iteration using DF arithmetic.
// Returns iteration count (u32).
// -------------------------------
fn mandelbrot_df_from_z(z: ComplexDf, c: ComplexDf) -> u32 {
    var zx: Df = z.r;
    var zy: Df = z.i;
    var i: u32 = 0u;
    let max_i: u32 = uni.max_iter;

    loop {
        // compute squares and cross product (all double-float)
        let zx2 = df_mul(zx, zx);      // zx*zx
        let zy2 = df_mul(zy, zy);      // zy*zy
        let zxy = df_mul(zx, zy);      // zx*zy

        // real = zx2 - zy2 + c.r
        let real_part = df_add(df_sub(zx2, zy2), c.r);

        // imag = 2*zx*zy + c.i  -> 2*zxy + c.i
        let imag_part = df_add(df_add(zxy, zxy), c.i);

        // update z
        zx = real_part;
        zy = imag_part;

        // Bailout
        let mag2 = df_mag2_upper(zx, zy);
        if (mag2 > 16.0) {
            break;
        }

        i = i + 1u;
        if (i >= max_i) { break; }
    }

    return i;
}

const K = 0.25;
const validity_radius2 = 0.01; // or 0.001 to be stricter
const PERTURB_DELTA_C_LIMIT = 1e-3;

// OrbitMeta Flags From CPU
const ORBIT_USABLE: u32 = 0x00000010u;   // Derived orbit meta flag

// Perturbance feedback flags
const PERTURB_ATTEMPTED = 1u << 0; // perturbation path taken
const PERTURB_VALID     = 1u << 1; // perturbation stayed valid to user max_iter
const PERTURB_COLLAPSED = 1u << 2; // |dz| exceeded validity radius
const PERTURB_ESCAPED   = 1u << 3; // escaped during perturb
const MAX_ITER_REACHED  = 1u << 4; // User max_iter reached
const ABSOLUTE_FALLBACK = 1u << 5; // required absolute continuation
const ABSOLUTE_ESCAPED  = 1u << 6; // escaped during absolute iteration

fn mandelbrot_perturb(orbit_idx: u32, c: ComplexDf, delta_c: ComplexDf, pix: vec2<i32>) -> vec3<u32> {
    var dz = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var i: u32 = 0u;
    let max_i = uni.max_iter;
    var flags: u32 = 0u;

    let scale = Df(uni.scale_hi, uni.scale_lo);

    // Track last iteration where perturbation was valid
    var last_valid_i: u32 = 0u;
    var last_valid_l = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var last_valid_dz = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var last_valid_z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var abs_i: u32 = 0u;

    loop {
        // Load reference orbit Z_n
        let Z = load_ref_orbit(orbit_idx, i);

        // λ_n = 2 * Z_n
        let lambda = cdf_add(Z, Z);

        // dz_{n+1} = λ_n * dz_n + Δc
        dz = cdf_add(cdf_mul(lambda, dz), delta_c);

        // Absolute z for escape testing
        let z = cdf_add(Z, dz);

        // Standard bailout
        if (df_mag2_upper(z.r, z.i) > 16.0) {
            flags |= PERTURB_ESCAPED;
            break;
        }

        // Perturbation validity collapse
        if (df_mag2_upper(dz.r, dz.i) > validity_radius2) {
            flags |= PERTURB_COLLAPSED;
            break;
        }

        i = i + 1u;
        last_valid_i = i;
        last_valid_l = lambda;
        last_valid_dz = dz;
        last_valid_z = z;
        flags |= PERTURB_ATTEMPTED;
        
        if (i >= max_i) { 
            flags = flags | PERTURB_VALID | MAX_ITER_REACHED;
            break; 
        }
    }

    if (i < max_i) {
        flags |= ABSOLUTE_FALLBACK;
        // Continue ABSOLUTE from last valid
        abs_i = mandelbrot_df_from_z(last_valid_z, c); 
        i = last_valid_i + abs_i;

        if (i >= max_i) { 
            flags |= MAX_ITER_REACHED;
        } else {
            flags |= ABSOLUTE_ESCAPED;
        }
    }

    record_perturb_feedback(pix, last_valid_i, orbit_idx, flags);

    // -- DEBUG --
    if (pix.x == i32(uni.screen_width * 0.5) && pix.y == i32(uni.screen_height * 0.5)) {
        debug_out.perturb_escape_seq = flags;
        debug_out.last_valid_i = last_valid_i;
        debug_out.abs_i = abs_i;
        debug_out.last_valid_z_re_hi = last_valid_z.r.hi;
        debug_out.last_valid_z_re_lo = last_valid_z.r.lo;
        debug_out.last_valid_z_im_hi = last_valid_z.i.hi;
        debug_out.last_valid_z_im_lo = last_valid_z.i.lo;
        frame_fb.max_lambda_re_hi = last_valid_l.r.hi;
        frame_fb.max_lambda_re_lo = last_valid_l.r.lo;
        frame_fb.max_lambda_im_hi = last_valid_l.i.hi;
        frame_fb.max_lambda_im_lo = last_valid_l.i.lo;
        frame_fb.max_delta_z_re_hi = last_valid_dz.r.hi;
        frame_fb.max_delta_z_re_lo = last_valid_dz.r.lo;
        frame_fb.max_delta_z_im_hi = last_valid_dz.i.hi;
        frame_fb.max_delta_z_im_lo = last_valid_dz.i.lo;
    }

    return vec3<u32>(i, last_valid_i, abs_i);
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

    // Build complex point c with DF precision using pixel offsets.
    let c = build_c_from_frag(pix);
    var it: u32 = 0u;
    var t: f32 = 0.0;
    var color = vec3<f32>(0.0, 0.0, 0.0);

    var c_ref = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var delta_c = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));

    // lookup an orbit index for use with orbit reference atlas texture,
    // by way of determining the screen-space tile for this pixel
    let orbit_idx = load_tile_orbit_index(pix);
    var orb_meta = OrbitMeta(0u, 0u, 0u, 0u);
    var use_perturbation = false;

    // If the orbit index is out-of-bounds with our current atlas, then 
    // there is no valid reference orbit for the pixel and perturbation
    // cannot be used.
    if (orbit_idx < uni.ref_orb_count) {
        c_ref = load_ref_orbit(orbit_idx, 1u);
        delta_c = cdf_sub(c, c_ref);

        orb_meta = orbit_meta[orbit_idx];
        if ((orb_meta.flags & ORBIT_USABLE) != 0u) {
            let delta_c_mag = df_mag2_upper(delta_c.r, delta_c.i);
            let scale2 = uni.scale_hi * uni.scale_hi;

            let delta_c_ok =
                delta_c_mag < PERTURB_DELTA_C_LIMIT * scale2;

            let orbit_len_ok =
                orb_meta.ref_len >= (uni.max_iter >> 1u);

            if (delta_c_ok && orbit_len_ok) {
                use_perturbation = true;
            }
        }
    }

    if (use_perturbation) {
        let p_res = mandelbrot_perturb(orbit_idx, c, delta_c, pix);
        it = p_res.x;

        let pt = f32(p_res.y) / f32(uni.max_iter);
        let at = f32(p_res.z) / f32(uni.max_iter);
        let tt = f32(p_res.x) / f32(uni.max_iter);
        color = vec3<f32>(pt, at, tt);

        // -- DEBUG --
        if (pix.x == i32(uni.screen_width * 0.5) && pix.y == i32(uni.screen_height * 0.5)) {
            debug_out.c_ref_re_hi = c_ref.r.hi;
            debug_out.c_ref_re_lo = c_ref.r.lo;
            debug_out.c_ref_im_hi = c_ref.i.hi;
            debug_out.c_ref_im_lo = c_ref.i.lo;
            debug_out.delta_c_re_hi = delta_c.r.hi;
            debug_out.delta_c_re_lo = delta_c.r.lo;
            debug_out.delta_c_im_hi = delta_c.i.hi;
            debug_out.delta_c_im_lo = delta_c.i.lo;
            debug_out.orbit_idx = orbit_idx;
            debug_out.orbit_meta_ref_len = orb_meta.ref_len;
        }
    } else {
        var z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
        it = mandelbrot_df_from_z(z, c);
        
        t = f32(it) / f32(uni.max_iter);
        color = vec3<f32>(t, t*t, pow(t, 0.5));

        // -- DEBUG --
        if (pix.x == i32(uni.screen_width * 0.5) && pix.y == i32(uni.screen_height * 0.5)) {
            // Reset debug outs that mandelbrot_perturb will later overwrite (if used)
            debug_out.perturb_escape_seq = 0u;
            debug_out.last_valid_i = 0u;
            debug_out.abs_i = 0u;
            debug_out.orbit_idx = 0u;
            debug_out.orbit_meta_ref_len = 0u;
            debug_out.perturb_escape_seq = 0u;
            debug_out.c_ref_re_hi = 0.0;
            debug_out.c_ref_re_lo = 0.0;
            debug_out.c_ref_im_hi = 0.0;
            debug_out.c_ref_im_lo = 0.0;
            debug_out.delta_c_re_hi = 0.0;
            debug_out.delta_c_re_lo = 0.0;
            debug_out.delta_c_im_hi = 0.0;
            debug_out.delta_c_im_lo = 0.0;
            debug_out.last_valid_z_re_hi = 0.0;
            debug_out.last_valid_z_re_lo = 0.0;
            debug_out.last_valid_z_im_hi = 0.0;
            debug_out.last_valid_z_im_lo = 0.0;
        }
    }

    //t = f32(it) / f32(uni.max_iter);
    //color = vec3<f32>(t, t*t, pow(t, 0.5));

    if (it == uni.max_iter) {
        // inside set -> black
        color = vec3<f32>(0.0, 0.0, 0.0);
    }

    return vec4<f32>(color, 1.0);
}

