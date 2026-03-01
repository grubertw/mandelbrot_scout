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
    screen_width:       f32,
    screen_height:      f32,
    max_iter:           u32,
    tile_count:         u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

fn build_c_from_scene(pix: vec2<f32>) -> ComplexDf {
    let half_w = uni.screen_width * 0.5;
    let half_h = uni.screen_height * 0.5;

    let dx_i = i32(pix.x) - i32(half_w);
    let dy_i = i32(half_h) - i32(pix.y); // y-axis increases downward, so must flip!

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

// ----------------------------
// Anchor Orbit Tiles from ScoutEngine
// ----------------------------
// The Reference (anchor) Orbit data
// Iteration count is on the x-azis (i.e. the RefOrb's Vec<Complex>'s index)
// OrbitId/TileId is in the y-axis
// Note that each orbit takes 4x y-indicies (i.e. complex re(hi+lo) + im(hi+lo))
@group(1) @binding(0)
var ref_orbit_tex : texture_2d<f32>;

struct GpuTileGeometry {
    anchor_c_ref_re_hi:             f32,
    anchor_c_ref_re_lo:             f32,
    anchor_c_ref_im_hi:             f32,
    anchor_c_ref_im_lo:             f32,
    center_offset_re_hi:            f32,
    center_offset_re_lo:            f32,
    center_offset_im_hi:            f32,
    center_offset_im_lo:            f32,
    tile_screen_min_x:              f32,
    tile_screen_min_y:              f32,
    tile_screen_max_x:              f32,
    tile_screen_max_y:              f32,
};

// The 'geommetry' of each OrbitId/TileId. 
// Constructing Delta-C from c_ref (the orbit seed), and in a way that avoids loss
// of precision is the name of the game. tile_delta_from_anchor is computed in
// high-presision FIRST, and preserved the entire time the orbit remains anchor 
// for the tile. The CPU ALSO pre-computes the distance from scene-center to 
// tile-center in high-precision, and only at the very end is rounded back down 
// into tile-pixel mapping.
@group(1) @binding(1)
var<storage, read> tile_geometry : array<GpuTileGeometry>;

// Pertubation feedback into the reduce (compute) shader
@group(1) @binding(2)
var flags_tex: texture_storage_2d<r32uint, write>; 

// Test if pixel is in a Tile
fn pixel_in_tile(pix: vec2<f32>, tile: GpuTileGeometry) -> bool {
    return pix.x >= tile.tile_screen_min_x &&
           pix.x <= tile.tile_screen_max_x &&
           pix.y >= tile.tile_screen_min_y &&
           pix.y <= tile.tile_screen_max_y;
}

// Find the index of the tile, based on pixel (fragment) cordinates
fn find_tile_index(pix: vec2<f32>) -> i32 {
    for (var i: u32 = 0u; i < uni.tile_count; i = i + 1u) {
        let tile = tile_geometry[i];
        if (pixel_in_tile(pix, tile)) {
            return i32(i);
        }
    }
    return -1;
}

// delta_c - along with Zref-n0 - are needed to start pertubation.
fn build_delta_c_from_tile_geometry(pix: vec2<f32>, tile_idx: u32) -> ComplexDf {
    let half_w = uni.screen_width * 0.5;
    let half_h = uni.screen_height * 0.5;
    let scale = Df(uni.scale_hi, uni.scale_lo);

    let tile = tile_geometry[tile_idx];

    let dx_i = i32(pix.x) - i32(half_w);
    let dy_i = i32(half_h) - i32(pix.y);

    let dx_df = df_from_i32(dx_i);
    let dy_df = df_from_i32(dy_i);

    let off_x = df_mul(dx_df, scale);
    let off_y = df_mul(dy_df, scale);

    let delta_from_center_to_anchor = ComplexDf(
        Df(tile.center_offset_re_hi, tile.center_offset_re_lo),
        Df(tile.center_offset_im_hi, tile.center_offset_im_lo)
    );

    let delta_c = ComplexDf(
        df_add(delta_from_center_to_anchor.r, off_x),
        df_add(delta_from_center_to_anchor.i, off_y)
    );

    return delta_c;
}

fn load_ref_orbit(tile_idx: u32, it: u32) -> ComplexDf {
    let base_y = tile_idx * 4u;

    let z_re_hi = textureLoad(ref_orbit_tex, vec2<i32>(i32(it), i32(base_y + 0u)), 0).x;
    let z_re_lo = textureLoad(ref_orbit_tex, vec2<i32>(i32(it), i32(base_y + 1u)), 0).x;
    let z_im_hi = textureLoad(ref_orbit_tex, vec2<i32>(i32(it), i32(base_y + 2u)), 0).x;
    let z_im_lo = textureLoad(ref_orbit_tex, vec2<i32>(i32(it), i32(base_y + 3u)), 0).x;

    return ComplexDf(Df(z_re_hi, z_re_lo), Df(z_im_hi, z_im_lo));
}

const ITER_MASK: u32        = 0x0000FFFFu;
const ESCAPED_BIT: u32      = 1u << 16u;
const PERTURB_BIT: u32      = 1u << 17u;
const PERTURB_ERR_BIT: u32  = 1u << 18u;
const MAX_ITER_BIT: u32     = 1u << 19u;
const TILE_SHIFT: u32       = 20u;

fn record_pix_feedback(pix: vec2<f32>, tile_idx: i32, it: u32, flags_in: u32) {
    var flags = flags_in | (it & ITER_MASK);
    if (it < uni.max_iter) {
        flags = flags | ESCAPED_BIT;
    }
    else {
        flags = flags | MAX_ITER_BIT;
    }
    if (tile_idx >= 0) {
        flags = flags | PERTURB_BIT;
        flags = flags | (u32(tile_idx) << TILE_SHIFT);
    }

    textureStore(flags_tex, vec2<i32>(i32(pix.x), i32(pix.y)), vec4<u32>(flags, 0u, 0u, 0u));
}

const BETA = 0.1; // Used for dynamic tracking of perturbation error

// -------------------------------
// Mandelbrot Perturbance 
// Inputs:
// 1) orbit/tile index (into reference orbit texture)
// 2) delta_c (from pixel to uni.center to tile.anchor_ref_c)
// Outputs:
// 1) Iteration count
// -------------------------------
fn mandelbrot_perturb(tile_idx: u32, delta_c: ComplexDf) -> vec2<u32> {
    var dz = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var i: u32 = 0u;
    var flags: u32 = 0u;
    let max_i = uni.max_iter;

    for (i = 0u; i < max_i; i = i + 1u) {
        // Load reference orbit Z_n
        let Z = load_ref_orbit(tile_idx, i);

        // Track error dynamicly by comparing |dz_n| to |Z|
        let mag2_dz = cdf_mag2(dz);
        let mag2_Z = cdf_mag2(Z);
        if (mag2_dz > BETA * mag2_Z) {
            flags |= PERTURB_ERR_BIT;
        }

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
        if (cdf_mag2(z) > 16.0) {
            break;
        }
    }

    return vec2<u32>(i, flags);
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
    let pix = vec2<f32>(coords.x, coords.y);
    var it: u32 = 0u;
    var flags: u32 = 0u;
    var c = build_c_from_scene(pix);
    var r_valid_crossed = false;

    // Check if the pixel/frag co-ordinates fall within a tile.
    // If they do, use perurbation. If not, use absoluate mandelbrot
    // calculation.
    let tile_idx = find_tile_index(pix);
    if (tile_idx >= 0) {
        let delta_c = build_delta_c_from_tile_geometry(pix, u32(tile_idx));

        let p_res = mandelbrot_perturb(u32(tile_idx), delta_c);
        it = p_res.x;
        flags = p_res.y;
        c = delta_c;
    }
    else {
        it = mandelbrot(c);
    }

    // Color logic
    var t = f32(it) / f32(uni.max_iter);
    var color = vec3<f32>(t, t*t, pow(t, 0.5));

    if (it == uni.max_iter) {
        // If in the set, color black
        color = vec3(0.0, 0.0, 0.0);
    }
    // TileId/idx diagnostic (blue)
    if (tile_idx >= 0 && uni.tile_count > 0u) {
        let tile_c = f32(tile_idx) / f32(uni.tile_count);
        color = mix(color, vec3(0.0, 0.0, tile_c), 0.3);
    }

    // Record per-pixel flags for the reduce/compute shader
    record_pix_feedback(pix, tile_idx, it, flags);

    // -- DEBUG --
    if (   i32(pix.x) == i32(uni.screen_width * 0.5) 
        && i32(pix.y) == i32(uni.screen_height * 0.5) ) {
        debug_out.center_x_hi = c.r.hi;
        debug_out.center_x_lo = c.r.lo;
        debug_out.center_y_hi = c.i.hi;
        debug_out.center_y_lo = c.i.lo;
        debug_out.scale_hi = uni.scale_hi;
        debug_out.scale_lo = uni.scale_lo;
        debug_out.screen_width = uni.screen_width;
        debug_out.screen_height = uni.screen_height;
        debug_out.tile_count = uni.tile_count;
        debug_out.tile_idx = tile_idx;
    }

    return vec4<f32>(color, 1.0);
}

struct DebugOut {
    center_x_hi:        f32,
    center_x_lo:        f32,
    center_y_hi:        f32,
    center_y_lo:        f32,
    scale_hi:           f32,
    scale_lo:           f32,
    screen_width:       f32,
    screen_height:      f32,
    tile_count:         u32,
    tile_idx:           i32,
};

@group(2) @binding(0)
var<storage, read_write> debug_out: DebugOut;
