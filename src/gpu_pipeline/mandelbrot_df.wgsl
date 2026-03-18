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

// -------------------------------
// Mandelbrot iteration using DF arithmetic.
// Returns iteration count (u32).
// -------------------------------
fn mandelbrot(c: ComplexDf) -> vec3<f32> {
    var z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));

    var i: u32 = 0u;
    let max_i: u32 = uni.max_iter;
    var mag2: f32 = 0.0;
    // Extra tracking for DE/surface normals
    var escaped: bool = false;
    var escape_mag2: f32 = 0.0;
    var dz = vec2<f32>(0.0, 0.0);
    var extra: u32 = 0u; // Iterate a few past bailout for better DE values.
    // For stripe-averaging
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;

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
            let zr = z.r.hi;
            let zi = z.i.hi;
            let dz_new = vec2<f32>(
                2.0 * (zr * dz.x - zi * dz.y) + 1.0,
                2.0 * (zr * dz.y + zi * dz.x)
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
            escaped = true;
            // Make extra iterations past escape for better DE approximation.
            if (extra >= 2) {
                break;
            }
            extra += 1;
        }

        if (!escaped) {
            escape_mag2 = mag2;
        }
    }

    var fi = f32(i - extra);
    // Replace with smooth iters if enabled
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        let safe_mag2 = max(escape_mag2, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag2) * 0.5), 0.0, f32(max_i));
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
    return vec3<f32>(fi, distance, stripe_avg);
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

fn record_pix_feedback(pix: vec2<i32>, orbit_idx: u32, it: u32) {
    var flags: u32 = it & ITER_MASK;
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

// -------------------------------
// Mandelbrot Perturbance 
//
// -------------------------------
fn mandelbrot_perturb(delta_c: ComplexDf) -> vec3<f32> {
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
    var escaped: bool = false;
    var escape_mag2: f32 = 0.0;
    var extra: u32 = 0u; // Iterate a few past bailout for better DE values.
    // For stripe-averaging
    var stripe_sum: f32 = 0.0;
    var stripe_count: f32 = 0.0;

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
            let zr = z.r.hi;
            let zi = z.i.hi;
            let dz_new = vec2<f32>(
                2.0 * (zr * dzdc.x - zi * dzdc.y) + 1.0,
                2.0 * (zr * dzdc.y + zi * dzdc.x)
            );
            dzdc = dz_new;
        }

        // Absolute z for escape testing
        z = cdf_add(Z, dz);

        if ((uni.render_flags & USE_STRIPES) != 0) {
            // for stripe-averaging
            let angle = atan2(zi, zr);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
            stripe_sum += stripe;
            stripe_count += 1.0;
        }

        // Standard bailout
        mag2 = cdf_mag2(z);
        if (mag2 > 1024.0) {
             escaped = true;
            // Make extra iterations past escape for better DE approximation.
            if (extra >= 2) {
                break;
            }
            extra += 1;
        }

        if (!escaped) {
            escape_mag2 = mag2;
        }
    }

    var fi = f32(i - extra);
    // For smooth iterations
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        let safe_mag2 = max(escape_mag2, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag2) * 0.5), 0.0, f32(max_i));
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
    return vec3<f32>(fi, distance, stripe_avg);
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

fn calculate_surface_normals(pix: vec2<i32>) -> vec3<f32> {
    var eps: i32 = i32(uni.neighbor_scale_multiplier);
    if (eps <= 0) {
        eps = 1;
    }

    let c_dx_r = build_c_from_scene(pix + vec2<i32>(eps, 0));
    let c_dx_l = build_c_from_scene(pix - vec2<i32>(eps, 0));
    let c_dy_t = build_c_from_scene(pix + vec2<i32>(0, eps));
    let c_dy_b = build_c_from_scene(pix - vec2<i32>(0, eps));
    let dx = mandelbrot(c_dx_r).z - mandelbrot(c_dx_l).z;
    let dy = mandelbrot(c_dy_t).z - mandelbrot(c_dy_b).z;

    eps *= 2;
    let grad = vec3<f32>(dx, dy, uni.scale_hi * f32(eps));
    return normalize(grad);
}

fn calculate_surface_normals_perturb(delta_c: ComplexDf) -> vec3<f32> {
    var eps = Df(uni.scale_hi, uni.scale_lo);
    eps = df_mul(eps, df_from_f32(uni.neighbor_scale_multiplier));

    let dx_offset = ComplexDf(eps, df_from_f32(0.0));
    let dy_offset = ComplexDf(df_from_f32(0.0), eps);
    let dx =
        mandelbrot_perturb(cdf_add(delta_c, dx_offset)).z -
        mandelbrot_perturb(cdf_sub(delta_c, dx_offset)).z;
    let dy =
        mandelbrot_perturb(cdf_add(delta_c, dy_offset)).z -
        mandelbrot_perturb(cdf_sub(delta_c, dy_offset)).z;

    var eps2: f32 = eps.hi * 2.0;
    let grad = vec3<f32>(dx, dy, uni.scale_hi * eps2);
    return normalize(grad);
}

// Calculate light direction from azimuth and elevation, in degres
fn light_dir(az_deg: f32, el_deg: f32) -> vec3<f32> {
    let az = radians(az_deg);
    let el = radians(el_deg);

    let x = cos(el) * cos(az);
    let y = cos(el) * sin(az);
    let z = sin(el);

    return normalize(vec3<f32>(x,y,z));
}

fn calculate_diffuse(d: f32, N: vec3<f32>) -> f32 {
    let key_light  = light_dir(uni.key_light_azimuth, uni.key_light_elevation);
    let fill_light = light_dir(uni.fill_light_azimuth, uni.fill_light_elevation);
    var diffuse = uni.ambient_intensity;

    if ((uni.render_flags & ENABLE_KEY_LIGHT) != 0) {
        diffuse += uni.key_light_intensity * max(dot(N, key_light),0.0);
    }
    if ((uni.render_flags & ENABLE_FILL_LIGHT) != 0) {
        diffuse += uni.fill_light_intensity * max(dot(N, fill_light),0.0);
    }

    let view = vec3<f32>(0.0, 0.0, 1.0);

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
    var it: f32 = 0.0;
    var nu: f32 = 0.0;
    var d: f32 = 0.0;
    var stripe_avg: f32 = 0.0;
    var f_max_iters = f32(uni.max_iter);
    var N = vec3<f32>(0.0, 0.0, 0.0);

    // Check if there is a (qualified) reference orbit available and,
    // use the perturbance path, if found.
    if (uni.ref_orb_count > 0) {
        // Attempt perturbance with the 1st qualified orbit
        let delta_c = build_delta_c_from_orbit_location(pix, 0u);
        let mres = mandelbrot_perturb(delta_c);
        it = mres.x;
        nu = mres.y;
        d = mres.z;
        stripe_avg = mres.w;

        if ((uni.render_flags & USE_DE) != 0) {
            N = calculate_surface_normals_perturb(delta_c);
        }
    }
    else {
        let c = build_c_from_scene(pix);
        let mres = mandelbrot(c);
        it = mres.x;
        nu = mres.y;
        d = mres.z;
        stripe_avg = mres.w;

        if ((uni.render_flags & USE_DE) != 0) {
            N = calculate_surface_normals(pix);
        }
    }

    // Color logic
    var t = it / f_max_iters;
    if ((uni.render_flags & SMOOTH_COLORING) != 0) {
        t = nu / (f_max_iters + 1.0);
    }

    if ((uni.render_flags & USE_STRIPES) != 0) {
        t = mix(t, t + (stripe_avg - 0.5), uni.stripe_strength);
    }

    t = pow(t, uni.palette_gamma);
    var color = palette_lookup(t);

    if ((uni.render_flags & DEBUG_COLORING) != 0) {
        color = vec3<f32>(t, t*t, pow(t, 0.5));
    }

    if ((uni.render_flags & USE_DE) != 0) {
        var diffuse = calculate_diffuse(d, N);
        color *= diffuse;
    }

    d /= uni.scale_hi; // Glow and AO seem to work better with scale as a factor

    if ((uni.render_flags & ENABLE_GLOW) != 0) {
        let glow = 1.0 / (1.0 + d * pow(2.0, uni.distance_multiplier));
        color += glow * uni.glow_intensity;
    }

    if ((uni.render_flags & ENABLE_AO) != 0) {
        // AO lighting
        let ao = exp(-d * pow(2.0, uni.distance_multiplier));
        color *= mix(uni.ao_darkness, 1.0, ao);
    }

    if (it >= f_max_iters) {
        // If in the set, color black
        color = vec3(0.0, 0.0, 0.0);
    }

    // Record per-pixel flags for the reduce/compute shader
    record_pix_feedback(pix, 0u, u32(it));

    // -- DEBUG --
    if (   pix.x == i32(f32(uni.screen_width) * 0.5)
        && pix.y == i32(f32(uni.screen_height) * 0.5) ) {
        debug_out.max_iters = uni.max_iter;
        debug_out.iter = u32(it);
        debug_out.nu_iter = nu;
        debug_out.distance = d;
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
    nu_iter:            f32,
    distance:           f32,
    t:                  f32,
};

@group(3) @binding(0)
var<storage, read_write> debug_out: DebugOut;
