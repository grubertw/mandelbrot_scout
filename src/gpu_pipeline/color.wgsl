// -------------------------------
// Uniforms
// -------------------------------
struct Uniforms {
    center_x:           f32,
    center_y:           f32,
    scale:              f32,
    max_iter:           u32,
    ref_orb_count:      u32,
    perturb_err_thresh: f32,
    grid_feedback_scale:f32,
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
    stripe_density:     f32,
    stripe_strength:    f32,
    stripe_gamma:       f32,
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
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

// Render flags from scene uniforms
const DEBUG_COLORING: u32   = 1u << 0;
const SHOW_GLITCH: u32      = 1u << 1;
const SMOOTH_COLORING: u32  = 1u << 2;
const USE_DE: u32           = 1u << 3;
const USE_STRIPES: u32      = 1u << 4;
const ENABLE_GLOW: u32      = 1u << 5;
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

fn palette_lookup(t_in: f32) -> vec3f {
    let tex_width = f32(uni.palette_tex_width);
    let palette_len = f32(uni.palette_len);

    var t = t_in * palette_len * uni.palette_cycles + (uni.palette_offset * palette_len);
    t = t % tex_width;

    let i0 = i32(floor(t));
    let i1 = i32(min(f32(i0 + 1), tex_width - 1.0));

    let frac = fract(t);
    //let frac = smoothstep(0.0, 1.0, fract(x));
    //let frac_smooth = frac * frac * (3.0 - 2.0 * frac); // smootherstep-lite

    // Fetch texels
    let c0 = textureLoad(palette_tex, vec2<i32>(i0, 0), 0).rgb;
    let c1 = textureLoad(palette_tex, vec2<i32>(i1, 0), 0).rgb;

    // Linear interpolation
    return mix(c0, c1, frac);

    // Gamma correct interpolation
    //let c0_lin = pow(c0, vec3f(2.2));
    //let c1_lin = pow(c1, vec3f(2.2));
    //let mixed = mix(c0_lin, c1_lin, frac);
    //return pow(mixed, vec3f(1.0 / 2.2));
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

    eps *= 2;
    let grad = vec3f(dx, dy, uni.scale * f32(eps));
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

    if ((uni.render_flags & USE_STRIPES) != 0) {
        t = mix(t, t + (stripe_avg - 0.5), uni.stripe_strength);
    }

    // User-costimizable mapping functions on normalized escape-time - i.e color scalar
    t = map_color_scalar(t);
    
    var color = palette_lookup(t);

    if ((uni.render_flags & DEBUG_COLORING) != 0) {
        color = vec3f(t, t*t, pow(t, 0.5));
    }

    color = pow(color, vec3f(1.0 / uni.palette_gamma));

    if ((uni.render_flags & USE_DE) != 0) {
        var diffuse = calculate_diffuse(d, N);
        color *= diffuse;
    }

    d /= uni.scale; // Glow and AO seem to work better with scale as a factor

    if ((uni.render_flags & ENABLE_GLOW) != 0) {
        let glow = 1.0 / (1.0 + d * pow(2.0, uni.distance_multiplier));
        color += glow * uni.glow_intensity;
    }

    if ((uni.render_flags & ENABLE_AO) != 0) {
        // AO lighting
        let ao = exp(-d * pow(2.0, uni.distance_multiplier));
        color *= mix(uni.ao_darkness, 1.0, ao);
    }

    if (it >= uni.max_iter) {
        // If in the set, color black
        color = vec3f(0.0, 0.0, 0.0);
    }

    textureStore(render_tex, pix, vec4f(color, 1.0));
}
