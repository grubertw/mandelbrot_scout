// -------------------------------
// Uniforms
// -------------------------------
struct Uniforms {
    center_x:           f32,
    center_y:           f32,
    scale:              f32,
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
const RECALC_FRACTAL: u32   = 1u << 11;

// From mandelbrot calculation shader (per-pixel calculations)
@group(0) @binding(1)
var calc_tex: texture_2d<f32>;

@group(0) @binding(2)
var palette_tex: texture_2d<f32>;

@group(0) @binding(3)
var palette_sampler: sampler;

fn palette_lookup(t: f32) -> vec3f {
    return textureSample(palette_tex, palette_sampler,
        vec2f(t * uni.palette_frequency + uni.palette_offset, 0.5)).rgb;
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

// -------------------------------
// Fullscreen triangle VS
// -------------------------------
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    var pos: vec2f;
    switch (vid) {
        case 0u: { pos = vec2f(-1.0, -1.0); }
        case 1u: { pos = vec2f( 3.0, -1.0); }
        case 2u: { pos = vec2f(-1.0,  3.0); }
        default: { pos = vec2f(0.0, 0.0); }
    }
    return vec4f(pos, 0.0, 1.0);
}

// -------------------------------
// Fragment shader
// -------------------------------
@fragment
fn fs_main(@builtin(position) coords: vec4f) -> @location(0) vec4f {
    let pix = vec2i(i32(coords.x), i32(coords.y));
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

    t = pow(t, uni.palette_gamma);
    var color = palette_lookup(t);

    if ((uni.render_flags & DEBUG_COLORING) != 0) {
        color = vec3f(t, t*t, pow(t, 0.5));
    }

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

    return vec4f(color, 1.0);
}
